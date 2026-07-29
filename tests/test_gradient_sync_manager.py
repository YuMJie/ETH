import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_gradient_executor import load_executor
from tests.test_gradient_fragment_planner import descriptor
from tests.test_optimizer_semantics import load_optimizer_semantics


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAD_SYNC_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync"


def load_manager():
    executor, bucket_plan, planner = load_executor()
    load_optimizer_semantics()
    full_name = "megatron.grad_sync.manager"
    path = GRAD_SYNC_DIR / "manager.py"
    assert path.exists(), "gradient synchronization manager is missing"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module, planner, bucket_plan


class GradientSyncManagerTest(unittest.TestCase):
    def test_initialization_gathers_metadata_validates_hash_and_caches_groups(self):
        manager_module, planner, unused_bucket_plan = load_manager()
        runtime_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(runtime_tmp.cleanup)
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        rank0 = descriptor(
            descriptor_cls, "bias", 0, 0, 1, 0, (4,), (4,), False, -1
        )
        rank1 = descriptor(
            descriptor_cls, "bias", 1, 1, 1, 0, (4,), (4,), False, -1
        )
        main_grad = torch.arange(4, dtype=torch.float32)
        runtime_binding = SimpleNamespace(
            logical_param_id="bias",
            parameter=torch.zeros_like(main_grad),
            main_grad=main_grad,
            device=main_grad.device,
        )
        gather_calls = []

        def all_gather_object(outputs, value):
            gather_calls.append(value)
            if isinstance(value, tuple):
                outputs[:] = [(rank0,), (rank1,)]
            else:
                outputs[:] = [value, value]

        group_calls = []

        def create_group(ranks):
            group_calls.append(ranks)
            return "handle-" + "-".join(map(str, ranks))

        collective_calls = []

        def all_reduce_sum(payload, handle):
            collective_calls.append((payload.clone(), handle))

        group_gather_calls = []

        def all_gather_group_object(outputs, value, handle):
            group_gather_calls.append((value, handle))
            outputs[:] = [value, value]

        topology = SimpleNamespace(
            world_size=2,
            stages=(
                SimpleNamespace(pipeline_id=0),
                SimpleNamespace(pipeline_id=1),
            ),
            fingerprint="topology",
        )
        manager = manager_module.initialize_gradient_sync(
            local_descriptors=(rank0,),
            local_runtime_bindings=(runtime_binding,),
            topology=topology,
            model_impl="gpt",
            manifest_hash="manifest",
            sync_mode="tensor",
            bucket_cap_bytes=1024,
            global_rank=0,
            all_gather_object=all_gather_object,
            all_gather_group_object=all_gather_group_object,
            create_group=create_group,
            all_reduce_sum=all_reduce_sum,
            control_device="cpu",
            runtime_evidence_dir=runtime_tmp.name,
        )

        self.assertEqual(len(gather_calls), 2)
        self.assertIsInstance(gather_calls[0], tuple)
        self.assertEqual(gather_calls[1], manager.plan.global_plan_hash)
        self.assertEqual(group_calls, [(0, 1)])
        self.assertEqual(manager.registry.group_count, 1)
        self.assertEqual(len(group_gather_calls), 1)
        manifest, handle = group_gather_calls[0]
        self.assertEqual(handle, "handle-0-1")
        self.assertEqual(manifest[0][0], manager.plan.global_bucket_order[0].semantic_hash)
        self.assertEqual(manifest[0][1], manager.plan.global_bucket_order[0].exact_numel)
        self.assertEqual(manifest[0][2], manager.plan.global_bucket_order[0].grad_dtype)
        self.assertEqual(manifest[0][3], len(manager.plan.global_bucket_order[0].placements))

        manager.execute()
        self.assertEqual(len(collective_calls), 1)
        self.assertEqual(collective_calls[0][1], "handle-0-1")
        runtime_report = Path(runtime_tmp.name) / "rank-00000.json"
        self.assertTrue(runtime_report.exists())

        broadcasts = []
        manager.synchronize_parameters(
            lambda payload, source, handle: broadcasts.append(
                (payload.clone(), source, handle)
            )
        )
        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(broadcasts[0][1:], (0, "handle-0-1"))
        self.assertIsNotNone(manager.optimizer_semantics)

        optimizer_state = {"exp_avg": torch.ones_like(main_grad)}
        manager.synchronize_optimizer_state(
            SimpleNamespace(
                state_for_model_parameter=lambda unused_parameter: optimizer_state
            ),
            lambda payload, source, handle: broadcasts.append(
                (payload.clone(), source, handle)
            ),
            lambda outputs, value, unused_handle: outputs.__setitem__(
                slice(None), [value, value]
            ),
        )
        self.assertEqual(len(broadcasts), 2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "plan.json"
            manager_module.dump_gradient_sync_plan(manager.plan, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["global_plan_hash"], manager.plan.global_plan_hash)
        self.assertEqual(
            payload["global_fragment_hash"], manager.plan.global_fragment_hash
        )
        self.assertEqual(payload["padding_payload_bytes"], 0)
        self.assertEqual(len(payload["process_groups"]), 1)
        self.assertEqual(len(payload["global_bucket_order"]), 1)
        self.assertNotIn("tensor_data", payload)

    def test_hash_mismatch_fails_before_group_creation(self):
        manager_module, unused_planner, unused_bucket_plan = load_manager()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        rank0 = descriptor(
            descriptor_cls, "bias", 0, 0, 1, 0, (1,), (1,), False, -1
        )
        rank1 = descriptor(
            descriptor_cls, "bias", 1, 1, 1, 0, (1,), (1,), False, -1
        )
        main_grad = torch.zeros(1)
        runtime_binding = SimpleNamespace(
            logical_param_id="bias", main_grad=main_grad, device=main_grad.device
        )

        def all_gather_object(outputs, value):
            if isinstance(value, tuple):
                outputs[:] = [(rank0,), (rank1,)]
            else:
                outputs[:] = [value, "different-plan-hash"]

        topology = SimpleNamespace(
            world_size=2,
            stages=(
                SimpleNamespace(pipeline_id=0),
                SimpleNamespace(pipeline_id=1),
            ),
            fingerprint="topology",
        )
        group_calls = []
        with self.assertRaisesRegex(ValueError, "gradient plan hash mismatch"):
            manager_module.initialize_gradient_sync(
                local_descriptors=(rank0,),
                local_runtime_bindings=(runtime_binding,),
                topology=topology,
                model_impl="gpt",
                manifest_hash="manifest",
                sync_mode="tensor",
                bucket_cap_bytes=1024,
                global_rank=0,
                all_gather_object=all_gather_object,
                all_gather_group_object=lambda outputs, value, group: None,
                create_group=lambda ranks: group_calls.append(ranks),
                all_reduce_sum=lambda payload, handle: None,
            )
        self.assertEqual(group_calls, [])

    def test_group_bucket_manifest_mismatch_fails_before_executor(self):
        manager_module, unused_planner, unused_bucket_plan = load_manager()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        rank0 = descriptor(
            descriptor_cls, "bias", 0, 0, 1, 0, (1,), (1,), False, -1
        )
        rank1 = descriptor(
            descriptor_cls, "bias", 1, 1, 1, 0, (1,), (1,), False, -1
        )
        main_grad = torch.zeros(1)
        runtime_binding = SimpleNamespace(
            logical_param_id="bias", main_grad=main_grad, device=main_grad.device
        )

        def all_gather_object(outputs, value):
            if isinstance(value, tuple):
                outputs[:] = [(rank0,), (rank1,)]
            else:
                outputs[:] = [value, value]

        def all_gather_group_object(outputs, value, unused_group):
            outputs[:] = [value, (("different", 1, "float32", 1),)]

        topology = SimpleNamespace(
            world_size=2,
            stages=(
                SimpleNamespace(pipeline_id=0),
                SimpleNamespace(pipeline_id=1),
            ),
            fingerprint="topology",
        )
        with self.assertRaisesRegex(ValueError, "bucket manifest mismatch.*pg-"):
            manager_module.initialize_gradient_sync(
                local_descriptors=(rank0,),
                local_runtime_bindings=(runtime_binding,),
                topology=topology,
                model_impl="gpt",
                manifest_hash="manifest",
                sync_mode="tensor",
                bucket_cap_bytes=1024,
                global_rank=0,
                all_gather_object=all_gather_object,
                all_gather_group_object=all_gather_group_object,
                create_group=lambda ranks: "handle",
                all_reduce_sum=lambda payload, handle: None,
            )


if __name__ == "__main__":
    unittest.main()
