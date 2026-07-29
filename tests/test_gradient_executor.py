import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_gradient_bucket_plan import load_bucket_plan
from tests.test_gradient_fragment_planner import descriptor


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAD_SYNC_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync"


def load_executor():
    bucket_plan, planner = load_bucket_plan()
    evidence_name = "megatron.grad_sync.runtime_evidence"
    evidence_path = GRAD_SYNC_DIR / "runtime_evidence.py"
    evidence_spec = importlib.util.spec_from_file_location(
        evidence_name, evidence_path
    )
    evidence_module = importlib.util.module_from_spec(evidence_spec)
    sys.modules[evidence_name] = evidence_module
    evidence_spec.loader.exec_module(evidence_module)
    full_name = "megatron.grad_sync.executor"
    path = GRAD_SYNC_DIR / "executor.py"
    assert path.exists(), "gradient executor is missing"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module, bucket_plan, planner


def make_fragment(
    planner, fragment_id, logical_id, group_id, numel, order, dtype="float32"
):
    return planner.CanonicalGradientFragmentSpec(
        fragment_id=fragment_id,
        logical_param_id=logical_id,
        canonical_shard_id=order,
        process_group_id=group_id,
        grad_dtype=dtype,
        reduction_semantics="sum",
        logical_region_key=logical_id,
        global_regions=((0, numel),),
        numel=numel,
        semantic_key=(order,),
        optimizer_semantic_id=logical_id,
        include_in_optimizer_norm=True,
        norm_owner_rank=0,
    )


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def handle(self, process_group_id):
        self.calls.append(process_group_id)
        return "cached-" + process_group_id


class GradientExecutorTest(unittest.TestCase):
    def test_runtime_evidence_observes_four_to_two_collectives_and_equal_payload(self):
        executor, bucket_plan, planner = load_executor()
        groups = (
            planner.ProcessGroupSpec("pg-a", (0,), "gradient_sync"),
            planner.ProcessGroupSpec("pg-b", (0,), "gradient_sync"),
        )
        fragments = (
            make_fragment(planner, "f0", "p0", "pg-a", 2, 0),
            make_fragment(planner, "f1", "p1", "pg-b", 2, 1),
            make_fragment(planner, "f2", "p2", "pg-a", 2, 2),
            make_fragment(planner, "f3", "p3", "pg-b", 2, 3),
        )
        segment = planner.TensorSegmentSpec(0, 2, (2,), 0, 2, 2)
        bindings = tuple(
            planner.LocalFragmentBindingSpec(
                0, fragment.fragment_id, fragment.logical_param_id, (segment,), 1.0
            )
            for fragment in fragments
        )
        source = planner.CanonicalFragmentPlan(
            groups, fragments, ((0, bindings),), "fragment-hash"
        )

        reports = {}
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("tensor", "block"):
                plan = bucket_plan.build_gradient_sync_plan(
                    source, mode, 1024, "gpt", "manifest", "topology"
                )
                gradients = {
                    f"p{index}": torch.tensor([index + 1.0, index + 2.0])
                    for index in range(4)
                }
                runtime_bindings = tuple(
                    SimpleNamespace(
                        logical_param_id=logical_id,
                        main_grad=value,
                        device=value.device,
                    )
                    for logical_id, value in gradients.items()
                )
                descriptors = tuple(
                    SimpleNamespace(logical_param_id=f"p{index}", partition_dim=0)
                    for index in range(4)
                )

                class Registry(FakeRegistry):
                    group_count = 2
                    creation_count = 2

                runner = executor.GradientBucketExecutor(
                    plan,
                    0,
                    descriptors,
                    runtime_bindings,
                    Registry(),
                    lambda unused_payload, unused_handle: None,
                    runtime_evidence_dir=Path(tmp) / mode,
                )
                for _ in range(2):
                    for index, value in enumerate(gradients.values()):
                        value.copy_(torch.tensor([index + 1.0, index + 2.0]))
                    runner.execute()
                reports[mode] = json.loads(
                    (Path(tmp) / mode / "rank-00000.json").read_text(
                        encoding="utf-8"
                    )
                )

        self.assertEqual(reports["tensor"]["iterations"][0]["collective_count"], 4)
        self.assertEqual(reports["block"]["iterations"][0]["collective_count"], 2)
        self.assertEqual(
            reports["tensor"]["iterations"][0]["packed_payload_sha256"],
            reports["block"]["iterations"][0]["packed_payload_sha256"],
        )
        self.assertEqual(
            reports["tensor"]["iterations"][0]["payload_bytes"],
            reports["block"]["iterations"][0]["payload_bytes"],
        )
        self.assertTrue(reports["block"]["scratch_stable"])
        self.assertTrue(reports["block"]["process_group_count_stable"])
        self.assertTrue(reports["block"]["process_group_creation_count_stable"])

    def test_replicated_matrix_packs_the_whole_gradient(self):
        executor_module, bucket_plan, planner = load_executor()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        descriptors = tuple(
            descriptor(
                descriptor_cls, "position_embeddings", rank, rank, 1, 0,
                (2, 3), (2, 3), False, -1,
            )
            for rank in range(2)
        )
        fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
        )
        gradients = (torch.ones(2, 3), torch.full((2, 3), 2.0))
        executors = tuple(
            executor_module.GradientBucketExecutor(
                plan,
                rank,
                (descriptors[rank],),
                (
                    SimpleNamespace(
                        logical_param_id="position_embeddings",
                        parameter=torch.zeros(2, 3),
                        main_grad=gradients[rank],
                        device=torch.device("cpu"),
                    ),
                ),
                FakeRegistry(),
                lambda payload, unused_handle: payload.mul_(3.0),
            )
            for rank in range(2)
        )

        for item in executors:
            item.execute()

        self.assertTrue(torch.equal(gradients[0], torch.full((2, 3), 3.0)))
        self.assertTrue(torch.equal(gradients[1], torch.full((2, 3), 6.0)))

    def test_tp_one_reconstructs_all_shards_while_tp_max_writes_owned_shard(self):
        executor, bucket_plan, planner = load_executor()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        descriptors = (
            descriptor(
                descriptor_cls,
                "weight",
                0,
                0,
                1,
                0,
                (8, 2),
                (8, 2),
                True,
                0,
            ),
            descriptor(
                descriptor_cls,
                "weight",
                1,
                1,
                2,
                0,
                (8, 2),
                (4, 2),
                True,
                0,
            ),
            descriptor(
                descriptor_cls,
                "weight",
                2,
                1,
                2,
                1,
                (8, 2),
                (4, 2),
                True,
                0,
            ),
        )
        fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
        )
        rank0 = torch.arange(16, dtype=torch.float32).view(8, 2)
        rank1 = torch.arange(8, dtype=torch.float32).view(4, 2) + 100
        rank2 = torch.arange(8, dtype=torch.float32).view(4, 2) + 200
        gradients = (rank0, rank1, rank2)
        groups = {item.process_group_id: item.ranks for item in plan.process_groups}
        condition = threading.Condition()
        pending = {}

        def all_reduce_sum(payload, handle):
            with condition:
                entries = pending.setdefault(handle, [])
                entries.append(payload)
                if len(entries) == len(groups[handle]):
                    total = torch.stack([item.clone() for item in entries]).sum(dim=0)
                    for item in entries:
                        item.copy_(total)
                    condition.notify_all()
                else:
                    condition.wait_for(lambda: len(entries) == len(groups[handle]))

        class Registry:
            def handle(self, process_group_id):
                return process_group_id

        runners = []
        for rank, main_grad in enumerate(gradients):
            runtime_binding = SimpleNamespace(
                logical_param_id="weight",
                main_grad=main_grad,
                device=main_grad.device,
            )
            runners.append(
                executor.GradientBucketExecutor(
                    plan,
                    rank,
                    (descriptors[rank],),
                    (runtime_binding,),
                    Registry(),
                    all_reduce_sum,
                )
            )

        errors = []

        def run(runner):
            try:
                runner.execute()
            except Exception as exc:  # pragma: no cover - surfaced after join
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(runner,)) for runner in runners]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        expected_first = torch.arange(8, dtype=torch.float32).view(4, 2) * 2 + 100
        expected_second = (
            torch.arange(8, 16, dtype=torch.float32).view(4, 2)
            + torch.arange(8, dtype=torch.float32).view(4, 2)
            + 200
        )
        self.assertTrue(torch.equal(rank0[:4], expected_first))
        self.assertTrue(torch.equal(rank0[4:], expected_second))
        self.assertTrue(torch.equal(rank1, expected_first))
        self.assertTrue(torch.equal(rank2, expected_second))

    def test_pack_sum_normalize_unpack_handles_non_last_axis_and_stride(self):
        executor, bucket_plan, planner = load_executor()
        group = planner.ProcessGroupSpec("pg", (0, 1), "gradient_sync")
        first = make_fragment(planner, "f-even", "weight", "pg", 6, 0)
        second = make_fragment(planner, "f-odd", "weight", "pg", 6, 1)
        even_segments = (
            planner.TensorSegmentSpec(0, 1, (4, 3), 0, 1, 3),
            planner.TensorSegmentSpec(2, 1, (4, 3), 2, 3, 3),
        )
        odd_segments = (
            planner.TensorSegmentSpec(1, 1, (4, 3), 1, 2, 3),
            planner.TensorSegmentSpec(3, 1, (4, 3), 3, 4, 3),
        )
        bindings = (
            planner.LocalFragmentBindingSpec(0, "f-even", "weight", even_segments, 1.0),
            planner.LocalFragmentBindingSpec(0, "f-odd", "weight", odd_segments, 1.0),
        )
        fragment_plan = planner.CanonicalFragmentPlan(
            (group,), (first, second), ((0, bindings),), "fragment-hash"
        )
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "block", 1024, "gpt", "manifest", "topology"
        )
        main_grad = torch.arange(12, dtype=torch.float32).view(4, 3)
        runtime_binding = SimpleNamespace(
            logical_param_id="weight",
            main_grad=main_grad,
            device=main_grad.device,
        )
        descriptor = SimpleNamespace(logical_param_id="weight", partition_dim=0)
        registry = FakeRegistry()
        packed = []

        def all_reduce_sum(payload, handle):
            self.assertEqual(handle, "cached-pg")
            packed.append(payload.clone())
            payload.mul_(2)

        runner = executor.GradientBucketExecutor(
            plan, 0, (descriptor,), (runtime_binding,), registry, all_reduce_sum
        )
        runner.execute(normalization_scale=0.5)

        self.assertEqual(registry.calls, ["pg"])
        self.assertTrue(
            torch.equal(
                packed[0],
                torch.tensor([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11], dtype=torch.float32),
            )
        )
        self.assertTrue(torch.equal(main_grad, torch.arange(12).view(4, 3)))

    def test_replicated_pack_scales_duplicates_and_unused_gradients_are_zero(self):
        executor, bucket_plan, planner = load_executor()
        group = planner.ProcessGroupSpec("pg", (0, 1), "gradient_sync")
        fragment = make_fragment(planner, "f", "bias", "pg", 4, 0, "float16")
        segment = planner.TensorSegmentSpec(0, 4, (4,), 0, 4, 4)
        binding = planner.LocalFragmentBindingSpec(0, "f", "bias", (segment,), 0.5)
        source = planner.CanonicalFragmentPlan(
            (group,), (fragment,), ((0, (binding,)),), "fragment-hash"
        )
        plan = bucket_plan.build_gradient_sync_plan(
            source, "tensor", 1024, "gpt", "manifest", "topology"
        )
        main_grad = torch.zeros(4, dtype=torch.float16)
        runtime_binding = SimpleNamespace(
            logical_param_id="bias", main_grad=main_grad, device=main_grad.device
        )
        descriptor = SimpleNamespace(logical_param_id="bias", partition_dim=0)

        def all_reduce_sum(payload, unused_handle):
            self.assertEqual(payload.dtype, torch.float16)
            payload.mul_(2)

        runner = executor.GradientBucketExecutor(
            plan, 0, (descriptor,), (runtime_binding,), FakeRegistry(), all_reduce_sum
        )
        runner.execute()
        self.assertTrue(torch.equal(main_grad, torch.zeros_like(main_grad)))

        main_grad.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.float16))
        runner.execute()
        self.assertTrue(
            torch.equal(main_grad, torch.tensor([1, 2, 3, 4], dtype=torch.float16))
        )

    def test_two_iterations_reuse_scratch_without_touching_short_bucket_tail(self):
        executor, bucket_plan, planner = load_executor()
        group = planner.ProcessGroupSpec("pg", (0,), "gradient_sync")
        large = make_fragment(planner, "large", "large-param", "pg", 8, 0)
        small = make_fragment(planner, "small", "small-param", "pg", 4, 1)
        large_segment = planner.TensorSegmentSpec(0, 8, (8,), 0, 8, 8)
        small_segment = planner.TensorSegmentSpec(0, 4, (4,), 0, 4, 4)
        bindings = (
            planner.LocalFragmentBindingSpec(0, "large", "large-param", (large_segment,), 1.0),
            planner.LocalFragmentBindingSpec(0, "small", "small-param", (small_segment,), 1.0),
        )
        source = planner.CanonicalFragmentPlan(
            (group,), (large, small), ((0, bindings),), "fragment-hash"
        )
        plan = bucket_plan.build_gradient_sync_plan(
            source, "tensor", 1024, "gpt", "manifest", "topology"
        )
        large_grad = torch.arange(8, dtype=torch.float32)
        small_grad = torch.arange(4, dtype=torch.float32) + 10
        runtime_bindings = (
            SimpleNamespace(logical_param_id="large-param", main_grad=large_grad, device=large_grad.device),
            SimpleNamespace(logical_param_id="small-param", main_grad=small_grad, device=small_grad.device),
        )
        descriptors = (
            SimpleNamespace(logical_param_id="large-param", partition_dim=0),
            SimpleNamespace(logical_param_id="small-param", partition_dim=0),
        )
        seen = []

        def all_reduce_sum(payload, unused_handle):
            seen.append((payload.numel(), payload.clone()))
            payload.add_(100)

        runner = executor.GradientBucketExecutor(
            plan, 0, descriptors, runtime_bindings, FakeRegistry(), all_reduce_sum
        )
        for _ in range(2):
            large_grad.copy_(torch.arange(8, dtype=torch.float32))
            small_grad.copy_(torch.arange(4, dtype=torch.float32) + 10)
            runner.execute()
            self.assertTrue(torch.equal(runner.scratch("float32")[4:], torch.arange(4, 8) + 100))

        self.assertEqual([numel for numel, unused in seen], [8, 4, 8, 4])

    def test_parameterless_rank_allocates_no_scratch_and_invalid_binding_has_context(self):
        executor, bucket_plan, planner = load_executor()
        empty_plan = bucket_plan.GradientSyncPlan(
            version=1,
            sync_mode="tensor",
            bucket_cap_bytes=1024,
            model_impl="gpt",
            op_manifest_hash="manifest",
            process_groups=(),
            process_group_index_by_id=(),
            canonical_fragments=(),
            local_fragment_bindings_by_rank=(),
            global_bucket_order=(),
            max_scratch_numel_by_rank_and_dtype=(),
            exact_payload_bytes_by_rank=(),
            padding_payload_bytes=0,
            global_topology_hash="topology",
            global_manifest_hash="manifest",
            global_fragment_hash="fragment",
            global_plan_hash="plan",
        )
        runner = executor.GradientBucketExecutor(
            empty_plan, 7, (), (), FakeRegistry(), lambda payload, group: None
        )
        runner.execute()
        self.assertEqual(runner.scratch_dtypes, ())

        group = planner.ProcessGroupSpec("pg", (0,), "gradient_sync")
        fragment = make_fragment(planner, "missing-fragment", "missing-param", "pg", 1, 0)
        segment = planner.TensorSegmentSpec(0, 1, (1,), 0, 1, 1)
        binding = planner.LocalFragmentBindingSpec(
            0, "missing-fragment", "missing-param", (segment,), 1.0
        )
        source = planner.CanonicalFragmentPlan(
            (group,), (fragment,), ((0, (binding,)),), "fragment-hash"
        )
        plan = bucket_plan.build_gradient_sync_plan(
            source, "tensor", 1024, "gpt", "manifest", "topology"
        )
        with self.assertRaisesRegex(
            ValueError, "bucket-.*rank 0.*missing-param.*missing-fragment"
        ):
            executor.GradientBucketExecutor(
                plan, 0, (), (), FakeRegistry(), lambda payload, group: None
            )

        main_grad = torch.zeros(1)
        wrong_device_binding = SimpleNamespace(
            logical_param_id="missing-param",
            main_grad=main_grad,
            device=torch.device("cuda:0"),
        )
        descriptor = SimpleNamespace(
            logical_param_id="missing-param", partition_dim=0
        )
        with self.assertRaisesRegex(
            ValueError, "bucket-.*rank 0.*missing-param.*device mismatch"
        ):
            executor.GradientBucketExecutor(
                plan,
                0,
                (descriptor,),
                (wrong_device_binding,),
                FakeRegistry(),
                lambda payload, group: None,
            )


if __name__ == "__main__":
    unittest.main()
