import importlib.util
import sys
import threading
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_gradient_bucket_plan import load_bucket_plan
from tests.test_gradient_fragment_planner import descriptor
from tests.test_training_loss_contract import load_optimizer_module


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "eth"
    / "runtime"
    / "megatron"
    / "grad_sync"
    / "optimizer_semantics.py"
)


def load_optimizer_semantics():
    bucket_plan, planner = load_bucket_plan()
    assert MODULE_PATH.exists(), "canonical optimizer semantics module is missing"
    full_name = "megatron.grad_sync.optimizer_semantics"
    spec = importlib.util.spec_from_file_location(full_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module, bucket_plan, planner


def build_plan(bucket_plan, planner):
    descriptor_cls = sys.modules[
        "megatron.grad_sync.metadata"
    ].SerializableGradientDescriptor
    descriptors = (
        descriptor(
            descriptor_cls, "weight", 0, 0, 1, 0,
            (4,), (4,), True, 0,
        ),
        descriptor(
            descriptor_cls, "weight", 1, 1, 2, 0,
            (4,), (2,), True, 0,
        ),
        descriptor(
            descriptor_cls, "weight", 2, 1, 2, 1,
            (4,), (2,), True, 0,
        ),
    )
    fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
    plan = bucket_plan.build_gradient_sync_plan(
        fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
    )
    return plan, descriptors


class _Collectives:
    def __init__(self, world_size):
        self.world_size = world_size
        self.condition = threading.Condition()
        self.reductions = {}
        self.broadcasts = {}

    def sum(self, payload, operation):
        self._reduce(payload, operation, torch.sum)

    def maximum(self, payload, operation):
        self._reduce(payload, operation, torch.max)

    def _reduce(self, payload, operation, reducer):
        with self.condition:
            entries = self.reductions.setdefault(operation, [])
            entries.append(payload)
            if len(entries) == self.world_size:
                result = reducer(torch.stack([item.clone() for item in entries]))
                for item in entries:
                    item.copy_(result)
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: len(entries) == self.world_size)

    def broadcast(
        self, payload, source_rank, unused_group_id, group_ranks,
        fragment_id, current_rank
    ):
        key = (tuple(group_ranks), fragment_id)
        with self.condition:
            entries = self.broadcasts.setdefault(key, {})
            entries[current_rank] = payload
            if len(entries) == len(group_ranks):
                source = entries[source_rank].clone()
                for item in entries.values():
                    item.copy_(source)
                self.condition.notify_all()
            else:
                self.condition.wait_for(
                    lambda: len(entries) == len(group_ranks)
                )


class _ObjectGather:
    def __init__(self, world_size):
        self.world_size = world_size
        self.condition = threading.Condition()
        self.entries = []

    def __call__(self, outputs, value):
        with self.condition:
            self.entries.append((outputs, value))
            if len(self.entries) == self.world_size:
                values = [item[1] for item in self.entries]
                for target, unused_value in self.entries:
                    target[:] = values
                self.condition.notify_all()
            else:
                self.condition.wait_for(
                    lambda: len(self.entries) == self.world_size
                )


class OptimizerSemanticsTest(unittest.TestCase):
    def test_parameter_pack_mismatch_reports_logical_layout(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        plan, descriptors = build_plan(bucket_plan, planner)
        semantics = module.CanonicalOptimizerSemantics(
            plan,
            0,
            (descriptors[0],),
            (
                SimpleNamespace(
                    logical_param_id="weight",
                    parameter=torch.zeros(4),
                    main_grad=torch.zeros(4),
                    device=torch.device("cpu"),
                ),
            ),
            "cpu",
        )
        fragment = plan.canonical_fragments[0]
        binding = semantics._binding_specs[fragment.fragment_id]
        bad_fragment = SimpleNamespace(
            fragment_id=fragment.fragment_id,
            numel=fragment.numel + 1,
        )

        with self.assertRaisesRegex(
            ValueError, r"logical_param_id weight.*segments=.*expected 3"
        ):
            semantics._pack_fragment(torch.zeros(4), binding, bad_fragment)

    def test_training_wires_parameter_sync_and_world_statistics(self):
        source = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "training.py"
        ).read_text(encoding="utf-8")

        self.assertGreaterEqual(
            source.count("gradient_sync_manager.synchronize_parameters("), 2
        )
        self.assertGreaterEqual(
            source.count("synchronize_shared_weight_parameters(model)"), 2
        )
        self.assertIn("control_device=control_device", source)
        self.assertIn("set_canonical_gradient_statistics(", source)
        self.assertIn("synchronize_optimizer_state(", source)
        self.assertIn("synchronize_shared_optimizer_state(", source)
        self.assertIn("ReduceOp.SUM", source)
        self.assertIn("ReduceOp.MAX", source)

    def test_shared_weight_transport_is_device_aware_and_validates_splits(self):
        source = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "training.py"
        ).read_text(encoding="utf-8")
        shared_transport = source[
            source.index("def send_shared_tensors"):
            source.index("def initialize_weights_sharing")
        ]

        self.assertNotIn("torch.cuda.current_device()", shared_transport)
        self.assertNotIn("torch.cuda.synchronize()", shared_transport)
        self.assertIn("shared tensor split mismatch", shared_transport)
        self.assertIn("device=shared_tensor[key].device", shared_transport)

    def test_parameterless_control_device_comes_from_launcher(self):
        training = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "training.py"
        ).read_text(encoding="utf-8")
        initialize = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "initialize.py"
        ).read_text(encoding="utf-8")

        self.assertIn("args.distributed_device =", initialize)
        self.assertIn('torch.device("cpu")', initialize)
        self.assertIn("control_device = args.distributed_device", training)
        self.assertIn("if args.fp16 or args.bf16:", training)
        optimizer_source = (
            REPO_ROOT
            / "eth"
            / "runtime"
            / "megatron"
            / "optimizer"
            / "optimizer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.control_device = torch.device(control_device)", optimizer_source)
        self.assertNotIn("self._scale = torch.cuda.FloatTensor", optimizer_source)

    def test_shared_scalar_state_rejects_conflicting_shard_sources(self):
        module, unused_bucket_plan, unused_planner = load_optimizer_semantics()
        key = ("tied", None, "step", "scalar", "torch.float32")

        with self.assertRaisesRegex(ValueError, "conflicting shared optimizer"):
            module.merge_shared_optimizer_state_records(
                [((key, torch.tensor([7.0])),), ((key, torch.tensor([8.0])),)]
            )

        merged = module.merge_shared_optimizer_state_records(
            [((key, torch.tensor([7.0])),), ((key, torch.tensor([7.0])),)]
        )
        self.assertEqual(merged[key].item(), 7.0)

    def test_shared_parameter_state_reassembles_mixed_tp_regions(self):
        module, unused_bucket_plan, unused_planner = load_optimizer_semantics()
        sources = {
            ("tied", ((0, 2),), "exp_avg", "parameter", "torch.float32",
             (4,), 0):
                torch.tensor([10.0, 20.0]),
            ("tied", ((2, 4),), "exp_avg", "parameter", "torch.float32",
             (4,), 0):
                torch.tensor([30.0, 40.0]),
        }

        payload = module.assemble_shared_optimizer_state_payload(
            sources,
            "tied",
            "exp_avg",
            "torch.float32",
            ((0, 4),),
            4,
        )

        self.assertTrue(
            torch.equal(payload, torch.tensor([10.0, 20.0, 30.0, 40.0]))
        )

    def test_shared_gradient_copy_preserves_persistent_main_grad_binding(self):
        module, unused_bucket_plan, unused_planner = load_optimizer_semantics()
        target = torch.zeros(4)
        source = torch.arange(4, dtype=torch.float32)
        identity = id(target)

        result = module.copy_shared_tensor_(target, source, "word_embeddings")

        self.assertIs(result, target)
        self.assertEqual(id(result), identity)
        self.assertTrue(torch.equal(target, source))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            module.copy_shared_tensor_(target, torch.zeros(2), "word_embeddings")

    def test_shared_root_gradient_replicates_after_tied_sync(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        descriptors = tuple(
            descriptor(
                descriptor_cls,
                logical_id,
                rank,
                0 if rank == 0 else 1,
                1,
                0,
                (2,),
                (2,),
                False,
                -1,
                dp_degree=1 if rank == 0 else 2,
                dp_rank=0 if rank == 0 else rank - 1,
                dp_group=(0,) if rank == 0 else (1, 2),
                shared_weight_id="tied",
                shared_weight_root=(logical_id == "embedding"),
            )
            for logical_id in ("embedding", "head")
            for rank in range(3)
        )
        fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
        )
        root_gradients = (
            torch.tensor([9.0, 8.0]),
            torch.zeros(2),
            torch.zeros(2),
        )
        head_gradients = tuple(torch.full((2,), 20.0 + rank) for rank in range(3))
        semantics = []
        for rank in range(3):
            local_descriptors = tuple(
                item for item in descriptors if item.global_rank == rank
            )
            parameters = {
                logical_id: torch.zeros(2)
                for logical_id in ("embedding", "head")
            }
            semantics.append(
                module.CanonicalOptimizerSemantics(
                    plan,
                    rank,
                    local_descriptors,
                    (
                        SimpleNamespace(
                            logical_param_id="embedding",
                            parameter=parameters["embedding"],
                            main_grad=root_gradients[rank],
                            device=torch.device("cpu"),
                        ),
                        SimpleNamespace(
                            logical_param_id="head",
                            parameter=parameters["head"],
                            main_grad=head_gradients[rank],
                            device=torch.device("cpu"),
                        ),
                    ),
                    "cpu",
                )
            )

        collectives = _Collectives(world_size=3)
        threads = [
            threading.Thread(
                target=item.synchronize_shared_root_gradients,
                args=(collectives.broadcast,),
            )
            for item in semantics
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        for gradient in root_gradients:
            self.assertTrue(torch.equal(gradient, torch.tensor([9.0, 8.0])))
        for rank, gradient in enumerate(head_gradients):
            self.assertTrue(
                torch.equal(gradient, torch.full((2,), 20.0 + rank))
            )

    def test_parameter_fragments_broadcast_from_one_canonical_source(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        plan, descriptors = build_plan(bucket_plan, planner)
        parameters = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.zeros(2),
            torch.zeros(2),
        )
        synchronizers = []
        for rank in range(3):
            synchronizers.append(
                module.CanonicalOptimizerSemantics(
                    plan,
                    rank,
                    (descriptors[rank],),
                    (
                        SimpleNamespace(
                            logical_param_id="weight",
                            parameter=parameters[rank],
                            main_grad=torch.zeros_like(parameters[rank]),
                            device=parameters[rank].device,
                        ),
                    ),
                    control_device="cpu",
                )
            )
        collectives = _Collectives(world_size=3)
        threads = [
            threading.Thread(
                target=item.synchronize_parameters,
                args=(collectives.broadcast,),
            )
            for item in synchronizers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(torch.equal(parameters[1], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(parameters[2], torch.tensor([3.0, 4.0])))

    def test_replicated_matrix_parameter_sync_uses_the_whole_tensor(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        descriptors = tuple(
            descriptor(
                descriptor_cls,
                "position_embeddings",
                rank,
                rank,
                1,
                0,
                (2, 3),
                (2, 3),
                False,
                -1,
            )
            for rank in range(2)
        )
        fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
        )
        parameters = (torch.arange(6.0).view(2, 3), torch.zeros(2, 3))
        semantics = tuple(
            module.CanonicalOptimizerSemantics(
                plan,
                rank,
                (descriptors[rank],),
                (
                    SimpleNamespace(
                        logical_param_id="position_embeddings",
                        parameter=parameters[rank],
                        main_grad=torch.zeros_like(parameters[rank]),
                        device=torch.device("cpu"),
                    ),
                ),
                "cpu",
            )
            for rank in range(2)
        )
        collectives = _Collectives(world_size=2)
        threads = [
            threading.Thread(
                target=item.synchronize_parameters,
                args=(collectives.broadcast,),
            )
            for item in semantics
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(torch.equal(parameters[1], parameters[0]))

    def test_canonical_norm_counts_once_and_clips_every_local_replica(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        plan, descriptors = build_plan(bucket_plan, planner)
        gradients = (
            torch.tensor([3.0, 4.0, 0.0, 0.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([0.0, 0.0]),
        )
        semantics = []
        for rank in range(3):
            semantics.append(
                module.CanonicalOptimizerSemantics(
                    plan,
                    rank,
                    (descriptors[rank],),
                    (
                        SimpleNamespace(
                            logical_param_id="weight",
                            parameter=torch.zeros_like(gradients[rank]),
                            main_grad=gradients[rank],
                            device=gradients[rank].device,
                        ),
                    ),
                    control_device="cpu",
                )
            )
        collectives = _Collectives(world_size=3)
        results = [None] * 3

        def run(rank):
            results[rank] = semantics[rank].clip_grad_norm(
                2.0,
                2.0,
                collectives.sum,
                collectives.maximum,
            )

        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(results, [5.0, 5.0, 5.0])
        self.assertTrue(torch.allclose(gradients[0], torch.tensor([1.2, 1.6, 0.0, 0.0])))
        self.assertTrue(torch.allclose(gradients[1], torch.tensor([1.2, 1.6])))
        self.assertTrue(torch.equal(gradients[2], torch.zeros(2)))

        zero_results = [None] * 3
        threads = [
            threading.Thread(
                target=lambda rank=rank: zero_results.__setitem__(
                    rank, semantics[rank].count_zeros(collectives.sum)
                )
            )
            for rank in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(zero_results, [2.0, 2.0, 2.0])

        inf_results = [None] * 3
        threads = [
            threading.Thread(
                target=lambda rank=rank: inf_results.__setitem__(
                    rank,
                    semantics[rank].clip_grad_norm(
                        10.0,
                        float("inf"),
                        collectives.sum,
                        collectives.maximum,
                    ),
                )
            )
            for rank in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        for result in inf_results:
            self.assertAlmostEqual(result, 1.6, places=5)

    def test_optimizer_moments_follow_canonical_parameter_fragments(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        plan, descriptors = build_plan(bucket_plan, planner)
        parameters = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.zeros(2),
            torch.zeros(2),
        )
        states = (
            {"exp_avg": torch.tensor([10.0, 20.0, 30.0, 40.0]),
             "step": torch.tensor(7.0)},
            {"exp_avg": torch.zeros(2), "step": torch.tensor(0.0)},
            {"exp_avg": torch.zeros(2), "step": torch.tensor(0.0)},
        )
        semantics = []
        for rank in range(3):
            semantics.append(
                module.CanonicalOptimizerSemantics(
                    plan,
                    rank,
                    (descriptors[rank],),
                    (
                        SimpleNamespace(
                            logical_param_id="weight",
                            parameter=parameters[rank],
                            main_grad=torch.zeros_like(parameters[rank]),
                            device=parameters[rank].device,
                        ),
                    ),
                    control_device="cpu",
                )
            )
        collectives = _Collectives(world_size=3)
        threads = [
            threading.Thread(
                target=semantics[rank].synchronize_optimizer_state,
                args=(lambda unused_parameter, rank=rank: states[rank],
                      collectives.broadcast),
            )
            for rank in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(
            torch.equal(states[1]["exp_avg"], torch.tensor([10.0, 20.0]))
        )
        self.assertTrue(
            torch.equal(states[2]["exp_avg"], torch.tensor([30.0, 40.0]))
        )
        self.assertEqual(states[1]["step"].item(), 7.0)
        self.assertEqual(states[2]["step"].item(), 7.0)

        group_id = plan.canonical_fragments[0].process_group_id
        manifest = semantics[0].optimizer_state_manifest_for_group(
            group_id, lambda unused_parameter: states[0]
        )
        self.assertEqual(manifest[0][0], plan.canonical_fragments[0].fragment_id)
        self.assertEqual(
            manifest[0][1],
            (("exp_avg", "parameter", "torch.float32"),
             ("step", "scalar", "torch.float32")),
        )

    def test_tied_optimizer_state_uses_shared_semantic_root(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        descriptors = tuple(
            descriptor(
                descriptor_cls,
                logical_id,
                rank,
                rank,
                1,
                0,
                (2,),
                (2,),
                False,
                -1,
                shared_weight_id="tied",
                shared_weight_root=(logical_id == "embedding"),
            )
            for logical_id in ("embedding", "head")
            for rank in range(2)
        )
        fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
        )
        semantics = []
        states = []
        for rank in range(2):
            local_descriptors = tuple(
                item for item in descriptors if item.global_rank == rank
            )
            parameters = {
                logical_id: torch.zeros(2)
                for logical_id in ("embedding", "head")
            }
            bindings = tuple(
                SimpleNamespace(
                    logical_param_id=logical_id,
                    parameter=parameters[logical_id],
                    main_grad=torch.zeros(2),
                    device=torch.device("cpu"),
                )
                for logical_id in ("embedding", "head")
            )
            semantics.append(
                module.CanonicalOptimizerSemantics(
                    plan, rank, local_descriptors, bindings, "cpu"
                )
            )
            states.append(
                {
                    id(parameters["embedding"]): {
                        "exp_avg": torch.tensor([10.0, 20.0]) if rank == 0 else torch.zeros(2),
                        "step": torch.tensor(7.0 if rank == 0 else 0.0),
                    },
                    id(parameters["head"]): {
                        "exp_avg": torch.tensor([90.0, 99.0]),
                        "step": torch.tensor(3.0),
                    },
                }
            )
        gather = _ObjectGather(world_size=2)
        threads = [
            threading.Thread(
                target=semantics[rank].synchronize_shared_optimizer_state,
                args=(
                    lambda parameter, rank=rank: states[rank][id(parameter)],
                    gather,
                    2,
                ),
            )
            for rank in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        for rank in range(2):
            for state in states[rank].values():
                self.assertTrue(
                    torch.equal(state["exp_avg"], torch.tensor([10.0, 20.0]))
                )
                self.assertEqual(state["step"].item(), 7.0)

    def test_parameterless_rank_contributes_zero_statistics(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        plan, unused_descriptors = build_plan(bucket_plan, planner)
        semantics = module.CanonicalOptimizerSemantics(
            plan, 3, (), (), control_device="cpu"
        )
        calls = []

        def identity(payload, operation):
            calls.append((operation, payload.item()))

        norm = semantics.clip_grad_norm(1.0, 2.0, identity, identity)
        zeros = semantics.count_zeros(identity)

        self.assertEqual(norm, 0.0)
        self.assertEqual(zeros, 0.0)
        self.assertEqual(calls, [("l2_norm_sq", 0.0), ("zero_count", 0.0)])

    def test_megatron_optimizer_delegates_norm_and_zero_count(self):
        optimizer_module = load_optimizer_module()
        optimizer = optimizer_module.Float16OptimizerWithFloat16Params.__new__(
            optimizer_module.Float16OptimizerWithFloat16Params
        )
        calls = []

        class Statistics:
            def clip_grad_norm(
                self, max_norm, norm_type, all_reduce_sum, all_reduce_max
            ):
                calls.append(("clip", max_norm, norm_type))
                return 7.0

            def count_zeros(self, all_reduce_sum):
                calls.append(("zeros",))
                return 11.0

        optimizer.set_canonical_gradient_statistics(
            Statistics(), lambda payload, operation: None,
            lambda payload, operation: None,
        )

        self.assertEqual(optimizer.clip_grad_norm(2.0), 7.0)
        self.assertEqual(optimizer.count_zeros(), 11.0)
        self.assertEqual(calls, [("clip", 2.0, 2.0), ("zeros",)])

    def test_canonical_statistics_bind_optimizer_visible_fp32_gradient(self):
        module, bucket_plan, planner = load_optimizer_semantics()
        plan, descriptors = build_plan(bucket_plan, planner)
        model_parameter = torch.zeros(4)
        scaled_model_grad = torch.tensor([30.0, 40.0, 0.0, 0.0])
        optimizer_grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
        semantics = module.CanonicalOptimizerSemantics(
            plan,
            0,
            (descriptors[0],),
            (
                SimpleNamespace(
                    logical_param_id="weight",
                    parameter=model_parameter,
                    main_grad=scaled_model_grad,
                    device=model_parameter.device,
                ),
            ),
            control_device="cpu",
        )
        semantics.bind_optimizer_gradient(model_parameter, optimizer_grad)

        norm = semantics.clip_grad_norm(
            2.0,
            2.0,
            lambda payload, unused_operation: None,
            lambda payload, unused_operation: None,
        )

        self.assertEqual(norm, 5.0)
        self.assertTrue(
            torch.allclose(
                optimizer_grad, torch.tensor([1.2, 1.6, 0.0, 0.0])
            )
        )
        self.assertTrue(
            torch.equal(
                scaled_model_grad, torch.tensor([30.0, 40.0, 0.0, 0.0])
            )
        )

    def test_megatron_optimizer_exposes_state_for_model_parameter(self):
        optimizer_module = load_optimizer_module()
        optimizer = optimizer_module.Float16OptimizerWithFloat16Params.__new__(
            optimizer_module.Float16OptimizerWithFloat16Params
        )
        model_parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer_parameter = torch.nn.Parameter(torch.zeros(2))
        expected_state = {"exp_avg": torch.ones(2)}
        optimizer.optimizer = SimpleNamespace(
            state={optimizer_parameter: expected_state},
            param_groups=[{
                "params": [optimizer_parameter],
                "betas": (0.9, 0.95),
                "eps": 1.0e-8,
                "weight_decay": 0.1,
            }],
        )
        optimizer._model_to_optimizer_param = {
            id(model_parameter): optimizer_parameter
        }

        state = optimizer.state_for_model_parameter(model_parameter)

        self.assertIs(state, expected_state)
        self.assertIs(
            optimizer.optimizer_parameter_for_model_parameter(model_parameter),
            optimizer_parameter,
        )
        self.assertEqual(
            optimizer.hyperparameters_for_model_parameter(model_parameter),
            {
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1.0e-8,
                "weight_decay": 0.1,
            },
        )
        with self.assertRaisesRegex(ValueError, "not owned"):
            optimizer.state_for_model_parameter(torch.nn.Parameter(torch.zeros(1)))

    def test_parameterless_fp16_rank_skips_empty_unscale_but_world_reduces(self):
        optimizer_module = load_optimizer_module()
        optimizer = optimizer_module.Float16OptimizerWithFloat16Params.__new__(
            optimizer_module.Float16OptimizerWithFloat16Params
        )
        optimizer.fp32_from_float16_groups = []
        optimizer.fp32_from_fp32_groups = []
        optimizer.found_inf = torch.tensor(0.0)
        optimizer.grad_scaler = SimpleNamespace(inv_scale=torch.tensor(1.0))
        reductions = []
        world_group = object()
        distributed = SimpleNamespace(
            ReduceOp=SimpleNamespace(MAX="max"),
            group=SimpleNamespace(WORLD=world_group),
            all_reduce=lambda payload, op, group: reductions.append(
                (payload.item(), op, group)
            ),
        )

        with mock.patch.object(
            torch,
            "_amp_foreach_non_finite_check_and_unscale_",
            side_effect=AssertionError("empty gradients must bypass AMP foreach"),
        ), mock.patch.object(
            torch, "distributed", distributed
        ):
            found_inf = optimizer._unscale_main_grads_and_check_for_nan()

        self.assertFalse(found_inf)
        self.assertEqual(len(reductions), 1)
        self.assertIs(reductions[0][2], world_group)

    def test_fp32_optimizer_scale_uses_launcher_control_device(self):
        optimizer_module = load_optimizer_module()
        base_optimizer = SimpleNamespace(param_groups=[], state={})

        optimizer = optimizer_module.FP32Optimizer(
            base_optimizer,
            clip_grad=0.0,
            log_num_zeros_in_grad=False,
            params_have_main_grad=True,
            use_contiguous_buffers_in_ddp=True,
            control_device="cpu",
        )

        self.assertEqual(optimizer.get_loss_scale().device.type, "cpu")
        self.assertEqual(optimizer.get_loss_scale().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
