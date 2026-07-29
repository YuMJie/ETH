import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path

from tests.test_gpt_gradient_metadata import load_adapter
from tests.test_config_normalization import load_normalizer
from tests.test_grad_sync_topology import load_topology


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAD_SYNC_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync"


def load_planner():
    adapter = load_adapter()
    full_name = "megatron.grad_sync.planner"
    path = GRAD_SYNC_DIR / "planner.py"
    assert path.exists(), "canonical fragment planner is missing"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module, adapter.SerializableGradientDescriptor


def descriptor(
    cls,
    logical_id,
    rank,
    pipeline_id,
    tp_degree,
    tp_rank,
    full_shape,
    local_shape,
    tensor_parallel,
    partition_dim,
    partition_stride=1,
    dtype="float32",
    dp_degree=1,
    dp_rank=0,
    shared_weight_id=None,
    shared_weight_root=False,
    tp_group=None,
    dp_group=None,
    stage_ranks=None,
):
    stage_start = rank - tp_rank
    tp_group = tp_group or tuple(range(stage_start, stage_start + tp_degree))
    dp_group = dp_group or (rank,)
    stage_ranks = stage_ranks or tuple(sorted(set(tp_group + dp_group)))
    return cls(
        logical_param_id=logical_id,
        shared_weight_id=shared_weight_id,
        shared_weight_root=shared_weight_root,
        logical_region_key=logical_id,
        semantic_order_key=(0, 0, logical_id),
        global_rank=rank,
        model_impl="gpt",
        op_manifest_hash="manifest",
        pipeline_id=pipeline_id,
        stage_id=0,
        model_chunk_id=0,
        global_op_index=1,
        manifest_op_name="op",
        runtime_op_name="op",
        relative_parameter_name="weight",
        stage_global_op_begin=0,
        stage_global_op_end=1,
        local_tp_degree=tp_degree,
        local_tp_rank=tp_rank,
        local_dp_degree=dp_degree,
        local_dp_rank=dp_rank,
        local_cp_degree=1,
        local_cp_rank=0,
        stage_ranks=stage_ranks,
        local_tp_group_ranks=tp_group,
        local_dp_group_ranks=dp_group,
        full_shape=full_shape,
        local_shape=local_shape,
        tensor_parallel=tensor_parallel,
        tp_replica_semantics=(None if tensor_parallel else "identical_duplicate"),
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        parameter_dtype="float16",
        gradient_dtype=dtype,
        requires_grad=True,
    )


class GradientFragmentPlannerTest(unittest.TestCase):
    def test_shape_mismatch_reports_rank_layouts(self):
        planner, cls = load_planner()
        descriptors = [
            descriptor(cls, "weight", 0, 0, 1, 0, (8, 2), (8, 2), True, 0),
            descriptor(cls, "weight", 1, 1, 1, 0, (4, 2), (4, 2), True, 0),
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"shape mismatch for weight: .*rank=0.*full=\(8, 2\).*rank=1.*full=\(4, 2\)",
        ):
            planner.plan_canonical_fragments(descriptors, (0, 1))

    def test_dim0_mixed_tp_layouts_build_canonical_shards_and_bindings(self):
        planner, cls = load_planner()
        descriptors = [
            descriptor(cls, "weight", 0, 0, 1, 0, (8, 2), (8, 2), True, 0),
            descriptor(cls, "weight", 1, 1, 2, 0, (8, 2), (4, 2), True, 0),
            descriptor(cls, "weight", 2, 1, 2, 1, (8, 2), (4, 2), True, 0),
        ]

        plan = planner.plan_canonical_fragments(descriptors, (0, 1))

        self.assertEqual([item.ranks for item in plan.process_groups], [(0, 1), (0, 2)])
        self.assertEqual([item.canonical_shard_id for item in plan.fragments], [0, 1])
        self.assertEqual([item.numel for item in plan.fragments], [8, 8])
        rank0 = plan.bindings_for_rank(0)
        self.assertEqual(
            [(item.source_segments[0].source_dim_offset, item.source_segments[0].source_dim_length) for item in rank0],
            [(0, 4), (4, 4)],
        )

    def test_dim1_stride_two_preserves_multiple_global_intervals(self):
        planner, cls = load_planner()
        descriptors = []
        for pipeline_id, start_rank in enumerate((0, 2)):
            descriptors.extend(
                [
                    descriptor(cls, "weight", start_rank, pipeline_id, 2, 0, (2, 8), (2, 4), True, 1, 2),
                    descriptor(cls, "weight", start_rank + 1, pipeline_id, 2, 1, (2, 8), (2, 4), True, 1, 2),
                ]
            )

        plan = planner.plan_canonical_fragments(descriptors, (0, 1))

        first = plan.fragments[0]
        self.assertEqual(first.global_regions, ((0, 2), (4, 6)))
        binding = plan.bindings_for_rank(0)[0]
        self.assertEqual(
            [(item.source_dim_offset, item.source_dim_length) for item in binding.source_segments],
            [(0, 2), (2, 2)],
        )

    def test_replicated_parameters_scale_every_tp_holder(self):
        planner, cls = load_planner()
        descriptors = [
            descriptor(cls, "bias", 0, 0, 1, 0, (4,), (4,), False, -1),
            descriptor(cls, "bias", 1, 1, 2, 0, (4,), (4,), False, -1),
            descriptor(cls, "bias", 2, 1, 2, 1, (4,), (4,), False, -1),
        ]

        plan = planner.plan_canonical_fragments(descriptors, (0, 1))

        self.assertEqual(plan.process_groups[0].ranks, (0, 1, 2))
        self.assertIsNone(plan.fragments[0].canonical_shard_id)
        self.assertEqual(
            [plan.bindings_for_rank(rank)[0].contribution_scale for rank in range(3)],
            [1.0, 0.5, 0.5],
        )

    def test_shared_root_is_the_only_optimizer_norm_fragment(self):
        planner, cls = load_planner()
        descriptors = [
            descriptor(cls, "embedding", 0, 0, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared", shared_weight_root=True),
            descriptor(cls, "embedding", 1, 1, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared", shared_weight_root=True),
            descriptor(cls, "head", 0, 0, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared", shared_weight_root=False),
            descriptor(cls, "head", 1, 1, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared", shared_weight_root=False),
        ]

        plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        by_id = {item.logical_param_id: item for item in plan.fragments}

        self.assertTrue(by_id["embedding"].include_in_optimizer_norm)
        self.assertEqual(by_id["embedding"].norm_owner_rank, 0)
        self.assertFalse(by_id["head"].include_in_optimizer_norm)
        self.assertIsNone(by_id["head"].norm_owner_rank)

    def test_reuses_rank_tuple_groups_and_validates_pipeline_dp_and_shared_roots(self):
        planner, cls = load_planner()
        mixed = []
        for rank, pipeline_id in ((0, 0), (1, 1)):
            mixed.append(
                descriptor(cls, "weight", rank, pipeline_id, 1, 0, (4,), (4,), True, 0)
            )
            mixed.append(
                descriptor(cls, "bias", rank, pipeline_id, 1, 0, (4,), (4,), False, -1)
            )
        mixed_plan = planner.plan_canonical_fragments(mixed, expected_pipeline_ids=(0, 1))
        self.assertEqual(len(mixed_plan.process_groups), 1)
        self.assertEqual(len(mixed_plan.fragments), 2)

        with self.assertRaisesRegex(ValueError, "missing owner pipeline.*1"):
            planner.plan_canonical_fragments(mixed[:2], expected_pipeline_ids=(0, 1))

        dp_descriptors = [
            descriptor(cls, "bias", 0, 0, 1, 0, (4,), (4,), False, -1, dp_degree=2, dp_rank=0, dp_group=(0, 1), stage_ranks=(0, 1)),
            descriptor(cls, "bias", 1, 0, 1, 0, (4,), (4,), False, -1, dp_degree=2, dp_rank=1, dp_group=(0, 1), stage_ranks=(0, 1)),
            descriptor(cls, "bias", 2, 1, 1, 0, (4,), (4,), False, -1, dp_degree=2, dp_rank=0, dp_group=(2, 3), stage_ranks=(2, 3)),
            descriptor(cls, "bias", 3, 1, 1, 0, (4,), (4,), False, -1, dp_degree=2, dp_rank=1, dp_group=(2, 3), stage_ranks=(2, 3)),
        ]
        dp_plan = planner.plan_canonical_fragments(dp_descriptors, expected_pipeline_ids=(0, 1))
        self.assertEqual(dp_plan.process_groups[0].ranks, (0, 1, 2, 3))

        no_root = [
            descriptor(cls, "head", rank, pipeline_id, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared")
            for rank, pipeline_id in ((0, 0), (1, 1))
        ]
        with self.assertRaisesRegex(ValueError, "shared root mismatch"):
            planner.plan_canonical_fragments(no_root, expected_pipeline_ids=(0, 1))

        two_roots = no_root + [
            descriptor(cls, "embedding", rank, pipeline_id, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared", shared_weight_root=True)
            for rank, pipeline_id in ((0, 0), (1, 1))
        ] + [
            descriptor(cls, "other_root", rank, pipeline_id, 1, 0, (4,), (4,), False, -1, shared_weight_id="shared", shared_weight_root=True)
            for rank, pipeline_id in ((0, 0), (1, 1))
        ]
        with self.assertRaisesRegex(ValueError, "shared root mismatch"):
            planner.plan_canonical_fragments(two_roots, expected_pipeline_ids=(0, 1))

    def test_rejects_invalid_layouts_and_hash_is_input_order_independent(self):
        planner, cls = load_planner()
        valid = [
            descriptor(cls, "weight", 0, 0, 1, 0, (8, 2), (8, 2), True, 0),
            descriptor(cls, "weight", 1, 1, 2, 0, (8, 2), (4, 2), True, 0),
            descriptor(cls, "weight", 2, 1, 2, 1, (8, 2), (4, 2), True, 0),
        ]
        shuffled = list(valid)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(
            planner.plan_canonical_fragments(valid, (0, 1)).fragment_hash,
            planner.plan_canonical_fragments(shuffled, (0, 1)).fragment_hash,
        )

        invalid_cases = {
            "nondivisible": [
                descriptor(cls, "weight", rank, 0, 3, rank, (8, 2), (3, 2), True, 0)
                for rank in range(3)
            ],
            "axis mismatch": [valid[0], descriptor(cls, "weight", 1, 1, 1, 0, (8, 2), (8, 2), True, 1)],
            "dtype mismatch": [valid[0], descriptor(cls, "weight", 1, 1, 1, 0, (8, 2), (8, 2), True, 0, dtype="float16")],
            "shape mismatch": [
                descriptor(cls, "weight", rank, 0, 2, rank, (2, 8, 3), (3, 4, 2), True, 1)
                for rank in range(2)
            ],
            "replica semantics": [descriptor(cls, "bias", 0, 0, 1, 0, (4,), (4,), False, -1)],
        }
        invalid_cases["replica semantics"][0] = invalid_cases["replica semantics"][0].__class__(
            **{
                **invalid_cases["replica semantics"][0].__dict__,
                "tp_replica_semantics": "replicated",
            }
        )
        for message, case in invalid_cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    expected = (0, 1) if len({item.pipeline_id for item in case}) > 1 else (0,)
                    planner.plan_canonical_fragments(case, expected)

    def test_fixture_sweep_keeps_parameterless_ranks_out_of_bindings(self):
        planner, cls = load_planner()
        normalizer = load_normalizer()
        topology_module = load_topology()
        covered_world_sizes = set()
        for config in normalizer.generate_heterogeneous_config_fixtures():
            normalized = normalizer.normalize_flexpipe_config(config)
            topology = topology_module.build_canonical_topology(normalized)
            descriptors = []
            participating_ranks = set()
            for pipeline_id in range(len(normalized["pipe"])):
                stage = topology.stage(pipeline_id, 0)
                for coordinate in stage.rank_coordinates:
                    rank = coordinate.global_rank
                    tp_group = next(group for group in stage.tp_group_ranks_by_coordinate if rank in group)
                    dp_group = next(group for group in stage.dp_group_ranks_by_coordinate if rank in group)
                    participating_ranks.add(rank)
                    descriptors.append(
                        descriptor(
                            cls,
                            "bias",
                            rank,
                            pipeline_id,
                            stage.tp_degree,
                            coordinate.tp_rank,
                            (4,),
                            (4,),
                            False,
                            -1,
                            dp_degree=stage.dp_degree,
                            dp_rank=coordinate.dp_rank,
                            tp_group=tp_group,
                            dp_group=dp_group,
                            stage_ranks=stage.stage_ranks,
                        )
                    )
            plan = planner.plan_canonical_fragments(
                descriptors,
                expected_pipeline_ids=range(len(normalized["pipe"])),
            )
            binding_ranks = {rank for rank, _ in plan.local_bindings_by_rank}
            self.assertEqual(binding_ranks, participating_ranks)
            self.assertTrue(binding_ranks.issubset(set(range(topology.world_size))))
            covered_world_sizes.add(topology.world_size)
        self.assertTrue({4, 6, 8}.issubset(covered_world_sizes))


if __name__ == "__main__":
    unittest.main()
