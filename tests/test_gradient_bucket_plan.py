import importlib.util
import sys
import unittest
from pathlib import Path

from tests.test_gradient_fragment_planner import load_planner


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAD_SYNC_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync"


def load_bucket_plan():
    planner, _ = load_planner()
    full_name = "megatron.grad_sync.bucket_plan"
    path = GRAD_SYNC_DIR / "bucket_plan.py"
    assert path.exists(), "bucket planner is missing"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module, planner


def fragment(planner, index, group_id, dtype="float32", numel=8):
    return planner.CanonicalGradientFragmentSpec(
        fragment_id=f"fragment-{index}",
        logical_param_id=f"param-{index}",
        canonical_shard_id=0,
        process_group_id=group_id,
        grad_dtype=dtype,
        reduction_semantics="sum",
        logical_region_key=f"region-{index}",
        global_regions=((0, numel),),
        numel=numel,
        semantic_key=(index,),
        optimizer_semantic_id=f"param-{index}",
        include_in_optimizer_norm=True,
        norm_owner_rank=0,
    )


def fragment_plan(planner):
    groups = (
        planner.ProcessGroupSpec("pg-a", (0, 1), "gradient_sync"),
        planner.ProcessGroupSpec("pg-b", (0, 2), "gradient_sync"),
    )
    fragments = (
        fragment(planner, 0, "pg-a"),
        fragment(planner, 1, "pg-b"),
        fragment(planner, 2, "pg-a"),
        fragment(planner, 3, "pg-b"),
    )
    return planner.CanonicalFragmentPlan(groups, fragments, (), "fragment-hash")


class GradientBucketPlanTest(unittest.TestCase):
    def test_tensor_and_block_share_semantics_while_four_blocks_fuse_to_two(self):
        bucket, planner = load_bucket_plan()
        fragments = fragment_plan(planner)

        tensor = bucket.build_gradient_sync_plan(
            fragments, "tensor", 32 * 1024 * 1024, "gpt", "manifest", "topology"
        )
        block = bucket.build_gradient_sync_plan(
            fragments, "block", 32 * 1024 * 1024, "gpt", "manifest", "topology"
        )

        self.assertEqual(len(tensor.global_bucket_order), 4)
        self.assertEqual(len(block.global_bucket_order), 2)
        self.assertEqual(tensor.canonical_fragments, block.canonical_fragments)
        self.assertEqual(tensor.process_groups, block.process_groups)
        self.assertEqual(tensor.global_fragment_hash, block.global_fragment_hash)
        self.assertEqual(tensor.exact_payload_bytes_by_rank, block.exact_payload_bytes_by_rank)
        self.assertEqual(tensor.padding_payload_bytes, 0)
        self.assertEqual(block.padding_payload_bytes, 0)
        self.assertNotEqual(tensor.global_plan_hash, block.global_plan_hash)

    def test_cap_dtype_and_group_boundaries_never_split_fragments(self):
        bucket, planner = load_bucket_plan()
        groups = (
            planner.ProcessGroupSpec("pg-a", (0, 1), "gradient_sync"),
            planner.ProcessGroupSpec("pg-b", (0, 2), "gradient_sync"),
        )
        fragments = (
            fragment(planner, 0, "pg-a", "float16", 4),
            fragment(planner, 1, "pg-a", "float32", 4),
            fragment(planner, 2, "pg-b", "float32", 4),
            fragment(planner, 3, "pg-b", "float32", 20),
        )
        source = planner.CanonicalFragmentPlan(groups, fragments, (), "hash")

        plan = bucket.build_gradient_sync_plan(
            source, "block", 16, "gpt", "manifest", "topology"
        )

        self.assertEqual(len(plan.global_bucket_order), 4)
        self.assertTrue(plan.global_bucket_order[-1].oversized)
        self.assertEqual(plan.global_bucket_order[-1].exact_numel, 20)
        self.assertTrue(
            all(
                placement.destination_offset == 0
                for item in plan.global_bucket_order
                for placement in item.placements[:1]
            )
        )

    def test_input_order_and_cap_variants_are_deterministic(self):
        bucket, planner = load_bucket_plan()
        source = fragment_plan(planner)
        reversed_source = planner.CanonicalFragmentPlan(
            tuple(reversed(source.process_groups)),
            tuple(reversed(source.fragments)),
            (),
            source.fragment_hash,
        )

        for cap_mib in (1, 32, 256):
            first = bucket.build_gradient_sync_plan(
                source, "block", cap_mib * 1024 * 1024, "gpt", "manifest", "topology"
            )
            second = bucket.build_gradient_sync_plan(
                reversed_source, "block", cap_mib * 1024 * 1024, "gpt", "manifest", "topology"
            )
            self.assertEqual(first, second)

    def test_collective_order_and_hashes_cover_full_layout_and_bindings(self):
        bucket, planner = load_bucket_plan()
        groups = (
            planner.ProcessGroupSpec("z-first", (0, 1), "gradient_sync"),
            planner.ProcessGroupSpec("a-second", (0, 2), "gradient_sync"),
        )
        fragments = (
            fragment(planner, 0, "a-second", numel=4),
            fragment(planner, 1, "z-first", numel=4),
        )
        segment = planner.TensorSegmentSpec(0, 4, (4,), 0, 4, 4)
        binding = planner.LocalFragmentBindingSpec(0, "fragment-0", "param-0", (segment,), 1.0)
        source = planner.CanonicalFragmentPlan(
            groups, fragments, ((0, (binding,)),), "fragment-hash"
        )
        plan = bucket.build_gradient_sync_plan(
            source, "tensor", 1024, "gpt", "manifest", "topology"
        )
        self.assertEqual(
            [item.process_group_id for item in plan.global_bucket_order],
            ["z-first", "a-second"],
        )

        changed_numel = planner.CanonicalFragmentPlan(
            groups,
            (fragment(planner, 0, "a-second", numel=5), fragments[1]),
            source.local_bindings_by_rank,
            source.fragment_hash,
        )
        changed_numel_plan = bucket.build_gradient_sync_plan(
            changed_numel, "tensor", 1024, "gpt", "manifest", "topology"
        )
        self.assertNotEqual(
            plan.global_bucket_order[1].semantic_hash,
            changed_numel_plan.global_bucket_order[1].semantic_hash,
        )

        changed_binding = planner.LocalFragmentBindingSpec(
            0, "fragment-0", "param-0", (segment,), 0.5
        )
        binding_source = planner.CanonicalFragmentPlan(
            groups, fragments, ((0, (changed_binding,)),), "fragment-hash"
        )
        binding_plan = bucket.build_gradient_sync_plan(
            binding_source, "tensor", 1024, "gpt", "manifest", "topology"
        )
        self.assertNotEqual(plan.global_fragment_hash, binding_plan.global_fragment_hash)
        self.assertNotEqual(plan.global_plan_hash, binding_plan.global_plan_hash)

    def test_process_group_registry_creates_one_sorted_global_order(self):
        bucket, planner = load_bucket_plan()
        groups = (
            planner.ProcessGroupSpec("pg-b", (2, 3), "gradient_sync"),
            planner.ProcessGroupSpec("pg-a", (0, 2), "gradient_sync"),
        )
        calls = []
        registry = bucket.ProcessGroupRegistry(groups, lambda ranks: calls.append(ranks) or f"handle-{ranks}")

        registry.initialize()

        self.assertEqual(calls, [(0, 2), (2, 3)])
        self.assertEqual(registry.group_count, 2)
        self.assertEqual(registry.creation_count, 2)
        self.assertEqual(registry.handle("pg-a"), "handle-(0, 2)")
        registry.initialize()
        self.assertEqual(calls, [(0, 2), (2, 3)])
        self.assertEqual(registry.creation_count, 2)


if __name__ == "__main__":
    unittest.main()
