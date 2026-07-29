import math
import socket
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.grad_sync_correctness_matrix import build_cpu_matrix
from tests.grad_sync_correctness_matrix import build_distributed_matrix
from tests.grad_sync_correctness_matrix import build_rank_contract
from tests.grad_sync_correctness_matrix import expected_iteration
from tests.test_gradient_sync_manager import load_manager


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _spawn_with_timeout(fixture, timeout_seconds=45):
    context = mp.spawn(
        _gloo_worker,
        args=(fixture.fixture_id, _free_port()),
        nprocs=fixture.world_size,
        join=False,
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if context.join(timeout=1.0):
            return
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=5.0)
    raise AssertionError(
        f"{fixture.fixture_id} exceeded {timeout_seconds}s; possible deadlock"
    )


def _gloo_worker(rank, fixture_id, port):
    fixture = {
        item.fixture_id: item for item in build_distributed_matrix()
    }[fixture_id]
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=fixture.world_size,
    )
    try:
        manager_module, unused_planner, unused_bucket_plan = load_manager()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        contract = build_rank_contract(fixture, rank, descriptor_cls)
        created_groups = []

        def create_group(ranks):
            created_groups.append(tuple(ranks))
            return dist.new_group(ranks=list(ranks), backend="gloo")

        collective_count = 0

        def all_reduce_sum(payload, group):
            nonlocal collective_count
            collective_count += 1
            dist.all_reduce(payload, op=dist.ReduceOp.SUM, group=group)

        manager = manager_module.initialize_gradient_sync(
            local_descriptors=contract.descriptors,
            local_runtime_bindings=contract.runtime_bindings,
            topology=SimpleNamespace(
                world_size=fixture.world_size,
                stages=tuple(
                    SimpleNamespace(pipeline_id=pipeline_id)
                    for pipeline_id in range(fixture.pipeline_count)
                ),
                fingerprint=fixture.topology_fingerprint,
            ),
            model_impl="gpt",
            manifest_hash="synthetic-gpt-v1",
            sync_mode="block",
            bucket_cap_bytes=32 * 1024 * 1024,
            global_rank=rank,
            all_gather_object=dist.all_gather_object,
            all_gather_group_object=lambda outputs, value, group: (
                dist.all_gather_object(outputs, value, group=group)
            ),
            create_group=create_group,
            all_reduce_sum=all_reduce_sum,
            control_device="cpu",
        )
        initial_group_count = manager.registry.group_count
        expected_rank_collectives = sum(
            rank in next(
                group.ranks
                for group in manager.plan.process_groups
                if group.process_group_id == bucket.process_group_id
            )
            for bucket in manager.plan.global_bucket_order
        )

        for iteration in range(2):
            contract.load_iteration(iteration)
            local_weight = torch.tensor(
                contract.local_loss_weight(iteration), dtype=torch.float32
            )
            dist.all_reduce(local_weight, op=dist.ReduceOp.SUM)
            manager.execute(
                normalization_scale=fixture.nominal_global_loss_scale
                / float(local_weight.item())
            )

            expected = expected_iteration(fixture, iteration)
            contract.assert_matches(expected)
            self_norm = manager.optimizer_semantics.clip_grad_norm(
                expected.l2_norm,
                float("inf"),
                lambda payload, unused_operation: dist.all_reduce(
                    payload, op=dist.ReduceOp.SUM
                ),
                lambda payload, unused_operation: dist.all_reduce(
                    payload, op=dist.ReduceOp.MAX
                ),
            )
            if not math.isclose(
                self_norm, expected.inf_norm, rel_tol=1.0e-6, abs_tol=1.0e-3
            ):
                raise AssertionError((self_norm, expected.inf_norm))

            measured_l2 = manager.optimizer_semantics.clip_grad_norm(
                expected.l2_norm / 2.0,
                2.0,
                lambda payload, unused_operation: dist.all_reduce(
                    payload, op=dist.ReduceOp.SUM
                ),
                lambda payload, unused_operation: dist.all_reduce(
                    payload, op=dist.ReduceOp.MAX
                ),
            )
            if not math.isclose(
                measured_l2, expected.l2_norm, rel_tol=1.0e-6, abs_tol=1.0e-3
            ):
                raise AssertionError((measured_l2, expected.l2_norm))
            contract.assert_clipped(expected, expected.l2_norm / 2.0)

        if manager.registry.group_count != initial_group_count:
            raise AssertionError("process-group registry grew during execution")
        if len(created_groups) != initial_group_count:
            raise AssertionError((len(created_groups), initial_group_count))
        if collective_count != expected_rank_collectives * 2:
            raise AssertionError((collective_count, expected_rank_collectives * 2))
    finally:
        dist.destroy_process_group()


class GradientSyncCorrectnessMatrixTest(unittest.TestCase):
    def test_cpu_matrix_tracks_six_distinct_topology_fingerprints(self):
        fixtures = build_cpu_matrix()

        self.assertGreaterEqual(len(fixtures), 6)
        self.assertGreaterEqual(
            len({fixture.topology_fingerprint for fixture in fixtures}), 6
        )
        self.assertTrue({4, 6, 8}.issubset({fixture.world_size for fixture in fixtures}))

    def test_cpu_matrix_builds_stable_tensor_and_block_plans(self):
        saw_fusion = False
        for fixture in build_cpu_matrix():
            with self.subTest(fixture=fixture.fixture_id):
                tensor, block = fixture.build_plans()
                self.assertEqual(tensor.canonical_fragments, block.canonical_fragments)
                self.assertEqual(tensor.process_groups, block.process_groups)
                self.assertEqual(
                    tensor.exact_payload_bytes_by_rank,
                    block.exact_payload_bytes_by_rank,
                )
                self.assertEqual(tensor.padding_payload_bytes, 0)
                self.assertEqual(block.padding_payload_bytes, 0)
                self.assertLessEqual(
                    len(block.global_bucket_order), len(tensor.global_bucket_order)
                )
                saw_fusion |= len(block.global_bucket_order) < len(
                    tensor.global_bucket_order
                )
        self.assertTrue(saw_fusion)

    def test_cpu_matrix_materializes_every_topology_stage(self):
        for fixture in build_cpu_matrix():
            with self.subTest(fixture=fixture.fixture_id):
                descriptors = fixture.build_descriptors()
                actual = {
                    (item.pipeline_id, item.stage_id) for item in descriptors
                }
                expected = {
                    (stage.pipeline_id, stage.stage_id)
                    for stage in fixture.build_topology().stages
                }
                self.assertEqual(actual, expected)

    def test_cpu_matrix_accepts_additional_valid_generated_fixtures(self):
        normalizer = build_cpu_matrix()[0].normalized_config
        module = sys.modules[build_cpu_matrix.__module__]
        generated = module.load_normalizer().generate_heterogeneous_config_fixtures()
        generated.append(build_cpu_matrix()[0].config)

        fake_normalizer = module.load_normalizer()
        fake_normalizer.generate_heterogeneous_config_fixtures = lambda: generated
        with mock.patch.object(module, "load_normalizer", return_value=fake_normalizer):
            fixtures = build_cpu_matrix()

        self.assertEqual(len(fixtures), len(generated))
        self.assertEqual(fixtures[-1].normalized_config, normalizer)

    def test_four_rank_fixture_contains_the_hand_calculated_oracle(self):
        fixture = build_distributed_matrix()[0]
        expected = expected_iteration(fixture, 0)

        torch.testing.assert_close(
            expected.weights[0], torch.tensor([111.0, 222.0, 3033.0, 4044.0])
        )
        torch.testing.assert_close(expected.bias, torch.tensor([6.0]))

    def test_dp_replicas_partition_each_pipeline_loss_weight(self):
        for fixture in build_distributed_matrix():
            for iteration in range(2):
                for layout in fixture.loss_layouts:
                    contributions = []
                    for rank in layout.stage_ranks:
                        contribution = fixture.local_loss_weight(rank, iteration)
                        if contribution:
                            contributions.append(contribution)
                    self.assertEqual(len(contributions), layout.dp_degree)
                    self.assertAlmostEqual(
                        sum(contributions),
                        fixture.loss_weights_by_iteration[iteration][
                            layout.pipeline_id
                        ],
                    )

    def test_swapped_stage_contract_preserves_stage_metadata(self):
        load_manager()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        fixture = build_distributed_matrix()[1]

        for rank in range(fixture.world_size):
            contract = build_rank_contract(fixture, rank, descriptor_cls)
            self.assertTrue(contract.descriptors)
            for item in contract.descriptors:
                parameter = fixture.parameter(item.logical_param_id)
                layout = parameter.layout_for_rank(rank)
                self.assertIsNotNone(layout)
                self.assertEqual(item.global_op_index, parameter.global_op_index)
                self.assertEqual(
                    (
                        item.stage_id,
                        item.stage_global_op_begin,
                        item.stage_global_op_end,
                    ),
                    (
                        layout.stage_id,
                        layout.global_op_begin,
                        layout.global_op_end,
                    ),
                )

    def test_gloo_matrix_matches_independent_oracles_for_two_iterations(self):
        fixtures = build_distributed_matrix()
        self.assertEqual(
            {fixture.fixture_id for fixture in fixtures},
            {
                "canonical-4r-equal-gbs",
                "hetero-6r-swapped-stages",
                "canonical-6r-three-pipeline",
            },
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture.fixture_id):
                _spawn_with_timeout(fixture)


if __name__ == "__main__":
    unittest.main()
