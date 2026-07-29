import dataclasses
import importlib.util
import unittest
from pathlib import Path

from tests.test_config_normalization import expanded
from tests.test_config_normalization import load_normalizer
from tests.test_config_normalization import make_config
from tests.test_config_normalization import make_pipe


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync" / "topology.py"


def load_topology():
    assert MODULE_PATH.exists(), "canonical topology adapter is missing"
    spec = importlib.util.spec_from_file_location("grad_sync_topology", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GradSyncTopologyTest(unittest.TestCase):
    def setUp(self):
        normalizer = load_normalizer()
        self.normalized = normalizer.normalize_flexpipe_config(
            make_config(
                [
                    make_pipe([21, 22], [1, 2], [1, 1]),
                    make_pipe([11, 32], [1, 1], [2, 1]),
                ]
            ),
            world_size=6,
        )

    def test_builds_immutable_stage_records_and_rank_coordinates(self):
        topology = load_topology()
        result = topology.build_canonical_topology(self.normalized)

        self.assertEqual(result.world_size, 6)
        self.assertEqual(result.fingerprint, self.normalized["topology_fingerprint"])
        self.assertEqual(len(result.stages), 4)
        stage = result.stage(0, 1)
        self.assertEqual((stage.global_op_begin, stage.global_op_end), (21, 43))
        self.assertEqual(stage.stage_ranks, (1, 2))
        self.assertEqual((stage.tp_degree, stage.dp_degree, stage.cp_degree), (2, 1, 1))
        self.assertEqual(stage.tp_group_ranks_by_coordinate, ((1, 2),))
        self.assertEqual(stage.dp_group_ranks_by_coordinate, ((1,), (2,)))
        self.assertEqual(
            [(item.global_rank, item.tp_rank, item.dp_rank, item.cp_rank) for item in stage.rank_coordinates],
            [(1, 0, 0, 0), (2, 1, 0, 0)],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            stage.stage_id = 9

    def test_covers_every_rank_for_two_and_three_pipeline_sweeps(self):
        normalizer = load_normalizer()
        topology = load_topology()
        fixtures = normalizer.generate_heterogeneous_config_fixtures()

        for config in fixtures:
            normalized = normalizer.normalize_flexpipe_config(config)
            result = topology.build_canonical_topology(normalized)
            with self.subTest(world_size=result.world_size, fingerprint=result.fingerprint):
                ranks = [
                    coordinate.global_rank
                    for stage in result.stages
                    for coordinate in stage.rank_coordinates
                ]
                self.assertEqual(sorted(ranks), list(range(result.world_size)))
                self.assertTrue(
                    all(tuple(sorted(group)) == group for group in result.all_group_rank_tuples())
                )

    def test_brief_and_expanded_configs_build_identical_topology(self):
        normalizer = load_normalizer()
        topology = load_topology()
        brief = make_config(
            [
                make_pipe([21, 22], [1, 2], [1, 1]),
                make_pipe([11, 32], [1, 1], [2, 1]),
            ]
        )
        brief_topology = topology.build_canonical_topology(
            normalizer.normalize_flexpipe_config(brief)
        )
        expanded_topology = topology.build_canonical_topology(
            normalizer.normalize_flexpipe_config(expanded(brief))
        )

        self.assertEqual(brief_topology, expanded_topology)

    def test_cross_validation_reports_rank_and_stage_context(self):
        topology = load_topology()
        result = topology.build_canonical_topology(self.normalized)
        facts = topology.build_mpu_rank_facts(
            global_rank=4,
            pipeline_id=1,
            stage_id=0,
            global_op_begin=0,
            global_op_end=11,
            stage_ranks=(3, 4),
            tp_group_ranks=(4,),
            dp_group_ranks=(3, 4),
            cp_group_ranks=(4,),
        )
        topology.validate_mpu_rank_facts(result, facts)

        facts["tp_group_ranks"] = (3, 5)
        with self.assertRaisesRegex(ValueError, "rank 4.*pipeline 1 stage 0.*tp_group_ranks"):
            topology.validate_mpu_rank_facts(result, facts)


if __name__ == "__main__":
    unittest.main()
