import copy
import importlib.util
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "config_normalization.py"


def load_normalizer():
    assert MODULE_PATH.exists(), "canonical config normalizer is missing"
    spec = importlib.util.spec_from_file_location("config_normalization", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pipe(stage_ops, tp, dp, gbs=24, mbs=8, cp=None):
    cp = cp or [1] * len(stage_ops)
    return {
        "global_batch_size": gbs,
        "micro_batch_size": mbs,
        "num_stages": len(stage_ops),
        "num_gpus": [t * d * c for t, d, c in zip(tp, dp, cp)],
        "checkpoint_activations": [False] * len(stage_ops),
        "resharding_stages": [False] * len(stage_ops),
        "num_ops_in_each_stage": stage_ops,
        "model_parallel_size_of_each_stage": tp,
        "data_parallel_size_of_each_stage": dp,
        "context_parallel_size_of_each_stage": cp,
        "algo_of_each_stage": [0] * len(stage_ops),
        "recompute_ops": [[0] * count for count in stage_ops],
    }


def make_config(pipes, model_impl="gpt", num_layers=4):
    config = {
        "model_name": "gpt",
        "model_size": "test",
        "num_layers": num_layers,
        "seq_length": 32,
        "max_position_embeddings": 32,
        "num_attention_heads": 8,
        "hidden_size": 64,
        "pipe": pipes,
    }
    if model_impl is not None:
        config["model_impl"] = model_impl
    return config


def expanded(config):
    result = copy.deepcopy(config)
    for pipe in result["pipe"]:
        counts = pipe["num_ops_in_each_stage"]
        for stem in (
            "model_parallel_size",
            "data_parallel_size",
            "context_parallel_size",
            "algo",
        ):
            stage_key = f"{stem}_of_each_stage"
            op_key = f"{stem}_of_each_op"
            pipe[op_key] = [
                [value] * count
                for value, count in zip(pipe.pop(stage_key), counts)
            ]
    return result


class ConfigNormalizationTest(unittest.TestCase):
    def test_rejects_unknown_top_level_field(self):
        normalizer = load_normalizer()
        config = make_config([make_pipe([43], [1], [1])])
        config["hidden_sze"] = 64

        with self.assertRaisesRegex(
            ValueError, "unknown top-level config field.*hidden_sze"
        ):
            normalizer.normalize_flexpipe_config(config)

    def test_rejects_unknown_pipeline_field(self):
        normalizer = load_normalizer()
        config = make_config([make_pipe([43], [1], [1])])
        config["pipe"][0]["micro_bacth_size"] = 8

        with self.assertRaisesRegex(
            ValueError, "pipeline 0 has unknown field.*micro_bacth_size"
        ):
            normalizer.normalize_flexpipe_config(config)

    def test_rejects_unsupported_tensor_parallel_value_even_when_divisible(self):
        normalizer = load_normalizer()
        config = make_config([make_pipe([43], [3], [1])])
        config["hidden_size"] = 96
        config["num_attention_heads"] = 12

        with self.assertRaisesRegex(
            ValueError, "unsupported TP value 3.*1, 2, 4, 6, 8"
        ):
            normalizer.normalize_flexpipe_config(config)

    def test_selects_explicit_manifest_and_infers_unversioned_gpt(self):
        normalizer = load_normalizer()
        config = make_config([make_pipe([43], [1], [1])])
        inferred = make_config([make_pipe([43], [1], [1])], model_impl=None)

        self.assertEqual(normalizer.normalize_flexpipe_config(config)["model_impl"], "gpt")
        manifest = normalizer.normalize_flexpipe_config(config)["manifest"]
        self.assertEqual(manifest["ordered_ops"][1]["name"], "enc-1st-layernorm")
        self.assertEqual(manifest["ordered_ops"][2]["name"], "enc-attention-qkv")
        self.assertEqual(manifest["ordered_ops"][3]["name"], "enc-flash-attention")
        self.assertEqual(manifest["ordered_ops"][4]["name"], "enc-attention-dense")
        self.assertEqual(manifest["ordered_ops"][41]["name"], "final-layernorm")
        self.assertEqual(manifest["ordered_ops"][42]["name"], "gpt-post-process")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalized = normalizer.normalize_flexpipe_config(inferred)
        self.assertEqual(normalized["model_impl"], "gpt")
        self.assertTrue(any("inferred" in str(item.message) for item in caught))

    def test_brief_and_expanded_forms_have_identical_canonical_hashes(self):
        normalizer = load_normalizer()
        brief = make_config(
            [
                make_pipe([21, 22], [1, 2], [1, 1]),
                make_pipe([11, 32], [1, 1], [2, 1]),
            ]
        )

        brief_result = normalizer.normalize_flexpipe_config(brief, world_size=6)
        expanded_result = normalizer.normalize_flexpipe_config(
            expanded(brief), world_size=6
        )
        reordered = {
            key: brief[key]
            for key in reversed(list(brief))
        }
        reordered_result = normalizer.normalize_flexpipe_config(
            reordered, world_size=6
        )

        self.assertEqual(brief_result, expanded_result)
        self.assertEqual(brief_result, reordered_result)
        self.assertEqual(brief_result["total_gbs"], 48)
        self.assertEqual(brief_result["pipeline_sample_offsets"], [0, 24])
        self.assertEqual(
            brief_result["pipe"][0]["model_parallel_size_of_each_op"][1],
            [2] * 22,
        )

    def test_rejects_manifest_topology_and_batch_contract_violations(self):
        normalizer = load_normalizer()
        valid = make_config([make_pipe([43], [1], [1])])
        cases = {}
        cases["manifest size"] = copy.deepcopy(valid)
        cases["manifest size"]["pipe"][0]["num_ops_in_each_stage"] = [42]
        cases["stage product"] = copy.deepcopy(valid)
        cases["stage product"]["pipe"][0]["num_gpus"] = [2]
        cases["world size"] = copy.deepcopy(valid)
        cases["context parallel"] = copy.deepcopy(valid)
        cases["context parallel"]["pipe"][0]["context_parallel_size_of_each_stage"] = [2]
        cases["TP constraint"] = copy.deepcopy(valid)
        cases["TP constraint"]["hidden_size"] = 66
        cases["TP constraint"]["pipe"][0]["model_parallel_size_of_each_stage"] = [4]
        cases["TP constraint"]["pipe"][0]["num_gpus"] = [4]
        cases["micro batch divisibility"] = copy.deepcopy(valid)
        cases["micro batch divisibility"]["pipe"][0]["micro_batch_size"] = 7
        inconsistent = expanded(valid)
        inconsistent["pipe"][0]["model_parallel_size_of_each_op"][0][-1] = 2
        cases["intra-stage constancy"] = inconsistent

        for name, config in cases.items():
            with self.subTest(name=name):
                world_size = 2 if name == "world size" else None
                with self.assertRaisesRegex(ValueError, name):
                    normalizer.normalize_flexpipe_config(config, world_size=world_size)

    def test_rejects_removed_or_mismatched_manifests(self):
        normalizer = load_normalizer()
        explicit_conflict = make_config([make_pipe([55], [1], [1])])
        mixed_manifest = make_config(
            [make_pipe([43], [1], [1]), make_pipe([55], [1], [1])],
            model_impl=None,
        )

        with self.assertRaisesRegex(ValueError, "manifest size"):
            normalizer.normalize_flexpipe_config(explicit_conflict)
        with self.assertRaisesRegex(ValueError, "manifest size"):
            normalizer.normalize_flexpipe_config(mixed_manifest)

    def test_fixture_sweep_is_stable_and_covers_required_world_sizes(self):
        normalizer = load_normalizer()
        first = normalizer.generate_heterogeneous_config_fixtures()
        second = normalizer.generate_heterogeneous_config_fixtures()

        self.assertGreaterEqual(len(first), 6)
        self.assertEqual(first, second)
        normalized = [normalizer.normalize_flexpipe_config(item) for item in first]
        fingerprints = {item["topology_fingerprint"] for item in normalized}
        world_sizes = {item["world_size"] for item in normalized}
        self.assertGreaterEqual(len(fingerprints), 6)
        self.assertTrue({4, 6, 8}.issubset(world_sizes))
        self.assertTrue(
            any(
                len({sum(pipe["num_gpus"]) for pipe in item["pipe"]}) > 1
                for item in normalized
            )
        )

    def test_named_gpt_e2e_fixtures_are_tracked_and_distinct(self):
        normalizer = load_normalizer()
        fixtures = normalizer.generate_gpt_e2e_fixtures()

        self.assertEqual(
            set(fixtures),
            {
                "canonical-4r-equal-gbs",
                "reference-1r-canonical4-global-batch",
                "brief-6r-equal-gbs",
                "hetero-6r-swapped-stages",
                "canonical-6r-three-pipeline",
            },
        )
        normalized = {
            name: normalizer.normalize_flexpipe_config(config)
            for name, config in fixtures.items()
        }
        self.assertEqual(normalized["canonical-4r-equal-gbs"]["world_size"], 4)
        self.assertEqual(
            normalized["reference-1r-canonical4-global-batch"]["world_size"],
            1,
        )
        self.assertEqual(
            normalized["reference-1r-canonical4-global-batch"]["total_gbs"],
            normalized["canonical-4r-equal-gbs"]["total_gbs"],
        )
        self.assertTrue(
            all(item["model_impl"] == "gpt" for item in normalized.values())
        )
        self.assertEqual(
            len({item["topology_fingerprint"] for item in normalized.values()}), 5
        )

    def test_unequal_pipeline_totals_map_every_rank_without_legacy_overrun(self):
        normalizer = load_normalizer()
        config = make_config(
            [
                make_pipe([43], [1], [1]),
                make_pipe([11, 32], [1, 1], [1, 2]),
            ]
        )
        normalized = normalizer.normalize_flexpipe_config(config, world_size=4)

        for rank, expected_pipe in enumerate((0, 1, 1, 1)):
            args = SimpleNamespace(rank=rank, world_size=4, mbs_seq="[]")
            normalizer.apply_normalized_config_to_args(args, normalized)
            self.assertEqual(args.current_pipe, expected_pipe)
        self.assertEqual(args.rank_to_stage, [0, 0, 1, 1])

    def test_applies_canonical_config_to_rank_local_runtime_arguments(self):
        normalizer = load_normalizer()
        config = make_config(
            [
                make_pipe([21, 22], [1, 2], [1, 1]),
                make_pipe([11, 32], [1, 1], [2, 1]),
            ]
        )
        normalized = normalizer.normalize_flexpipe_config(config, world_size=6)
        args = SimpleNamespace(rank=4, world_size=6, mbs_seq="[]")

        normalizer.apply_normalized_config_to_args(args, normalized)

        self.assertEqual(args.model_impl, "gpt")
        self.assertEqual(args.num_ops_each_layer, 10)
        self.assertEqual(args.current_pipe, 1)
        self.assertEqual(args.offset, 3)
        self.assertEqual(args.rank_to_pipe, [0, 0, 0, 1, 1, 1])
        self.assertEqual(args.total_gbs, 48)
        self.assertEqual(args.pipeline_sample_offsets, [0, 24])
        self.assertEqual(args.config_hash, normalized["config_hash"])
        self.assertEqual(args.context_parallel_size_of_each_op, [[1] * 11, [1] * 32])

    def test_runtime_manifest_must_match_config(self):
        normalizer = load_normalizer()
        normalized = normalizer.normalize_flexpipe_config(
            make_config([make_pipe([43], [1], [1])])
        )
        runtime_ops = [
            {"op_index": item["op_index"], "name": item["name"]}
            for item in normalized["manifest"]["ordered_ops"]
        ]

        normalizer.validate_runtime_manifest(runtime_ops, normalized["manifest"])
        runtime_ops[3]["name"] = "wrong-op"
        with self.assertRaisesRegex(ValueError, "runtime GPT manifest mismatch.*op 3"):
            normalizer.validate_runtime_manifest(runtime_ops, normalized["manifest"])


if __name__ == "__main__":
    unittest.main()
