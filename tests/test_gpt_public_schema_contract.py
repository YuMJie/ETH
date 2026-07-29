import ast
import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZER_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "config_normalization.py"


def load_normalizer():
    spec = importlib.util.spec_from_file_location("gpt_public_normalizer", NORMALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config(model_impl):
    return {
        "model_name": "gpt",
        "model_impl": model_impl,
        "num_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "pipe": [
            {
                "global_batch_size": 8,
                "micro_batch_size": 8,
                "num_stages": 1,
                "num_gpus": [1],
                "checkpoint_activations": [False],
                "resharding_stages": [False],
                "num_ops_in_each_stage": [13],
                "model_parallel_size_of_each_stage": [1],
                "data_parallel_size_of_each_stage": [1],
                "context_parallel_size_of_each_stage": [1],
                "algo_of_each_stage": [0],
                "recompute_ops": [[0] * 13],
            }
        ],
    }


class GptPublicSchemaContractTest(unittest.TestCase):
    def test_only_unversioned_gpt_model_impl_is_accepted(self):
        normalizer = load_normalizer()
        normalized = normalizer.normalize_flexpipe_config(make_config("gpt"))
        self.assertEqual("gpt", normalized["model_impl"])
        self.assertEqual(10, normalized["manifest"]["ops_per_layer"])
        self.assertEqual(
            [
                "encoder-embedding",
                "enc-1st-layernorm",
                "enc-attention-qkv",
                "enc-flash-attention",
            ],
            [item["name"] for item in normalized["manifest"]["ordered_ops"][:4]],
        )
        for removed in ("gpt_v1", "gpt_legacy", "unknown"):
            with self.subTest(removed=removed):
                with self.assertRaisesRegex(ValueError, "unsupported model_impl"):
                    normalizer.normalize_flexpipe_config(make_config(removed))

    def test_public_examples_use_unversioned_names(self):
        config_dir = REPO_ROOT / "examples" / "configs"
        self.assertEqual(
            {"gpt_4rank.json", "gpt_6rank.json", "gpt_stage_brief_6rank.json", "node_example_gpu.json"},
            {path.name for path in config_dir.glob("*.json")},
        )
        for path in config_dir.glob("gpt*.json"):
            self.assertEqual("gpt", json.loads(path.read_text())["model_impl"])

    def test_runtime_and_profiler_expose_no_version_switch(self):
        paths = [
            REPO_ROOT / "eth" / "runtime" / "megatron" / "arguments.py",
            REPO_ROOT / "eth" / "runtime" / "megatron" / "model" / "flex_gpt.py",
            REPO_ROOT / "eth" / "runtime" / "megatron" / "config_normalization.py",
            REPO_ROOT / "eth" / "searching" / "config_adapter.py",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for removed in ("gpt_v1", "gpt_legacy", "GPT v1", "prof-new-gpt", "prof_new_gpt"):
                self.assertNotIn(removed, source, f"{removed} remains in {path}")
        flex_gpt = paths[1].read_text(encoding="utf-8")
        self.assertIn("def _build_gpt_ops(", flex_gpt)

    def test_unversioned_ops_dispatch_to_ten_op_runtime_classes(self):
        path = REPO_ROOT / "eth" / "runtime" / "megatron" / "model" / "flex_ops.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "gen_op"
        )

        class Recorder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        scope = {
            name: type(name, (Recorder,), {})
            for name in (
                "Embedding", "Embedding_v1", "ParallelLayerNormOp",
                "ParallelLayerNormOp_v1", "ParallelQKVOp", "ParallelQKVOp_v1",
                "ParallelDropoutOp", "ParallelDropoutOp_v1",
            )
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        gen_op = scope["gen_op"]
        common = {"op_index": 0, "prev_name": None}
        embedding = gen_op(
            {**common, "name": "encoder-embedding", "embedding_dropout_prob": 0.0,
             "init_method": None, "num_tokentypes": 0},
            0,
        )
        layernorm = gen_op({**common, "name": "enc-1st-layernorm"}, 0)
        qkv = gen_op(
            {**common, "name": "enc-attention-qkv", "init_method": None,
             "attention_type": None},
            0,
        )
        dropout = gen_op({**common, "name": "enc-post-attention-dropout"}, 0)
        self.assertEqual(
            [type(item).__name__ for item in (embedding, layernorm, qkv, dropout)],
            ["Embedding_v1", "ParallelLayerNormOp_v1", "ParallelQKVOp_v1", "ParallelDropoutOp_v1"],
        )

        source = path.read_text(encoding="utf-8")
        self.assertIn("sharing_with_ops\": [10*args.num_layers +2]", source)
        self.assertIn("extra_tensor_send_to = 4", source)
        self.assertIn("recv_from = -3", source)


if __name__ == "__main__":
    unittest.main()
