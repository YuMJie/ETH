import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "tokenizer"


def load_tokenizer_module():
    package = types.ModuleType("megatron.tokenizer")
    package.__path__ = [str(TOKENIZER_DIR)]
    sys.modules["megatron.tokenizer"] = package
    for name, class_name in (
        ("bert_tokenization", "FullTokenizer"),
        ("gpt2_tokenization", "GPT2Tokenizer"),
    ):
        dependency = types.ModuleType(f"megatron.tokenizer.{name}")
        setattr(dependency, class_name, object)
        sys.modules[dependency.__name__] = dependency
    spec = importlib.util.spec_from_file_location(
        "megatron.tokenizer.tokenizer", TOKENIZER_DIR / "tokenizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TokenizerPaddingTest(unittest.TestCase):
    def test_heterogeneous_pipelines_use_global_max_tp_padding(self):
        tokenizer = load_tokenizer_module()
        all_pipeline_tp = [
            [[1, 1]],
            [[2, 2]],
        ]
        common = {
            "rank": 1,
            "make_vocab_size_divisible_by": 128,
            "model_parallel_size_of_each_op_each_pipe": all_pipeline_tp,
        }
        tp1_rank = types.SimpleNamespace(
            **common,
            num_stages=1,
            model_parallel_size_of_each_op=[[1, 1]],
        )
        tp2_rank = types.SimpleNamespace(
            **common,
            num_stages=1,
            model_parallel_size_of_each_op=[[2, 2]],
        )

        self.assertEqual(
            tokenizer._vocab_size_with_padding(50257, tp1_rank), 50432
        )
        self.assertEqual(
            tokenizer._vocab_size_with_padding(50257, tp2_rank), 50432
        )


if __name__ == "__main__":
    unittest.main()
