import importlib.util
import sys
import types
import unittest
from pathlib import Path

from tests.test_config_normalization import load_normalizer
from tests.test_config_normalization import make_config
from tests.test_config_normalization import make_pipe
from tests.test_grad_sync_topology import load_topology


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAD_SYNC_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync"


def load_adapter():
    megatron = types.ModuleType("megatron")
    megatron.__path__ = [str(GRAD_SYNC_DIR.parent)]
    grad_sync = types.ModuleType("megatron.grad_sync")
    grad_sync.__path__ = [str(GRAD_SYNC_DIR)]
    sys.modules["megatron"] = megatron
    sys.modules["megatron.grad_sync"] = grad_sync
    for name in ("metadata", "gpt_adapter"):
        full_name = f"megatron.grad_sync.{name}"
        path = GRAD_SYNC_DIR / f"{name}.py"
        assert path.exists(), f"{name} module is missing"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules["megatron.grad_sync.gpt_adapter"]


class FakeGradient:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"


class FakeParameter:
    def __init__(
        self,
        shape,
        dtype="float16",
        tensor_parallel=False,
        partition_dim=-1,
        partition_stride=1,
        requires_grad=True,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"
        self.requires_grad = requires_grad
        self.tensor_model_parallel = tensor_parallel
        self.partition_dim = partition_dim
        self.partition_stride = partition_stride
        self.main_grad = FakeGradient(shape, "float32")


def module(**kwargs):
    return types.SimpleNamespace(**kwargs)


def runtime_op(name, op_index, **kwargs):
    return FakeRuntimeOp(name=name, op_index=op_index, **kwargs)


class FakeRuntimeOp(types.SimpleNamespace):
    def parameters(self, recurse=True):
        result = []

        def visit(value):
            if isinstance(value, FakeParameter):
                result.append(value)
            elif isinstance(value, types.SimpleNamespace):
                for nested in vars(value).values():
                    visit(nested)

        for value in vars(self).values():
            visit(value)
        return result


class GptGradientMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        normalizer = load_normalizer()
        cls.v1 = normalizer.normalize_flexpipe_config(
            make_config([make_pipe([43], [2], [1])]), world_size=2
        )
        cls.v1_topology = load_topology().build_canonical_topology(cls.v1)

    def test_adapts_column_qkv_weight_and_bias_from_explicit_paths(self):
        adapter = load_adapter()
        op = runtime_op(
            "enc-attention-qkv",
            2,
            algo="column",
            query_key_value=module(
                weight=FakeParameter((96, 64), tensor_parallel=True, partition_dim=0),
                bias=FakeParameter((96,), tensor_parallel=True, partition_dim=0),
            ),
        )

        descriptors, bindings = adapter.adapt_gpt_runtime_op(
            op, self.v1["manifest"], self.v1_topology, global_rank=0
        )

        self.assertEqual(len(descriptors), 2)
        weight = descriptors[0]
        self.assertEqual(weight.logical_param_id, "gpt.layers.0.attention.query_key_value.weight")
        self.assertEqual(weight.full_shape, (192, 64))
        self.assertEqual(weight.local_shape, (96, 64))
        self.assertEqual((weight.partition_dim, weight.partition_stride), (0, 1))
        self.assertEqual(weight.local_tp_group_ranks, (0, 1))
        self.assertEqual(weight.gradient_dtype, "float32")
        self.assertEqual(bindings[0].logical_param_id, weight.logical_param_id)

    def test_adapts_row_weight_and_replicated_bias(self):
        adapter = load_adapter()
        op = runtime_op(
            "enc-attention-dense",
            4,
            algo="row",
            dense=module(
                weight=FakeParameter((64, 32), tensor_parallel=True, partition_dim=1),
                bias=FakeParameter((64,)),
            ),
        )

        descriptors, _ = adapter.adapt_gpt_runtime_op(
            op, self.v1["manifest"], self.v1_topology, global_rank=1
        )

        self.assertEqual(descriptors[0].full_shape, (64, 64))
        self.assertIsNone(descriptors[0].tp_replica_semantics)
        self.assertEqual(descriptors[1].full_shape, (64,))
        self.assertEqual(descriptors[1].tp_replica_semantics, "identical_duplicate")

    def test_adapts_embeddings_layernorm_tied_head_and_parameterless_ops(self):
        adapter = load_adapter()
        embedding = runtime_op(
            "encoder-embedding",
            0,
            word_embeddings=module(
                weight=FakeParameter((128, 64), tensor_parallel=True, partition_dim=0)
            ),
            position_embeddings=module(weight=FakeParameter((32, 64))),
            tokentype_embeddings=None,
        )
        head = runtime_op(
            "gpt-post-process",
            42,
            word_embeddings=module(
                weight=FakeParameter((128, 64), tensor_parallel=True, partition_dim=0)
            ),
        )
        layernorm = runtime_op(
            "enc-1st-layernorm",
            1,
            input_layernorm=module(
                weight=FakeParameter((64,), dtype="bfloat16"),
                bias=FakeParameter((64,), dtype="bfloat16"),
            ),
        )
        parameterless = runtime_op("enc-flash-attention", 3)

        embedding_desc, _ = adapter.adapt_gpt_runtime_op(
            embedding, self.v1["manifest"], self.v1_topology, 0
        )
        head_desc, _ = adapter.adapt_gpt_runtime_op(
            head, self.v1["manifest"], self.v1_topology, 0
        )
        norm_desc, _ = adapter.adapt_gpt_runtime_op(
            layernorm, self.v1["manifest"], self.v1_topology, 0
        )
        empty_desc, empty_bindings = adapter.adapt_gpt_runtime_op(
            parameterless, self.v1["manifest"], self.v1_topology, 0
        )

        self.assertEqual(embedding_desc[0].shared_weight_id, "gpt.word_embeddings")
        self.assertNotEqual(head_desc[0].logical_param_id, embedding_desc[0].logical_param_id)
        self.assertEqual(head_desc[0].shared_weight_id, embedding_desc[0].shared_weight_id)
        self.assertTrue(embedding_desc[0].shared_weight_root)
        self.assertFalse(head_desc[0].shared_weight_root)
        self.assertEqual(norm_desc[0].parameter_dtype, "bfloat16")
        self.assertEqual((empty_desc, empty_bindings), ((), ()))

    def test_rejects_runtime_name_missing_tp_metadata_and_duplicate_parameters(self):
        adapter = load_adapter()
        bad_name = runtime_op("wrong", 1)
        with self.assertRaisesRegex(ValueError, "rank 0.*op 1.*runtime name"):
            adapter.adapt_gpt_runtime_op(
                bad_name, self.v1["manifest"], self.v1_topology, 0
            )

        missing = FakeParameter((64,))
        del missing.partition_stride
        layernorm = runtime_op(
            "enc-1st-layernorm",
            1,
            input_layernorm=module(weight=missing, bias=FakeParameter((64,))),
        )
        with self.assertRaisesRegex(ValueError, "logical_param_id.*rank 0.*partition_stride"):
            adapter.adapt_gpt_runtime_op(
                layernorm, self.v1["manifest"], self.v1_topology, 0
            )

        shared = FakeParameter((64,))
        duplicate = runtime_op(
            "enc-1st-layernorm",
            1,
            input_layernorm=module(weight=shared, bias=shared),
        )
        with self.assertRaisesRegex(ValueError, "duplicate parameter.*rank 0"):
            adapter.adapt_gpt_runtime_op(
                duplicate, self.v1["manifest"], self.v1_topology, 0
            )

        inconsistent = runtime_op(
            "enc-attention-dense",
            4,
            algo="row",
            dense=module(
                weight=FakeParameter((32, 64), tensor_parallel=True, partition_dim=0),
                bias=FakeParameter((64,), tensor_parallel=True, partition_dim=0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "logical_param_id.*expected sharded_dim1"):
            adapter.adapt_gpt_runtime_op(
                inconsistent, self.v1["manifest"], self.v1_topology, 0
            )

        unexpected = runtime_op(
            "enc-flash-attention",
            3,
            unexpected=FakeParameter((1,)),
        )
        with self.assertRaisesRegex(ValueError, "unlisted trainable parameter.*op 3"):
            adapter.adapt_gpt_runtime_op(
                unexpected, self.v1["manifest"], self.v1_topology, 0
            )

    def test_collects_runtime_model_chunks_after_main_grad_binding(self):
        adapter = load_adapter()
        op = runtime_op(
            "enc-1st-layernorm",
            1,
            input_layernorm=module(
                weight=FakeParameter((64,)),
                bias=FakeParameter((64,)),
            ),
        )
        chunks = [module(language_model=module(ops=[op]))]
        actual_stage = self.v1_topology.stage_for_rank(0)
        stage = module(
            pipeline_id=actual_stage.pipeline_id,
            stage_id=actual_stage.stage_id,
            global_op_begin=1,
            global_op_end=2,
        )
        single_op_topology = module(
            stage_for_rank=lambda rank: stage,
            expected_rank_facts=self.v1_topology.expected_rank_facts,
        )

        descriptors, bindings = adapter.collect_gpt_gradient_metadata(
            chunks,
            self.v1["manifest"],
            single_op_topology,
            global_rank=0,
        )

        self.assertEqual(len(descriptors), 2)
        self.assertEqual(len(bindings), 2)
        self.assertEqual(descriptors[0].model_chunk_id, 0)

        wider_stage = module(
            pipeline_id=actual_stage.pipeline_id,
            stage_id=actual_stage.stage_id,
            global_op_begin=1,
            global_op_end=3,
        )
        wider_topology = module(
            stage_for_rank=lambda rank: wider_stage,
            expected_rank_facts=self.v1_topology.expected_rank_facts,
        )
        with self.assertRaisesRegex(ValueError, "rank 0.*missing runtime op indices.*2"):
            adapter.collect_gpt_gradient_metadata(
                chunks, self.v1["manifest"], wider_topology, 0
            )
        duplicate_chunks = [module(language_model=module(ops=[op, op]))]
        with self.assertRaisesRegex(ValueError, "rank 0.*duplicate runtime op index 1"):
            adapter.collect_gpt_gradient_metadata(
                duplicate_chunks, self.v1["manifest"], single_op_topology, 0
            )


if __name__ == "__main__":
    unittest.main()
