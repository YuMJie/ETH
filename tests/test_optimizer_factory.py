import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_INIT = REPO_ROOT / "eth" / "runtime" / "megatron" / "optimizer" / "__init__.py"


def load_optimizer_factory():
    apex = types.ModuleType("apex")
    apex_optimizers = types.ModuleType("apex.optimizers")
    apex_optimizers.FusedAdam = torch.optim.Adam
    apex_optimizers.FusedSGD = torch.optim.SGD
    apex.optimizers = apex_optimizers
    sys.modules["apex"] = apex
    sys.modules["apex.optimizers"] = apex_optimizers

    megatron = types.ModuleType("megatron")
    megatron.get_args = lambda: None
    megatron_model = types.ModuleType("megatron.model")
    megatron_model.LayerNorm = torch.nn.LayerNorm
    sys.modules["megatron"] = megatron
    sys.modules["megatron.model"] = megatron_model

    grad_scaler = types.ModuleType("megatron.optimizer.grad_scaler")
    grad_scaler.ConstantGradScaler = object
    grad_scaler.DynamicGradScaler = object
    optimizer = types.ModuleType("megatron.optimizer.optimizer")
    optimizer.Float16OptimizerWithFloat16Params = object
    optimizer.FP32Optimizer = object
    sys.modules[grad_scaler.__name__] = grad_scaler
    sys.modules[optimizer.__name__] = optimizer

    spec = importlib.util.spec_from_file_location(
        "megatron.optimizer", OPTIMIZER_INIT,
        submodule_search_locations=[str(OPTIMIZER_INIT.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OptimizerFactoryTest(unittest.TestCase):
    def test_missing_fused_extension_falls_back_to_torch_adam(self):
        factory = load_optimizer_factory()

        class MissingFusedAdam:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "apex.optimizers.FusedAdam requires cuda extensions"
                )

        class FallbackOptimizer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        result = factory._construct_optimizer(
            MissingFusedAdam,
            FallbackOptimizer,
            [{"params": []}],
            lr=0.1,
        )

        self.assertIsInstance(result, FallbackOptimizer)
        self.assertEqual(result.kwargs["lr"], 0.1)

    def test_unrelated_fused_optimizer_error_is_not_hidden(self):
        factory = load_optimizer_factory()

        class BrokenAdam:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("invalid optimizer configuration")

        with self.assertRaisesRegex(RuntimeError, "invalid optimizer"):
            factory._construct_optimizer(
                BrokenAdam, torch.optim.Adam, [], lr=0.1
            )


if __name__ == "__main__":
    unittest.main()
