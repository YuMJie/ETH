import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DDP_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "model" / "distributed.py"


def load_distributed():
    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    megatron.mpu = types.SimpleNamespace(
        get_data_parallel_world_size=lambda: 1,
        get_data_parallel_group=lambda: None,
    )
    model_package = types.ModuleType("megatron.model")
    model_package.__path__ = []
    module = types.ModuleType("megatron.model.module")
    module.MegatronModule = torch.nn.Module
    sys.modules["megatron"] = megatron
    sys.modules["megatron.model"] = model_package
    sys.modules["megatron.model.module"] = module

    full_name = "megatron.model.distributed"
    spec = importlib.util.spec_from_file_location(full_name, DDP_PATH)
    distributed = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = distributed
    spec.loader.exec_module(distributed)
    return distributed


class ModuleWithUnusedParameter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.unused = torch.nn.Parameter(torch.tensor([3.0]))

    def forward(self, value):
        return (self.used * value).sum()


class LocalDDPRuntimeTest(unittest.TestCase):
    def test_main_grad_is_unconditional_accumulates_and_keeps_unused_zero(self):
        distributed = load_distributed()
        module = ModuleWithUnusedParameter()
        ddp = distributed.DistributedDataParallel(
            module,
            accumulate_allreduce_grads_in_fp32=False,
            use_contiguous_buffers=False,
        )

        self.assertTrue(hasattr(module.used, "main_grad"))
        self.assertTrue(hasattr(module.unused, "main_grad"))
        self.assertEqual(module.used.main_grad.device, module.used.device)
        self.assertEqual(module.used.main_grad.dtype, module.used.dtype)

        ddp(torch.tensor([2.0, 4.0])).backward()
        self.assertTrue(torch.equal(module.used.main_grad, torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(module.unused.main_grad, torch.zeros(1)))
        self.assertIsNone(module.used.grad)

        ddp(torch.tensor([1.0, 1.0])).backward()
        self.assertTrue(torch.equal(module.used.main_grad, torch.tensor([3.0, 5.0])))

        ddp.zero_grad_buffer()
        self.assertTrue(torch.equal(module.used.main_grad, torch.zeros(2)))
        self.assertTrue(torch.equal(module.unused.main_grad, torch.zeros(1)))


if __name__ == "__main__":
    unittest.main()
