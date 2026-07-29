import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MPU_DIR = REPO_ROOT / "eth" / "runtime" / "megatron" / "mpu"


def load_mpu_initialize():
    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    megatron.get_args = lambda: None
    grad_sync = types.ModuleType("megatron.grad_sync")
    grad_sync.__path__ = []
    topology = types.ModuleType("megatron.grad_sync.topology")
    topology.build_mpu_rank_facts = lambda **kwargs: kwargs
    topology.validate_mpu_rank_facts = lambda expected, actual: None
    mpu_package = types.ModuleType("megatron.mpu")
    mpu_package.__path__ = [str(MPU_DIR)]
    utils = types.ModuleType("megatron.mpu.utils")
    utils.ensure_divisibility = lambda numerator, denominator: None
    sys.modules["megatron"] = megatron
    sys.modules["megatron.grad_sync"] = grad_sync
    sys.modules["megatron.grad_sync.topology"] = topology
    sys.modules["megatron.mpu"] = mpu_package
    sys.modules["megatron.mpu.utils"] = utils

    full_name = "megatron.mpu.initialize"
    path = MPU_DIR / "initialize.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


class FakeDistributed:
    def __init__(self):
        self.calls = []

    def get_world_size(self):
        return 4

    def new_group(self, ranks):
        ranks = tuple(ranks)
        self.calls.append(ranks)
        return "handle-" + "-".join(map(str, ranks))


class MpuGroupRegistryRuntimeTest(unittest.TestCase):
    def test_groups_are_created_once_in_sorted_order_then_lookup_only(self):
        initialize = load_mpu_initialize()
        fake = FakeDistributed()
        initialize.torch.distributed = fake
        initialize.all_groups.clear()

        initialize.initialize_process_groups(((2, 3), (0, 2), (3, 2)))
        initialize.initialize_process_groups(((0, 2), (2, 3)))

        self.assertEqual(fake.calls, [(0, 2), (2, 3)])
        self.assertEqual(initialize.get_group((2, 3)), "handle-2-3")
        with self.assertRaisesRegex(RuntimeError, "is not initialized"):
            initialize.get_group((0, 1))
        self.assertEqual(fake.calls, [(0, 2), (2, 3)])


if __name__ == "__main__":
    unittest.main()
