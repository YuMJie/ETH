import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_shared_sync():
    path = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync" / "shared.py"
    spec = importlib.util.spec_from_file_location("test_shared_sync", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedWeightSyncTest(unittest.TestCase):
    def test_gradient_merge_averages_dp_replicas_after_concatenating_tp_shards(self):
        shared = load_shared_sync()
        groups = [
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        ]

        merged = shared.merge_shared_tensor_groups(
            groups, split_dim=0, average_replicas=True
        )

        torch.testing.assert_close(
            merged, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )


if __name__ == "__main__":
    unittest.main()
