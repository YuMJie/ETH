import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "dependency_build.py"


def load_dependency_build():
    assert MODULE_PATH.exists(), "node-local dependency build coordinator is missing"
    name = "megatron.dependency_build"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DependencyBuildTest(unittest.TestCase):
    def test_local_rank_zero_builds_before_both_global_barriers(self):
        coordinator = load_dependency_build()
        events = []

        coordinator.coordinate_node_local_build(
            local_rank=0,
            compile_dataset=lambda: events.append("dataset"),
            load_fused=lambda: events.append("fused"),
            barrier=lambda: events.append("barrier"),
        )

        self.assertEqual(events, ["dataset", "fused", "barrier", "barrier"])

    def test_local_peer_waits_then_loads_completed_cache(self):
        coordinator = load_dependency_build()
        events = []

        coordinator.coordinate_node_local_build(
            local_rank=2,
            compile_dataset=lambda: events.append("dataset"),
            load_fused=lambda: events.append("fused"),
            barrier=lambda: events.append("barrier"),
        )

        self.assertEqual(events, ["barrier", "fused", "barrier"])

    def test_builder_ownership_does_not_depend_on_global_rank(self):
        coordinator = load_dependency_build()
        events = []

        coordinator.coordinate_node_local_build(
            local_rank=0,
            compile_dataset=lambda: events.append("remote-node-dataset"),
            load_fused=lambda: events.append("remote-node-fused"),
            barrier=lambda: events.append("barrier"),
        )

        self.assertEqual(
            events,
            ["remote-node-dataset", "remote-node-fused", "barrier", "barrier"],
        )


if __name__ == "__main__":
    unittest.main()
