import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "data" / "dataset_utils.py"


def load_dataset_utils():
    spec = importlib.util.spec_from_file_location("dataset_utils_contract", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DatasetBuildContractTest(unittest.TestCase):
    def test_helper_build_uses_the_active_python_interpreter(self):
        dataset_utils = load_dataset_utils()
        completed = mock.Mock(returncode=0)

        with mock.patch.object(dataset_utils.subprocess, "run", return_value=completed) as run:
            dataset_utils.compile_helper()

        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["make", "-C", str(MODULE_PATH.parent)])
        self.assertIn(f"PYTHON={sys.executable}", command)

    def test_makefile_derives_pybind_and_extension_flags_from_python(self):
        makefile = (MODULE_PATH.parent / "Makefile").read_text(encoding="utf-8")

        self.assertIn("PYTHON ?= python3", makefile)
        self.assertIn('"$(PYTHON)" -m pybind11 --includes', makefile)
        self.assertIn('"$(PYTHON)" -c', makefile)


if __name__ == "__main__":
    unittest.main()
