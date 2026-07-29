import importlib.util
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validation" / "validate_cuda_toolchain.py"


def load_validator():
    assert VALIDATOR.exists(), "CUDA toolchain validator is missing"
    name = "validate_cuda_toolchain_test"
    spec = importlib.util.spec_from_file_location(name, VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CudaToolchainTest(unittest.TestCase):
    def test_accepts_release_cuda_version(self):
        validator = load_validator()

        versions = validator.validate_versions(
            "12.4", "Cuda compilation tools, release 12.4, V12.4.131"
        )

        self.assertEqual(versions, (12, 4, 12, 4))

    def test_rejects_non_release_cuda_minor(self):
        validator = load_validator()

        with self.assertRaisesRegex(ValueError, "release environment requires CUDA 12.4"):
            validator.validate_versions(
                "12.4", "Cuda compilation tools, release 12.8, V12.8.93"
            )

    def test_rejects_mismatched_major_versions(self):
        validator = load_validator()

        with self.assertRaisesRegex(ValueError, "major version mismatch"):
            validator.validate_versions(
                "12.4", "Cuda compilation tools, release 11.8, V11.8.89"
            )

    def test_cli_rejects_incompatible_nvcc_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = Path(tmp) / "nvcc"
            nvcc.write_text(
                "#!/usr/bin/env bash\necho 'Cuda compilation tools, release 11.8, V11.8.89'\n",
                encoding="utf-8",
            )
            nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)

            result = subprocess.run(
                [sys.executable, str(VALIDATOR), "--nvcc", str(nvcc)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("major version mismatch", result.stdout)
        self.assertIn("PyTorch CUDA", result.stdout)
        self.assertIn("nvcc", result.stdout)


if __name__ == "__main__":
    unittest.main()
