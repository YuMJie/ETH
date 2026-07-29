import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicEnvironmentContractTest(unittest.TestCase):
    def test_docker_python_cuda_torch_and_apex_pin_are_consistent(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        launcher = (REPO_ROOT / "scripts" / "train_gpt.sh").read_text(encoding="utf-8")

        self.assertIn("nvidia/cuda:12.4", dockerfile)
        self.assertIn("ubuntu22.04", dockerfile)
        self.assertIn("python3.10", dockerfile)
        self.assertIn("torch==2.6.0", dockerfile)
        self.assertIn("cu124", dockerfile)
        self.assertIn("requirements.txt", dockerfile)
        self.assertIn("requirements-dev.txt", dockerfile)
        self.assertIn("nvidia-ml-py", requirements)
        self.assertNotIn("pynvml==", requirements)
        self.assertIn("a651e2c24ecf97cbf367fd3f330df36760e1c597", launcher)

    def test_docker_apex_build_loads_runtime_compatibility_shims(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

        runtime_copy = dockerfile.index("COPY eth /opt/eth/eth")
        apex_shim = dockerfile.index(
            "ENV PYTHONPATH=/opt/eth/eth/runtime:/opt/eth", runtime_copy
        )
        apex_install = dockerfile.index("python3.10 -m pip install", apex_shim)
        self.assertLess(runtime_copy, apex_install)
        self.assertLess(apex_shim, apex_install)
        self.assertIn("python3.10 -c \"import apex", dockerfile)
        self.assertNotIn("APEX_CPP_EXT", dockerfile)
        self.assertNotIn("APEX_CUDA_EXT", dockerfile)

    def test_neutral_examples_cover_four_rank_six_rank_and_stage_brief(self):
        examples = REPO_ROOT / "examples" / "configs"
        expected = {
            "gpt_4rank.json": 4,
            "gpt_6rank.json": 6,
            "gpt_stage_brief_6rank.json": 6,
        }
        for name, world_size in expected.items():
            with self.subTest(name=name):
                path = examples / name
                self.assertTrue(path.is_file())
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/home/", text)
                self.assertNotIn("MASTER_ADDR", text)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json, sys; sys.path.insert(0, 'eth/runtime'); "
                            "from megatron.config_normalization import normalize_flexpipe_config; "
                            "config=json.load(open(sys.argv[1])); "
                            "normalized=normalize_flexpipe_config(config, world_size=int(sys.argv[2])); "
                            "assert normalized['model_impl'] == 'gpt'"
                        ),
                        str(path),
                        str(world_size),
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
