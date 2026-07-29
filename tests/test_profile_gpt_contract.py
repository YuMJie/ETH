import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "profile_gpt.sh"


def run_script(*args, cwd=REPO_ROOT, env=None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )


class ProfileGptContractTest(unittest.TestCase):
    def test_profiler_runtime_path_resolves_bundled_apex_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "profile"
            wrapper = tmp_path / "python-wrapper"
            wrapper.write_text(
                f"""#!{sys.executable}
import os
import sys
from pathlib import Path

import apex
import eth.profiler.op_profiler

expected = Path({str(REPO_ROOT / 'external' / 'apex')!r}).resolve()
assert Path(apex.__file__).resolve().is_relative_to(expected)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
""",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)

            result = run_script(
                "--model", "gpt",
                "--model-size", "1_3B",
                "--hardware", "example-gpu",
                "--tp-sizes", "1,2",
                "--num-nodes", "1",
                "--gpus-per-node", "2",
                "--node-rank", "0",
                "--master-addr", "127.0.0.1",
                "--master-port", "29500",
                "--output", str(output),
                "--python", str(wrapper),
                "--dry-run",
                cwd=tmp_path,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue((output / "profile_metadata.json").is_file())

    def test_missing_required_arguments_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile"
            result = run_script("--output", str(output))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Usage:", result.stdout)
            self.assertIn("--model-size", result.stdout)
            self.assertIn("--hardware", result.stdout)
            self.assertIn("--tp-sizes", result.stdout)
            self.assertFalse(output.exists())

    def test_invalid_topology_fails_before_profiler_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "profile"
            marker = tmp_path / "python-called"
            fake_python = tmp_path / "python3"
            fake_python.write_text(
                f"#!/usr/bin/env bash\ntouch '{marker}'\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{tmp_path}:{env['PATH']}"

            result = run_script(
                "--model", "gpt",
                "--model-size", "1_3B",
                "--hardware", "example-gpu",
                "--tp-sizes", "1,3",
                "--num-nodes", "1",
                "--gpus-per-node", "2",
                "--node-rank", "0",
                "--master-addr", "127.0.0.1",
                "--master-port", "29500",
                "--output", str(output),
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("power of two", result.stdout)
            self.assertFalse(marker.exists())
            self.assertFalse(output.exists())

    def test_missing_cuda_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "profile"
            fake_python = tmp_path / "python3"
            fake_python.write_text(
                "#!/usr/bin/env bash\nexit 1\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{tmp_path}:{env['PATH']}"

            result = run_script(
                "--model", "gpt",
                "--model-size", "1_3B",
                "--hardware", "example-gpu",
                "--tp-sizes", "1,2",
                "--num-nodes", "1",
                "--gpus-per-node", "2",
                "--node-rank", "0",
                "--master-addr", "127.0.0.1",
                "--master-port", "29500",
                "--output", str(output),
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA", result.stdout)
            self.assertFalse(output.exists())

    def test_dry_run_writes_compatible_metadata_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile"
            result = run_script(
                "--model", "gpt",
                "--model-size", "1_3B",
                "--hardware", "example-gpu",
                "--tp-sizes", "1,2,4",
                "--num-nodes", "2",
                "--gpus-per-node", "4",
                "--node-rank", "1",
                "--master-addr", "profile-master.example",
                "--master-port", "29500",
                "--output", str(output),
                "--dry-run",
                cwd=Path(tmp),
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            metadata = json.loads(
                (output / "profile_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["profile_status"], "planned")
            self.assertEqual(metadata["model"], {"name": "gpt", "size": "1_3B"})
            self.assertEqual(metadata["hardware"], "example-gpu")
            self.assertEqual(metadata["tp_sizes"], [1, 2, 4])
            self.assertEqual(metadata["distributed"]["num_nodes"], 2)
            self.assertEqual(metadata["distributed"]["gpus_per_node"], 4)
            self.assertIn("python", metadata["environment"])
            self.assertIn("torch", metadata["environment"])
            self.assertIn("cuda", metadata["environment"])

            plan = (output / "profile_plan.txt").read_text(encoding="utf-8")
            self.assertIn("--module", plan)
            self.assertIn("eth.profiler.op_profiler", plan)
            self.assertIn("-m eth.profiler.comm_profiler", plan)
            self.assertIn("-m eth.profiler.p2p_band_profiler", plan)
            self.assertIn("--prof-tp-size 4", plan)
            self.assertNotIn("/home/", plan)

    def test_legacy_profile_shell_families_are_removed(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)
        for path in (
            REPO_ROOT / "eth" / "profiler" / "scripts",
            REPO_ROOT / "eth" / "profiler" / "our_scripts",
            REPO_ROOT / "eth" / "profiler" / "our_scripts_hete",
        ):
            self.assertFalse(path.exists(), path)

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/profile_gpt.sh", readme)
        self.assertNotIn("profile_small.sh", readme)
        self.assertNotIn("profile_large.sh", readme)
        self.assertNotIn("profiler/our_scripts", readme)

    def test_profiler_is_only_available_from_the_eth_package(self):
        self.assertTrue((REPO_ROOT / "eth" / "profiler" / "__init__.py").is_file())

        tracked = subprocess.run(
            ["git", "ls-files", "profiler"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(tracked.stdout, "")


if __name__ == "__main__":
    unittest.main()
