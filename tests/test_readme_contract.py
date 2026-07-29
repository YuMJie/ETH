import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


class ReadmeContractTest(unittest.TestCase):
    def test_release_identity_scope_and_paper_status_are_precise(self):
        text = README.read_text(encoding="utf-8")
        lower = " ".join(text.lower().split())
        self.assertIn("# ETH", text)
        self.assertNotIn("accepted by NSDI 2027", text)
        self.assertNotIn("not public yet", text)
        self.assertIn("Microsoft SuperScaler", text)
        self.assertIn("Megatron-LM", text)
        self.assertIn("GPT", text)
        self.assertIn("HPP", text)
        self.assertIn("Block-wise", text)
        self.assertIn("no profiling measurements", lower)
        for unsupported_scope in (
            "other model families",
            "external baseline comparisons",
            "paper-figure reproduction",
        ):
            self.assertIn(unsupported_scope, lower)
        for unsupported in ("arxiv", "doi", "bibtex", "gpt v1", "alpa", "resnet", " t5"):
            self.assertNotIn(unsupported, lower)

    def test_only_canonical_workflow_and_release_environment_are_documented(self):
        text = README.read_text(encoding="utf-8")
        for script in (
            "scripts/profile_gpt.sh",
            "scripts/search_gpt.sh",
            "scripts/train_gpt.sh",
        ):
            self.assertIn(script, text)
        for legacy in ("our_scripts/", "runtime/scripts/", "npu_profile_new"):
            self.assertNotIn(legacy, text)
        for version in ("Ubuntu 22.04", "Python 3.10", "CUDA 12.4", "PyTorch 2.6.0"):
            self.assertIn(version, text)

    def test_documented_options_exist_in_script_help(self):
        text = README.read_text(encoding="utf-8")
        expectations = {
            "scripts/profile_gpt.sh": (
                "--model-size", "--hardware", "--tp-sizes", "--output",
            ),
            "scripts/search_gpt.sh": (
                "--profile-dir", "--search-budget", "--node-description", "--output",
            ),
            "scripts/train_gpt.sh": (
                "--config", "--gpus-per-node", "--grad-sync-mode", "--validate-only",
            ),
        }
        for script, options in expectations.items():
            result = subprocess.run(
                ["bash", script, "--help"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            for option in options:
                with self.subTest(script=script, option=option):
                    self.assertIn(option, result.stdout)
                    self.assertIn(option, text)


if __name__ == "__main__":
    unittest.main()
