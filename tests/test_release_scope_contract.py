import configparser
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseScopeContractTest(unittest.TestCase):
    def test_only_canonical_public_shell_entrypoints_are_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "*.sh"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        tracked_shells = {
            path
            for path in result.stdout.splitlines()
            if (REPO_ROOT / path).is_file()
        }
        self.assertEqual(
            tracked_shells,
            {
                "scripts/profile_gpt.sh",
                "scripts/search_gpt.sh",
                "scripts/train_gpt.sh",
            },
        )

    def test_external_baselines_and_comparison_tools_are_absent(self):
        forbidden_paths = [
            "external/alpa",
            "external/Megatron-LM",
            "external/aceso_ae_megatron.patch",
            "scripts/get_common.py",
            "scripts/get_e2e_performance.py",
            "scripts/get_perf_model_acc.py",
            "scripts/get_search_cost.py",
        ]
        for path in forbidden_paths:
            self.assertFalse((REPO_ROOT / path).exists(), path)

        plot_scripts = list((REPO_ROOT / "scripts").glob("plot_fig*.py"))
        self.assertEqual(plot_scripts, [])

    def test_apex_is_the_only_external_submodule(self):
        config = configparser.ConfigParser()
        config.read(REPO_ROOT / ".gitmodules")
        paths = {config[section]["path"] for section in config.sections()}
        self.assertEqual(paths, {"external/apex"})

    def test_public_entry_files_do_not_reference_removed_baselines(self):
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode().split("\0")
        excluded_prefixes = ("docs/", "issues/", "runtime/", "tests/")
        forbidden = (
            "external/alpa",
            "external/Megatron-LM",
            "aceso_ae_megatron",
            "alpa-projects",
            "Alpa",
        )

        findings = []
        for relative_path in tracked:
            if not relative_path or relative_path.startswith(excluded_prefixes):
                continue
            path = REPO_ROOT / relative_path
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in forbidden:
                if marker in text:
                    findings.append(f"{relative_path}: {marker}")

        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
