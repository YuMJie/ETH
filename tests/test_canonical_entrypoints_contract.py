import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CanonicalEntrypointsContractTest(unittest.TestCase):
    def test_only_three_public_shell_entrypoints_exist(self):
        shell_files = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob("*.sh")
            if "external" not in path.parts
        }
        self.assertEqual(
            {
                "scripts/profile_gpt.sh",
                "scripts/search_gpt.sh",
                "scripts/train_gpt.sh",
            },
            shell_files,
        )

    def test_validation_helpers_are_outside_scripts(self):
        self.assertEqual(
            {
                "compare_numerical_evidence.py",
                "compare_runtime_evidence.py",
                "validate_cuda_toolchain.py",
                "validate_train_config.py",
            },
            {
                path.name
                for path in (REPO_ROOT / "tools" / "validation").glob("*.py")
                if path.name != "__init__.py"
            },
        )
        self.assertEqual(
            {"profile_gpt.sh", "search_gpt.sh", "train_gpt.sh"},
            {path.name for path in (REPO_ROOT / "scripts").iterdir() if path.is_file()},
        )
        self.assertFalse((REPO_ROOT / "our_scripts").exists())

    def test_launchers_derive_repository_paths_from_their_location(self):
        for name in ("profile_gpt.sh", "search_gpt.sh", "train_gpt.sh"):
            source = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn('BASH_SOURCE[0]', source, name)
            self.assertIn("REPO_ROOT", source, name)
            self.assertNotIn("/home/", source, name)
            self.assertNotIn("conda activate", source, name)


if __name__ == "__main__":
    unittest.main()
