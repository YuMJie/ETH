import ast
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PACKAGE = REPO_ROOT / "eth" / "searching"


class ObsoleteSearchCleanupContractTest(unittest.TestCase):
    def test_public_sources_have_no_obsolete_project_names(self):
        forbidden_names = ("aceso", "metis", "whale")
        findings = []
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode().split("\0")
        text_suffixes = {
            ".py", ".sh", ".json", ".md", ".toml", ".txt", ".yaml", ".yml"
        }
        excluded_prefixes = ("tests/", "external/", "LICENSES/", "docs/superpowers/")
        for relative in tracked:
            if not relative or relative.startswith(excluded_prefixes):
                continue
            path = REPO_ROOT / relative
            if relative in {"LICENSE", "NOTICE"} or (
                path.suffix not in text_suffixes
                and path.name not in {"Dockerfile", ".gitignore"}
            ):
                continue
            if relative.startswith("eth/runtime/vocabs/"):
                continue
            source = path.read_text(encoding="utf-8").lower()
            for name in forbidden_names:
                if name in source:
                    findings.append(f"{relative}:{name}")
        self.assertEqual([], findings)

    def test_legacy_search_tree_is_absent(self):
        self.assertFalse((REPO_ROOT / "hetero_search").exists())

    def test_retained_search_modules_have_no_embedded_experiments(self):
        forbidden_definitions = {
            "simulator_config",
            "predict_time_breakdown_with_interleave",
            "gen_dgroups_recursive",
            "permute",
            "gen_device_group_shapes",
            "get_device_group_list",
            "gen_dgroups_for_stages_with_variance",
            "find_combinations",
            "find_powers_of_two",
            "generate_permutations",
            "generate_permutations_large",
            "unique_permutations",
            "split_devices_to_pipeline",
            "get_cached_throughput",
            "max_fun_sum_with_splits",
            "max_fun_sum",
            "test",
            *(f"test{index}" for index in range(1, 10)),
        }
        findings = []
        for path in SEARCH_PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name in forbidden_definitions:
                        findings.append(f"{path.name}:{node.lineno}:{node.name}")
        self.assertEqual([], findings)

    def test_public_tree_has_no_removed_search_or_simulator_references(self):
        removed_names = {
            "aceso_search.py",
            "aceso_search_dp_hete.py",
            "gen_megatron_plan.py",
            "pp_simulator",
            "simulator_config",
            "HyperConfig",
            "OneFOneBGenerator",
            "InterleavedOneFOneBGenerator",
            "OperationExecutor",
            "metis",
        }
        findings = []
        for root in (REPO_ROOT / "eth", REPO_ROOT / "scripts"):
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".sh"}:
                    continue
                source = path.read_text(encoding="utf-8")
                for name in removed_names:
                    if name in source:
                        findings.append(f"{path.relative_to(REPO_ROOT)}:{name}")
        self.assertEqual([], findings)


if __name__ == "__main__":
    unittest.main()
