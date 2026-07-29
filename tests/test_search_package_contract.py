import ast
import importlib
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PACKAGE = REPO_ROOT / "eth" / "searching"
EXPECTED_MODULES = {
    "actions.py",
    "arguments.py",
    "config.py",
    "config_adapter.py",
    "cost_model.py",
    "device_groups.py",
    "model_ops.py",
    "policy.py",
    "profile_validation.py",
    "reporting.py",
    "search.py",
    "serialization.py",
    "state.py",
}
REMOVED_SYMBOLS = {
    "AcesoStageInfo",
    "AcesoConfig",
    "AcesoPrim",
    "HybridConfig",
    "generate_initial_config_v1",
    "find_combinations_v1",
    "find_combinations_v2",
}


class SearchPackageContractTest(unittest.TestCase):
    def test_search_package_has_focused_modules_without_utils(self):
        self.assertTrue((SEARCH_PACKAGE / "__init__.py").is_file())
        self.assertEqual(
            EXPECTED_MODULES,
            {path.name for path in SEARCH_PACKAGE.glob("*.py") if path.name != "__init__.py"},
        )
        self.assertFalse((SEARCH_PACKAGE / "utils.py").exists())

    def test_public_symbols_use_behavioral_names_without_legacy_aliases(self):
        expected_symbols = {
            "eth.searching.state": {
                "SearchStage",
                "SearchConfig",
                "HeterogeneousPlan",
            },
            "eth.searching.actions": {"SearchAction"},
            "eth.searching.device_groups": {
                "enumerate_homogeneous_stage_sizes",
                "enumerate_node_aligned_stage_sizes",
            },
        }
        for module_name, symbols in expected_symbols.items():
            module = importlib.import_module(module_name)
            for symbol in symbols:
                self.assertTrue(hasattr(module, symbol), f"{module_name}.{symbol}")
            for symbol in REMOVED_SYMBOLS:
                self.assertFalse(hasattr(module, symbol), f"legacy alias {module_name}.{symbol}")

    def test_search_module_is_the_canonical_module_entrypoint(self):
        result = subprocess.run(
            [sys.executable, "-m", "eth.searching.search", "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--model-name", result.stdout)

    def test_active_search_tree_has_no_legacy_module_or_symbol_names(self):
        legacy_names = {
            "aceso_search_dp_hete_large",
            "aceso_cost_model",
            "aceso_policy",
            "aceso_prims",
            "aceso_var",
            "aceso_utils",
            "model_ops_info",
            "convert_search_config",
            "validate_profile",
            *REMOVED_SYMBOLS,
        }
        for path in SEARCH_PACKAGE.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for legacy_name in legacy_names:
                self.assertNotIn(legacy_name, source, f"{legacy_name} remains in {path}")

    def test_device_enumeration_helpers_are_owned_by_device_groups(self):
        owned_helpers = {
            "split_devices_to_pipeline_large",
        }
        definitions = {}
        for path in SEARCH_PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name in owned_helpers:
                        definitions.setdefault(node.name, []).append(path.name)

        self.assertEqual(
            {name: ["device_groups.py"] for name in owned_helpers},
            definitions,
        )


if __name__ == "__main__":
    unittest.main()
