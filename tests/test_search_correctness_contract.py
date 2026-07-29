import argparse
import ast
import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARGUMENTS_PATH = REPO_ROOT / "eth" / "searching" / "arguments.py"
SEARCH_PATH = REPO_ROOT / "eth" / "searching" / "search.py"
VARIABLES_PATH = REPO_ROOT / "eth" / "searching" / "state.py"
DEVICE_GROUPS_PATH = REPO_ROOT / "eth" / "searching" / "device_groups.py"
ACTIONS_PATH = REPO_ROOT / "eth" / "searching" / "actions.py"
MAINTAINED_PATHS = [
    REPO_ROOT / "eth" / "searching" / name
    for name in (
        "search.py",
        "arguments.py",
        "config.py",
        "serialization.py",
        "reporting.py",
        "device_groups.py",
        "state.py",
        "cost_model.py",
        "actions.py",
        "policy.py",
        "model_ops.py",
    )
]


def load_function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    scope = dict(namespace or {})
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class SearchCorrectnessContractTest(unittest.TestCase):
    def test_maintained_modules_avoid_implicit_import_and_exception_behavior(self):
        findings = []
        for path in MAINTAINED_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "*" for alias in node.names
                ):
                    findings.append(f"{path.name}:{node.lineno}: wildcard import")
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    findings.append(f"{path.name}:{node.lineno}: bare except")
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "parse_args"
                ):
                    findings.append(f"{path.name}:{node.lineno}: import-time parse_args")

        self.assertEqual(findings, [])

    def test_public_search_arguments_have_nonempty_help(self):
        tree = ast.parse(ARGUMENTS_PATH.read_text(encoding="utf-8"))
        empty_help = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            option = next(
                (
                    argument.value
                    for argument in node.args
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                ),
                None,
            )
            help_value = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "help"
                    and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
            if option and not help_value:
                empty_help.append((option, node.lineno))

        self.assertEqual(empty_help, [])

    def test_retained_search_symbols_use_clear_spellings(self):
        forbidden = (
            "determinated",
            "gloabl",
            "laset",
            "mamual",
            "pipline",
            "prex",
            "recieved",
            "seach",
            "supportted",
            "throughputt",
            "thpt",
        )
        findings = []
        for path in MAINTAINED_PATHS:
            text = path.read_text(encoding="utf-8").lower()
            for spelling in forbidden:
                if spelling in text:
                    findings.append(f"{path.name}: {spelling}")

        self.assertEqual(findings, [])

    def test_public_search_arguments_have_a_read_point(self):
        read_attributes = set()
        for path in MAINTAINED_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    read_attributes.add(node.attr)

        tree = ast.parse(ARGUMENTS_PATH.read_text(encoding="utf-8"))
        unread = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            option = next(
                (
                    argument.value
                    for argument in node.args
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                ),
                None,
            )
            if not option:
                continue
            destination = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "dest"
                    and isinstance(keyword.value, ast.Constant)
                ),
                option[2:].replace("-", "_").strip(),
            )
            if destination not in read_attributes:
                unread.append((option, node.lineno))

        self.assertEqual(unread, [])

    def test_recomputation_order_keeps_tp_6_and_tp_8_distinct(self):
        get_recomputation_order = load_function(
            VARIABLES_PATH,
            "get_recom_op_name_sort",
            {
                "sort_recom_op_name": {
                    "gpu": {0: {3: {0: ["tp6"]}, 4: {0: ["tp8"]}}}
                }
            },
        )

        self.assertEqual(get_recomputation_order("gpu", 0, 3, 0), ["tp6"])
        self.assertEqual(get_recomputation_order("gpu", 0, 4, 0), ["tp8"])
        with self.assertRaisesRegex(ValueError, "profile index 5"):
            get_recomputation_order("gpu", 0, 5, 0)

    def test_device_group_bounds_apply_to_both_execution_modes(self):
        source = SEARCH_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("selected_device_group_indices("), 3)
        selected_indices = load_function(
            SEARCH_PATH, "selected_device_group_indices"
        )
        args = SimpleNamespace(start_dg_idx=0, end_dg_idx=3)
        self.assertEqual(list(selected_indices(["a", "b", "c", "d"], args)), [1, 2])

        tree = ast.parse(source)
        direct_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_search"
        ]
        serial_call = next(node for node in direct_calls if len(node.args) == 6)
        self.assertIsInstance(serial_call.args[3], ast.Name)
        self.assertEqual(serial_call.args[3].id, "profile_time")
        self.assertIsInstance(serial_call.args[4], ast.Name)
        self.assertEqual(serial_call.args[4].id, "shared_dict")
        self.assertIsInstance(serial_call.args[5], ast.Name)
        self.assertEqual(serial_call.args[5].id, "args")
        self.assertNotIn("result_dict", source)

    def test_device_group_end_option_has_no_trailing_space(self):
        add_arguments = load_function(
            ARGUMENTS_PATH,
            "add_hetero_search_args",
            {"json": __import__("json"), "sys": sys},
        )
        parser = add_arguments(argparse.ArgumentParser())

        parsed = parser.parse_args(["--end-dg-idx", "7"])
        self.assertEqual(parsed.end_dg_idx, 7)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--end-dg-idx ", "7"])

    def test_search_ranges_reject_inverted_bounds(self):
        validate_stage_range = load_function(ARGUMENTS_PATH, "validate_stage_range")
        validate_device_group_range = load_function(
            ARGUMENTS_PATH, "validate_device_group_range"
        )

        valid = SimpleNamespace(start_num_stages=3, end_num_stages=7)
        validate_stage_range(valid)
        self.assertEqual((valid.start_num_stages, valid.end_num_stages), (3, 7))
        with self.assertRaisesRegex(ValueError, "start-num-stages"):
            validate_stage_range(
                SimpleNamespace(start_num_stages=8, end_num_stages=7)
            )
        with self.assertRaisesRegex(ValueError, "start-dg-idx"):
            validate_device_group_range(
                SimpleNamespace(start_dg_idx=7, end_dg_idx=7)
            )

    def test_tp_indices_follow_the_explicit_supported_values(self):
        get_tp_index = load_function(VARIABLES_PATH, "get_tp_index")
        supported = [1, 2, 4, 6, 8]

        self.assertEqual(get_tp_index(6, supported), 3)
        self.assertEqual(get_tp_index(8, supported), 4)
        with self.assertRaisesRegex(ValueError, "unsupported tensor parallel size 3"):
            get_tp_index(3, supported)

    def test_cost_model_does_not_derive_tp_indices_with_logarithms(self):
        source = (REPO_ROOT / "eth" / "searching" / "cost_model.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("math_log_2", source)
        self.assertNotRegex(source, r"math\.log\([^\n]*tp")

    def test_search_uses_the_configured_stage_range(self):
        tree = ast.parse(SEARCH_PATH.read_text(encoding="utf-8"))
        hard_coded_assignments = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id in {"start_num_stages", "end_num_stages"}
                and isinstance(node.value, ast.Constant)
                and node.value.value in {2, 12}
            ):
                hard_coded_assignments.append((target.id, node.value.value, node.lineno))

        self.assertEqual(hard_coded_assignments, [])

    def test_migration_candidates_follow_public_and_legacy_operator_boundaries(self):
        migration_candidate_sizes = load_function(
            ACTIONS_PATH, "migration_candidate_sizes"
        )
        public_layer = [
            "enc-1st-layernorm",
            "enc-attention-qkv",
            "enc-flash-attention",
            "enc-attention-dense",
            "enc-post-attention-dropout",
            "enc-2nd-layernorm",
            "enc-MLP-GEMM-1",
            "enc-MLP-gelu",
            "enc-MLP-GEMM-2",
            "enc-post-MLP-dropout",
        ]
        legacy_layer = [
            "enc-1st-layernorm",
            "enc-attention-qkv",
            "enc-attention-score",
            "enc-attention-softmax",
            "enc-attention-dropout",
            "enc-attention-context",
            "enc-attention-dense",
            "enc-post-attention-dropout",
            "enc-2nd-layernorm",
            "enc-MLP-GEMM-1",
            "enc-MLP-gelu",
            "enc-MLP-GEMM-2",
            "enc-post-MLP-dropout",
        ]

        self.assertEqual(
            migration_candidate_sizes(public_layer * 2, True, "layer"),
            [0, 10],
        )
        self.assertEqual(
            migration_candidate_sizes(legacy_layer * 2, False, "layer"),
            [0, 13],
        )
        self.assertEqual(
            migration_candidate_sizes(public_layer * 2, True, "block"),
            [0, 5, 10, 15],
        )
        self.assertEqual(
            migration_candidate_sizes(legacy_layer * 2, False, "block"),
            [0, 5, 13, 18],
        )

    def test_pipeline_variance_diagnostics_are_not_swapped(self):
        class ArrayMath:
            @staticmethod
            def var(values):
                mean = sum(values) / len(values)
                return sum((value - mean) ** 2 for value in values) / len(values)

        sort_pipelines = load_function(
            DEVICE_GROUPS_PATH,
            "sort_pipeline_split_list",
            {"np": ArrayMath},
        )

        class Args:
            sort_pipeline = True
            num_layers = 1
            global_batch_size = 1
            device_info = {
                "slow": {"TFLOPS": 1},
                "fast": {"TFLOPS": 5},
            }

        pipelines = [[[{"GPU": "slow", "GPU_NUM": 1}, {"GPU": "fast", "GPU_NUM": 1}]]]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sort_pipelines(pipelines, Args())

        self.assertIn("var_inter_gpu 0.0", output.getvalue())
        self.assertIn("var_intra_gpu 4.0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
