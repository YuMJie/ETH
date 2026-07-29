import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "search_gpt.sh"
LEGACY_SCRIPT = REPO_ROOT / "our_scripts" / "exp" / "gpt_hs_ft.sh"
PRIMITIVES = ("all_gather", "all_reduce", "reduce_scatter", "all_to_all")
OP_NAMES = (
    "encoder-embedding",
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
    "final-layernorm",
    "gpt-post-process",
)


def write_csv(path, header, row):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(row)


def make_profile(
    root,
    *,
    status="complete",
    model_size="1_3B",
    hardware="example-gpu",
    tp_sizes=(1, 2, 4),
    op_names=OP_NAMES,
):
    root.mkdir()
    metadata = {
        "schema_version": 1,
        "profile_status": status,
        "model": {"name": "gpt", "size": model_size},
        "hardware": hardware,
        "tp_sizes": list(tp_sizes),
        "distributed": {"num_nodes": 1, "gpus_per_node": 4, "node_rank": 0},
        "environment": {"python": "3.10", "torch": "2.6.0", "cuda": "12.4"},
    }
    (root / "profile_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    op_header = [
        "op_name", "forward-compute", "backward-compute", "input_size",
        "output_size", "weights", "activations", "fwd_reserved", "bwd_reserved",
    ]
    for mbs in (1, 2, 4, 8):
        for tp in tp_sizes:
            for algo in (0, 1):
                path = root / f"gpt_{model_size}_mbs{mbs}_tp{tp}_algo{algo}.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(op_header)
                    for op_name in op_names:
                        writer.writerow([op_name, 1, 1, 1, 1, 1, 1, 1, 1])
    for tp in tp_sizes:
        if tp == 1:
            continue
        for primitive in PRIMITIVES:
            write_csv(
                root / f"prim_gpt_{model_size}_{primitive}_{tp}gpus.csv",
                ["data_size(MB)", "time(ms)"],
                [1, 1],
            )
    sizes = [2**index for index in range(11)]
    write_csv(root / "p2p_intra_node.csv", sizes, [100] * len(sizes))


def make_node_description(path, *, hardware="example-gpu", gpu_num=4, node_num=1):
    description = {
        "node_info": {
            "0": {"GPU": hardware, "GPU_NUM": gpu_num, "NODE_NUM": node_num}
        },
        "device_info": {
            hardware: {"memory": 64000, "intra_bandwidth": 20, "TFLOPS": 100}
        },
    }
    path.write_text(json.dumps(description), encoding="utf-8")


def run_search(profile, nodes, output, *extra):
    return subprocess.run(
        [
            "bash", str(SCRIPT),
            "--profile-dir", str(profile),
            "--model-size", "1_3B",
            "--search-budget", "1",
            "--global-batch-size", "16",
            "--node-description", str(nodes),
            "--output", str(output),
            *extra,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )


class GptSearchContractTest(unittest.TestCase):
    def test_missing_required_args_print_usage_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "search"
            result = subprocess.run(
                ["bash", str(SCRIPT), "--output", str(output)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Usage:", result.stdout)
            self.assertIn("--profile-dir", result.stdout)
            self.assertFalse(output.exists())

    def test_profile_metadata_mismatches_fail_before_output(self):
        cases = (
            ({"status": "planned"}, "complete"),
            ({"model_size": "350M"}, "model size"),
            ({"hardware": "other-gpu"}, "hardware"),
            ({"tp_sizes": (1,)}, "TP coverage"),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                profile = tmp_path / "profile"
                nodes = tmp_path / "nodes.json"
                output = tmp_path / "search"
                make_profile(profile, **overrides)
                make_node_description(nodes)

                result = run_search(profile, nodes, output, "--dry-run")

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout)
                self.assertFalse(output.exists())

    def test_fractional_search_budget_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile = tmp_path / "profile"
            nodes = tmp_path / "nodes.json"
            output = tmp_path / "search"
            make_profile(profile)
            make_node_description(nodes)

            result = subprocess.run(
                [
                    "bash", str(SCRIPT),
                    "--profile-dir", str(profile),
                    "--model-size", "1_3B",
                    "--search-budget", "0.1",
                    "--global-batch-size", "16",
                    "--node-description", str(nodes),
                    "--output", str(output),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("positive integer", result.stdout)
            self.assertFalse(output.exists())

    def test_unsupported_profile_operator_manifest_fails_before_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile = tmp_path / "profile"
            nodes = tmp_path / "nodes.json"
            output = tmp_path / "search"
            make_profile(profile, op_names=OP_NAMES[:-1])
            make_node_description(nodes)

            result = run_search(profile, nodes, output, "--python", sys.executable)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("supported GPT operator manifest", result.stdout)
            self.assertEqual(list((output / "configs").rglob("*.json")), [])

    def test_valid_profile_dry_run_creates_normalized_compatibility_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile = tmp_path / "profile"
            nodes = tmp_path / "nodes.json"
            output = tmp_path / "search"
            make_profile(profile)
            make_node_description(nodes)

            result = run_search(profile, nodes, output, "--dry-run")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(
                json.loads((output / "node_info.json").read_text(encoding="utf-8"))["0"]["GPU"],
                "example-gpu",
            )
            self.assertIn(
                "example-gpu",
                json.loads((output / "device_info.json").read_text(encoding="utf-8")),
            )
            model_view = output / "profile_view" / "example-gpu" / "gpt_1_3B"
            self.assertTrue(model_view.is_symlink())
            self.assertEqual(model_view.resolve(), profile.resolve())
            self.assertTrue(
                (output / "profile_view" / "example-gpu" / "p2p_intra_node.csv").is_symlink()
            )

            plan = (output / "search_plan.txt").read_text(encoding="utf-8")
            self.assertIn("-m eth.searching.search", plan)
            self.assertIn("--profiled-time-path <output>/profile_view/", plan)
            self.assertIn("--max-tp 4", plan)
            self.assertNotIn("npu_profile_new", plan)

    def test_temporary_profile_drives_real_search_and_runtime_validation(self):
        workflow_root = None
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workflow_root = tmp_path
            profile = tmp_path / "profile"
            nodes = tmp_path / "nodes.json"
            output = tmp_path / "search"
            unrelated_cwd = tmp_path / "unrelated-cwd"
            unrelated_cwd.mkdir()
            make_profile(profile)
            make_node_description(nodes)

            result = subprocess.run(
                [
                    "bash", str(SCRIPT),
                    "--profile-dir", str(profile),
                    "--model-size", "1_3B",
                    "--search-budget", "1",
                    "--global-batch-size", "16",
                    "--node-description", str(nodes),
                    "--output", str(output),
                    "--python", sys.executable,
                ],
                cwd=unrelated_cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            config_path = output / "gpt_config.json"
            self.assertTrue(config_path.is_file())
            training_validation = subprocess.run(
                [
                    "bash", str(REPO_ROOT / "scripts" / "train_gpt.sh"),
                    "--config", str(config_path),
                    "--gpus-per-node", "4",
                    "--num-nodes", "1",
                    "--node-rank", "0",
                    "--master-addr", "localhost",
                    "--master-port", "29500",
                    "--grad-sync-mode", "block",
                    "--python", sys.executable,
                    "--validate-only",
                ],
                cwd=unrelated_cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            self.assertEqual(
                training_validation.returncode, 0, training_validation.stdout
            )
            self.assertIn("model_impl=gpt", training_validation.stdout)
            self.assertIn("distributed launch skipped", training_validation.stdout)

        self.assertIsNotNone(workflow_root)
        self.assertFalse(workflow_root.exists())

    def test_legacy_search_entry_is_removed(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)
        self.assertFalse(LEGACY_SCRIPT.exists())
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/search_gpt.sh", readme)
        self.assertNotIn("our_scripts/exp/gpt_hs_ft.sh", readme)


if __name__ == "__main__":
    unittest.main()
