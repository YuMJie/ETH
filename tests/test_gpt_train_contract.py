import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "train_gpt.sh"
LEGACY_SCRIPT = REPO_ROOT / "our_scripts" / "exp" / "gpt_train.sh"


class GptTrainContractTest(unittest.TestCase):
    def test_gpt_only_parser_accepts_public_four_rank_config(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "eth" / "runtime")
        env["WORLD_SIZE"] = "4"
        env["RANK"] = "0"
        probe = textwrap.dedent(
            """
            import sys
            from megatron.arguments import parse_args

            sys.argv = [
                "probe",
                "--flexpipe-config", "examples/configs/gpt_4rank.json",
                "--train-iters", "2",
                "--eval-iters", "0",
                "--lr-decay-iters", "320000",
                "--distributed-backend", "nccl",
                "--lr", "0.00015",
                "--lr-decay-style", "cosine",
                "--min-lr", "1e-5",
                "--weight-decay", "0.01",
                "--clip-grad", "1.0",
                "--lr-warmup-fraction", ".01",
                "--DDP-impl", "local",
            ]
            args = parse_args()
            assert args.model_name == "gpt"
            assert args.seq_length == 32
            print("gpt-parse-ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("gpt-parse-ok", result.stdout)

    def test_canonical_entry_replaces_legacy_launcher_families(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)
        self.assertFalse(LEGACY_SCRIPT.exists())
        self.assertFalse((REPO_ROOT / "eth" / "runtime" / "scripts").exists())
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/train_gpt.sh", readme)
        self.assertNotIn("our_scripts/exp/gpt_train.sh", readme)

    def test_invalid_config_is_rejected_before_cuda_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config.json"
            config.write_text("{}", encoding="utf-8")
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            marker = tmp_path / "python-called"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                f"#!/usr/bin/env bash\ntouch '{marker}'\nexit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

            result = subprocess.run(
                [
                    "bash", str(SCRIPT),
                    "--config", str(config),
                    "--gpus-per-node", "1",
                    "--num-nodes", "1",
                    "--node-rank", "0",
                    "--master-addr", "localhost",
                    "--master-port", "7012",
                    "--python", sys.executable,
                ],
                cwd=tmp_path,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("config validation failed", result.stdout)
            self.assertFalse(marker.exists())

    def test_validate_only_accepts_stage_brief_and_rejects_world_mismatch(self):
        config = REPO_ROOT / "examples" / "configs" / "gpt_stage_brief_6rank.json"
        base_args = [
            "bash", str(SCRIPT),
            "--config", str(config),
            "--num-nodes", "2",
            "--node-rank", "0",
            "--master-addr", "training-master.example",
            "--master-port", "29500",
            "--grad-sync-mode", "block",
            "--python", sys.executable,
            "--validate-only",
        ]

        accepted = subprocess.run(
            [*base_args, "--gpus-per-node", "3"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stdout)
        self.assertIn("model_impl=gpt", accepted.stdout)
        self.assertIn("world_size=6", accepted.stdout)
        self.assertIn("grad_sync_mode=block", accepted.stdout)
        self.assertIn("distributed launch skipped", accepted.stdout)

        rejected = subprocess.run(
            [*base_args, "--gpus-per-node", "2"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("world size mismatch", rejected.stdout)
    def test_toolchain_preflight_creates_python_shim_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config.json"
            config.write_text("{}", encoding="utf-8")
            log_dir = tmp_path / "logs"
            fake_bin = tmp_path / "python-bin"
            fake_bin.mkdir()
            scripts_bin = tmp_path / "scripts-bin"
            scripts_bin.mkdir()
            cuda_home = tmp_path / "cuda"
            (cuda_home / "bin").mkdir(parents=True)
            events = tmp_path / "events.txt"
            shim_capture = tmp_path / "python_shim.txt"
            ninja_capture = tmp_path / "ninja_path.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    if [[ "${{1:-}}" == "-c" ]]; then
                        printf '%s\n' {scripts_bin}
                        exit 0
                    fi
                    if [[ "${{1:-}}" == *validate_train_config.py ]]; then
                        exit 0
                    fi
                    if [[ "${{1:-}}" == *validate_cuda_toolchain.py ]]; then
                        echo preflight >> {events}
                        exit 0
                    fi
                    echo launch >> {events}
                    command -v python > {shim_capture}
                    command -v ninja > {ninja_capture}
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            fake_ninja = scripts_bin / "ninja"
            fake_ninja.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_ninja.chmod(fake_ninja.stat().st_mode | stat.S_IXUSR)
            fake_nvcc = cuda_home / "bin" / "nvcc"
            fake_nvcc.write_text(
                "#!/usr/bin/env bash\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n",
                encoding="utf-8",
            )
            fake_nvcc.chmod(fake_nvcc.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:/bin"
            env["CUDA_HOME"] = str(cuda_home)

            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--config",
                    str(config),
                    "--gpus-per-node",
                    "1",
                    "--num-nodes",
                    "1",
                    "--node-rank",
                    "0",
                    "--master-addr",
                    "localhost",
                    "--master-port",
                    "7010",
                    "--log-dir",
                    str(log_dir),
                    "--python",
                    str(fake_python),
                ],
                cwd=tmp_path,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(events.read_text(encoding="utf-8").splitlines(), ["preflight", "launch"])
            shim = Path(shim_capture.read_text(encoding="utf-8").strip())
            self.assertEqual(shim, log_dir / "toolchain-bin" / "python")
            self.assertEqual(shim.resolve(), fake_python.resolve())
            self.assertEqual(
                Path(ninja_capture.read_text(encoding="utf-8").strip()), fake_ninja
            )

    def test_missing_nvcc_is_rejected_before_distributed_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config.json"
            config.write_text("{}", encoding="utf-8")
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
            env["CUDA_HOME"] = str(tmp_path / "missing-cuda")

            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--config",
                    str(config),
                    "--gpus-per-node",
                    "1",
                    "--num-nodes",
                    "1",
                    "--node-rank",
                    "0",
                    "--master-addr",
                    "localhost",
                    "--master-port",
                    "7011",
                ],
                cwd=tmp_path,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nvcc not found", result.stdout)

    def test_missing_required_args_prints_usage_and_exits_nonzero(self):
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("--config <config_json>", result.stdout)
        self.assertIn("--gpus-per-node <gpus_per_node>", result.stdout)
        self.assertIn("--precision <fp32|fp16|bf16>", result.stdout)
        self.assertIn("--grad-sync-mode <tensor|block>", result.stdout)
        self.assertIn("--evidence-dir <snapshot_dir>", result.stdout)
        self.assertIn("--vocab-padding-multiple <value>", result.stdout)

    def test_valid_args_launch_pretrain_gpt_with_config_and_create_csv_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config.json"
            config.write_text("{}", encoding="utf-8")
            log_dir = tmp_path / "runtime_logs"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            capture = tmp_path / "python_args.txt"
            env_capture = tmp_path / "nccl_env.txt"
            pythonpath_capture = tmp_path / "pythonpath.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\n' "$@" > {capture}
                    printf '%s\\n' "${{NCCL_SOCKET_IFNAME:-}}" > {env_capture}
                    printf '%s\\n' "${{PYTHONPATH:-}}" > {pythonpath_capture}
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["NCCL_SOCKET_IFNAME"] = "eth-test0"
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--config",
                    str(config),
                    "--gpus-per-node",
                    "4",
                    "--num-nodes",
                    "1",
                    "--node-rank",
                    "0",
                    "--master-addr",
                    "localhost",
                    "--master-port",
                    "7000",
                    "--log-dir",
                    str(log_dir),
                    "--train-iters",
                    "3",
                ],
                cwd=tmp_path,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue((log_dir / "csv").is_dir())
            args = capture.read_text(encoding="utf-8")
            self.assertIn("-m\ntorch.distributed.launch", args)
            self.assertIn("--nproc_per_node\n4", args)
            self.assertIn("--nnodes\n1", args)
            self.assertIn("--node_rank\n0", args)
            self.assertIn("--master_addr\nlocalhost", args)
            self.assertIn("--master_port\n7000", args)
            self.assertIn("pretrain_gpt.py", args)
            self.assertIn(f"--flexpipe-config\n{config}", args)
            self.assertIn("--train-iters\n3", args)
            self.assertIn("--eval-iters\n0", args)
            self.assertIn("--vocab-file\nvocabs/gpt2-vocab.json", args)
            self.assertIn("--merge-file\nvocabs/gpt2-merges.txt", args)
            self.assertIn(f"--log-path\n{log_dir}", args)
            arg_lines = args.splitlines()
            self.assertEqual(
                arg_lines[arg_lines.index("--log-path") + 1],
                f"{log_dir}{os.sep}",
            )
            self.assertEqual(env_capture.read_text(encoding="utf-8").strip(), "eth-test0")
            self.assertIn(
                str(REPO_ROOT / "external" / "apex"),
                pythonpath_capture.read_text(encoding="utf-8"),
            )
            self.assertIn(
                str(REPO_ROOT / "eth" / "runtime"),
                pythonpath_capture.read_text(encoding="utf-8"),
            )
            self.assertNotIn("192.168.", args)
            self.assertNotIn("eno12399np0", args)
            self.assertNotIn("aceso", args.lower())

    def test_precision_and_grad_sync_options_map_without_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config.json"
            config.write_text('{"model_impl":"gpt"}', encoding="utf-8")
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            capture = tmp_path / "python_args.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\n' "$@" > {capture}
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            for precision, expected_flag in (
                ("fp32", None),
                ("fp16", "--fp16"),
                ("bf16", "--bf16"),
            ):
                plan_path = tmp_path / f"{precision}-plan.json"
                evidence_dir = tmp_path / f"{precision}-evidence"
                runtime_evidence_dir = tmp_path / f"{precision}-runtime-evidence"
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPT),
                        "--config",
                        str(config),
                        "--gpus-per-node",
                        "4",
                        "--num-nodes",
                        "1",
                        "--node-rank",
                        "0",
                        "--master-addr",
                        "localhost",
                        "--master-port",
                        "7001",
                        "--log-dir",
                        str(tmp_path / f"logs-{precision}"),
                        "--train-iters",
                        "2",
                        "--precision",
                        precision,
                        "--loss-scale",
                        "1024",
                        "--attention-dropout",
                        "0",
                        "--hidden-dropout",
                        "0",
                        "--clip-grad",
                        "0.1",
                        "--grad-sync-mode",
                        "block",
                        "--grad-sync-bucket-size-mb",
                        "32",
                        "--grad-sync-dump-plan",
                        str(plan_path),
                        "--grad-sync-debug-replicated-check",
                        "--grad-sync-metadata-transport",
                        "object",
                        "--grad-sync-runtime-evidence-dir",
                        str(runtime_evidence_dir),
                        "--evidence-dir",
                        str(evidence_dir),
                        "--evidence-canonical-init",
                        "--vocab-padding-multiple",
                        "256",
                    ],
                    cwd=tmp_path,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                )

                self.assertEqual(result.returncode, 0, result.stdout)
                args = capture.read_text(encoding="utf-8").splitlines()
                self.assertEqual("--fp16" in args, expected_flag == "--fp16")
                self.assertEqual("--bf16" in args, expected_flag == "--bf16")
                self.assertIn("--loss-scale", args)
                self.assertIn("1024", args)
                self.assertIn("--attention-dropout", args)
                self.assertIn("--hidden-dropout", args)
                self.assertIn("--clip-grad", args)
                self.assertIn("0.1", args)
                self.assertIn("--grad-sync-mode", args)
                self.assertIn("block", args)
                self.assertIn("--grad-sync-bucket-size-mb", args)
                self.assertIn("32", args)
                self.assertIn("--grad-sync-dump-plan", args)
                self.assertIn(str(plan_path), args)
                self.assertIn("--grad-sync-debug-replicated-check", args)
                self.assertIn("--grad-sync-metadata-transport", args)
                self.assertIn("object", args)
                self.assertIn("--grad-sync-runtime-evidence-dir", args)
                self.assertIn(str(runtime_evidence_dir), args)
                self.assertIn("--grad-sync-evidence-dir", args)
                self.assertIn(str(evidence_dir), args)
                self.assertIn("--grad-sync-evidence-canonical-init", args)
                self.assertIn("--make-vocab-size-divisible-by", args)
                self.assertIn("256", args)

    def test_invalid_precision_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--config",
                    str(config),
                    "--gpus-per-node",
                    "1",
                    "--num-nodes",
                    "1",
                    "--node-rank",
                    "0",
                    "--master-addr",
                    "localhost",
                    "--master-port",
                    "7002",
                    "--precision",
                    "tf32",
                ],
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported precision", result.stdout)

    def test_runtime_parser_accepts_grad_sync_options(self):
        probe = textwrap.dedent(
            """
            import argparse
            import json
            from megatron.arguments import _add_distributed_args
            from megatron.arguments import _add_grad_sync_args

            parser = _add_distributed_args(_add_grad_sync_args(argparse.ArgumentParser()))
            args = parser.parse_args([
                "--local-rank", "3",
                "--grad-sync-mode", "block",
                "--grad-sync-bucket-size-mb", "64",
                "--grad-sync-dump-plan", "/tmp/plan.json",
                "--grad-sync-debug-replicated-check",
                "--grad-sync-metadata-transport", "object",
                "--grad-sync-runtime-evidence-dir", "/tmp/runtime-evidence",
                "--grad-sync-evidence-dir", "/tmp/evidence",
                "--grad-sync-evidence-canonical-init",
            ])
            print(json.dumps(vars(args), sort_keys=True))
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "eth" / "runtime")
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('"grad_sync_mode": "block"', result.stdout)
        self.assertIn('"grad_sync_bucket_size_mb": 64', result.stdout)
        self.assertIn('"grad_sync_dump_plan": "/tmp/plan.json"', result.stdout)
        self.assertIn('"grad_sync_debug_replicated_check": true', result.stdout)
        self.assertIn('"grad_sync_metadata_transport": "object"', result.stdout)
        self.assertIn(
            '"grad_sync_runtime_evidence_dir": "/tmp/runtime-evidence"',
            result.stdout,
        )
        self.assertIn('"grad_sync_evidence_dir": "/tmp/evidence"', result.stdout)
        self.assertIn('"grad_sync_evidence_canonical_init": true', result.stdout)
        self.assertIn('"local_rank": 3', result.stdout)

    def test_runtime_data_import_supports_current_numpy(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "eth" / "runtime")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from megatron.data import indexed_dataset; "
                "print(indexed_dataset.dtypes[6].__name__)",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("float32", result.stdout)

    def test_runtime_imports_without_compiled_apex_extensions(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(REPO_ROOT / "external" / "apex"),
                str(REPO_ROOT / "eth" / "runtime"),
            ]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import megatron.model; import megatron.optimizer.optimizer; "
                "print('runtime-import-ok')",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("runtime-import-ok", result.stdout)

    def test_runtime_sitecustomize_restores_torch_six_for_bundled_apex(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "eth" / "runtime")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from torch._six import string_classes; "
                "assert string_classes == (str,); print('torch-six-shim-ok')",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("torch-six-shim-ok", result.stdout)

    def test_gpt_gelu_has_a_non_npu_runtime_path(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(REPO_ROOT / "external" / "apex"),
                str(REPO_ROOT / "eth" / "runtime"),
            ]
        )
        probe = textwrap.dedent(
            """
            import torch
            import torch.nn.functional as F
            from megatron.model.flex_ops import fast_gelu
            value = torch.tensor([-1.0, 0.0, 1.0])
            torch.testing.assert_close(fast_gelu(value), F.gelu(value, approximate="tanh"))
            print("gelu-fallback-ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("gelu-fallback-ok", result.stdout)

    def test_gpt_flash_attention_has_a_causal_non_npu_runtime_path(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(REPO_ROOT / "external" / "apex"),
                str(REPO_ROOT / "eth" / "runtime"),
            ]
        )
        probe = textwrap.dedent(
            """
            import torch
            import torch.nn.functional as F
            import megatron.model.flex_ops as flex_ops

            torch.manual_seed(7)
            query = torch.randn(4, 2, 3, 5)
            key = torch.randn(4, 2, 3, 5)
            value = torch.randn(4, 2, 3, 5)
            class UnexpectedNpu:
                def npu_fusion_attention(self, *args, **kwargs):
                    raise AssertionError("CPU tensors must not use NPU attention")

            flex_ops.torch_npu = UnexpectedNpu()
            actual = flex_ops.flash_attention(query, key, value)
            expected = F.scaled_dot_product_attention(
                query.permute(1, 2, 0, 3),
                key.permute(1, 2, 0, 3),
                value.permute(1, 2, 0, 3),
                is_causal=True,
            ).permute(2, 0, 1, 3).reshape(4, 2, 15)
            torch.testing.assert_close(actual, expected)
            print("flash-attention-fallback-ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("flash-attention-fallback-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
