import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class GptOnlyScopeContractTest(unittest.TestCase):
    def test_unsupported_model_files_are_absent(self):
        forbidden_paths = [
            "runtime/pretrain_t5.py",
            "runtime/pretrain_resnet.py",
            "eth/runtime/megatron/model/flex_t5.py",
            "eth/runtime/megatron/model/flex_resnet.py",
            "eth/runtime/megatron/data/t5_dataset.py",
            "eth/runtime/megatron/data/vit_dataset.py",
            "eth/runtime/megatron/data/autoaugment.py",
            "eth/runtime/megatron/tokenizer/bert_tokenization.py",
            "eth/runtime/vocabs/bert-large-uncased-vocab.txt",
        ]
        for relative_path in forbidden_paths:
            self.assertFalse((REPO_ROOT / relative_path).exists(), relative_path)

        forbidden_globs = [
            "runtime/scripts/*t5*",
            "runtime/scripts/*resnet*",
            "eth/profiler/**/*t5*",
            "eth/profiler/**/*resnet*",
            "search/**/*t5*",
            "search/**/*resnet*",
        ]
        for pattern in forbidden_globs:
            self.assertEqual(list(REPO_ROOT.glob(pattern)), [], pattern)

    def test_shared_model_registries_are_gpt_only(self):
        registry_files = [
            "eth/runtime/megatron/arguments.py",
            "eth/runtime/megatron/model/__init__.py",
            "eth/runtime/megatron/tokenizer/tokenizer.py",
            "eth/profiler/model_configs.py",
            "eth/profiler/op_profiler.py",
            "eth/profiler/comm_profiler.py",
            "eth/searching/arguments.py",
            "eth/searching/config.py",
            "eth/searching/model_ops.py",
        ]
        findings = []
        for relative_path in registry_files:
            text = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
            for marker in ("t5", "resnet", "llama2", "bertwordpiece"):
                if marker in text:
                    findings.append(f"{relative_path}: {marker}")
        self.assertEqual(findings, [])

    def test_public_source_contains_no_t5_or_resnet_branches(self):
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode().split("\0")
        findings = []
        for relative_path in tracked:
            if not relative_path or relative_path.startswith(("docs/", "issues/", "tests/")):
                continue
            path = REPO_ROOT / relative_path
            if path.suffix not in {".py", ".sh", ".md"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").lower()
            for marker in ("t5", "resnet"):
                if marker in text:
                    findings.append(f"{relative_path}: {marker}")
        self.assertEqual(findings, [])

    def test_retained_model_exports_and_profiler_help_are_gpt_only(self):
        python_path = os.pathsep.join(
            [
                str(REPO_ROOT),
                str(REPO_ROOT / "eth" / "runtime"),
                str(REPO_ROOT / "external" / "apex"),
            ]
        )
        env = {**os.environ, "PYTHONPATH": python_path}
        import_result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import megatron.model as m; "
                "assert hasattr(m, 'FlexGPTModel'); "
                "assert not hasattr(m, 'FlexT5Model'); "
                "assert not hasattr(m, 'FlexResNet')",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(import_result.returncode, 0, import_result.stderr)

        help_result = subprocess.run(
            [sys.executable, "-m", "eth.profiler.op_profiler", "--help"],
            cwd=REPO_ROOT.parent,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        help_text = help_result.stdout.lower()
        self.assertIn("--prof-model-name {gpt}", help_text)
        self.assertNotIn("t5", help_text)
        self.assertNotIn("resnet", help_text)


if __name__ == "__main__":
    unittest.main()
