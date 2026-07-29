import subprocess
import sys
import unittest
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def tracked_files():
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return [path for path in output.split("\0") if path]


class RepositoryHygieneContractTest(unittest.TestCase):
    def test_tracked_sources_do_not_expose_private_paths_or_credentials(self):
        patterns = {
            "home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
            "data path": re.compile(r"/(?:data|mnt)/[A-Za-z0-9._-]+/"),
            "private IPv4": re.compile(
                r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
                r"192\.168\.\d{1,3}\.\d{1,3}|"
                r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
            ),
            "internal hostname": re.compile(r"\b[A-Za-z0-9._-]+\.(?:corp|internal|local)\b"),
            "authenticated URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
            "SSH repository URL": re.compile(r"\bgit@[A-Za-z0-9._-]+:[^\s]+"),
            "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
            "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
            "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
            "assigned password": re.compile(
                r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|secret)\b"
                r"\s*[:=]\s*['\"][^'\"]{4,}['\"]"
            ),
            "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
        }
        findings = []
        for relative_path in tracked_files():
            path = REPO_ROOT / relative_path
            if not path.is_file():
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in patterns.items():
                if pattern.search(source):
                    findings.append(f"{relative_path}: {label}")
        self.assertEqual([], findings)

    def test_release_legal_and_governance_files_are_precise(self):
        for path in (
            "LICENSE",
            "NOTICE",
            "LICENSES/MIT.txt",
            "LICENSES/Apache-2.0.txt",
            "LICENSES/BSD-3-Clause.txt",
        ):
            self.assertTrue((REPO_ROOT / path).is_file(), path)
        for path in ("SECURITY.md", "THIRD_PARTY_NOTICES.md"):
            self.assertFalse((REPO_ROOT / path).exists(), path)
        notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
        for provenance in ("Microsoft SuperScaler", "NVIDIA Megatron-LM", "NVIDIA Apex"):
            self.assertIn(provenance, notice)

    def test_ruff_is_pinned_and_scoped_to_maintained_code(self):
        requirements = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        self.assertIn("ruff==0.15.12", requirements.splitlines())
        config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('extend-exclude = ["eth/runtime/megatron/**"]', config)
        self.assertNotIn("mypy", requirements.lower())

    def test_maintained_scope_passes_ruff(self):
        grad_sync_files = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync").glob("*.py")
        )
        self.assertGreaterEqual(len(grad_sync_files), 10)
        checked_paths = [
            "eth/searching",
            "eth/profiler",
            "tools/validation",
            "tests",
            "eth/runtime/pretrain_gpt.py",
            "eth/runtime/megatron/arguments.py",
            "eth/runtime/megatron/config_normalization.py",
            *grad_sync_files,
            "eth/runtime/megatron/model/flex_gpt.py",
            "eth/runtime/megatron/model/flex_ops.py",
        ]
        result = subprocess.run(
            [
                sys.executable, "-m", "ruff", "check",
                *checked_paths,
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        listed = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--show-files", *grad_sync_files],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        ).stdout
        for path in grad_sync_files:
            self.assertIn(str(REPO_ROOT / path), listed)

    def test_internal_and_generated_artifacts_are_not_tracked(self):
        forbidden_prefixes = (
            "issues/",
            "docs/superpowers/",
            ".agents/",
            ".codex/",
            ".mission/",
            "logs/",
            "logs-large/",
            "results/",
        )
        forbidden_suffixes = (
            ".csv", ".pkl", ".log", ".out", ".err", ".pt", ".pth",
            ".ckpt", ".so", ".pyc", ".ipynb",
        )
        findings = []
        for path in tracked_files():
            lower = path.lower()
            if path.startswith(forbidden_prefixes) or lower.endswith(forbidden_suffixes):
                findings.append(path)
            if any(part in {"__pycache__", "build", "checkpoints", "evidence"}
                   for part in Path(path).parts):
                findings.append(path)
        self.assertEqual(sorted(set(findings)), [])

    def test_json_allowlist_and_gpt_tokenizer_are_explicit(self):
        allowed_json = {"eth/runtime/vocabs/gpt2-vocab.json"}
        findings = []
        paths = tracked_files()
        for path in paths:
            if path.endswith(".json") and path not in allowed_json \
                    and not path.startswith("examples/configs/"):
                findings.append(path)
        self.assertEqual(findings, [])
        self.assertIn("eth/runtime/vocabs/gpt2-vocab.json", paths)
        self.assertIn("eth/runtime/vocabs/gpt2-merges.txt", paths)

        ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertNotIn("*.json", ignore_text.splitlines())

    def test_tracked_modes_match_file_roles(self):
        output = subprocess.run(
            ["git", "ls-files", "-s", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode()
        findings = []
        for record in output.split("\0"):
            if not record:
                continue
            metadata, path = record.split("\t", 1)
            mode = metadata.split()[0]
            if mode == "100755" and not path.endswith(".sh"):
                findings.append(f"unexpected executable: {path}")
            if path.endswith(".sh") and mode != "100755":
                findings.append(f"non-executable shell script: {path}")
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
