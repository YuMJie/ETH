import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "tools" / "validation" / "compare_runtime_evidence.py"


def make_report(rank, mode, collective_count):
    payload_bytes = 16 + rank * 4
    digest = f"rank-{rank}-payload"
    scratch = [{"dtype": "float32", "numel": 8, "data_ptr": 1000 + rank}]
    return {
        "version": 1,
        "global_rank": rank,
        "sync_mode": mode,
        "global_topology_hash": "topology",
        "global_manifest_hash": "manifest",
        "global_fragment_hash": "fragments",
        "global_plan_hash": f"{mode}-plan",
        "exact_payload_bytes_by_rank": [[0, 16], [1, 20]],
        "padding_payload_bytes": 0,
        "process_group_count": 2,
        "process_group_creation_count": 2,
        "scratch": scratch,
        "process_group_count_stable": True,
        "process_group_creation_count_stable": True,
        "scratch_stable": True,
        "iterations": [
            {
                "iteration": iteration,
                "normalization_scale": 0.5,
                "collective_count": collective_count,
                "payload_bytes": payload_bytes,
                "packed_payload_sha256": digest,
                "process_group_count": 2,
                "process_group_creation_count": 2,
                "scratch_signature": scratch,
                "buckets": [],
            }
            for iteration in (1, 2)
        ],
    }


class GptRuntimeEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.tensor_dir = root / "tensor"
        self.block_dir = root / "block"
        self.output = root / "comparison.json"
        self.tensor_reports = [make_report(rank, "tensor", 4) for rank in range(2)]
        self.block_reports = [make_report(rank, "block", 2) for rank in range(2)]

    @staticmethod
    def _write_reports(path, reports):
        path.mkdir(parents=True, exist_ok=True)
        for report in reports:
            rank = report["global_rank"]
            (path / f"rank-{rank:05d}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

    def _run(self, tensor_reports=None, block_reports=None):
        self._write_reports(
            self.tensor_dir,
            self.tensor_reports if tensor_reports is None else tensor_reports,
        )
        self._write_reports(
            self.block_dir,
            self.block_reports if block_reports is None else block_reports,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(COMPARE_SCRIPT),
                "--tensor",
                str(self.tensor_dir),
                "--block",
                str(self.block_dir),
                "--output",
                str(self.output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        report = (
            json.loads(self.output.read_text(encoding="utf-8"))
            if self.output.exists()
            else None
        )
        return result, report

    def assert_rejected(self, expected_error, tensor_reports=None, block_reports=None):
        result, report = self._run(tensor_reports, block_reports)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNotNone(report)
        self.assertFalse(report["passed"])
        self.assertIn(expected_error, report["error"])

    def test_accepts_complete_semantically_equal_runs_with_lower_block_count(self):
        result, report = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rank_count"], 2)
        self.assertEqual(report["iterations"], 2)
        self.assertEqual(report["tensor_collectives"], 8)
        self.assertEqual(report["block_collectives"], 4)
        self.assertTrue(report["payload_bytes_equal"])
        self.assertTrue(report["packed_payload_sha256_equal"])
        self.assertEqual(report["padding_payload_bytes"], 0)
        self.assertTrue(report["process_group_count_stable"])
        self.assertTrue(report["process_group_creation_count_stable"])
        self.assertTrue(report["scratch_stable"])

    def test_rejects_incomplete_rank_set(self):
        self.assert_rejected("incomplete rank set", block_reports=self.block_reports[:1])

    def test_rejects_semantic_hash_mismatch(self):
        changed = copy.deepcopy(self.block_reports)
        changed[1]["global_fragment_hash"] = "different-fragments"
        self.assert_rejected("global_fragment_hash mismatch", block_reports=changed)

    def test_rejects_packed_payload_mismatch(self):
        changed = copy.deepcopy(self.block_reports)
        changed[0]["iterations"][1]["packed_payload_sha256"] = "different"
        self.assert_rejected("packed payload mismatch", block_reports=changed)

    def test_rejects_nonzero_padding(self):
        changed = copy.deepcopy(self.block_reports)
        changed[0]["padding_payload_bytes"] = 4
        self.assert_rejected("nonzero padding", block_reports=changed)

    def test_rejects_non_decreasing_collective_count(self):
        changed = [make_report(rank, "block", 4) for rank in range(2)]
        self.assert_rejected("collective count did not decrease", block_reports=changed)

    def test_rejects_unstable_resources(self):
        changed = copy.deepcopy(self.block_reports)
        changed[0]["scratch_stable"] = False
        self.assert_rejected("scratch allocation is not stable", block_reports=changed)

    def test_rejects_runtime_process_group_creation(self):
        changed = copy.deepcopy(self.block_reports)
        changed[0]["process_group_creation_count_stable"] = False
        self.assert_rejected("process-group creation count is not stable", block_reports=changed)


if __name__ == "__main__":
    unittest.main()
