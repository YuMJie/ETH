#!/usr/bin/env python3
"""Compare rank-local GPT tensor/block collective runtime evidence."""

import argparse
import json
import sys
from pathlib import Path


SEMANTIC_FIELDS = (
    "global_topology_hash",
    "global_manifest_hash",
    "global_fragment_hash",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _rank_bytes(report):
    return {
        int(rank): int(payload_bytes)
        for rank, payload_bytes in report["exact_payload_bytes_by_rank"]
    }


def _validate_resources(report, path):
    if not report.get("process_group_count_stable", False):
        raise ValueError(f"process-group count is not stable in {path}")
    if not report.get("process_group_creation_count_stable", False):
        raise ValueError(f"process-group creation count is not stable in {path}")
    if not report.get("scratch_stable", False):
        raise ValueError(f"scratch allocation is not stable in {path}")
    expected_groups = int(report["process_group_count"])
    expected_creations = int(report["process_group_creation_count"])
    expected_scratch = report["scratch"]
    for iteration in report["iterations"]:
        if int(iteration["process_group_count"]) != expected_groups:
            raise ValueError(f"process-group count changed in {path}")
        if int(iteration["process_group_creation_count"]) != expected_creations:
            raise ValueError(f"process-group creation count changed in {path}")
        if iteration["scratch_signature"] != expected_scratch:
            raise ValueError(f"scratch allocation changed in {path}")


def load_run(path, expected_mode):
    root = Path(path)
    files = sorted(root.glob("rank-*.json"))
    if not files:
        raise ValueError(f"no rank evidence found under {root}")

    reports = {}
    for file_path in files:
        report = json.loads(file_path.read_text(encoding="utf-8"))
        rank = int(report["global_rank"])
        if rank in reports:
            raise ValueError(f"duplicate global rank {rank} under {root}")
        if report.get("version") != 1:
            raise ValueError(f"unsupported evidence version in {file_path}")
        if report.get("sync_mode") != expected_mode:
            raise ValueError(
                f"expected {expected_mode} evidence, got {report.get('sync_mode')} "
                f"in {file_path}"
            )
        reports[rank] = report

    first = reports[min(reports)]
    declared_bytes = _rank_bytes(first)
    declared_ranks = set(declared_bytes)
    actual_ranks = set(reports)
    if actual_ranks != declared_ranks:
        raise ValueError(
            f"incomplete rank set under {root}: actual={sorted(actual_ranks)} "
            f"declared={sorted(declared_ranks)}"
        )

    first_iterations = [item["iteration"] for item in first["iterations"]]
    if len(first_iterations) < 2 or first_iterations != list(
        range(1, len(first_iterations) + 1)
    ):
        raise ValueError(
            f"runtime evidence requires at least two contiguous iterations, "
            f"got {first_iterations}"
        )

    for rank, report in sorted(reports.items()):
        file_path = root / f"rank-{rank:05d}.json"
        for field in SEMANTIC_FIELDS:
            if report[field] != first[field]:
                raise ValueError(f"{field} mismatch within {expected_mode} run")
        if _rank_bytes(report) != declared_bytes:
            raise ValueError(
                f"exact_payload_bytes_by_rank mismatch within {expected_mode} run"
            )
        if int(report["padding_payload_bytes"]) != 0:
            raise ValueError(f"nonzero padding in {file_path}")
        iteration_numbers = [item["iteration"] for item in report["iterations"]]
        if iteration_numbers != first_iterations:
            raise ValueError(f"iteration set mismatch in {file_path}")
        for iteration in report["iterations"]:
            if int(iteration["payload_bytes"]) != declared_bytes[rank]:
                raise ValueError(
                    f"payload byte count mismatch for rank {rank} iteration "
                    f"{iteration['iteration']}"
                )
        _validate_resources(report, file_path)

    return reports, first_iterations


def compare_runs(tensor_path, block_path):
    tensor, tensor_iterations = load_run(tensor_path, "tensor")
    block, block_iterations = load_run(block_path, "block")
    if set(tensor) != set(block):
        raise ValueError(
            f"incomplete rank set across runs: tensor={sorted(tensor)} "
            f"block={sorted(block)}"
        )
    if tensor_iterations != block_iterations:
        raise ValueError("iteration set mismatch across tensor and block runs")

    first_rank = min(tensor)
    for field in SEMANTIC_FIELDS:
        if tensor[first_rank][field] != block[first_rank][field]:
            raise ValueError(f"{field} mismatch across tensor and block runs")
    if _rank_bytes(tensor[first_rank]) != _rank_bytes(block[first_rank]):
        raise ValueError("exact_payload_bytes_by_rank mismatch across runs")

    tensor_totals = []
    block_totals = []
    payload_bytes_equal = True
    packed_payload_sha256_equal = True
    by_rank = []
    for rank in sorted(tensor):
        tensor_report = tensor[rank]
        block_report = block[rank]
        rank_tensor_counts = []
        rank_block_counts = []
        for tensor_iteration, block_iteration in zip(
            tensor_report["iterations"], block_report["iterations"]
        ):
            iteration = tensor_iteration["iteration"]
            if (
                float(tensor_iteration["normalization_scale"])
                != float(block_iteration["normalization_scale"])
            ):
                raise ValueError(
                    f"normalization scale mismatch for rank {rank} iteration {iteration}"
                )
            if tensor_iteration["payload_bytes"] != block_iteration["payload_bytes"]:
                payload_bytes_equal = False
            if (
                tensor_iteration["packed_payload_sha256"]
                != block_iteration["packed_payload_sha256"]
            ):
                packed_payload_sha256_equal = False
            tensor_count = int(tensor_iteration["collective_count"])
            block_count = int(block_iteration["collective_count"])
            if block_count >= tensor_count:
                raise ValueError(
                    f"collective count did not decrease for rank {rank} "
                    f"iteration {iteration}: tensor={tensor_count} block={block_count}"
                )
            rank_tensor_counts.append(tensor_count)
            rank_block_counts.append(block_count)
        by_rank.append(
            {
                "global_rank": rank,
                "tensor_collectives": rank_tensor_counts,
                "block_collectives": rank_block_counts,
            }
        )

    if not payload_bytes_equal:
        raise ValueError("packed payload byte count mismatch across runs")
    if not packed_payload_sha256_equal:
        raise ValueError("packed payload mismatch across runs")

    for index in range(len(tensor_iterations)):
        tensor_totals.append(
            sum(item["tensor_collectives"][index] for item in by_rank)
        )
        block_totals.append(sum(item["block_collectives"][index] for item in by_rank))
    if len(set(tensor_totals)) != 1 or len(set(block_totals)) != 1:
        raise ValueError(
            f"collective count changed across iterations: "
            f"tensor={tensor_totals} block={block_totals}"
        )

    return {
        "passed": True,
        "rank_count": len(tensor),
        "iterations": len(tensor_iterations),
        "tensor_collectives": tensor_totals[0],
        "block_collectives": block_totals[0],
        "collectives_by_rank": by_rank,
        "payload_bytes_equal": payload_bytes_equal,
        "packed_payload_sha256_equal": packed_payload_sha256_equal,
        "padding_payload_bytes": 0,
        "process_group_count_stable": True,
        "process_group_creation_count_stable": True,
        "scratch_stable": True,
        "global_topology_hash": tensor[first_rank]["global_topology_hash"],
        "global_manifest_hash": tensor[first_rank]["global_manifest_hash"],
        "global_fragment_hash": tensor[first_rank]["global_fragment_hash"],
        "tensor_plan_hash": tensor[first_rank]["global_plan_hash"],
        "block_plan_hash": block[first_rank]["global_plan_hash"],
    }


def write_report(path, report):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    try:
        report = compare_runs(args.tensor, args.block)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        write_report(args.output, {"passed": False, "error": str(error)})
        print(str(error), file=sys.stderr)
        return 1
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
