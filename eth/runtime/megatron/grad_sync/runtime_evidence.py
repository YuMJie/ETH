"""Opt-in observations around real gradient collectives."""

import hashlib
import json
from pathlib import Path

import torch


def _scratch_signature(scratch):
    return tuple(
        (
            dtype,
            int(tensor.numel()),
            int(tensor.untyped_storage().data_ptr()),
        )
        for dtype, tensor in sorted(scratch.items())
    )


class GradientRuntimeEvidenceRecorder:
    def __init__(self, plan, global_rank, registry, scratch, output_dir):
        self._plan = plan
        self._rank = int(global_rank)
        self._registry = registry
        self._scratch = scratch
        self._output_dir = Path(output_dir)
        self._initial_group_count = int(registry.group_count)
        self._initial_group_creation_count = int(registry.creation_count)
        self._initial_scratch = _scratch_signature(scratch)
        self._iterations = []
        self._current = None
        self._hasher = None

    def begin_execution(self, normalization_scale):
        if self._current is not None:
            raise RuntimeError("runtime evidence execution already active")
        self._hasher = hashlib.sha256()
        self._current = {
            "iteration": len(self._iterations) + 1,
            "normalization_scale": float(normalization_scale),
            "collective_count": 0,
            "payload_bytes": 0,
            "buckets": [],
        }

    def observe_packed_bucket(self, bucket, payload):
        if self._current is None:
            raise RuntimeError("runtime evidence execution is not active")
        raw = (
            payload.detach()
            .cpu()
            .contiguous()
            .view(dtype=torch.uint8)
            .numpy()
            .tobytes()
        )
        digest = hashlib.sha256(raw).hexdigest()
        self._hasher.update(raw)
        self._current["collective_count"] += 1
        self._current["payload_bytes"] += len(raw)
        self._current["buckets"].append(
            {
                "bucket_id": bucket.bucket_id,
                "semantic_hash": bucket.semantic_hash,
                "process_group_id": bucket.process_group_id,
                "exact_numel": int(bucket.exact_numel),
                "exact_bytes": int(bucket.exact_bytes),
                "packed_payload_sha256": digest,
            }
        )

    def finish_execution(self):
        if self._current is None:
            raise RuntimeError("runtime evidence execution is not active")
        self._current["packed_payload_sha256"] = self._hasher.hexdigest()
        self._current["process_group_count"] = int(self._registry.group_count)
        self._current["process_group_creation_count"] = int(
            self._registry.creation_count
        )
        self._current["scratch_signature"] = [
            {"dtype": dtype, "numel": numel, "data_ptr": data_ptr}
            for dtype, numel, data_ptr in _scratch_signature(self._scratch)
        ]
        self._iterations.append(self._current)
        self._current = None
        self._hasher = None
        self._write()

    def _write(self):
        current_scratch = _scratch_signature(self._scratch)
        report = {
            "version": 1,
            "global_rank": self._rank,
            "sync_mode": self._plan.sync_mode,
            "global_topology_hash": self._plan.global_topology_hash,
            "global_manifest_hash": self._plan.global_manifest_hash,
            "global_fragment_hash": self._plan.global_fragment_hash,
            "global_plan_hash": self._plan.global_plan_hash,
            "exact_payload_bytes_by_rank": [
                [int(rank), int(value)]
                for rank, value in self._plan.exact_payload_bytes_by_rank
            ],
            "padding_payload_bytes": int(self._plan.padding_payload_bytes),
            "process_group_count": self._initial_group_count,
            "process_group_creation_count": self._initial_group_creation_count,
            "scratch": [
                {"dtype": dtype, "numel": numel, "data_ptr": data_ptr}
                for dtype, numel, data_ptr in self._initial_scratch
            ],
            "process_group_count_stable": all(
                item["process_group_count"] == self._initial_group_count
                for item in self._iterations
            ),
            "process_group_creation_count_stable": all(
                item["process_group_creation_count"]
                == self._initial_group_creation_count
                for item in self._iterations
            ),
            "scratch_stable": all(
                tuple(
                    (entry["dtype"], entry["numel"], entry["data_ptr"])
                    for entry in item["scratch_signature"]
                )
                == self._initial_scratch
                for item in self._iterations
            )
            and current_scratch == self._initial_scratch,
            "iterations": self._iterations,
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"rank-{self._rank:05d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
