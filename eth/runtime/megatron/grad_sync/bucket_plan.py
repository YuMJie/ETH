"""Bucket placement and deterministic process-group registry."""

import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass


_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}


def _hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BucketFragmentPlacement:
    fragment_id: str
    destination_offset: int


@dataclass(frozen=True)
class CommunicationBucketSpec:
    bucket_id: str
    process_group_id: str
    grad_dtype: str
    reduction_semantics: str
    placements: tuple
    exact_numel: int
    exact_bytes: int
    logical_region_begin: str
    logical_region_end: str
    semantic_hash: str
    oversized: bool


@dataclass(frozen=True)
class GradientSyncPlan:
    version: int
    sync_mode: str
    bucket_cap_bytes: int
    model_impl: str
    op_manifest_hash: str
    process_groups: tuple
    process_group_index_by_id: tuple
    canonical_fragments: tuple
    local_fragment_bindings_by_rank: tuple
    global_bucket_order: tuple
    max_scratch_numel_by_rank_and_dtype: tuple
    exact_payload_bytes_by_rank: tuple
    padding_payload_bytes: int
    global_topology_hash: str
    global_manifest_hash: str
    global_fragment_hash: str
    global_plan_hash: str


def _dtype_bytes(dtype):
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported gradient dtype {dtype!r}") from exc


def _make_bucket(fragments, cap_bytes, group_by_id):
    dtype = fragments[0].grad_dtype
    bytes_per_element = _dtype_bytes(dtype)
    placements = []
    destination_offset = 0
    for item in fragments:
        placements.append(
            BucketFragmentPlacement(item.fragment_id, destination_offset)
        )
        destination_offset += item.numel
    exact_bytes = destination_offset * bytes_per_element
    semantic_payload = {
        "version": 1,
        "process_group_id": fragments[0].process_group_id,
        "process_group_ranks": group_by_id[fragments[0].process_group_id].ranks,
        "grad_dtype": dtype,
        "reduction_semantics": fragments[0].reduction_semantics,
        "fragments": [asdict(item) for item in fragments],
        "placements": [asdict(item) for item in placements],
        "exact_numel": destination_offset,
        "exact_bytes": exact_bytes,
    }
    semantic_hash = _hash(semantic_payload)
    return CommunicationBucketSpec(
        bucket_id="bucket-" + semantic_hash[:20],
        process_group_id=fragments[0].process_group_id,
        grad_dtype=dtype,
        reduction_semantics=fragments[0].reduction_semantics,
        placements=tuple(placements),
        exact_numel=destination_offset,
        exact_bytes=exact_bytes,
        logical_region_begin=fragments[0].logical_region_key,
        logical_region_end=fragments[-1].logical_region_key,
        semantic_hash=semantic_hash,
        oversized=exact_bytes > cap_bytes,
    )


def _bucketize(fragments, sync_mode, cap_bytes, group_by_id):
    if sync_mode not in {"tensor", "block"}:
        raise ValueError(f"unsupported sync mode {sync_mode!r}")
    if cap_bytes <= 0:
        raise ValueError("bucket_cap_bytes must be positive")
    if sync_mode == "tensor":
        return tuple(
            _make_bucket((item,), cap_bytes, group_by_id) for item in fragments
        )

    buckets = []
    current = []
    current_bytes = 0
    current_key = None
    for item in fragments:
        item_bytes = item.numel * _dtype_bytes(item.grad_dtype)
        item_key = (
            item.process_group_id,
            item.grad_dtype,
            item.reduction_semantics,
        )
        if current and (
            item_key != current_key or current_bytes + item_bytes > cap_bytes
        ):
            buckets.append(_make_bucket(tuple(current), cap_bytes, group_by_id))
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
        current_key = item_key
        if item_bytes > cap_bytes:
            buckets.append(_make_bucket(tuple(current), cap_bytes, group_by_id))
            current = []
            current_bytes = 0
            current_key = None
    if current:
        buckets.append(_make_bucket(tuple(current), cap_bytes, group_by_id))
    return tuple(buckets)


def build_gradient_sync_plan(
    fragment_plan,
    sync_mode,
    bucket_cap_bytes,
    model_impl,
    manifest_hash,
    topology_hash,
):
    process_groups = tuple(sorted(fragment_plan.process_groups, key=lambda item: item.ranks))
    fragments = tuple(sorted(fragment_plan.fragments, key=lambda item: item.semantic_key))
    group_by_id = {item.process_group_id: item for item in process_groups}
    group_index = {
        item.process_group_id: index for index, item in enumerate(process_groups)
    }
    dtype_order = {
        dtype: index
        for index, dtype in enumerate(sorted({item.grad_dtype for item in fragments}))
    }
    placement_fragments = tuple(
        sorted(
            fragments,
            key=lambda item: (
                group_index[item.process_group_id],
                dtype_order[item.grad_dtype],
                item.reduction_semantics,
                item.semantic_key,
            ),
        )
    )
    buckets = _bucketize(
        placement_fragments, sync_mode, bucket_cap_bytes, group_by_id
    )
    payload_by_rank = {}
    scratch = {}
    for item in buckets:
        group = group_by_id[item.process_group_id]
        for rank in group.ranks:
            payload_by_rank[rank] = payload_by_rank.get(rank, 0) + item.exact_bytes
            key = (rank, item.grad_dtype)
            scratch[key] = max(scratch.get(key, 0), item.exact_numel)
    process_group_index = tuple(
        (item.process_group_id, index) for index, item in enumerate(process_groups)
    )
    global_fragment_hash = _hash(
        {
            "canonical_fragments": [asdict(item) for item in fragments],
            "local_fragment_bindings_by_rank": [
                (rank, [asdict(binding) for binding in bindings])
                for rank, bindings in fragment_plan.local_bindings_by_rank
            ],
        }
    )
    plan_payload = {
        "version": 1,
        "sync_mode": sync_mode,
        "bucket_cap_bytes": bucket_cap_bytes,
        "model_impl": model_impl,
        "manifest_hash": manifest_hash,
        "topology_hash": topology_hash,
        "fragment_hash": global_fragment_hash,
        "process_groups": [asdict(item) for item in process_groups],
        "canonical_fragments": [asdict(item) for item in fragments],
        "local_fragment_bindings_by_rank": [
            (rank, [asdict(binding) for binding in bindings])
            for rank, bindings in fragment_plan.local_bindings_by_rank
        ],
        "buckets": [asdict(item) for item in buckets],
    }
    return GradientSyncPlan(
        version=1,
        sync_mode=sync_mode,
        bucket_cap_bytes=bucket_cap_bytes,
        model_impl=model_impl,
        op_manifest_hash=manifest_hash,
        process_groups=process_groups,
        process_group_index_by_id=process_group_index,
        canonical_fragments=fragments,
        local_fragment_bindings_by_rank=fragment_plan.local_bindings_by_rank,
        global_bucket_order=buckets,
        max_scratch_numel_by_rank_and_dtype=tuple(sorted(scratch.items())),
        exact_payload_bytes_by_rank=tuple(sorted(payload_by_rank.items())),
        padding_payload_bytes=0,
        global_topology_hash=topology_hash,
        global_manifest_hash=manifest_hash,
        global_fragment_hash=global_fragment_hash,
        global_plan_hash=_hash(plan_payload),
    )


class ProcessGroupRegistry:
    def __init__(self, process_groups, create_group):
        self._groups = tuple(sorted(process_groups, key=lambda item: item.ranks))
        self._create_group = create_group
        self._handles = {}
        self._creation_count = 0

    def initialize(self):
        if self._handles:
            return
        seen_ids = set()
        seen_ranks = set()
        for item in self._groups:
            if item.process_group_id in seen_ids:
                raise ValueError(f"duplicate process group id {item.process_group_id}")
            if item.ranks in seen_ranks:
                raise ValueError(f"duplicate process group ranks {item.ranks}")
            if item.ranks != tuple(sorted(set(item.ranks))):
                raise ValueError(f"process group ranks must be sorted and unique: {item.ranks}")
            seen_ids.add(item.process_group_id)
            seen_ranks.add(item.ranks)
            handle = self._create_group(item.ranks)
            self._creation_count += 1
            self._handles[item.process_group_id] = handle

    @property
    def group_count(self):
        return len(self._handles)

    @property
    def creation_count(self):
        return self._creation_count

    def handle(self, process_group_id):
        if process_group_id not in self._handles:
            raise KeyError(f"process group {process_group_id} is not initialized")
        return self._handles[process_group_id]
