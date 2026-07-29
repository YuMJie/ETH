"""Deterministic canonical gradient fragment planning."""

import hashlib
import json
import math
from dataclasses import asdict
from dataclasses import dataclass


def _hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TensorSegmentSpec:
    source_dim_offset: int
    source_dim_length: int
    source_shape: tuple
    global_coordinate_begin: int
    global_coordinate_end: int
    flattened_numel: int


@dataclass(frozen=True)
class ProcessGroupSpec:
    process_group_id: str
    ranks: tuple
    semantic_kind: str


@dataclass(frozen=True)
class CanonicalGradientFragmentSpec:
    fragment_id: str
    logical_param_id: str
    canonical_shard_id: object
    process_group_id: str
    grad_dtype: str
    reduction_semantics: str
    logical_region_key: str
    global_regions: tuple
    numel: int
    semantic_key: tuple
    optimizer_semantic_id: str
    include_in_optimizer_norm: bool
    norm_owner_rank: object


@dataclass(frozen=True)
class LocalFragmentBindingSpec:
    global_rank: int
    fragment_id: str
    logical_param_id: str
    source_segments: tuple
    contribution_scale: float


@dataclass(frozen=True)
class CanonicalFragmentPlan:
    process_groups: tuple
    fragments: tuple
    local_bindings_by_rank: tuple
    fragment_hash: str

    def bindings_for_rank(self, global_rank):
        for rank, bindings in self.local_bindings_by_rank:
            if rank == global_rank:
                return bindings
        return ()


def _numel(shape):
    return math.prod(shape) if shape else 1


def _normalized_axis(descriptor):
    axis = descriptor.partition_dim
    if axis < 0:
        axis += len(descriptor.full_shape)
    if not 0 <= axis < len(descriptor.full_shape):
        raise ValueError(
            f"axis mismatch for {descriptor.logical_param_id}: {descriptor.partition_dim}"
        )
    return axis


def _rank_intervals(axis_size, tp_degree, tp_rank, stride, logical_id):
    divisor = tp_degree * stride
    if axis_size % divisor:
        raise ValueError(
            f"nondivisible TP layout for {logical_id}: axis {axis_size}, "
            f"TP {tp_degree}, stride {stride}"
        )
    chunk = axis_size // divisor
    return tuple(
        ((stride_index * tp_degree + tp_rank) * chunk,
         (stride_index * tp_degree + tp_rank + 1) * chunk)
        for stride_index in range(stride)
    )


def _source_segments(descriptor, target_regions, axis):
    local_regions = _rank_intervals(
        descriptor.full_shape[axis],
        descriptor.local_tp_degree,
        descriptor.local_tp_rank,
        descriptor.partition_stride,
        descriptor.logical_param_id,
    )
    other_numel = _numel(descriptor.full_shape) // descriptor.full_shape[axis]
    segments = []
    local_offset = 0
    for local_begin, local_end in local_regions:
        for target_begin, target_end in target_regions:
            begin = max(local_begin, target_begin)
            end = min(local_end, target_end)
            if begin < end:
                segments.append(
                    TensorSegmentSpec(
                        source_dim_offset=local_offset + begin - local_begin,
                        source_dim_length=end - begin,
                        source_shape=descriptor.local_shape,
                        global_coordinate_begin=begin,
                        global_coordinate_end=end,
                        flattened_numel=(end - begin) * other_numel,
                    )
                )
        local_offset += local_end - local_begin
    if tuple((item.global_coordinate_begin, item.global_coordinate_end) for item in segments) != target_regions:
        return ()
    return tuple(segments)


def _validate_common(logical_id, descriptors):
    seen_ranks = set()
    reference = descriptors[0]
    for item in descriptors:
        if item.global_rank in seen_ranks:
            raise ValueError(f"duplicate owner rank {item.global_rank} for {logical_id}")
        seen_ranks.add(item.global_rank)
        for field, message in (
            ("full_shape", "shape mismatch"),
            ("gradient_dtype", "dtype mismatch"),
            ("parameter_dtype", "dtype mismatch"),
            ("tensor_parallel", "replica semantics mismatch"),
            ("partition_stride", "stride mismatch"),
            ("model_impl", "model implementation mismatch"),
            ("op_manifest_hash", "manifest mismatch"),
            ("logical_region_key", "logical region mismatch"),
            ("semantic_order_key", "semantic order mismatch"),
            ("shared_weight_id", "shared weight mismatch"),
            ("shared_weight_root", "shared root mismatch"),
        ):
            if getattr(item, field) != getattr(reference, field):
                if field == "full_shape":
                    layouts = ", ".join(
                        "rank={rank} full={full} local={local} tp={tp_rank}/{tp_degree}".format(
                            rank=descriptor.global_rank,
                            full=descriptor.full_shape,
                            local=descriptor.local_shape,
                            tp_rank=descriptor.local_tp_rank,
                            tp_degree=descriptor.local_tp_degree,
                        )
                        for descriptor in descriptors
                    )
                    raise ValueError(f"{message} for {logical_id}: {layouts}")
                raise ValueError(f"{message} for {logical_id}")
        if item.tensor_parallel and _normalized_axis(item) != _normalized_axis(reference):
            raise ValueError(f"axis mismatch for {logical_id}")
        topology_groups = (
            ("TP", item.local_tp_group_ranks, item.local_tp_degree, item.local_tp_rank),
            ("DP", item.local_dp_group_ranks, item.local_dp_degree, item.local_dp_rank),
        )
        for label, ranks, degree, coordinate in topology_groups:
            if tuple(sorted(ranks)) != tuple(ranks) or len(ranks) != degree:
                raise ValueError(f"{label} group mismatch for {logical_id} rank {item.global_rank}")
            if item.global_rank not in ranks or ranks.index(item.global_rank) != coordinate:
                raise ValueError(f"{label} coordinate mismatch for {logical_id} rank {item.global_rank}")
        if item.local_cp_degree != 1 or item.local_cp_rank != 0:
            raise ValueError(f"CP coordinate mismatch for {logical_id} rank {item.global_rank}")
        if tuple(sorted(item.stage_ranks)) != tuple(item.stage_ranks) or item.global_rank not in item.stage_ranks:
            raise ValueError(f"stage ranks mismatch for {logical_id} rank {item.global_rank}")
    pipelines = {}
    for item in descriptors:
        pipelines.setdefault(item.pipeline_id, {}).setdefault(item.local_dp_rank, []).append(item)
    for pipeline_id, replicas in pipelines.items():
        pipeline_descriptors = [
            item for item in descriptors if item.pipeline_id == pipeline_id
        ]
        expected_dp = pipeline_descriptors[0].local_dp_degree
        if any(item.local_dp_degree != expected_dp for item in pipeline_descriptors):
            raise ValueError(f"DP degree mismatch for {logical_id} pipeline {pipeline_id}")
        if set(replicas) != set(range(expected_dp)):
            raise ValueError(f"missing owner DP replica for {logical_id} pipeline {pipeline_id}")
        for dp_rank, holders in replicas.items():
            degree = holders[0].local_tp_degree
            if len(holders) != degree or {item.local_tp_rank for item in holders} != set(range(degree)):
                raise ValueError(
                    f"missing owner TP rank for {logical_id} pipeline {pipeline_id} DP {dp_rank}"
                )
            if any(item.local_tp_degree != degree for item in holders):
                raise ValueError(f"TP degree mismatch for {logical_id}")
            expected_tp_group = tuple(sorted(item.global_rank for item in holders))
            if any(item.local_tp_group_ranks != expected_tp_group for item in holders):
                raise ValueError(f"TP group mismatch for {logical_id}")
        tp_degree = next(iter(replicas.values()))[0].local_tp_degree
        if any(
            item.local_tp_degree != tp_degree
            for holders in replicas.values()
            for item in holders
        ):
            raise ValueError(f"TP degree mismatch for {logical_id} pipeline {pipeline_id}")
        for tp_rank in range(tp_degree):
            lane = sorted(
                (
                    item
                    for holders in replicas.values()
                    for item in holders
                    if item.local_tp_rank == tp_rank
                ),
                key=lambda item: item.local_dp_rank,
            )
            expected_dp_group = tuple(item.global_rank for item in lane)
            if any(item.local_dp_group_ranks != expected_dp_group for item in lane):
                raise ValueError(f"DP group mismatch for {logical_id} pipeline {pipeline_id}")
    return pipelines


def _group_id(ranks):
    return "pg-" + _hash({"ranks": ranks})[:16]


def _fragment_id(fields):
    return "fragment-" + _hash(fields)[:20]


def _plan_sharded(logical_id, descriptors, pipelines):
    reference = descriptors[0]
    axis = _normalized_axis(reference)
    canonical_tp_degree = max(item.local_tp_degree for item in descriptors)
    _rank_intervals(
        reference.full_shape[axis],
        canonical_tp_degree,
        0,
        reference.partition_stride,
        logical_id,
    )
    for item in descriptors:
        if canonical_tp_degree % item.local_tp_degree:
            raise ValueError(
                f"nondivisible canonical TP degree for {logical_id}: "
                f"{canonical_tp_degree} % {item.local_tp_degree}"
            )
        expected_local_shape = list(reference.full_shape)
        expected_local_shape[axis] //= item.local_tp_degree
        if item.local_shape != tuple(expected_local_shape):
            raise ValueError(f"shape mismatch for {logical_id} rank {item.global_rank}")
    other_numel = _numel(reference.full_shape) // reference.full_shape[axis]
    fragments = []
    bindings = []
    groups = []
    for shard_id in range(canonical_tp_degree):
        regions = _rank_intervals(
            reference.full_shape[axis],
            canonical_tp_degree,
            shard_id,
            reference.partition_stride,
            logical_id,
        )
        owners = []
        owner_segments = []
        for pipeline_id in sorted(pipelines):
            for dp_rank in sorted(pipelines[pipeline_id]):
                matches = []
                for item in pipelines[pipeline_id][dp_rank]:
                    segments = _source_segments(item, regions, axis)
                    if segments:
                        matches.append((item, segments))
                if len(matches) != 1:
                    raise ValueError(
                        f"missing owner for {logical_id} shard {shard_id} "
                        f"pipeline {pipeline_id} DP {dp_rank}"
                    )
                owners.append(matches[0][0].global_rank)
                owner_segments.append(matches[0])
        ranks = tuple(sorted(owners))
        process_group_id = _group_id(ranks)
        groups.append(ProcessGroupSpec(process_group_id, ranks, "gradient_sync"))
        include_norm = reference.shared_weight_id is None or reference.shared_weight_root
        semantic_key = (
            reference.semantic_order_key,
            shard_id,
            regions,
        )
        fragment_fields = {
            "logical_param_id": logical_id,
            "canonical_shard_id": shard_id,
            "process_group_id": process_group_id,
            "grad_dtype": reference.gradient_dtype,
            "global_regions": regions,
            "semantic_key": semantic_key,
        }
        fragment_id = _fragment_id(fragment_fields)
        fragments.append(
            CanonicalGradientFragmentSpec(
                fragment_id=fragment_id,
                logical_param_id=logical_id,
                canonical_shard_id=shard_id,
                process_group_id=process_group_id,
                grad_dtype=reference.gradient_dtype,
                reduction_semantics="sum",
                logical_region_key=reference.logical_region_key,
                global_regions=regions,
                numel=sum(end - begin for begin, end in regions) * other_numel,
                semantic_key=semantic_key,
                optimizer_semantic_id=reference.shared_weight_id or logical_id,
                include_in_optimizer_norm=include_norm,
                norm_owner_rank=min(ranks) if include_norm else None,
            )
        )
        for item, segments in owner_segments:
            bindings.append(
                LocalFragmentBindingSpec(
                    global_rank=item.global_rank,
                    fragment_id=fragment_id,
                    logical_param_id=logical_id,
                    source_segments=segments,
                    contribution_scale=1.0,
                )
            )
    local_coverage = {}
    for binding in bindings:
        local_coverage[binding.global_rank] = local_coverage.get(binding.global_rank, 0) + sum(
            segment.flattened_numel for segment in binding.source_segments
        )
    for item in descriptors:
        if local_coverage.get(item.global_rank, 0) != _numel(item.local_shape):
            raise ValueError(f"local coverage mismatch for {logical_id} rank {item.global_rank}")
        source_intervals = sorted(
            (
                segment.source_dim_offset,
                segment.source_dim_offset + segment.source_dim_length,
            )
            for binding in bindings
            if binding.global_rank == item.global_rank
            for segment in binding.source_segments
        )
        expected_begin = 0
        for begin, end in source_intervals:
            if begin != expected_begin or end <= begin:
                raise ValueError(f"local coverage overlap or gap for {logical_id} rank {item.global_rank}")
            expected_begin = end
        if expected_begin != item.local_shape[axis]:
            raise ValueError(f"local coverage gap for {logical_id} rank {item.global_rank}")
    return groups, fragments, bindings


def _plan_replicated(logical_id, descriptors):
    reference = descriptors[0]
    for item in descriptors:
        if item.tp_replica_semantics != "identical_duplicate":
            raise ValueError(f"replica semantics invalid for {logical_id}")
        if item.local_shape != item.full_shape:
            raise ValueError(f"shape mismatch for replicated {logical_id}")
    ranks = tuple(sorted(item.global_rank for item in descriptors))
    process_group_id = _group_id(ranks)
    group = ProcessGroupSpec(process_group_id, ranks, "gradient_sync")
    regions = ((0, reference.full_shape[0] if reference.full_shape else 1),)
    include_norm = reference.shared_weight_id is None or reference.shared_weight_root
    semantic_key = (reference.semantic_order_key, None, regions)
    fields = {
        "logical_param_id": logical_id,
        "canonical_shard_id": None,
        "process_group_id": process_group_id,
        "grad_dtype": reference.gradient_dtype,
        "global_regions": regions,
        "semantic_key": semantic_key,
    }
    fragment_id = _fragment_id(fields)
    fragment = CanonicalGradientFragmentSpec(
        fragment_id=fragment_id,
        logical_param_id=logical_id,
        canonical_shard_id=None,
        process_group_id=process_group_id,
        grad_dtype=reference.gradient_dtype,
        reduction_semantics="sum_identical_duplicates",
        logical_region_key=reference.logical_region_key,
        global_regions=regions,
        numel=_numel(reference.full_shape),
        semantic_key=semantic_key,
        optimizer_semantic_id=reference.shared_weight_id or logical_id,
        include_in_optimizer_norm=include_norm,
        norm_owner_rank=min(ranks) if include_norm else None,
    )
    bindings = tuple(
        LocalFragmentBindingSpec(
            global_rank=item.global_rank,
            fragment_id=fragment_id,
            logical_param_id=logical_id,
            source_segments=(
                TensorSegmentSpec(
                    source_dim_offset=0,
                    source_dim_length=item.local_shape[0] if item.local_shape else 1,
                    source_shape=item.local_shape,
                    global_coordinate_begin=0,
                    global_coordinate_end=item.full_shape[0] if item.full_shape else 1,
                    flattened_numel=_numel(item.local_shape),
                ),
            ),
            contribution_scale=1.0 / item.local_tp_degree,
        )
        for item in sorted(descriptors, key=lambda value: value.global_rank)
    )
    return (group,), (fragment,), bindings


def _validate_shared_roots(by_logical_id):
    by_shared_id = {}
    for logical_id, descriptors in by_logical_id.items():
        shared_id = descriptors[0].shared_weight_id
        if shared_id is not None:
            by_shared_id.setdefault(shared_id, []).append((logical_id, descriptors))
    for shared_id, logical_sets in by_shared_id.items():
        roots = [
            logical_id
            for logical_id, descriptors in logical_sets
            if descriptors[0].shared_weight_root
        ]
        if len(roots) != 1:
            raise ValueError(
                f"shared root mismatch for {shared_id}: expected one root, got {roots}"
            )
        reference = logical_sets[0][1][0]
        for logical_id, descriptors in logical_sets[1:]:
            item = descriptors[0]
            for field in ("full_shape", "parameter_dtype", "gradient_dtype", "tensor_parallel", "partition_dim", "partition_stride"):
                if getattr(item, field) != getattr(reference, field):
                    raise ValueError(f"shared copy mismatch for {shared_id} at {logical_id}")


def plan_canonical_fragments(descriptors, expected_pipeline_ids):
    descriptors = tuple(descriptors)
    if len({item.model_impl for item in descriptors}) > 1:
        raise ValueError("model implementation mismatch across descriptors")
    if len({item.op_manifest_hash for item in descriptors}) > 1:
        raise ValueError("manifest mismatch across descriptors")
    by_logical_id = {}
    for item in descriptors:
        if not item.requires_grad:
            raise ValueError(f"non-trainable descriptor {item.logical_param_id}")
        by_logical_id.setdefault(item.logical_param_id, []).append(item)
    expected_pipelines = set(expected_pipeline_ids)
    if not expected_pipelines:
        raise ValueError("expected_pipeline_ids must not be empty")
    _validate_shared_roots(by_logical_id)
    all_groups = {}
    fragments = []
    bindings = []
    for logical_id in sorted(by_logical_id):
        logical_descriptors = sorted(
            by_logical_id[logical_id],
            key=lambda item: (item.pipeline_id, item.local_dp_rank, item.local_tp_rank, item.global_rank),
        )
        actual_pipelines = {item.pipeline_id for item in logical_descriptors}
        if actual_pipelines != expected_pipelines:
            missing = sorted(expected_pipelines - actual_pipelines)
            raise ValueError(
                f"missing owner pipeline {missing} for {logical_id}"
            )
        pipelines = _validate_common(logical_id, logical_descriptors)
        if logical_descriptors[0].tensor_parallel:
            groups, logical_fragments, logical_bindings = _plan_sharded(
                logical_id, logical_descriptors, pipelines
            )
        else:
            groups, logical_fragments, logical_bindings = _plan_replicated(
                logical_id, logical_descriptors
            )
        for group in groups:
            existing = all_groups.get(group.process_group_id)
            if existing is not None and existing.ranks != group.ranks:
                raise ValueError(f"process group hash collision for {group.ranks}")
            all_groups[group.process_group_id] = group
        fragments.extend(logical_fragments)
        bindings.extend(logical_bindings)
    fragments.sort(key=lambda item: item.semantic_key)
    process_groups = tuple(sorted(all_groups.values(), key=lambda item: item.ranks))
    bindings_by_rank = {}
    fragment_order = {item.fragment_id: index for index, item in enumerate(fragments)}
    for binding in bindings:
        bindings_by_rank.setdefault(binding.global_rank, []).append(binding)
    frozen_bindings = tuple(
        (
            rank,
            tuple(sorted(rank_bindings, key=lambda item: fragment_order[item.fragment_id])),
        )
        for rank, rank_bindings in sorted(bindings_by_rank.items())
    )
    fragment_payload = [asdict(item) for item in fragments]
    return CanonicalFragmentPlan(
        process_groups=process_groups,
        fragments=tuple(fragments),
        local_bindings_by_rank=frozen_bindings,
        fragment_hash=_hash(fragment_payload),
    )
