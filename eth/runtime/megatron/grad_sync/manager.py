"""Initialization lifecycle for canonical gradient synchronization."""

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from .bucket_plan import ProcessGroupRegistry
from .bucket_plan import build_gradient_sync_plan
from .executor import GradientBucketExecutor
from .optimizer_semantics import CanonicalOptimizerSemantics
from .planner import plan_canonical_fragments


def dump_gradient_sync_plan(plan, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class GradientSyncManager:
    plan: object
    registry: ProcessGroupRegistry
    executor: GradientBucketExecutor
    optimizer_semantics: CanonicalOptimizerSemantics

    def execute(self, normalization_scale=1.0):
        self.executor.execute(normalization_scale=normalization_scale)

    def synchronize_parameters(self, broadcast):
        def cached_broadcast(
            payload,
            source_rank,
            process_group_id,
            unused_group_ranks,
            unused_fragment_id,
            unused_current_rank,
        ):
            broadcast(
                payload,
                source_rank,
                self.registry.handle(process_group_id),
            )

        self.optimizer_semantics.synchronize_parameters(cached_broadcast)

    def synchronize_shared_root_gradients(self, broadcast):
        def cached_broadcast(
            payload,
            source_rank,
            process_group_id,
            unused_group_ranks,
            unused_fragment_id,
            unused_current_rank,
        ):
            broadcast(
                payload,
                source_rank,
                self.registry.handle(process_group_id),
            )

        self.optimizer_semantics.synchronize_shared_root_gradients(
            cached_broadcast
        )

    def synchronize_optimizer_state(
        self, optimizer, broadcast, all_gather_group_object
    ):
        def cached_broadcast(
            payload,
            source_rank,
            process_group_id,
            unused_group_ranks,
            unused_fragment_id,
            unused_current_rank,
        ):
            broadcast(
                payload,
                source_rank,
                self.registry.handle(process_group_id),
            )

        for group in self.plan.process_groups:
            if self.optimizer_semantics.global_rank not in group.ranks:
                continue
            manifest = (
                self.optimizer_semantics.optimizer_state_manifest_for_group(
                    group.process_group_id,
                    optimizer.state_for_model_parameter,
                )
            )
            gathered = [None] * len(group.ranks)
            all_gather_group_object(
                gathered,
                manifest,
                self.registry.handle(group.process_group_id),
            )
            if any(item != manifest for item in gathered):
                raise ValueError(
                    f"optimizer state manifest mismatch for process group "
                    f"{group.process_group_id}: {tuple(gathered)}"
                )
        self.optimizer_semantics.synchronize_optimizer_state(
            optimizer.state_for_model_parameter,
            cached_broadcast,
        )

    def synchronize_shared_optimizer_state(
        self, optimizer, all_gather_object, world_size
    ):
        self.optimizer_semantics.synchronize_shared_optimizer_state(
            optimizer.state_for_model_parameter,
            all_gather_object,
            world_size,
        )


def initialize_gradient_sync(
    *,
    local_descriptors,
    local_runtime_bindings,
    topology,
    model_impl,
    manifest_hash,
    sync_mode,
    bucket_cap_bytes,
    global_rank,
    all_gather_object,
    all_gather_group_object,
    create_group,
    all_reduce_sum,
    control_device="cpu",
    runtime_evidence_dir=None,
):
    gathered_descriptors = [None] * topology.world_size
    all_gather_object(gathered_descriptors, tuple(local_descriptors))
    if any(items is None for items in gathered_descriptors):
        raise ValueError("gradient metadata gather returned an incomplete world")
    global_descriptors = tuple(
        descriptor
        for rank_descriptors in gathered_descriptors
        for descriptor in rank_descriptors
    )
    expected_pipeline_ids = tuple(
        sorted({stage.pipeline_id for stage in topology.stages})
    )
    fragment_plan = plan_canonical_fragments(
        global_descriptors, expected_pipeline_ids=expected_pipeline_ids
    )
    plan = build_gradient_sync_plan(
        fragment_plan,
        sync_mode,
        bucket_cap_bytes,
        model_impl,
        manifest_hash,
        topology.fingerprint,
    )

    gathered_hashes = [None] * topology.world_size
    all_gather_object(gathered_hashes, plan.global_plan_hash)
    if len(set(gathered_hashes)) != 1:
        raise ValueError(
            f"gradient plan hash mismatch across ranks: {tuple(gathered_hashes)}"
        )

    registry = ProcessGroupRegistry(plan.process_groups, create_group)
    registry.initialize()
    for group in plan.process_groups:
        if global_rank not in group.ranks:
            continue
        manifest = tuple(
            (
                bucket.semantic_hash,
                bucket.exact_numel,
                bucket.grad_dtype,
                len(bucket.placements),
            )
            for bucket in plan.global_bucket_order
            if bucket.process_group_id == group.process_group_id
        )
        gathered_manifests = [None] * len(group.ranks)
        all_gather_group_object(
            gathered_manifests,
            manifest,
            registry.handle(group.process_group_id),
        )
        if any(item != manifest for item in gathered_manifests):
            raise ValueError(
                f"bucket manifest mismatch for process group "
                f"{group.process_group_id}: {tuple(gathered_manifests)}"
            )
    executor = GradientBucketExecutor(
        plan,
        global_rank,
        local_descriptors,
        local_runtime_bindings,
        registry,
        all_reduce_sum,
        runtime_evidence_dir=runtime_evidence_dir,
    )
    optimizer_semantics = CanonicalOptimizerSemantics(
        plan,
        global_rank,
        local_descriptors,
        local_runtime_bindings,
        control_device,
    )
    return GradientSyncManager(
        plan, registry, executor, optimizer_semantics
    )
