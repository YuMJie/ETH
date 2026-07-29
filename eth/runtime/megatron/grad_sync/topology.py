"""Immutable canonical stage topology derived from normalized config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankCoordinate:
    global_rank: int
    tp_rank: int
    dp_rank: int
    cp_rank: int


@dataclass(frozen=True)
class CanonicalStageTopology:
    pipeline_id: int
    stage_id: int
    global_op_begin: int
    global_op_end: int
    stage_ranks: tuple
    tp_degree: int
    dp_degree: int
    cp_degree: int
    rank_coordinates: tuple
    tp_group_ranks_by_coordinate: tuple
    dp_group_ranks_by_coordinate: tuple
    cp_group_ranks_by_coordinate: tuple

    def coordinate(self, global_rank):
        for item in self.rank_coordinates:
            if item.global_rank == global_rank:
                return item
        raise KeyError(f"rank {global_rank} is not in pipeline {self.pipeline_id} stage {self.stage_id}")


@dataclass(frozen=True)
class CanonicalTopology:
    world_size: int
    stages: tuple
    fingerprint: str

    def stage(self, pipeline_id, stage_id):
        for item in self.stages:
            if item.pipeline_id == pipeline_id and item.stage_id == stage_id:
                return item
        raise KeyError(f"unknown pipeline {pipeline_id} stage {stage_id}")

    def stage_for_rank(self, global_rank):
        for item in self.stages:
            if global_rank in item.stage_ranks:
                return item
        raise KeyError(f"rank {global_rank} is not present in canonical topology")

    def all_group_rank_tuples(self):
        groups = set()
        for stage in self.stages:
            groups.update(stage.tp_group_ranks_by_coordinate)
            groups.update(stage.dp_group_ranks_by_coordinate)
            groups.update(stage.cp_group_ranks_by_coordinate)
        return tuple(sorted(groups))

    def expected_rank_facts(self, global_rank):
        stage = self.stage_for_rank(global_rank)
        coordinate = stage.coordinate(global_rank)
        tp_group = next(
            group for group in stage.tp_group_ranks_by_coordinate if global_rank in group
        )
        dp_group = next(
            group for group in stage.dp_group_ranks_by_coordinate if global_rank in group
        )
        cp_group = next(
            group for group in stage.cp_group_ranks_by_coordinate if global_rank in group
        )
        return {
            "global_rank": global_rank,
            "pipeline_id": stage.pipeline_id,
            "stage_id": stage.stage_id,
            "global_op_begin": stage.global_op_begin,
            "global_op_end": stage.global_op_end,
            "stage_ranks": stage.stage_ranks,
            "tp_degree": stage.tp_degree,
            "dp_degree": stage.dp_degree,
            "cp_degree": stage.cp_degree,
            "tp_rank": coordinate.tp_rank,
            "dp_rank": coordinate.dp_rank,
            "cp_rank": coordinate.cp_rank,
            "tp_group_ranks": tp_group,
            "dp_group_ranks": dp_group,
            "cp_group_ranks": cp_group,
        }


def _stage_groups(stage_start, tp_degree, dp_degree, cp_degree):
    def rank_for(tp_rank, dp_rank, cp_rank):
        return stage_start + dp_rank * cp_degree * tp_degree + cp_rank * tp_degree + tp_rank

    tp_groups = tuple(
        tuple(rank_for(tp_rank, dp_rank, cp_rank) for tp_rank in range(tp_degree))
        for dp_rank in range(dp_degree)
        for cp_rank in range(cp_degree)
    )
    dp_groups = tuple(
        tuple(rank_for(tp_rank, dp_rank, cp_rank) for dp_rank in range(dp_degree))
        for cp_rank in range(cp_degree)
        for tp_rank in range(tp_degree)
    )
    cp_groups = tuple(
        tuple(rank_for(tp_rank, dp_rank, cp_rank) for cp_rank in range(cp_degree))
        for dp_rank in range(dp_degree)
        for tp_rank in range(tp_degree)
    )
    coordinates = tuple(
        RankCoordinate(
            global_rank=rank_for(tp_rank, dp_rank, cp_rank),
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            cp_rank=cp_rank,
        )
        for dp_rank in range(dp_degree)
        for cp_rank in range(cp_degree)
        for tp_rank in range(tp_degree)
    )
    return coordinates, tp_groups, dp_groups, cp_groups


def build_canonical_topology(normalized_config):
    stages = []
    stage_start = 0
    for pipeline_id, pipe in enumerate(normalized_config["pipe"]):
        global_op_begin = 0
        for stage_id, op_count in enumerate(pipe["num_ops_in_each_stage"]):
            tp_degree = pipe["model_parallel_size_of_each_op"][stage_id][0]
            dp_degree = pipe["data_parallel_size_of_each_op"][stage_id][0]
            cp_degree = pipe["context_parallel_size_of_each_op"][stage_id][0]
            stage_size = tp_degree * dp_degree * cp_degree
            expected_size = pipe["num_gpus"][stage_id]
            if stage_size != expected_size:
                raise ValueError(
                    f"pipeline {pipeline_id} stage {stage_id} has topology size {stage_size}, "
                    f"expected {expected_size}"
                )
            coordinates, tp_groups, dp_groups, cp_groups = _stage_groups(
                stage_start, tp_degree, dp_degree, cp_degree
            )
            stage_ranks = tuple(range(stage_start, stage_start + stage_size))
            stages.append(
                CanonicalStageTopology(
                    pipeline_id=pipeline_id,
                    stage_id=stage_id,
                    global_op_begin=global_op_begin,
                    global_op_end=global_op_begin + op_count,
                    stage_ranks=stage_ranks,
                    tp_degree=tp_degree,
                    dp_degree=dp_degree,
                    cp_degree=cp_degree,
                    rank_coordinates=coordinates,
                    tp_group_ranks_by_coordinate=tp_groups,
                    dp_group_ranks_by_coordinate=dp_groups,
                    cp_group_ranks_by_coordinate=cp_groups,
                )
            )
            stage_start += stage_size
            global_op_begin += op_count
    if stage_start != normalized_config["world_size"]:
        raise ValueError(
            f"canonical topology covers {stage_start} ranks, "
            f"expected world size {normalized_config['world_size']}"
        )
    return CanonicalTopology(
        world_size=normalized_config["world_size"],
        stages=tuple(stages),
        fingerprint=normalized_config["topology_fingerprint"],
    )


def build_mpu_rank_facts(
    global_rank,
    pipeline_id,
    stage_id,
    global_op_begin,
    global_op_end,
    stage_ranks,
    tp_group_ranks,
    dp_group_ranks,
    cp_group_ranks,
):
    groups = {
        "stage_ranks": tuple(stage_ranks),
        "tp_group_ranks": tuple(tp_group_ranks),
        "dp_group_ranks": tuple(dp_group_ranks),
        "cp_group_ranks": tuple(cp_group_ranks),
    }
    for name, ranks in groups.items():
        if ranks != tuple(sorted(ranks)):
            raise ValueError(f"rank {global_rank} {name} must be sorted")
        if global_rank not in ranks:
            raise ValueError(f"rank {global_rank} is missing from {name}")
    return {
        "global_rank": global_rank,
        "pipeline_id": pipeline_id,
        "stage_id": stage_id,
        "global_op_begin": global_op_begin,
        "global_op_end": global_op_end,
        "stage_ranks": groups["stage_ranks"],
        "tp_degree": len(groups["tp_group_ranks"]),
        "dp_degree": len(groups["dp_group_ranks"]),
        "cp_degree": len(groups["cp_group_ranks"]),
        "tp_rank": groups["tp_group_ranks"].index(global_rank),
        "dp_rank": groups["dp_group_ranks"].index(global_rank),
        "cp_rank": groups["cp_group_ranks"].index(global_rank),
        "tp_group_ranks": groups["tp_group_ranks"],
        "dp_group_ranks": groups["dp_group_ranks"],
        "cp_group_ranks": groups["cp_group_ranks"],
    }


def validate_mpu_rank_facts(topology, actual_facts):
    global_rank = actual_facts.get("global_rank")
    expected = topology.expected_rank_facts(global_rank)
    context = (
        f"rank {global_rank} pipeline {expected['pipeline_id']} "
        f"stage {expected['stage_id']}"
    )
    for key, expected_value in expected.items():
        actual_value = actual_facts.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"MPU topology mismatch for {context}: {key} "
                f"expected {expected_value}, got {actual_value}"
            )
