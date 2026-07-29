import hashlib
import json
import math
import sys
from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace

import torch

from tests.test_config_normalization import load_normalizer


def _with_stage_metadata(item, stage, global_op_index):
    return replace(
        item,
        stage_id=stage.stage_id,
        global_op_index=global_op_index,
        stage_global_op_begin=stage.global_op_begin,
        stage_global_op_end=stage.global_op_end,
    )


def _with_layout_metadata(item, layout, global_op_index):
    return replace(
        item,
        stage_id=layout.stage_id,
        global_op_index=global_op_index,
        stage_global_op_begin=layout.global_op_begin,
        stage_global_op_end=layout.global_op_end,
    )


@dataclass(frozen=True)
class CpuMatrixFixture:
    fixture_id: str
    config: dict
    normalized_config: dict
    topology_fingerprint: str
    world_size: int

    def build_topology(self):
        from tests.test_grad_sync_topology import load_topology

        return load_topology().build_canonical_topology(self.normalized_config)

    def build_descriptors(self):
        from tests.test_gradient_bucket_plan import load_bucket_plan
        from tests.test_gradient_fragment_planner import descriptor

        load_bucket_plan()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        topology = self.build_topology()
        descriptors = []
        global_op_anchors = sorted(
            {
                stage.global_op_begin
                for stage in topology.stages
            }
        )
        for global_op_index in global_op_anchors:
            for pipeline_id in range(len(self.normalized_config["pipe"])):
                stage = next(
                    item
                    for item in topology.stages
                    if item.pipeline_id == pipeline_id
                    and item.global_op_begin <= global_op_index < item.global_op_end
                )
                for coordinate in stage.rank_coordinates:
                    rank = coordinate.global_rank
                    tp_group = next(
                        group
                        for group in stage.tp_group_ranks_by_coordinate
                        if rank in group
                    )
                    dp_group = next(
                        group
                        for group in stage.dp_group_ranks_by_coordinate
                        if rank in group
                    )
                    for parameter_index in range(2):
                        logical_id = (
                            f"op-{global_op_index}.weight-{parameter_index}"
                        )
                        item = descriptor(
                            descriptor_cls,
                            logical_id,
                            rank,
                            pipeline_id,
                            stage.tp_degree,
                            coordinate.tp_rank,
                            (8,),
                            (8 // stage.tp_degree,),
                            True,
                            0,
                            dp_degree=stage.dp_degree,
                            dp_rank=coordinate.dp_rank,
                            tp_group=tp_group,
                            dp_group=dp_group,
                            stage_ranks=stage.stage_ranks,
                        )
                        descriptors.append(
                            _with_stage_metadata(item, stage, global_op_index)
                        )
                    logical_id = f"op-{global_op_index}.bias"
                    item = descriptor(
                        descriptor_cls,
                        logical_id,
                        rank,
                        pipeline_id,
                        stage.tp_degree,
                        coordinate.tp_rank,
                        (2,),
                        (2,),
                        False,
                        -1,
                        dp_degree=stage.dp_degree,
                        dp_rank=coordinate.dp_rank,
                        tp_group=tp_group,
                        dp_group=dp_group,
                        stage_ranks=stage.stage_ranks,
                    )
                    descriptors.append(
                        _with_stage_metadata(item, stage, global_op_index)
                    )
        return tuple(descriptors)

    def build_plans(self):
        from tests.test_gradient_bucket_plan import load_bucket_plan

        bucket_plan, planner = load_bucket_plan()
        fragments = planner.plan_canonical_fragments(
            self.build_descriptors(),
            expected_pipeline_ids=range(len(self.normalized_config["pipe"])),
        )
        plans = []
        for mode in ("tensor", "block"):
            plans.append(
                bucket_plan.build_gradient_sync_plan(
                    fragments,
                    mode,
                    32 * 1024 * 1024,
                    self.normalized_config["model_impl"],
                    self.normalized_config["manifest"]["hash"],
                    self.topology_fingerprint,
                )
            )
        return tuple(plans)


@dataclass(frozen=True)
class PipelineLayout:
    pipeline_id: int
    stage_id: int
    stage_ranks: tuple
    tp_degree: int
    dp_degree: int
    global_op_begin: int
    global_op_end: int

    def __post_init__(self):
        if len(self.stage_ranks) != self.tp_degree * self.dp_degree:
            raise ValueError(
                f"pipeline {self.pipeline_id} layout size mismatch: "
                f"{len(self.stage_ranks)} != {self.tp_degree}*{self.dp_degree}"
            )

    def coordinate(self, rank):
        index = self.stage_ranks.index(rank)
        return index % self.tp_degree, index // self.tp_degree

    def tp_group(self, dp_rank):
        begin = dp_rank * self.tp_degree
        return self.stage_ranks[begin : begin + self.tp_degree]

    def dp_group(self, tp_rank):
        return tuple(
            self.stage_ranks[dp_rank * self.tp_degree + tp_rank]
            for dp_rank in range(self.dp_degree)
        )


@dataclass(frozen=True)
class LogicalParameterSpec:
    logical_id: str
    global_op_index: int
    full_shape: tuple
    tensor_parallel: bool
    layouts: tuple

    def layout_for_rank(self, rank):
        matches = [layout for layout in self.layouts if rank in layout.stage_ranks]
        if len(matches) > 1:
            raise ValueError(f"rank {rank} has duplicate layouts for {self.logical_id}")
        return matches[0] if matches else None


@dataclass(frozen=True)
class DistributedMatrixFixture:
    fixture_id: str
    world_size: int
    parameters: tuple
    loss_layouts: tuple
    pipeline_gbs: tuple
    loss_weights_by_iteration: tuple
    rank_inputs_by_iteration: tuple
    topology_fingerprint: str

    @property
    def pipeline_count(self):
        return len(self.pipeline_gbs)

    @property
    def nominal_global_loss_scale(self):
        return float(sum(self.pipeline_gbs))

    def parameter(self, logical_id):
        return next(item for item in self.parameters if item.logical_id == logical_id)

    def loss_layout_for_rank(self, rank):
        matches = [item for item in self.loss_layouts if rank in item.stage_ranks]
        if len(matches) > 1:
            raise ValueError(f"rank {rank} appears in multiple loss layouts")
        return matches[0] if matches else None

    def local_loss_weight(self, rank, iteration):
        layout = self.loss_layout_for_rank(rank)
        if layout is None:
            return 0.0
        tp_rank, unused_dp_rank = layout.coordinate(rank)
        if tp_rank != 0:
            return 0.0
        return (
            self.loss_weights_by_iteration[iteration][layout.pipeline_id]
            / layout.dp_degree
        )


@dataclass(frozen=True)
class ExpectedIteration:
    values: tuple
    l2_norm: float
    inf_norm: float

    def value(self, logical_id):
        return next(value for name, value in self.values if name == logical_id)

    @property
    def weights(self):
        return tuple(value for name, value in self.values if ".weight" in name)

    @property
    def bias(self):
        return next(value for name, value in self.values if ".bias" in name)


class RankContract:
    def __init__(self, fixture, rank, descriptors, runtime_bindings, layouts):
        self.fixture = fixture
        self.rank = rank
        self.descriptors = tuple(descriptors)
        self.runtime_bindings = tuple(runtime_bindings)
        self._layouts = dict(layouts)
        self._gradients = {
            item.logical_param_id: item.main_grad for item in runtime_bindings
        }

    def local_loss_weight(self, iteration):
        return self.fixture.local_loss_weight(self.rank, iteration)

    def load_iteration(self, iteration):
        rank_inputs = self.fixture.rank_inputs_by_iteration[iteration][self.rank]
        for logical_id, gradient in self._gradients.items():
            gradient.copy_(torch.tensor(rank_inputs[logical_id], dtype=torch.float32))

    def assert_matches(self, expected):
        if not self.runtime_bindings:
            return
        for logical_id, gradient in self._gradients.items():
            parameter = self.fixture.parameter(logical_id)
            layout = self._layouts[logical_id]
            target = expected.value(logical_id)
            if parameter.tensor_parallel:
                tp_rank, unused_dp_rank = layout.coordinate(self.rank)
                shard_size = parameter.full_shape[0] // layout.tp_degree
                begin = tp_rank * shard_size
                target = target[begin : begin + shard_size]
            torch.testing.assert_close(gradient, target)

    def assert_clipped(self, expected, max_norm):
        coefficient = max_norm / (expected.l2_norm + 1.0e-6)
        clipped = ExpectedIteration(
            values=tuple(
                (logical_id, value * coefficient)
                for logical_id, value in expected.values
            ),
            l2_norm=expected.l2_norm,
            inf_norm=expected.inf_norm,
        )
        self.assert_matches(clipped)


def build_cpu_matrix():
    normalizer = load_normalizer()
    fixtures = []
    for config in normalizer.generate_heterogeneous_config_fixtures():
        normalized = normalizer.normalize_flexpipe_config(config)
        fixture_id = (
            f"generated-{normalized['world_size']}r-"
            f"{len(normalized['pipe'])}p-"
            f"{normalized['topology_fingerprint'][:12]}"
        )
        fixtures.append(
            CpuMatrixFixture(
                fixture_id=fixture_id,
                config=config,
                normalized_config=normalized,
                topology_fingerprint=normalized["topology_fingerprint"],
                world_size=normalized["world_size"],
            )
        )
    return tuple(fixtures)


def _fingerprint(parameters, loss_layouts):
    payload = {
        "parameters": [
            {
                "logical_id": item.logical_id,
                "global_op_index": item.global_op_index,
                "full_shape": item.full_shape,
                "tensor_parallel": item.tensor_parallel,
                "layouts": [_layout_payload(layout) for layout in item.layouts],
            }
            for item in parameters
        ],
        "loss_layouts": [_layout_payload(item) for item in loss_layouts],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _layout_payload(layout):
    return {
        "pipeline_id": layout.pipeline_id,
        "stage_id": layout.stage_id,
        "global_op_begin": layout.global_op_begin,
        "global_op_end": layout.global_op_end,
        "stage_ranks": layout.stage_ranks,
        "tp_degree": layout.tp_degree,
        "dp_degree": layout.dp_degree,
    }


def _distributed_fixture(
    fixture_id,
    world_size,
    parameters,
    loss_layouts,
    pipeline_gbs,
    loss_weights_by_iteration,
    rank_inputs_by_iteration=None,
):
    parameters = tuple(parameters)
    loss_layouts = tuple(loss_layouts)
    loss_weights_by_iteration = tuple(
        tuple(item) for item in loss_weights_by_iteration
    )
    if rank_inputs_by_iteration is None:
        rank_inputs_by_iteration = _default_rank_inputs(
            world_size, parameters, loss_weights_by_iteration
        )
    return DistributedMatrixFixture(
        fixture_id=fixture_id,
        world_size=world_size,
        parameters=parameters,
        loss_layouts=loss_layouts,
        pipeline_gbs=tuple(pipeline_gbs),
        loss_weights_by_iteration=loss_weights_by_iteration,
        rank_inputs_by_iteration=tuple(rank_inputs_by_iteration),
        topology_fingerprint=_fingerprint(parameters, loss_layouts),
    )


def _parameter(logical_id, global_op_index, tensor_parallel, layouts):
    return LogicalParameterSpec(
        logical_id=logical_id,
        global_op_index=global_op_index,
        full_shape=(4,) if tensor_parallel else (1,),
        tensor_parallel=tensor_parallel,
        layouts=tuple(layouts),
    )


def _canonical_rank_inputs():
    names = ("op-0.weight-0", "op-0.weight-1", "op-0.bias")
    iterations = (
        (
            ((1, 2, 3, 4), (2, 4, 6, 8), (1,)),
            ((10, 20, 30, 40), (20, 40, 60, 80), (2,)),
            ((100, 200), (200, 400), (3,)),
            ((3000, 4000), (6000, 8000), (3,)),
        ),
        (
            ((2, 4, 6, 8), (3, 6, 9, 12), (2,)),
            ((20, 40, 60, 80), (30, 60, 90, 120), (4,)),
            ((200, 400), (300, 600), (6,)),
            ((6000, 8000), (9000, 12000), (6,)),
        ),
    )
    return tuple(
        tuple(dict(zip(names, rank_values)) for rank_values in iteration)
        for iteration in iterations
    )


def build_distributed_matrix():
    canonical_layouts = (
        PipelineLayout(0, 0, (0, 1), 1, 2, 0, 43),
        PipelineLayout(1, 0, (2, 3), 2, 1, 0, 43),
    )
    swapped_p0_stage0 = PipelineLayout(0, 0, (0, 1), 2, 1, 0, 11)
    swapped_p0_stage1 = PipelineLayout(0, 1, (2,), 1, 1, 11, 43)
    swapped_p1_stage0 = PipelineLayout(1, 0, (3,), 1, 1, 0, 21)
    swapped_p1_stage1 = PipelineLayout(1, 1, (4, 5), 1, 2, 21, 43)
    three_pipeline_layouts = (
        PipelineLayout(0, 0, (0, 1), 1, 2, 0, 43),
        PipelineLayout(1, 0, (2, 3), 2, 1, 0, 43),
        PipelineLayout(2, 0, (4, 5), 1, 2, 0, 43),
    )
    return (
        _distributed_fixture(
            "canonical-4r-equal-gbs",
            4,
            (
                _parameter("op-0.weight-0", 0, True, canonical_layouts),
                _parameter("op-0.weight-1", 0, True, canonical_layouts),
                _parameter("op-0.bias", 0, False, canonical_layouts),
            ),
            canonical_layouts,
            (2, 2),
            ((2, 2), (2, 2)),
            _canonical_rank_inputs(),
        ),
        _distributed_fixture(
            "hetero-6r-swapped-stages",
            6,
            (
                _parameter(
                    "op-0.weight", 0, True,
                    (swapped_p0_stage0, swapped_p1_stage0),
                ),
                _parameter(
                    "op-11.weight", 11, True,
                    (swapped_p0_stage1, swapped_p1_stage0),
                ),
                _parameter(
                    "op-21.weight", 21, True,
                    (swapped_p0_stage1, swapped_p1_stage1),
                ),
                _parameter(
                    "op-0.bias", 0, False,
                    (swapped_p0_stage0, swapped_p1_stage0),
                ),
            ),
            (swapped_p0_stage1, swapped_p1_stage1),
            (24, 24),
            ((12, 24), (18, 10)),
        ),
        _distributed_fixture(
            "canonical-6r-three-pipeline",
            6,
            (
                _parameter("op-0.weight-0", 0, True, three_pipeline_layouts),
                _parameter("op-0.weight-1", 0, True, three_pipeline_layouts),
                _parameter("op-0.bias", 0, False, three_pipeline_layouts),
            ),
            three_pipeline_layouts,
            (12, 16, 20),
            ((6, 12, 15), (10, 8, 18)),
        ),
    )


def build_rank_contract(fixture, rank, descriptor_cls):
    from tests.test_gradient_fragment_planner import descriptor

    descriptors = []
    runtime_bindings = []
    local_layouts = {}
    for parameter in fixture.parameters:
        layout = parameter.layout_for_rank(rank)
        if layout is None:
            continue
        tp_rank, dp_rank = layout.coordinate(rank)
        tp_group = layout.tp_group(dp_rank)
        dp_group = layout.dp_group(tp_rank)
        local_shape = (
            (parameter.full_shape[0] // layout.tp_degree,)
            if parameter.tensor_parallel
            else parameter.full_shape
        )
        item = descriptor(
            descriptor_cls,
            parameter.logical_id,
            rank,
            layout.pipeline_id,
            layout.tp_degree,
            tp_rank,
            parameter.full_shape,
            local_shape,
            parameter.tensor_parallel,
            0 if parameter.tensor_parallel else -1,
            dp_degree=layout.dp_degree,
            dp_rank=dp_rank,
            tp_group=tp_group,
            dp_group=dp_group,
            stage_ranks=layout.stage_ranks,
        )
        descriptors.append(
            _with_layout_metadata(item, layout, parameter.global_op_index)
        )
        gradient = torch.empty(local_shape, dtype=torch.float32)
        runtime_bindings.append(
            SimpleNamespace(
                logical_param_id=parameter.logical_id,
                parameter=torch.zeros_like(gradient),
                main_grad=gradient,
                device=gradient.device,
            )
        )
        local_layouts[parameter.logical_id] = layout
    return RankContract(
        fixture, rank, descriptors, runtime_bindings, local_layouts.items()
    )


def _parameter_value(
    parameter_index, iteration, pipeline_id, dp_rank, coordinate
):
    return (
        100.0 * (iteration + 1)
        + 20.0 * (parameter_index + 1)
        + 5.0 * (pipeline_id + 1)
        + float(dp_rank)
        + float(coordinate) / 10.0
    )


def _default_rank_inputs(world_size, parameters, loss_weights_by_iteration):
    iterations = []
    for iteration, loss_weights in enumerate(loss_weights_by_iteration):
        rank_inputs = [dict() for _ in range(world_size)]
        for parameter_index, parameter in enumerate(parameters):
            for layout in parameter.layouts:
                local_loss_weight = (
                    loss_weights[layout.pipeline_id] / layout.dp_degree
                )
                for rank in layout.stage_ranks:
                    tp_rank, dp_rank = layout.coordinate(rank)
                    if parameter.tensor_parallel:
                        shard_size = parameter.full_shape[0] // layout.tp_degree
                        begin = tp_rank * shard_size
                        coordinates = range(begin, begin + shard_size)
                    else:
                        coordinates = range(parameter.full_shape[0])
                    rank_inputs[rank][parameter.logical_id] = tuple(
                        _parameter_value(
                            parameter_index,
                            iteration,
                            layout.pipeline_id,
                            dp_rank,
                            coordinate,
                        )
                        * local_loss_weight
                        for coordinate in coordinates
                    )
        iterations.append(tuple(rank_inputs))
    return tuple(iterations)


def expected_iteration(fixture, iteration):
    global_weight = float(sum(fixture.loss_weights_by_iteration[iteration]))
    scale = fixture.nominal_global_loss_scale / global_weight
    expected_values = []
    for parameter in fixture.parameters:
        values = []
        for coordinate in range(parameter.full_shape[0]):
            numerator = 0.0
            for layout in parameter.layouts:
                for dp_rank in range(layout.dp_degree):
                    if parameter.tensor_parallel:
                        shard_size = parameter.full_shape[0] // layout.tp_degree
                        tp_rank = coordinate // shard_size
                        local_offset = coordinate % shard_size
                    else:
                        tp_rank = 0
                        local_offset = coordinate
                    rank = layout.stage_ranks[
                        dp_rank * layout.tp_degree + tp_rank
                    ]
                    numerator += fixture.rank_inputs_by_iteration[iteration][rank][
                        parameter.logical_id
                    ][local_offset]
            values.append(numerator * scale)
        expected_values.append(
            (parameter.logical_id, torch.tensor(values, dtype=torch.float32))
        )
    all_values = torch.cat([value for unused_name, value in expected_values])
    return ExpectedIteration(
        values=tuple(expected_values),
        l2_norm=math.sqrt(float(all_values.float().pow(2).sum().item())),
        inf_norm=float(all_values.abs().max().item()),
    )
