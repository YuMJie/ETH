"""Serializable metadata records for canonical gradient synchronization."""

from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Tuple


@dataclass(frozen=True)
class SerializableGradientDescriptor:
    logical_param_id: str
    shared_weight_id: Optional[str]
    shared_weight_root: bool
    logical_region_key: str
    semantic_order_key: Tuple[Any, ...]
    global_rank: int
    model_impl: str
    op_manifest_hash: str
    pipeline_id: int
    stage_id: int
    model_chunk_id: int
    global_op_index: int
    manifest_op_name: str
    runtime_op_name: str
    relative_parameter_name: str
    stage_global_op_begin: int
    stage_global_op_end: int
    local_tp_degree: int
    local_tp_rank: int
    local_dp_degree: int
    local_dp_rank: int
    local_cp_degree: int
    local_cp_rank: int
    stage_ranks: Tuple[int, ...]
    local_tp_group_ranks: Tuple[int, ...]
    local_dp_group_ranks: Tuple[int, ...]
    full_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    tensor_parallel: bool
    tp_replica_semantics: Optional[str]
    partition_dim: int
    partition_stride: int
    parameter_dtype: str
    gradient_dtype: str
    requires_grad: bool


@dataclass(frozen=True)
class LocalGradientBinding:
    logical_param_id: str
    parameter: Any
    main_grad: Any
    device: Any
