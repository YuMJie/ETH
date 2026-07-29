"""Explicit GPT runtime-op to canonical gradient metadata adapter."""

from .metadata import LocalGradientBinding
from .metadata import SerializableGradientDescriptor


_PARAMETERLESS_OPS = {
    "enc-attention-score",
    "enc-attention-softmax",
    "enc-attention-dropout",
    "enc-attention-context",
    "enc-flash-attention",
    "enc-post-attention-dropout",
    "enc-MLP-gelu",
    "enc-post-MLP-dropout",
}


def _linear_semantics(runtime_op):
    algo = getattr(runtime_op, "algo", None)
    if algo == "column":
        return "sharded_dim0", "sharded_dim0"
    if algo == "row":
        return "sharded_dim1", "replicated"
    raise ValueError(
        f"runtime op {getattr(runtime_op, 'op_index', None)} "
        f"has unsupported linear algo {algo!r}"
    )


def _parameter_schema(runtime_op, layer_index):
    op_name = runtime_op.name
    layer_prefix = f"gpt.layers.{layer_index}"
    if op_name == "encoder-embedding":
        return (
            ("word_embeddings.weight", "gpt.word_embeddings.weight", "gpt.word_embeddings", True, True, "sharded_dim0"),
            ("position_embeddings.weight", "gpt.position_embeddings.weight", None, False, True, "replicated"),
            ("tokentype_embeddings.weight", "gpt.tokentype_embeddings.weight", None, False, False, "replicated"),
        )
    if op_name == "enc-1st-layernorm":
        component = f"{layer_prefix}.attention.input_layernorm"
        return (
            ("input_layernorm.weight", f"{component}.weight", None, False, True, "replicated"),
            ("input_layernorm.bias", f"{component}.bias", None, False, True, "replicated"),
        )
    if op_name == "enc-2nd-layernorm":
        component = f"{layer_prefix}.mlp.input_layernorm"
        return (
            ("input_layernorm.weight", f"{component}.weight", None, False, True, "replicated"),
            ("input_layernorm.bias", f"{component}.bias", None, False, True, "replicated"),
        )
    if op_name == "final-layernorm":
        return (
            ("input_layernorm.weight", "gpt.final_layernorm.weight", None, False, True, "replicated"),
            ("input_layernorm.bias", "gpt.final_layernorm.bias", None, False, True, "replicated"),
        )
    if op_name == "enc-attention-qkv":
        component = f"{layer_prefix}.attention.query_key_value"
        weight_semantics, bias_semantics = _linear_semantics(runtime_op)
        return (
            ("query_key_value.weight", f"{component}.weight", None, False, True, weight_semantics),
            ("query_key_value.bias", f"{component}.bias", None, False, True, bias_semantics),
        )
    if op_name == "enc-attention-dense":
        component = f"{layer_prefix}.attention.dense"
        weight_semantics, bias_semantics = _linear_semantics(runtime_op)
        return (
            ("dense.weight", f"{component}.weight", None, False, True, weight_semantics),
            ("dense.bias", f"{component}.bias", None, False, True, bias_semantics),
        )
    if op_name == "enc-MLP-GEMM-1":
        component = f"{layer_prefix}.mlp.dense_h_to_4h"
        weight_semantics, bias_semantics = _linear_semantics(runtime_op)
        return (
            ("dense.weight", f"{component}.weight", None, False, True, weight_semantics),
            ("dense.bias", f"{component}.bias", None, False, True, bias_semantics),
        )
    if op_name == "enc-MLP-GEMM-2":
        component = f"{layer_prefix}.mlp.dense_4h_to_h"
        weight_semantics, bias_semantics = _linear_semantics(runtime_op)
        return (
            ("dense.weight", f"{component}.weight", None, False, True, weight_semantics),
            ("dense.bias", f"{component}.bias", None, False, True, bias_semantics),
        )
    if op_name == "gpt-post-process":
        return (
            ("word_embeddings.weight", "gpt.output_head.word_embeddings.weight", "gpt.word_embeddings", False, True, "sharded_dim0"),
        )
    if op_name in _PARAMETERLESS_OPS:
        return ()
    raise ValueError(f"unsupported GPT manifest op {op_name!r}")


def _resolve_path(root, path):
    value = root
    for part in path.split("."):
        if value is None or not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _dtype_name(value):
    return str(value)


def _parameter_metadata(
    parameter, logical_param_id, global_rank, tp_degree, expected_semantics
):
    for attribute in (
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
    ):
        if not hasattr(parameter, attribute):
            raise ValueError(
                f"logical_param_id {logical_param_id} rank {global_rank} "
                f"is missing {attribute}"
            )
    local_shape = tuple(int(size) for size in parameter.shape)
    tensor_parallel = bool(parameter.tensor_model_parallel)
    partition_dim = int(parameter.partition_dim)
    partition_stride = int(parameter.partition_stride)
    if partition_stride <= 0:
        raise ValueError(
            f"logical_param_id {logical_param_id} rank {global_rank} has invalid partition_stride"
        )
    if tensor_parallel:
        if not local_shape:
            raise ValueError(
                f"logical_param_id {logical_param_id} rank {global_rank} cannot shard a scalar"
            )
        if partition_dim < 0:
            partition_dim += len(local_shape)
        if not 0 <= partition_dim < len(local_shape):
            raise ValueError(
                f"logical_param_id {logical_param_id} rank {global_rank} has invalid partition_dim"
            )
        full_shape = list(local_shape)
        full_shape[partition_dim] *= tp_degree
        replica_semantics = None
    else:
        if partition_dim not in (-1, 0) and local_shape:
            raise ValueError(
                f"logical_param_id {logical_param_id} rank {global_rank} has replicated partition_dim"
            )
        partition_dim = -1
        full_shape = list(local_shape)
        replica_semantics = "identical_duplicate"
    actual_semantics = (
        f"sharded_dim{partition_dim}" if tensor_parallel else "replicated"
    )
    if actual_semantics != expected_semantics:
        raise ValueError(
            f"logical_param_id {logical_param_id} rank {global_rank} "
            f"expected {expected_semantics}, got {actual_semantics}"
        )
    return (
        local_shape,
        tuple(full_shape),
        tensor_parallel,
        partition_dim,
        partition_stride,
        replica_semantics,
    )


def adapt_gpt_runtime_op(
    runtime_op,
    manifest,
    topology,
    global_rank,
    model_chunk_id=0,
):
    op_index = getattr(runtime_op, "op_index", None)
    if not isinstance(op_index, int) or not 0 <= op_index < len(manifest["ordered_ops"]):
        raise ValueError(f"rank {global_rank} has invalid runtime op index {op_index}")
    manifest_op = manifest["ordered_ops"][op_index]
    runtime_name = getattr(runtime_op, "name", None)
    if runtime_name != manifest_op["name"]:
        raise ValueError(
            f"rank {global_rank} op {op_index} runtime name {runtime_name!r} "
            f"does not match manifest {manifest_op['name']!r}"
        )
    stage = topology.stage_for_rank(global_rank)
    if not stage.global_op_begin <= op_index < stage.global_op_end:
        raise ValueError(
            f"rank {global_rank} pipeline {stage.pipeline_id} stage {stage.stage_id} "
            f"does not own op {op_index}"
        )
    rank_facts = topology.expected_rank_facts(global_rank)
    schema = _parameter_schema(runtime_op, manifest_op["layer_index"])
    descriptors = []
    bindings = []
    seen_parameters = set()
    for parameter_order, (
        path,
        logical_id,
        shared_id,
        shared_root,
        required,
        expected_semantics,
    ) in enumerate(schema):
        parameter = _resolve_path(runtime_op, path)
        if parameter is None:
            if required:
                raise ValueError(
                    f"logical_param_id {logical_id} rank {global_rank} is missing {path}"
                )
            continue
        if id(parameter) in seen_parameters:
            raise ValueError(
                f"duplicate parameter for logical_param_id {logical_id} rank {global_rank}"
            )
        seen_parameters.add(id(parameter))
        if not getattr(parameter, "requires_grad", False):
            continue
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is None:
            raise ValueError(
                f"logical_param_id {logical_id} rank {global_rank} is missing main_grad"
            )
        if tuple(main_grad.shape) != tuple(parameter.shape):
            raise ValueError(
                f"logical_param_id {logical_id} rank {global_rank} main_grad shape mismatch"
            )
        (
            local_shape,
            full_shape,
            tensor_parallel,
            partition_dim,
            partition_stride,
            replica_semantics,
        ) = _parameter_metadata(
            parameter,
            logical_id,
            global_rank,
            rank_facts["tp_degree"],
            expected_semantics,
        )
        descriptors.append(
            SerializableGradientDescriptor(
                logical_param_id=logical_id,
                shared_weight_id=shared_id,
                shared_weight_root=shared_root,
                logical_region_key=logical_id,
                semantic_order_key=(op_index, parameter_order, logical_id),
                global_rank=global_rank,
                model_impl=manifest["id"],
                op_manifest_hash=manifest["hash"],
                pipeline_id=stage.pipeline_id,
                stage_id=stage.stage_id,
                model_chunk_id=model_chunk_id,
                global_op_index=op_index,
                manifest_op_name=manifest_op["name"],
                runtime_op_name=runtime_name,
                relative_parameter_name=path,
                stage_global_op_begin=stage.global_op_begin,
                stage_global_op_end=stage.global_op_end,
                local_tp_degree=rank_facts["tp_degree"],
                local_tp_rank=rank_facts["tp_rank"],
                local_dp_degree=rank_facts["dp_degree"],
                local_dp_rank=rank_facts["dp_rank"],
                local_cp_degree=rank_facts["cp_degree"],
                local_cp_rank=rank_facts["cp_rank"],
                stage_ranks=rank_facts["stage_ranks"],
                local_tp_group_ranks=rank_facts["tp_group_ranks"],
                local_dp_group_ranks=rank_facts["dp_group_ranks"],
                full_shape=full_shape,
                local_shape=local_shape,
                tensor_parallel=tensor_parallel,
                tp_replica_semantics=replica_semantics,
                partition_dim=partition_dim,
                partition_stride=partition_stride,
                parameter_dtype=_dtype_name(parameter.dtype),
                gradient_dtype=_dtype_name(main_grad.dtype),
                requires_grad=True,
            )
        )
        bindings.append(
            LocalGradientBinding(
                logical_param_id=logical_id,
                parameter=parameter,
                main_grad=main_grad,
                device=parameter.device,
            )
        )
    if hasattr(runtime_op, "parameters"):
        actual_trainable = {
            id(parameter)
            for parameter in runtime_op.parameters(recurse=True)
            if getattr(parameter, "requires_grad", False)
        }
        unlisted = actual_trainable - seen_parameters
        if unlisted:
            raise ValueError(
                f"unlisted trainable parameter on rank {global_rank} op {op_index} "
                f"({runtime_name})"
            )
    return tuple(descriptors), tuple(bindings)


def collect_gpt_gradient_metadata(
    model_chunks,
    manifest,
    topology,
    global_rank,
):
    descriptors = []
    bindings = []
    stage = topology.stage_for_rank(global_rank)
    seen_op_indices = set()
    for model_chunk_id, model_chunk in enumerate(model_chunks):
        language_model = getattr(model_chunk, "language_model", None)
        runtime_ops = getattr(language_model, "ops", None)
        if runtime_ops is None:
            raise ValueError(
                f"rank {global_rank} model chunk {model_chunk_id} is missing language_model.ops"
            )
        for runtime_op in runtime_ops:
            op_index = getattr(runtime_op, "op_index", None)
            if op_index in seen_op_indices:
                raise ValueError(
                    f"rank {global_rank} has duplicate runtime op index {op_index}"
                )
            seen_op_indices.add(op_index)
            op_descriptors, op_bindings = adapt_gpt_runtime_op(
                runtime_op,
                manifest,
                topology,
                global_rank,
                model_chunk_id=model_chunk_id,
            )
            descriptors.extend(op_descriptors)
            bindings.extend(op_bindings)
    expected_op_indices = set(range(stage.global_op_begin, stage.global_op_end))
    missing = sorted(expected_op_indices - seen_op_indices)
    unexpected = sorted(seen_op_indices - expected_op_indices)
    if missing:
        raise ValueError(
            f"rank {global_rank} is missing runtime op indices {missing} "
            f"for pipeline {stage.pipeline_id} stage {stage.stage_id}"
        )
    if unexpected:
        raise ValueError(
            f"rank {global_rank} has unexpected runtime op indices {unexpected} "
            f"for pipeline {stage.pipeline_id} stage {stage.stage_id}"
        )
    return tuple(descriptors), tuple(bindings)
