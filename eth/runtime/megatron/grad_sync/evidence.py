"""Test-only numerical evidence helpers for GPT gradient synchronization."""

import hashlib
import math
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path

import torch


def _logical_seed(logical_param_id):
    digest = hashlib.sha256(logical_param_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def deterministic_logical_tensor(logical_param_id, full_shape, dtype):
    shape = tuple(int(size) for size in full_shape)
    if any(size < 0 for size in shape):
        raise ValueError(
            f"{logical_param_id} has invalid full shape {shape}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_logical_seed(logical_param_id))
    values = torch.empty(shape, dtype=torch.float32, device="cpu")
    values.uniform_(-0.02, 0.02, generator=generator)
    return values.to(dtype=dtype)


def local_tensor_from_logical(full_tensor, descriptor):
    logical_id = descriptor.logical_param_id
    full_shape = tuple(int(size) for size in descriptor.full_shape)
    local_shape = tuple(int(size) for size in descriptor.local_shape)
    if tuple(full_tensor.shape) != full_shape:
        raise ValueError(
            f"{logical_id} full shape mismatch: "
            f"{tuple(full_tensor.shape)} != {full_shape}"
        )
    if not descriptor.tensor_parallel:
        local = full_tensor.clone()
    else:
        axis = int(descriptor.partition_dim)
        if axis < 0:
            axis += len(full_shape)
        if not 0 <= axis < len(full_shape):
            raise ValueError(f"{logical_id} has invalid partition axis {axis}")
        tp_degree = int(descriptor.local_tp_degree)
        tp_rank = int(descriptor.local_tp_rank)
        stride = int(descriptor.partition_stride)
        if tp_degree <= 0 or not 0 <= tp_rank < tp_degree or stride <= 0:
            raise ValueError(
                f"{logical_id} has invalid TP coordinate "
                f"rank={tp_rank} degree={tp_degree} stride={stride}"
            )
        local_axis_size = local_shape[axis]
        if local_axis_size % stride:
            raise ValueError(
                f"{logical_id} local shape {local_shape} is not divisible "
                f"by partition stride {stride} on axis {axis}"
            )
        chunk_size = local_axis_size // stride
        expected_axis_size = chunk_size * stride * tp_degree
        if full_shape[axis] != expected_axis_size:
            raise ValueError(
                f"{logical_id} local shape {local_shape} and TP layout imply "
                f"axis size {expected_axis_size}, got {full_shape[axis]}"
            )
        chunks = torch.split(full_tensor, chunk_size, dim=axis)
        selected = chunks[tp_rank::tp_degree]
        if len(selected) != stride:
            raise ValueError(
                f"{logical_id} expected {stride} strided chunks, "
                f"got {len(selected)}"
            )
        local = torch.cat(selected, dim=axis).contiguous()
    if tuple(local.shape) != local_shape:
        raise ValueError(
            f"{logical_id} local shape mismatch: "
            f"{tuple(local.shape)} != {local_shape}"
        )
    if not math.prod(local_shape) and local.numel():
        raise ValueError(f"{logical_id} empty local shape has data")
    return local


def _tensor_checksum(tensor):
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _by_logical_id(items, kind):
    result = {}
    for item in items:
        logical_id = item.logical_param_id
        if logical_id in result:
            raise ValueError(f"duplicate {kind} for logical parameter {logical_id}")
        result[logical_id] = item
    return result


def apply_canonical_initialization(descriptors, runtime_bindings):
    descriptor_by_id = _by_logical_id(descriptors, "descriptor")
    binding_by_id = _by_logical_id(runtime_bindings, "runtime binding")
    if set(descriptor_by_id) != set(binding_by_id):
        raise ValueError(
            "canonical initialization descriptor/binding IDs differ: "
            f"{sorted(descriptor_by_id)} != {sorted(binding_by_id)}"
        )
    checksums = {}
    for logical_id in sorted(descriptor_by_id):
        descriptor = descriptor_by_id[logical_id]
        binding = binding_by_id[logical_id]
        initialization_id = descriptor.shared_weight_id or logical_id
        full = deterministic_logical_tensor(
            initialization_id,
            descriptor.full_shape,
            binding.parameter.dtype,
        )
        local = local_tensor_from_logical(full, descriptor).to(
            device=binding.parameter.device,
            dtype=binding.parameter.dtype,
        )
        with torch.no_grad():
            binding.parameter.copy_(local)
        checksums[logical_id] = _tensor_checksum(full)
    return checksums


def _descriptor_dict(descriptor):
    if is_dataclass(descriptor):
        return asdict(descriptor)
    return dict(vars(descriptor))


def _cpu_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_value(item) for item in value)
    return value


def _validate_finite(value, context):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{context} contains non-finite values")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{context}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{context}[{index}]")


def build_rank_snapshot(
    *,
    phase,
    iteration,
    global_rank,
    descriptors,
    runtime_bindings,
    optimizer_state_by_logical=None,
    optimizer_parameter_by_logical=None,
    optimizer_hyperparameters_by_logical=None,
    clock=None,
    metrics=None,
    run_metadata=None,
):
    valid_phases = {"init", "pre_sync", "post_sync", "post_step"}
    if phase not in valid_phases:
        raise ValueError(f"unsupported evidence phase {phase!r}")
    descriptor_by_id = _by_logical_id(descriptors, "descriptor")
    binding_by_id = _by_logical_id(runtime_bindings, "runtime binding")
    if set(descriptor_by_id) != set(binding_by_id):
        raise ValueError(
            "snapshot descriptor/binding IDs differ: "
            f"{sorted(descriptor_by_id)} != {sorted(binding_by_id)}"
        )
    optimizer_state_by_logical = optimizer_state_by_logical or {}
    optimizer_parameter_by_logical = optimizer_parameter_by_logical or {}
    optimizer_hyperparameters_by_logical = (
        optimizer_hyperparameters_by_logical or {}
    )
    records = []
    for logical_id in sorted(descriptor_by_id):
        descriptor = descriptor_by_id[logical_id]
        binding = binding_by_id[logical_id]
        parameter = _cpu_value(binding.parameter)
        main_grad = _cpu_value(binding.main_grad)
        optimizer_state = _cpu_value(
            optimizer_state_by_logical.get(logical_id, {})
        )
        optimizer_parameter = _cpu_value(
            optimizer_parameter_by_logical.get(logical_id, binding.parameter)
        )
        optimizer_hyperparameters = _cpu_value(
            optimizer_hyperparameters_by_logical.get(logical_id, {})
        )
        _validate_finite(parameter, f"{logical_id}.parameter")
        if phase == "init":
            _validate_finite(main_grad, f"{logical_id}.main_grad")
        _validate_finite(optimizer_parameter, f"{logical_id}.optimizer_parameter")
        _validate_finite(optimizer_state, f"{logical_id}.optimizer_state")
        _validate_finite(
            optimizer_hyperparameters,
            f"{logical_id}.optimizer_hyperparameters",
        )
        records.append(
            {
                "descriptor": _descriptor_dict(descriptor),
                "parameter": parameter,
                "main_grad": main_grad,
                "optimizer_state": optimizer_state,
                "optimizer_parameter": optimizer_parameter,
                "optimizer_hyperparameters": optimizer_hyperparameters,
            }
        )
    snapshot = {
        "version": 1,
        "phase": phase,
        "iteration": int(iteration),
        "global_rank": int(global_rank),
        "records": records,
        "clock": _cpu_value(clock or {}),
        "metrics": _cpu_value(metrics or {}),
        "run_metadata": dict(run_metadata or {}),
    }
    _validate_finite(snapshot["clock"], "clock")
    _validate_finite(snapshot["metrics"], "metrics")
    return snapshot


def save_rank_snapshot(output_dir, **snapshot_args):
    snapshot = build_rank_snapshot(**snapshot_args)
    rank_dir = Path(output_dir) / f"rank-{snapshot['global_rank']:05d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    path = rank_dir / (
        f"iteration-{snapshot['iteration']:06d}-{snapshot['phase']}.pt"
    )
    torch.save(snapshot, path)
    return path


def load_rank_snapshot(path):
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if snapshot.get("version") != 1:
        raise ValueError(
            f"unsupported evidence version {snapshot.get('version')!r}"
        )
    return snapshot


def _descriptor_value(record, name, default=None):
    return record["descriptor"].get(name, default)


def _tensor_equal_with_matching_non_finite(actual, expected):
    return torch.allclose(
        actual, expected, rtol=0.0, atol=0.0, equal_nan=True
    )


def _replica_key(record):
    return (
        _descriptor_value(record, "pipeline_id", 0),
        _descriptor_value(record, "stage_id", 0),
        _descriptor_value(record, "model_chunk_id", 0),
        _descriptor_value(record, "local_dp_rank", 0),
    )


def _reconstruct_replica(records, field, average_replicated):
    first = records[0]["descriptor"]
    logical_id = first["logical_param_id"]
    full_shape = tuple(first["full_shape"])
    for record in records:
        descriptor = record["descriptor"]
        if descriptor["logical_param_id"] != logical_id:
            raise ValueError("replica records contain multiple logical IDs")
        if tuple(descriptor["full_shape"]) != full_shape:
            raise ValueError(f"{logical_id} replica full shapes differ")
    if not first["tensor_parallel"]:
        if any(record["descriptor"]["tensor_parallel"] for record in records):
            raise ValueError(f"{logical_id} replica mixes sharded and replicated records")
        values = [record[field].detach().cpu() for record in records]
        if any(tuple(value.shape) != full_shape for value in values):
            raise ValueError(f"{logical_id} replicated value shape mismatch")
        if average_replicated:
            return torch.stack(values).mean(dim=0)
        for value in values[1:]:
            if not _tensor_equal_with_matching_non_finite(value, values[0]):
                raise ValueError(f"{logical_id} replicated values differ")
        return values[0].clone()

    tp_degree = int(first["local_tp_degree"])
    axis = int(first["partition_dim"])
    if axis < 0:
        axis += len(full_shape)
    stride = int(first["partition_stride"])
    by_rank = {}
    for record in records:
        descriptor = record["descriptor"]
        layout = (
            descriptor["tensor_parallel"],
            int(descriptor["local_tp_degree"]),
            int(descriptor["partition_dim"]),
            int(descriptor["partition_stride"]),
        )
        if layout != (True, tp_degree, int(first["partition_dim"]), stride):
            raise ValueError(f"{logical_id} TP replica layouts differ")
        tp_rank = int(descriptor["local_tp_rank"])
        if tp_rank in by_rank:
            raise ValueError(f"{logical_id} duplicate TP rank {tp_rank}")
        by_rank[tp_rank] = record[field].detach().cpu()
    if set(by_rank) != set(range(tp_degree)):
        raise ValueError(
            f"{logical_id} missing TP ranks: {sorted(by_rank)} of {tp_degree}"
        )
    result = torch.empty(full_shape, dtype=by_rank[0].dtype)
    for tp_rank, local in sorted(by_rank.items()):
        local_shape = tuple(local.shape)
        local_axis_size = local_shape[axis]
        if local_axis_size % stride:
            raise ValueError(f"{logical_id} local TP shard is not stride divisible")
        chunk_size = local_axis_size // stride
        local_chunks = torch.split(local, chunk_size, dim=axis)
        for stride_index, chunk in enumerate(local_chunks):
            global_chunk_index = tp_rank + stride_index * tp_degree
            result.narrow(
                axis, global_chunk_index * chunk_size, chunk_size
            ).copy_(chunk)
    return result


def _records_by_logical_and_replica(records):
    grouped = {}
    for record in records:
        logical_id = record["descriptor"]["logical_param_id"]
        grouped.setdefault(logical_id, {}).setdefault(
            _replica_key(record), []
        ).append(record)
    return grouped


def reconstruct_equal_logical_values(records, field):
    result = {}
    for logical_id, replicas in sorted(
        _records_by_logical_and_replica(records).items()
    ):
        values = [
            _reconstruct_replica(replica, field, average_replicated=False)
            for unused_key, replica in sorted(replicas.items())
        ]
        for value in values[1:]:
            if not _tensor_equal_with_matching_non_finite(value, values[0]):
                raise ValueError(
                    f"{logical_id} reconstructed values differ across replicas"
                )
        result[logical_id] = values[0]
    return result


def reconstruct_equal_optimizer_states(records):
    result = {}
    for logical_id, replicas in sorted(
        _records_by_logical_and_replica(records).items()
    ):
        all_records = [record for replica in replicas.values() for record in replica]
        state_names = {tuple(sorted(record["optimizer_state"])) for record in all_records}
        if len(state_names) != 1:
            raise ValueError(f"{logical_id} optimizer state names differ")
        states = {}
        for state_name in next(iter(state_names), ()):
            sample = all_records[0]["optimizer_state"][state_name]
            local_shape = tuple(all_records[0]["descriptor"]["local_shape"])
            if torch.is_tensor(sample) and tuple(sample.shape) == local_shape:
                replica_values = []
                for unused_key, replica in sorted(replicas.items()):
                    state_records = []
                    for record in replica:
                        state_record = dict(record)
                        state_record["state_value"] = record["optimizer_state"][state_name]
                        state_records.append(state_record)
                    replica_values.append(
                        _reconstruct_replica(
                            state_records,
                            "state_value",
                            average_replicated=False,
                        )
                    )
                for value in replica_values[1:]:
                    if not torch.equal(value, replica_values[0]):
                        raise ValueError(
                            f"{logical_id}.{state_name} differs across replicas"
                        )
                states[state_name] = replica_values[0]
            else:
                values = [record["optimizer_state"][state_name] for record in all_records]
                first = values[0]
                for value in values[1:]:
                    equal = (
                        torch.equal(value, first)
                        if torch.is_tensor(first)
                        else value == first
                    )
                    if not equal:
                        raise ValueError(
                            f"{logical_id}.{state_name} differs across replicas"
                        )
                states[state_name] = _cpu_value(first)
        result[logical_id] = states
    return result


def full_gradient_reference(records, normalization_scale):
    result = {}
    grouped = _records_by_logical_and_replica(records)
    for logical_id, replicas in sorted(grouped.items()):
        contributions = [
            _reconstruct_replica(
                replica, "main_grad", average_replicated=True
            )
            for unused_key, replica in sorted(replicas.items())
        ]
        total = torch.zeros_like(contributions[0])
        for contribution in contributions:
            total.add_(contribution)
        result[logical_id] = total.mul(float(normalization_scale))
    shared_ids = {}
    for logical_id, replicas in grouped.items():
        first_replica = next(iter(replicas.values()))
        shared_id = first_replica[0]["descriptor"].get("shared_weight_id")
        if shared_id:
            shared_ids.setdefault(shared_id, []).append(logical_id)
    for shared_id, logical_ids in sorted(shared_ids.items()):
        if len(logical_ids) < 2:
            continue
        shapes = {tuple(result[logical_id].shape) for logical_id in logical_ids}
        if len(shapes) != 1:
            raise ValueError(
                f"shared weight {shared_id} has incompatible shapes {sorted(shapes)}"
            )
        combined = torch.zeros_like(result[logical_ids[0]])
        for logical_id in sorted(logical_ids):
            combined.add_(result[logical_id])
        for logical_id in logical_ids:
            result[logical_id] = combined.clone()
    return result


def clip_logical_gradients(gradients, max_norm, norm_owner_ids=None):
    owner_ids = set(gradients) if norm_owner_ids is None else set(norm_owner_ids)
    unknown = owner_ids - set(gradients)
    if unknown:
        raise ValueError(f"unknown norm owner IDs: {sorted(unknown)}")
    norm_squared = sum(
        gradient.detach().double().square().sum().item()
        for logical_id, gradient in gradients.items()
        if logical_id in owner_ids
    )
    total_norm = math.sqrt(norm_squared)
    coefficient = min(1.0, float(max_norm) / (total_norm + 1.0e-6))
    return (
        {
            logical_id: gradient.detach().clone().mul(coefficient)
            for logical_id, gradient in gradients.items()
        },
        total_norm,
        coefficient,
    )


def adamw_reference_step(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    *,
    step,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    weight_decay,
):
    step = int(step)
    if step <= 0:
        raise ValueError(f"AdamW step must be positive, got {step}")
    next_exp_avg = exp_avg.mul(beta1).add(gradient, alpha=1.0 - beta1)
    next_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(
        gradient, gradient, value=1.0 - beta2
    )
    bias_correction1 = 1.0 - beta1**step
    bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
    denominator = next_exp_avg_sq.sqrt().div(bias_correction2_sqrt).add(
        epsilon
    )
    next_parameter = parameter.mul(1.0 - learning_rate * weight_decay)
    next_parameter = next_parameter.addcdiv(
        next_exp_avg,
        denominator,
        value=-learning_rate / bias_correction1,
    )
    return next_parameter, next_exp_avg, next_exp_avg_sq


def _flat_index(shape, index):
    if not shape:
        return []
    flat_index = torch.tensor(index, dtype=torch.int64)
    return [int(item) for item in torch.unravel_index(flat_index, shape)]


def compare_logical_tensors(actual, expected, *, rtol, atol, equal_nan=False):
    logical_ids = sorted(set(actual) | set(expected))
    reports = []
    for logical_id in logical_ids:
        if logical_id not in actual or logical_id not in expected:
            reports.append(
                {
                    "logical_param_id": logical_id,
                    "passed": False,
                    "error": "missing actual" if logical_id not in actual else "missing expected",
                }
            )
            continue
        actual_value = actual[logical_id].detach().cpu()
        expected_value = expected[logical_id].detach().cpu()
        if tuple(actual_value.shape) != tuple(expected_value.shape):
            reports.append(
                {
                    "logical_param_id": logical_id,
                    "passed": False,
                    "error": "shape mismatch",
                    "actual_shape": list(actual_value.shape),
                    "expected_shape": list(expected_value.shape),
                }
            )
            continue
        finite_pair = torch.isfinite(actual_value) & torch.isfinite(expected_value)
        matching_infinity = (
            torch.isinf(actual_value)
            & torch.isinf(expected_value)
            & (torch.signbit(actual_value) == torch.signbit(expected_value))
        )
        matching_nan = (
            torch.isnan(actual_value) & torch.isnan(expected_value)
            if equal_nan
            else torch.zeros_like(finite_pair)
        )
        matching_special = matching_infinity | matching_nan
        difference = torch.where(
            finite_pair,
            (actual_value - expected_value).abs().double(),
            torch.where(
                matching_special,
                torch.zeros_like(actual_value, dtype=torch.float64),
                torch.full_like(actual_value, float("inf"), dtype=torch.float64),
            ),
        )
        flat_abs_index = int(difference.reshape(-1).argmax().item())
        max_abs_error = float(difference.reshape(-1)[flat_abs_index].item())
        denominator = expected_value.detach().abs().double().clamp_min(atol)
        relative = torch.where(
            finite_pair,
            difference / denominator,
            difference,
        )
        flat_rel_index = int(relative.reshape(-1).argmax().item())
        max_rel_error = float(relative.reshape(-1)[flat_rel_index].item())
        passed = bool(
            torch.allclose(
                actual_value,
                expected_value,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )
        reports.append(
            {
                "logical_param_id": logical_id,
                "passed": passed,
                "max_abs_error": max_abs_error,
                "max_abs_index": _flat_index(
                    tuple(difference.shape), flat_abs_index
                ),
                "max_abs_region": {
                    "shape": list(difference.shape),
                    "coordinate": _flat_index(
                        tuple(difference.shape), flat_abs_index
                    ),
                },
                "max_rel_error": max_rel_error,
                "max_rel_index": _flat_index(
                    tuple(relative.shape), flat_rel_index
                ),
            }
        )
    return {
        "passed": all(item["passed"] for item in reports),
        "rtol": float(rtol),
        "atol": float(atol),
        "parameters": reports,
    }
