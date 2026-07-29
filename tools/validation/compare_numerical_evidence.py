#!/usr/bin/env python3
"""Compare independently reconstructable GPT numerical evidence runs."""

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eth" / "runtime"))

from megatron.grad_sync.evidence import compare_logical_tensors
from megatron.grad_sync.evidence import adamw_reference_step
from megatron.grad_sync.evidence import clip_logical_gradients
from megatron.grad_sync.evidence import full_gradient_reference
from megatron.grad_sync.evidence import load_rank_snapshot
from megatron.grad_sync.evidence import reconstruct_equal_logical_values
from megatron.grad_sync.evidence import reconstruct_equal_optimizer_states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--full-gradient", action="store_true")
    return parser.parse_args()


def load_run(path):
    grouped = {}
    metadata = None
    files = sorted(Path(path).glob("rank-*/iteration-*.pt"))
    if not files:
        raise ValueError(f"no evidence snapshots found under {path}")
    for file_path in files:
        snapshot = load_rank_snapshot(file_path)
        if metadata is None:
            metadata = snapshot.get("run_metadata", {})
        elif snapshot.get("run_metadata", {}) != metadata:
            raise ValueError(f"run metadata differs in {file_path}")
        key = (snapshot["iteration"], snapshot["phase"])
        grouped.setdefault(key, []).append(snapshot)
    return grouped, metadata


def records(snapshots):
    return [
        record
        for snapshot in sorted(snapshots, key=lambda item: item["global_rank"])
        for record in snapshot["records"]
    ]


def scalar_comparison(actual, expected, names, rtol, atol):
    result = []
    for name in names:
        actual_values = [item.get(name) for item in actual]
        expected_values = [item.get(name) for item in expected]
        if any(value is None for value in actual_values + expected_values):
            result.append(
                {
                    "name": name,
                    "passed": False,
                    "error": "missing required scalar",
                    "actual": actual_values,
                    "expected": expected_values,
                }
            )
            continue
        first = expected_values[0]
        passed = all(
            value == first
            if isinstance(first, (bool, str))
            else math.isclose(float(value), float(first), rel_tol=rtol, abs_tol=atol)
            for value in actual_values + expected_values
        )
        result.append(
            {
                "name": name,
                "passed": passed,
                "actual": actual_values,
                "expected": expected_values,
            }
        )
    return result


def validate_run(run):
    iterations = sorted({iteration for iteration, unused_phase in run if iteration})
    if len(iterations) < 2 or iterations != list(range(1, max(iterations) + 1)):
        raise ValueError(f"run requires at least two contiguous iterations, got {iterations}")
    required = {(0, "init")}
    for iteration in iterations:
        required.update(
            (iteration, phase)
            for phase in ("pre_sync", "post_sync", "post_step")
        )
    missing = sorted(required - set(run))
    extra = sorted(set(run) - required)
    if missing or extra:
        raise ValueError(f"incomplete evidence phases: missing={missing} extra={extra}")
    expected_ranks = {
        snapshot["global_rank"] for snapshot in run[(0, "init")]
    }
    metadata = run[(0, "init")][0].get("run_metadata", {})
    declared_ranks = set(range(int(metadata.get("world_size", -1))))
    if expected_ranks != declared_ranks:
        raise ValueError(
            f"init rank set {sorted(expected_ranks)} does not match "
            f"declared world size {metadata.get('world_size')}"
        )
    for key in sorted(required):
        ranks = {snapshot["global_rank"] for snapshot in run[key]}
        if ranks != expected_ranks or len(run[key]) != len(expected_ranks):
            raise ValueError(
                f"incomplete rank set for {key}: {sorted(ranks)} != "
                f"{sorted(expected_ranks)}"
            )
    return iterations


def _learning_rate(metadata, num_steps):
    max_lr = float(metadata["max_lr"])
    min_lr = float(metadata["min_lr"])
    warmup_steps = float(metadata["lr_warmup_steps"])
    decay_steps = float(metadata["lr_decay_steps"])
    style = metadata["lr_decay_style"]
    if warmup_steps > 0 and num_steps <= warmup_steps:
        return max_lr * float(num_steps) / warmup_steps
    if style == "constant":
        return max_lr
    if num_steps > decay_steps:
        return min_lr
    ratio = float(num_steps - warmup_steps) / float(decay_steps - warmup_steps)
    if style == "linear":
        coefficient = 1.0 - ratio
    elif style == "cosine":
        coefficient = 0.5 * (math.cos(math.pi * ratio) + 1.0)
    else:
        raise ValueError(f"unsupported LR decay style {style!r}")
    return min_lr + coefficient * (max_lr - min_lr)


def flatten_optimizer_states(states):
    flattened = {}
    for logical_id, values in sorted(states.items()):
        for state_name, value in sorted(values.items()):
            if not hasattr(value, "detach"):
                import torch

                value = torch.tensor(value)
            flattened[f"{logical_id}.{state_name}"] = value
    return flattened


def unscale_gradients(values, loss_scale):
    return {
        logical_id: value.detach().cpu().double() / float(loss_scale)
        for logical_id, value in values.items()
    }


def compare_runs(actual, expected, rtol, atol):
    comparable_phases = {"init", "post_sync", "post_step"}
    validate_run(actual)
    validate_run(expected)
    common = sorted(key for key in actual if key[1] in comparable_phases)
    if set(common) != {key for key in expected if key[1] in comparable_phases}:
        raise ValueError("runs have different comparable evidence phases")
    comparisons = []
    for iteration, phase in common:
        field = "main_grad" if phase == "post_sync" else "parameter"
        overflow_phase = phase == "post_sync" and all(
            item["metrics"].get("overflow") is True
            for run in (actual, expected)
            for item in run[(iteration, "post_step")]
        )
        actual_values = reconstruct_equal_logical_values(
            records(actual[(iteration, phase)]), field
        )
        expected_values = reconstruct_equal_logical_values(
            records(expected[(iteration, phase)]), field
        )
        actual_topology = actual[(0, "init")][0]["run_metadata"].get(
            "topology_fingerprint"
        )
        expected_topology = expected[(0, "init")][0]["run_metadata"].get(
            "topology_fingerprint"
        )
        compare_unscaled = (
            overflow_phase
            and actual_topology is not None
            and expected_topology is not None
            and actual_topology != expected_topology
        )
        if compare_unscaled:
            actual_values = unscale_gradients(
                actual_values,
                actual[(iteration, phase)][0]["clock"]["loss_scale"],
            )
            expected_values = unscale_gradients(
                expected_values,
                expected[(iteration, phase)][0]["clock"]["loss_scale"],
            )
        tensor_report = compare_logical_tensors(
            actual_values,
            expected_values,
            rtol=rtol,
            atol=atol,
            equal_nan=overflow_phase,
        )
        tensor_report["comparison_mode"] = (
            "unscaled_values" if compare_unscaled else "values"
        )
        clock_report = scalar_comparison(
            [item["clock"] for item in actual[(iteration, phase)]],
            [item["clock"] for item in expected[(iteration, phase)]],
            (
                "learning_rate",
                "global_data_cursor",
                "successful_optimizer_steps",
                "loss_scale",
            ),
            rtol,
            atol,
        )
        metric_report = []
        if phase == "post_step":
            overflow_values = [
                item["metrics"].get("overflow")
                for item in actual[(iteration, phase)]
            ]
            metric_names = ("overflow", "update_successful")
            if not all(value is True for value in overflow_values):
                metric_names = (
                    "grad_norm",
                    "clip_coefficient",
                    "overflow",
                    "update_successful",
                )
            metric_report = scalar_comparison(
                [item["metrics"] for item in actual[(iteration, phase)]],
                [item["metrics"] for item in expected[(iteration, phase)]],
                metric_names,
                rtol,
                atol,
            )
        optimizer_report = None
        if phase == "post_step":
            actual_optimizer = flatten_optimizer_states(
                reconstruct_equal_optimizer_states(
                    records(actual[(iteration, phase)])
                )
            )
            expected_optimizer = flatten_optimizer_states(
                reconstruct_equal_optimizer_states(
                    records(expected[(iteration, phase)])
                )
            )
            optimizer_report = compare_logical_tensors(
                actual_optimizer,
                expected_optimizer,
                rtol=rtol,
                atol=atol,
            )
        comparisons.append(
            {
                "iteration": iteration,
                "phase": phase,
                "field": field,
                "passed": tensor_report["passed"]
                and all(item["passed"] for item in clock_report + metric_report)
                and (optimizer_report is None or optimizer_report["passed"]),
                "tensors": tensor_report,
                "clock": clock_report,
                "metrics": metric_report,
                "optimizer_state": optimizer_report,
            }
        )
    return comparisons


def compare_overflow_semantics(run, rtol, atol):
    validate_run(run)
    metadata = run[(0, "init")][0]["run_metadata"]
    total_gbs = int(metadata["total_gbs"])
    reports = []
    overflow_count = 0
    for iteration in sorted({key[0] for key in run if key[0]}):
        post_step = run[(iteration, "post_step")]
        overflow = [item["metrics"].get("overflow") for item in post_step]
        if not all(value is True for value in overflow):
            continue
        overflow_count += 1
        previous_key = (0, "init") if iteration == 1 else (iteration - 1, "post_step")
        previous_model_parameters = reconstruct_equal_logical_values(
            records(run[previous_key]), "parameter"
        )
        actual_model_parameters = reconstruct_equal_logical_values(
            records(post_step), "parameter"
        )
        model_parameter_report = compare_logical_tensors(
            actual_model_parameters,
            previous_model_parameters,
            rtol=rtol,
            atol=atol,
        )
        previous_parameters = reconstruct_equal_logical_values(
            records(run[previous_key]), "optimizer_parameter"
        )
        actual_parameters = reconstruct_equal_logical_values(
            records(post_step), "optimizer_parameter"
        )
        parameter_report = compare_logical_tensors(
            actual_parameters, previous_parameters, rtol=rtol, atol=atol
        )
        previous_states = flatten_optimizer_states(
            reconstruct_equal_optimizer_states(records(run[previous_key]))
        )
        actual_states = flatten_optimizer_states(
            reconstruct_equal_optimizer_states(records(post_step))
        )
        state_report = compare_logical_tensors(
            actual_states, previous_states, rtol=rtol, atol=atol
        )
        previous_clock = run[previous_key][0]["clock"]
        expected_loss_scale = previous_clock["loss_scale"]
        if metadata.get("loss_scale_mode") == "dynamic" and overflow_count >= int(
            metadata["hysteresis"]
        ):
            expected_loss_scale = max(
                float(metadata["min_loss_scale"]),
                float(metadata["initial_loss_scale"])
                * 0.5 ** (overflow_count - int(metadata["hysteresis"]) + 1),
            )
        expected_clock = {
            "learning_rate": previous_clock["learning_rate"],
            "global_data_cursor": previous_clock["global_data_cursor"] + total_gbs,
            "successful_optimizer_steps": previous_clock["successful_optimizer_steps"],
            "loss_scale": expected_loss_scale,
        }
        clock_report = scalar_comparison(
            [item["clock"] for item in post_step],
            [expected_clock],
            tuple(expected_clock),
            rtol,
            atol,
        )
        reports.append(
            {
                "iteration": iteration,
                "passed": model_parameter_report["passed"]
                and parameter_report["passed"]
                and state_report["passed"]
                and all(item["passed"] for item in clock_report),
                "model_parameters": model_parameter_report,
                "parameters": parameter_report,
                "optimizer_state": state_report,
                "clock": clock_report,
            }
        )
    return {
        "passed": all(item["passed"] for item in reports),
        "iterations": reports,
    }


def _shared_norm_owners(all_records):
    owners = set()
    shared = {}
    for record in all_records:
        descriptor = record["descriptor"]
        logical_id = descriptor["logical_param_id"]
        shared_id = descriptor.get("shared_weight_id")
        if shared_id:
            shared.setdefault(shared_id, {})[logical_id] = bool(
                descriptor.get("shared_weight_root")
            )
        else:
            owners.add(logical_id)
    for shared_id, logical_ids in sorted(shared.items()):
        roots = [logical_id for logical_id, root in logical_ids.items() if root]
        if len(roots) != 1:
            raise ValueError(
                f"shared weight {shared_id} requires one norm root, got {roots}"
            )
        owners.add(roots[0])
    return owners


def _optimizer_hyperparameters(all_records):
    result = {}
    for record in all_records:
        logical_id = record["descriptor"]["logical_param_id"]
        value = record.get("optimizer_hyperparameters", {})
        if not value or any(value.get(name) is None for name in ("beta1", "beta2", "epsilon")):
            raise ValueError(f"missing optimizer hyperparameters for {logical_id}")
        if logical_id in result and result[logical_id] != value:
            raise ValueError(f"optimizer hyperparameters differ for {logical_id}")
        result[logical_id] = value
    return result


def compare_full_gradient(run, rtol, atol):
    iterations = validate_run(run)
    metadata = run[(0, "init")][0]["run_metadata"]
    nominal_scale = float(metadata["nominal_global_loss_scale"])
    max_norm = float(metadata["clip_grad"])
    total_gbs = int(metadata["total_gbs"])
    initial_loss_scale = float(metadata["initial_loss_scale"])
    if metadata["loss_scale_mode"] == "dynamic" and len(iterations) > int(
        metadata["loss_scale_window"]
    ):
        raise ValueError("dynamic loss-scale growth beyond one window is unsupported")
    reports = []
    for iteration in iterations:
        pre_sync = run[(iteration, "pre_sync")]
        local_weights = [item["metrics"].get("local_loss_weight") for item in pre_sync]
        if any(value is None for value in local_weights):
            raise ValueError(f"iteration {iteration} missing local loss weights")
        global_weight = sum(float(value) for value in local_weights)
        if global_weight <= 0:
            raise ValueError(f"iteration {iteration} has non-positive global weight")
        normalization_scale = nominal_scale / global_weight
        for item in pre_sync:
            metrics = item["metrics"]
            if not math.isclose(
                float(metrics["global_loss_weight"]),
                global_weight,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            ):
                raise ValueError(f"iteration {iteration} global loss weight mismatch")
            if not math.isclose(
                float(metrics["normalization_scale"]),
                normalization_scale,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            ):
                raise ValueError(f"iteration {iteration} normalization mismatch")
        expected = full_gradient_reference(records(pre_sync), normalization_scale)
        actual = reconstruct_equal_logical_values(
            records(run[(iteration, "post_sync")]), "main_grad"
        )
        gradient_report = compare_logical_tensors(
            actual, expected, rtol=rtol, atol=atol
        )
        loss_scales = {
            float(item["clock"]["loss_scale"])
            for item in run[(iteration, "post_sync")]
        }
        if len(loss_scales) != 1:
            raise ValueError(f"iteration {iteration} loss scales differ")
        loss_scale = loss_scales.pop()
        unscaled = {
            logical_id: gradient / loss_scale
            for logical_id, gradient in expected.items()
        }
        clipped, norm, coefficient = clip_logical_gradients(
            unscaled,
            max_norm,
            _shared_norm_owners(records(pre_sync)),
        )
        metric_report = scalar_comparison(
            [item["metrics"] for item in run[(iteration, "post_step")]],
            [{"grad_norm": norm, "clip_coefficient": coefficient}],
            ("grad_norm", "clip_coefficient"),
            rtol,
            atol,
        )
        previous_key = (0, "init") if iteration == 1 else (iteration - 1, "post_step")
        previous_parameters = reconstruct_equal_logical_values(
            records(run[previous_key]), "optimizer_parameter"
        )
        previous_states = (
            {}
            if iteration == 1
            else reconstruct_equal_optimizer_states(records(run[previous_key]))
        )
        post_sync_records = records(run[(iteration, "post_sync")])
        hyperparameters = _optimizer_hyperparameters(post_sync_records)
        learning_rates = {
            float(item["clock"]["learning_rate"])
            for item in run[(iteration, "post_sync")]
        }
        if len(learning_rates) != 1:
            raise ValueError(f"iteration {iteration} learning rates differ")
        learning_rate = learning_rates.pop()
        expected_pre_clock = {
            "learning_rate": _learning_rate(metadata, (iteration - 1) * total_gbs),
            "global_data_cursor": (iteration - 1) * total_gbs,
            "successful_optimizer_steps": iteration - 1,
            "loss_scale": initial_loss_scale,
        }
        pre_clock_report = scalar_comparison(
            [item["clock"] for item in run[(iteration, "post_sync")]],
            [expected_pre_clock],
            tuple(expected_pre_clock),
            rtol,
            atol,
        )
        expected_parameters = {}
        expected_states = {}
        for logical_id, parameter in previous_parameters.items():
            state = previous_states.get(logical_id, {})
            exp_avg = state.get("exp_avg", parameter.new_zeros(parameter.shape))
            exp_avg_sq = state.get("exp_avg_sq", parameter.new_zeros(parameter.shape))
            previous_step = int(float(state.get("step", 0)))
            hyper = hyperparameters[logical_id]
            next_parameter, next_exp_avg, next_exp_avg_sq = adamw_reference_step(
                parameter,
                clipped[logical_id],
                exp_avg,
                exp_avg_sq,
                step=previous_step + 1,
                learning_rate=learning_rate,
                beta1=float(hyper["beta1"]),
                beta2=float(hyper["beta2"]),
                epsilon=float(hyper["epsilon"]),
                weight_decay=float(hyper["weight_decay"]),
            )
            expected_parameters[logical_id] = next_parameter
            expected_states[f"{logical_id}.exp_avg"] = next_exp_avg
            expected_states[f"{logical_id}.exp_avg_sq"] = next_exp_avg_sq
            expected_states[f"{logical_id}.step"] = parameter.new_tensor(
                float(previous_step + 1)
            )
        actual_post_parameters = reconstruct_equal_logical_values(
            records(run[(iteration, "post_step")]), "optimizer_parameter"
        )
        actual_post_states = flatten_optimizer_states(
            reconstruct_equal_optimizer_states(records(run[(iteration, "post_step")]))
        )
        parameter_report = compare_logical_tensors(
            actual_post_parameters, expected_parameters, rtol=rtol, atol=atol
        )
        optimizer_report = compare_logical_tensors(
            actual_post_states, expected_states, rtol=rtol, atol=atol
        )
        expected_post_clock = {
            "learning_rate": _learning_rate(metadata, iteration * total_gbs),
            "global_data_cursor": iteration * total_gbs,
            "successful_optimizer_steps": iteration,
            "loss_scale": initial_loss_scale,
        }
        post_clock_report = scalar_comparison(
            [item["clock"] for item in run[(iteration, "post_step")]],
            [expected_post_clock],
            tuple(expected_post_clock),
            rtol,
            atol,
        )
        passed = (
            gradient_report["passed"]
            and parameter_report["passed"]
            and optimizer_report["passed"]
            and all(item["passed"] for item in metric_report)
            and all(item["passed"] for item in pre_clock_report + post_clock_report)
        )
        reports.append(
            {
                "iteration": iteration,
                "passed": passed,
                "normalization_scale": normalization_scale,
                "global_loss_weight": global_weight,
                "grad_norm": norm,
                "clip_coefficient": coefficient,
                "gradients": gradient_report,
                "parameters": parameter_report,
                "optimizer_state": optimizer_report,
                "metrics": metric_report,
                "pre_step_clock": pre_clock_report,
                "post_step_clock": post_clock_report,
            }
        )
    return {"passed": all(item["passed"] for item in reports), "iterations": reports}


def main():
    args = parse_args()
    actual, actual_metadata = load_run(args.actual)
    expected, expected_metadata = load_run(args.expected)
    comparisons = compare_runs(actual, expected, args.rtol, args.atol)
    full_gradient = (
        compare_full_gradient(actual, args.rtol, args.atol)
        if args.full_gradient
        else None
    )
    overflow = compare_overflow_semantics(actual, args.rtol, args.atol)
    report = {
        "version": 1,
        "actual": str(Path(args.actual).resolve()),
        "expected": str(Path(args.expected).resolve()),
        "rtol": args.rtol,
        "atol": args.atol,
        "actual_metadata": actual_metadata,
        "expected_metadata": expected_metadata,
        "comparisons": comparisons,
        "full_gradient": full_gradient,
        "overflow": overflow,
    }
    report["passed"] = all(item["passed"] for item in comparisons) and (
        full_gradient is None or full_gradient["passed"]
    ) and overflow["passed"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
