"""Canonical FlexPipe configuration normalization before distributed setup."""

import ast
import copy
import hashlib
import json
import warnings


_MANIFESTS = {
    "gpt": {
        "id": "gpt",
        "ops_per_layer": 10,
        "layer_schema": (
            "enc-1st-layernorm",
            "enc-attention-qkv",
            "enc-flash-attention",
            "enc-attention-dense",
            "enc-post-attention-dropout",
            "enc-2nd-layernorm",
            "enc-MLP-GEMM-1",
            "enc-MLP-gelu",
            "enc-MLP-GEMM-2",
            "enc-post-MLP-dropout",
        ),
    },
}

_TOPOLOGY_FIELDS = (
    ("model_parallel_size", 1),
    ("data_parallel_size", 1),
    ("context_parallel_size", 1),
    ("algo", 0),
)

_SUPPORTED_TP_VALUES = (1, 2, 4, 6, 8)

_TOP_LEVEL_FIELDS = {
    "config_hash",
    "hidden_size",
    "manifest",
    "max_position_embeddings",
    "model_impl",
    "model_name",
    "model_size",
    "node_info",
    "num_attention_heads",
    "num_layers",
    "pipe",
    "pipeline_sample_offsets",
    "seq_length",
    "topology_fingerprint",
    "total_gbs",
    "world_size",
}

_PIPE_FIELDS = {
    "algo_of_each_op",
    "algo_of_each_stage",
    "checkpoint_activations",
    "context_parallel_size_of_each_op",
    "context_parallel_size_of_each_stage",
    "data_parallel_size_of_each_op",
    "data_parallel_size_of_each_stage",
    "global_batch_size",
    "micro_batch_size",
    "model_impl",
    "model_parallel_size_of_each_op",
    "model_parallel_size_of_each_stage",
    "node_info",
    "num_gpus",
    "num_ops_in_each_stage",
    "num_stages",
    "pipeline_index",
    "recompute_ops",
    "resharding_stages",
}


def _unknown_fields(value, allowed):
    return sorted(set(value) - allowed)


def _canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(model_impl, num_layers):
    try:
        template = _MANIFESTS[model_impl]
    except KeyError as exc:
        supported = ", ".join(sorted(_MANIFESTS))
        raise ValueError(
            f"unsupported model_impl {model_impl!r}; expected one of {supported}"
        ) from exc

    ordered_ops = [
        {
            "op_index": 0,
            "name": "encoder-embedding",
            "layer_index": None,
        }
    ]
    for layer_index in range(num_layers):
        for name in template["layer_schema"]:
            ordered_ops.append(
                {
                    "op_index": len(ordered_ops),
                    "name": name,
                    "layer_index": layer_index,
                }
            )
    ordered_ops.extend(
        (
            {
                "op_index": len(ordered_ops),
                "name": "final-layernorm",
                "layer_index": None,
            },
            {
                "op_index": len(ordered_ops) + 1,
                "name": "gpt-post-process",
                "layer_index": None,
            },
        )
    )
    manifest = {
        "id": template["id"],
        "ops_per_layer": template["ops_per_layer"],
        "ordered_ops": ordered_ops,
    }
    manifest["hash"] = _canonical_hash(manifest)
    return manifest


def validate_runtime_manifest(runtime_ops, manifest):
    expected = manifest["ordered_ops"]
    if len(runtime_ops) != len(expected):
        raise ValueError(
            "runtime GPT manifest mismatch: "
            f"expected {len(expected)} ops, got {len(runtime_ops)}"
        )
    for op_index, (runtime_op, expected_op) in enumerate(zip(runtime_ops, expected)):
        actual = (runtime_op.get("op_index"), runtime_op.get("name"))
        wanted = (expected_op["op_index"], expected_op["name"])
        if actual != wanted:
            raise ValueError(
                f"runtime GPT manifest mismatch at op {op_index}: "
                f"expected {wanted}, got {actual}"
            )


def _select_model_impl(config, pipeline_op_counts):
    num_layers = config.get("num_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")

    explicit = config.get("model_impl")
    per_pipeline = {pipe.get("model_impl") for pipe in config["pipe"] if pipe.get("model_impl")}
    if len(per_pipeline) > 1:
        raise ValueError("different manifests requested by per-pipeline model_impl values")
    if per_pipeline and explicit and explicit not in per_pipeline:
        raise ValueError("top-level model_impl conflicts with per-pipeline model_impl")
    if per_pipeline:
        explicit = next(iter(per_pipeline))

    inferred = []
    for total_ops in pipeline_op_counts:
        matches = [
            name
            for name, item in _MANIFESTS.items()
            if total_ops == 3 + num_layers * item["ops_per_layer"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"manifest size {total_ops} does not uniquely match a supported model_impl"
            )
        inferred.append(matches[0])
    if len(set(inferred)) != 1:
        raise ValueError("different manifests inferred across pipelines")

    inferred_impl = inferred[0]
    if explicit is not None:
        _manifest(explicit, num_layers)
        if explicit != inferred_impl:
            raise ValueError(
                f"model_impl {explicit!r} conflicts with pipeline manifest size"
            )
        return explicit

    warnings.warn(
        f"model_impl inferred as {inferred_impl}; regenerate the config with an explicit value",
        UserWarning,
        stacklevel=3,
    )
    return inferred_impl


def _validate_stage_list(pipe, key, num_stages):
    value = pipe.get(key)
    if not isinstance(value, list) or len(value) != num_stages:
        raise ValueError(f"{key} stage list length must equal num_stages")
    return value


def _expand_topology_field(pipe, stem, default, stage_ops):
    stage_key = f"{stem}_of_each_stage"
    op_key = f"{stem}_of_each_op"
    stage_values = pipe.get(stage_key)
    op_values = pipe.get(op_key)

    if stage_values is None and op_values is None and stem == "context_parallel_size":
        stage_values = [default] * len(stage_ops)
    if stage_values is not None:
        if not isinstance(stage_values, list) or len(stage_values) != len(stage_ops):
            raise ValueError(f"{stage_key} stage list length must equal num_stages")
        brief_expanded = [
            [value] * count for value, count in zip(stage_values, stage_ops)
        ]
    else:
        brief_expanded = None

    if op_values is not None:
        if not isinstance(op_values, list) or len(op_values) != len(stage_ops):
            raise ValueError(f"{op_key} stage list length must equal num_stages")
        canonical = []
        for stage_index, (values, count) in enumerate(zip(op_values, stage_ops)):
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(
                    f"{op_key} stage {stage_index} length must equal its op count"
                )
            if len(set(values)) != 1:
                raise ValueError(
                    f"intra-stage constancy violated for {op_key} at stage {stage_index}"
                )
            canonical.append(list(values))
        if brief_expanded is not None and canonical != brief_expanded:
            raise ValueError(f"brief and expanded {stem} values conflict")
        return canonical

    if brief_expanded is None:
        raise ValueError(f"missing {stage_key} or {op_key}")
    return brief_expanded


def _normalize_pipe(source, pipeline_index, manifest_size):
    pipe = copy.deepcopy(source)
    num_stages = pipe.get("num_stages")
    if not isinstance(num_stages, int) or num_stages <= 0:
        raise ValueError(f"pipeline {pipeline_index} num_stages must be positive")
    stage_ops = _validate_stage_list(pipe, "num_ops_in_each_stage", num_stages)
    if any(not isinstance(count, int) or count <= 0 for count in stage_ops):
        raise ValueError("num_ops_in_each_stage values must be positive integers")
    if sum(stage_ops) != manifest_size:
        raise ValueError(
            f"manifest size mismatch in pipeline {pipeline_index}: "
            f"expected {manifest_size}, got {sum(stage_ops)}"
        )

    canonical_fields = {}
    for stem, default in _TOPOLOGY_FIELDS:
        canonical_fields[f"{stem}_of_each_op"] = _expand_topology_field(
            pipe, stem, default, stage_ops
        )

    stage_tp = [values[0] for values in canonical_fields["model_parallel_size_of_each_op"]]
    stage_dp = [values[0] for values in canonical_fields["data_parallel_size_of_each_op"]]
    stage_cp = [values[0] for values in canonical_fields["context_parallel_size_of_each_op"]]
    for stage_index, cp in enumerate(stage_cp):
        if cp != 1:
            raise ValueError(
                f"context parallel stage {stage_index} must be 1 for block-wise sync"
            )
    for label, values in (("TP", stage_tp), ("DP", stage_dp), ("CP", stage_cp)):
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(f"{label} values must be positive integers")
    for stage_index, tp in enumerate(stage_tp):
        if tp not in _SUPPORTED_TP_VALUES:
            supported = ", ".join(str(value) for value in _SUPPORTED_TP_VALUES)
            raise ValueError(
                f"unsupported TP value {tp} in pipeline {pipeline_index} "
                f"stage {stage_index}; supported TP values are {supported}"
            )
    num_gpus = _validate_stage_list(pipe, "num_gpus", num_stages)
    for stage_index, (actual, tp, dp, cp) in enumerate(
        zip(num_gpus, stage_tp, stage_dp, stage_cp)
    ):
        expected = tp * dp * cp
        if actual != expected:
            raise ValueError(
                f"stage product mismatch in pipeline {pipeline_index} stage {stage_index}: "
                f"num_gpus={actual}, TP*DP*CP={expected}"
            )

    micro_batch_size = pipe.get("micro_batch_size")
    global_batch_size = pipe.get("global_batch_size")
    if not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be a positive integer")
    if not isinstance(global_batch_size, int) or global_batch_size <= 0:
        raise ValueError("global_batch_size must be a positive integer")
    if global_batch_size % micro_batch_size:
        raise ValueError("micro batch divisibility requires GBS % micro_batch_size == 0")
    for stage_index, dp in enumerate(stage_dp):
        if micro_batch_size % dp:
            raise ValueError(
                f"micro batch divisibility failed for pipeline {pipeline_index} "
                f"stage {stage_index}: micro_batch_size % DP != 0"
            )

    recompute_ops = _validate_stage_list(pipe, "recompute_ops", num_stages)
    for stage_index, (values, count) in enumerate(zip(recompute_ops, stage_ops)):
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(
                f"recompute_ops stage {stage_index} length must equal its op count"
            )

    normalized = {
        "pipeline_index": pipeline_index,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "num_stages": num_stages,
        "num_gpus": list(num_gpus),
        "num_ops_in_each_stage": list(stage_ops),
        "checkpoint_activations": list(
            _validate_stage_list(pipe, "checkpoint_activations", num_stages)
        ),
        "resharding_stages": list(
            _validate_stage_list(pipe, "resharding_stages", num_stages)
        ),
        "recompute_ops": recompute_ops,
    }
    normalized.update(canonical_fields)
    return normalized


def normalize_flexpipe_config(config, world_size=None):
    """Return the unique canonical representation of a FlexPipe GPT config."""
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    unknown = _unknown_fields(config, _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown top-level config field(s): {', '.join(unknown)}")
    pipelines = config.get("pipe")
    if not isinstance(pipelines, list) or not pipelines:
        raise ValueError("config must contain at least one pipeline")

    pipeline_op_counts = []
    for index, pipe in enumerate(pipelines):
        if not isinstance(pipe, dict):
            raise ValueError(f"pipeline {index} must be a mapping")
        unknown = _unknown_fields(pipe, _PIPE_FIELDS)
        if unknown:
            raise ValueError(
                f"pipeline {index} has unknown field(s): {', '.join(unknown)}"
            )
        counts = pipe.get("num_ops_in_each_stage")
        if not isinstance(counts, list) or not counts:
            raise ValueError(f"pipeline {index} is missing num_ops_in_each_stage")
        pipeline_op_counts.append(sum(counts))

    model_impl = _select_model_impl(config, pipeline_op_counts)
    manifest = _manifest(model_impl, config["num_layers"])
    normalized_pipes = [
        _normalize_pipe(pipe, index, len(manifest["ordered_ops"]))
        for index, pipe in enumerate(pipelines)
    ]
    hidden_size = config.get("hidden_size")
    attention_heads = config.get("num_attention_heads")
    for pipeline_index, pipe in enumerate(normalized_pipes):
        stage_tp = [
            values[0] for values in pipe["model_parallel_size_of_each_op"]
        ]
        for stage_index, tp in enumerate(stage_tp):
            if (
                not isinstance(hidden_size, int)
                or hidden_size % tp
                or not isinstance(attention_heads, int)
                or attention_heads % tp
            ):
                raise ValueError(
                    f"TP constraint failed in pipeline {pipeline_index} stage {stage_index}: "
                    f"hidden_size={hidden_size}, num_attention_heads={attention_heads}, TP={tp}"
                )
    derived_world_size = sum(sum(pipe["num_gpus"]) for pipe in normalized_pipes)
    if world_size is not None and world_size != derived_world_size:
        raise ValueError(
            f"world size mismatch: expected {derived_world_size}, got {world_size}"
        )

    offsets = []
    running_offset = 0
    for pipe in normalized_pipes:
        offsets.append(running_offset)
        running_offset += pipe["global_batch_size"]

    normalized = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in {"pipe", "model_impl"}
    }
    normalized.update(
        {
            "model_impl": model_impl,
            "manifest": manifest,
            "pipe": normalized_pipes,
            "world_size": derived_world_size,
            "total_gbs": running_offset,
            "pipeline_sample_offsets": offsets,
        }
    )
    topology_payload = {
        "world_size": derived_world_size,
        "pipelines": [
            {
                key: pipe[key]
                for key in (
                    "num_gpus",
                    "num_ops_in_each_stage",
                    "model_parallel_size_of_each_op",
                    "data_parallel_size_of_each_op",
                    "context_parallel_size_of_each_op",
                )
            }
            for pipe in normalized_pipes
        ],
    }
    normalized["topology_fingerprint"] = _canonical_hash(topology_payload)
    normalized["config_hash"] = _canonical_hash(normalized)
    return normalized


def _parse_mbs_sequence(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("mbs_seq must be a literal list") from exc
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("mbs_seq must be a list")
    return value


def apply_normalized_config_to_args(args, normalized):
    """Populate legacy runtime argument fields from one canonical config."""
    if args.world_size != normalized["world_size"]:
        raise ValueError(
            f"world size mismatch: expected {normalized['world_size']}, got {args.world_size}"
        )
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank {args.rank} is outside world size {args.world_size}")

    for key, value in normalized.items():
        if key != "pipe":
            setattr(args, key, copy.deepcopy(value))
    args.manifest_hash = normalized["manifest"]["hash"]
    args.num_ops_each_layer = normalized["manifest"]["ops_per_layer"]

    pipes = normalized["pipe"]
    args.pipe_num = len(pipes)
    mappings = {
        "num_gpus_each_pipe": "num_gpus",
        "num_ops_in_each_stage_each_pipe": "num_ops_in_each_stage",
        "micro_batch_size_each_pipe": "micro_batch_size",
        "num_stages_each_pipe": "num_stages",
        "algo_of_each_op_each_pipe": "algo_of_each_op",
        "model_parallel_size_of_each_op_each_pipe": "model_parallel_size_of_each_op",
        "data_parallel_size_of_each_op_each_pipe": "data_parallel_size_of_each_op",
        "context_parallel_size_of_each_op_each_pipe": "context_parallel_size_of_each_op",
        "recompute_ops_each_pipe": "recompute_ops",
        "resharding_stages_each_pipe": "resharding_stages",
        "checkpoint_activations_each_pipe": "checkpoint_activations",
        "global_batch_size_each_pipe": "global_batch_size",
    }
    for destination, source in mappings.items():
        setattr(args, destination, [copy.deepcopy(pipe[source]) for pipe in pipes])
    args.sum_gpus_each_pipe = [sum(pipe["num_gpus"]) for pipe in pipes]

    args.rank_to_pipe = []
    args.rank_to_stage = []
    args.rank_to_offset = []
    args.offset_list_each_pipe = []
    pipeline_offset = 0
    for pipeline_index, pipe in enumerate(pipes):
        args.offset_list_each_pipe.append(pipeline_offset)
        for stage_index, stage_size in enumerate(pipe["num_gpus"]):
            for _ in range(stage_size):
                args.rank_to_pipe.append(pipeline_index)
                args.rank_to_stage.append(stage_index)
                args.rank_to_offset.append(pipeline_offset)
        pipeline_offset += sum(pipe["num_gpus"])
    args.offset_list_each_pipe.append(pipeline_offset)

    args.current_pipe = args.rank_to_pipe[args.rank]
    args.offset = args.rank_to_offset[args.rank]
    current = pipes[args.current_pipe]
    local_mappings = {
        "global_batch_size": "global_batch_size",
        "micro_batch_size": "micro_batch_size",
        "num_gpus": "num_gpus",
        "num_ops_in_each_stage": "num_ops_in_each_stage",
        "num_stages": "num_stages",
        "algo_of_each_op": "algo_of_each_op",
        "model_parallel_size_of_each_op": "model_parallel_size_of_each_op",
        "data_parallel_size_of_each_op": "data_parallel_size_of_each_op",
        "context_parallel_size_of_each_op": "context_parallel_size_of_each_op",
        "recompute_ops": "recompute_ops",
        "resharding_stages": "resharding_stages",
        "checkpoint_activations": "checkpoint_activations",
    }
    for destination, source in local_mappings.items():
        setattr(args, destination, copy.deepcopy(current[source]))
    args.num_op = sum(args.num_ops_in_each_stage)

    configured_mbs_seq = _parse_mbs_sequence(getattr(args, "mbs_seq", []))
    if not configured_mbs_seq:
        configured_mbs_seq = [
            [[pipe["micro_batch_size"], normalized["seq_length"]]]
            * (pipe["global_batch_size"] // pipe["micro_batch_size"])
            for pipe in pipes
        ]
    if len(configured_mbs_seq) != len(pipes):
        raise ValueError("mbs_seq must contain one sequence per pipeline")
    for pipeline_index, (sequence, pipe) in enumerate(zip(configured_mbs_seq, pipes)):
        try:
            scheduled_gbs = sum(item[0] for item in sequence)
        except (IndexError, TypeError) as exc:
            raise ValueError(f"mbs_seq pipeline {pipeline_index} is malformed") from exc
        if scheduled_gbs != pipe["global_batch_size"]:
            raise ValueError(
                f"mbs_seq pipeline {pipeline_index} sums to {scheduled_gbs}, "
                f"expected {pipe['global_batch_size']}"
            )
    args.mbs_seq = configured_mbs_seq
    args.nominal_global_loss_scale = float(
        sum(microbatch_size * sequence_length
            for sequence in configured_mbs_seq
            for microbatch_size, sequence_length in sequence)
    )
    if args.nominal_global_loss_scale <= 0:
        raise ValueError("nominal_global_loss_scale must be positive")
    return args


def _fixture_pipe(stage_ops, tp, dp, gbs=24, mbs=8):
    return {
        "global_batch_size": gbs,
        "micro_batch_size": mbs,
        "num_stages": len(stage_ops),
        "num_gpus": [t * d for t, d in zip(tp, dp)],
        "checkpoint_activations": [False] * len(stage_ops),
        "resharding_stages": [False] * len(stage_ops),
        "num_ops_in_each_stage": list(stage_ops),
        "model_parallel_size_of_each_stage": list(tp),
        "data_parallel_size_of_each_stage": list(dp),
        "context_parallel_size_of_each_stage": [1] * len(stage_ops),
        "algo_of_each_stage": [0] * len(stage_ops),
        "recompute_ops": [[0] * count for count in stage_ops],
    }


def _fixture_config(pipes):
    return {
        "model_name": "gpt",
        "model_impl": "gpt",
        "model_size": "fixture",
        "num_layers": 4,
        "seq_length": 32,
        "max_position_embeddings": 32,
        "num_attention_heads": 8,
        "hidden_size": 64,
        "pipe": pipes,
    }


def generate_heterogeneous_config_fixtures():
    """Generate the tracked minimum topology sweep without local JSON inputs."""
    return [
        _fixture_config(
            [
                _fixture_pipe([43], [1], [1]),
                _fixture_pipe([11, 32], [1, 1], [1, 2]),
            ]
        ),
        _fixture_config(
            [
                _fixture_pipe([21, 22], [1, 2], [1, 1]),
                _fixture_pipe([11, 32], [1, 1], [2, 1]),
            ]
        ),
        _fixture_config(
            [
                _fixture_pipe([11, 32], [2, 1], [1, 1]),
                _fixture_pipe([21, 22], [1, 1], [1, 2]),
            ]
        ),
        _fixture_config(
            [
                _fixture_pipe([43], [1], [2], gbs=8),
                _fixture_pipe([43], [2], [1], gbs=16),
                _fixture_pipe([43], [1], [2], gbs=24),
            ]
        ),
        _fixture_config(
            [
                _fixture_pipe([11, 20, 12], [1, 2, 1], [1, 1, 1]),
                _fixture_pipe([21, 10, 12], [1, 1, 1], [2, 1, 1]),
            ]
        ),
        _fixture_config(
            [
                _fixture_pipe([10, 33], [1, 2], [2, 1]),
                _fixture_pipe([30, 13], [2, 1], [1, 2]),
            ]
        ),
    ]


def generate_gpt_e2e_fixtures():
    """Return the named tracked GPT configurations used by launchers."""
    return {
        "canonical-4r-equal-gbs": _fixture_config(
            [
                _fixture_pipe([43], [1], [2], gbs=24, mbs=8),
                _fixture_pipe([43], [2], [1], gbs=24, mbs=8),
            ]
        ),
        "reference-1r-canonical4-global-batch": _fixture_config(
            [_fixture_pipe([43], [1], [1], gbs=48, mbs=8)]
        ),
        "brief-6r-equal-gbs": _fixture_config(
            [
                _fixture_pipe([21, 22], [1, 2], [1, 1], gbs=24, mbs=8),
                _fixture_pipe([11, 32], [1, 1], [2, 1], gbs=24, mbs=8),
            ]
        ),
        "hetero-6r-swapped-stages": _fixture_config(
            [
                _fixture_pipe([11, 32], [2, 1], [1, 1], gbs=24, mbs=8),
                _fixture_pipe([21, 22], [1, 1], [1, 2], gbs=24, mbs=8),
            ]
        ),
        "canonical-6r-three-pipeline": _fixture_config(
            [
                _fixture_pipe([43], [1], [2], gbs=12, mbs=4),
                _fixture_pipe([43], [2], [1], gbs=16, mbs=4),
                _fixture_pipe([43], [1], [2], gbs=20, mbs=4),
            ]
        ),
    }
