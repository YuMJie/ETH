"""Convert a legacy GPT search result to the canonical GPT HPP contract."""

import argparse
import json
import sys
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME_DIR))

from megatron.config_normalization import normalize_flexpipe_config


TOPOLOGY_FIELDS = (
    "model_parallel_size_of_each_op",
    "data_parallel_size_of_each_op",
    "algo_of_each_op",
    "recompute_ops",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def flatten_stages(config, field):
    values = config.get(field)
    stage_counts = config["num_ops_in_each_stage"]
    if not isinstance(values, list) or len(values) != len(stage_counts):
        raise ValueError(f"{field} stage list is invalid")
    flattened = []
    for stage_index, (stage_values, count) in enumerate(zip(values, stage_counts)):
        if not isinstance(stage_values, list) or len(stage_values) != count:
            raise ValueError(f"{field} stage {stage_index} length is invalid")
        flattened.extend((stage_index, value) for value in stage_values)
    return flattened


def v1_source_indices(num_layers):
    indices = [0]
    for layer_index in range(num_layers):
        base = 1 + layer_index * 13
        indices.extend(
            (
                base,
                base + 1,
                base + 5,
                base + 6,
                base + 7,
                base + 8,
                base + 9,
                base + 10,
                base + 11,
                base + 12,
            )
        )
    tail = 1 + num_layers * 13
    indices.extend((tail, tail + 1))
    return indices


def convert(config):
    num_layers = config.get("num_layers")
    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("search config num_layers must be positive")
    stage_counts = config.get("num_ops_in_each_stage")
    if not isinstance(stage_counts, list):
        raise ValueError("search config stage counts are invalid")
    op_count = sum(stage_counts)
    public_op_count = 3 + 10 * num_layers
    legacy_op_count = 3 + 13 * num_layers
    if op_count == public_op_count:
        selected = range(op_count)
    elif op_count == legacy_op_count:
        selected = v1_source_indices(num_layers)
    else:
        raise ValueError("search config does not match a supported GPT manifest")
    converted_fields = {}
    selected_stages = None
    for field in TOPOLOGY_FIELDS:
        flattened = flatten_stages(config, field)
        chosen = [flattened[index] for index in selected]
        stages = [stage for stage, _ in chosen]
        if selected_stages is None:
            selected_stages = stages
        elif stages != selected_stages:
            raise ValueError("search topology fields disagree on stage assignment")
        converted_fields[field] = [value for _, value in chosen]

    retained_stages = sorted(set(selected_stages))
    stage_mapping = {old: new for new, old in enumerate(retained_stages)}
    grouped = {field: [[] for _ in retained_stages] for field in TOPOLOGY_FIELDS}
    for item_index, old_stage in enumerate(selected_stages):
        new_stage = stage_mapping[old_stage]
        for field in TOPOLOGY_FIELDS:
            grouped[field][new_stage].append(converted_fields[field][item_index])

    new_counts = [0] * len(retained_stages)
    for old_stage in selected_stages:
        new_counts[stage_mapping[old_stage]] += 1

    pipe = {
        "global_batch_size": config["global_batch_size"],
        "micro_batch_size": config["micro_batch_size"],
        "num_stages": len(retained_stages),
        "num_gpus": [config["num_gpus"][index] for index in retained_stages],
        "num_ops_in_each_stage": new_counts,
        "checkpoint_activations": [
            config["checkpoint_activations"][index] for index in retained_stages
        ],
        "resharding_stages": [
            config["resharding_stages"][index] for index in retained_stages
        ],
    }
    pipe.update(grouped)

    canonical = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "global_batch_size",
            "micro_batch_size",
            "num_stages",
            "num_gpus",
            "num_ops_in_each_stage",
            "checkpoint_activations",
            "resharding_stages",
            *TOPOLOGY_FIELDS,
        }
    }
    canonical["model_impl"] = "gpt"
    canonical["pipe"] = [pipe]
    return normalize_flexpipe_config(canonical)


def main():
    args = parse_args()
    config = json.loads(args.input.read_text(encoding="utf-8"))
    normalized = convert(config)
    args.output.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
