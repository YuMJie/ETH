"""Validate a canonical GPT profile before heterogeneous search."""

import argparse
import csv
import json
from pathlib import Path


MICRO_BATCH_SIZES = (1, 2, 4, 8)
ALGORITHMS = (0, 1)
PRIMITIVES = ("all_gather", "all_reduce", "reduce_scatter", "all_to_all")


def fail(message):
    raise SystemExit(f"profile validation failed: {message}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--node-description", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path, label):
    if not path.is_file():
        fail(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid {label}: {error}")


def required_tp_sizes(model_size, node_info):
    max_devices = max(entry["GPU_NUM"] for entry in node_info.values())
    max_tp = min(max_devices, 4 if model_size == "350M" else 8)
    sizes = []
    value = 1
    while value <= max_tp:
        sizes.append(value)
        value *= 2
    return sizes


def validate_node_description(description):
    if not isinstance(description, dict):
        fail("node description must be a JSON object")
    node_info = description.get("node_info")
    device_info = description.get("device_info")
    if not isinstance(node_info, dict) or not node_info:
        fail("node description requires non-empty node_info")
    if not isinstance(device_info, dict) or not device_info:
        fail("node description requires non-empty device_info")

    for key, entry in node_info.items():
        if not isinstance(entry, dict):
            fail(f"node_info[{key}] must be an object")
        for field in ("GPU", "GPU_NUM", "NODE_NUM"):
            if field not in entry:
                fail(f"node_info[{key}] is missing {field}")
        if not isinstance(entry["GPU"], str) or not entry["GPU"]:
            fail(f"node_info[{key}].GPU must be a non-empty string")
        if not isinstance(entry["GPU_NUM"], int) or entry["GPU_NUM"] < 1:
            fail(f"node_info[{key}].GPU_NUM must be positive")
        if not isinstance(entry["NODE_NUM"], int) or entry["NODE_NUM"] < 1:
            fail(f"node_info[{key}].NODE_NUM must be positive")
        if entry["GPU"] not in device_info:
            fail(f"device_info is missing hardware {entry['GPU']}")

    for hardware, entry in device_info.items():
        if not isinstance(entry, dict):
            fail(f"device_info[{hardware}] must be an object")
        for field in ("memory", "intra_bandwidth", "TFLOPS"):
            value = entry.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                fail(f"device_info[{hardware}].{field} must be positive")
    return node_info, device_info


def validate_csv(path, expected_header):
    if not path.is_file():
        fail(f"required profile record is missing: {path.name}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            row = next(reader)
    except (OSError, StopIteration, csv.Error) as error:
        fail(f"invalid profile record {path.name}: {error}")
    if header != expected_header or not row:
        fail(f"invalid profile record schema: {path.name}")


def validate_p2p(path):
    if not path.is_file():
        fail(f"required profile record is missing: {path.name}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError("expected size and bandwidth rows")
        width = len(rows[0])
        if any(len(row) != width for row in rows[1:]):
            raise ValueError("rows have inconsistent widths")
        for row in rows:
            for value in row:
                float(value)
    except (OSError, ValueError, csv.Error) as error:
        fail(f"invalid P2P profile record {path.name}: {error}")


def validate_records(profile_dir, model_size, tp_sizes, require_inter_node):
    op_header = [
        "op_name", "forward-compute", "backward-compute", "input_size",
        "output_size", "weights", "activations", "fwd_reserved", "bwd_reserved",
    ]
    for mbs in MICRO_BATCH_SIZES:
        for tp in tp_sizes:
            for algorithm in ALGORITHMS:
                validate_csv(
                    profile_dir / f"gpt_{model_size}_mbs{mbs}_tp{tp}_algo{algorithm}.csv",
                    op_header,
                )
    for tp in tp_sizes:
        if tp == 1:
            continue
        for primitive in PRIMITIVES:
            validate_csv(
                profile_dir / f"prim_gpt_{model_size}_{primitive}_{tp}gpus.csv",
                ["data_size(MB)", "time(ms)"],
            )
    validate_p2p(profile_dir / "p2p_intra_node.csv")
    if require_inter_node:
        validate_p2p(profile_dir / "p2p_inter_node.csv")


def main():
    args = parse_args()
    metadata = load_json(args.profile_dir / "profile_metadata.json", "profile metadata")
    description = load_json(args.node_description, "node description")
    node_info, device_info = validate_node_description(description)

    if metadata.get("schema_version") != 1:
        fail("profile schema_version must be 1")
    if metadata.get("profile_status") != "complete":
        fail("profile_status must be complete; planned dry-run data cannot be searched")
    model = metadata.get("model", {})
    if model.get("name") != "gpt" or model.get("size") != args.model_size:
        fail(f"profile model size does not match requested {args.model_size}")

    hardware_labels = {entry["GPU"] for entry in node_info.values()}
    profile_hardware = metadata.get("hardware")
    if hardware_labels != {profile_hardware}:
        fail(
            "profile hardware does not match node description; "
            "one canonical profile currently supports one hardware label"
        )

    tp_sizes = required_tp_sizes(args.model_size, node_info)
    covered = metadata.get("tp_sizes")
    if not isinstance(covered, list) or any(tp not in covered for tp in tp_sizes):
        fail(f"profile TP coverage must include {tp_sizes}")

    node_count = sum(entry["NODE_NUM"] for entry in node_info.values())
    validate_records(args.profile_dir, args.model_size, tp_sizes, node_count > 1)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "node_info.json").write_text(
        json.dumps(node_info, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "device_info.json").write_text(
        json.dumps(device_info, indent=2) + "\n", encoding="utf-8"
    )
    contract = {
        "hardware": profile_hardware,
        "max_tp": max(tp_sizes),
        "node_count": node_count,
        "total_gpus": sum(
            entry["GPU_NUM"] * entry["NODE_NUM"] for entry in node_info.values()
        ),
        "node_info": node_info,
    }
    (args.output / "search_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    print(profile_hardware)
    print(max(tp_sizes))
    print(contract["total_gpus"])
    print(json.dumps(node_info, separators=(",", ":")))


if __name__ == "__main__":
    main()
