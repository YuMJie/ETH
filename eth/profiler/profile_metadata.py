"""Write machine-readable metadata for a GPT profiling directory."""

import argparse
import json
import platform
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=("planned", "complete"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--tp-sizes", required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    return parser.parse_args()


def environment_metadata():
    try:
        import torch
    except ImportError:
        return {
            "python": platform.python_version(),
            "torch": None,
            "cuda": None,
        }

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def main():
    args = parse_args()
    metadata = {
        "schema_version": 1,
        "profile_status": args.status,
        "model": {"name": args.model, "size": args.model_size},
        "hardware": args.hardware,
        "tp_sizes": [int(value) for value in args.tp_sizes.split(",")],
        "distributed": {
            "num_nodes": args.num_nodes,
            "gpus_per_node": args.gpus_per_node,
            "node_rank": args.node_rank,
        },
        "environment": environment_metadata(),
    }
    destination = args.output / "profile_metadata.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
