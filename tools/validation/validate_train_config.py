#!/usr/bin/env python3
"""Validate a GPT HPP config before CUDA or distributed setup."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eth" / "runtime"))

from megatron.config_normalization import normalize_flexpipe_config
from megatron.grad_sync.topology import build_canonical_topology


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--grad-sync-mode", choices=("tensor", "block"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        normalized = normalize_flexpipe_config(config, world_size=args.world_size)
        if normalized["model_name"] != "gpt":
            raise ValueError("only GPT configs are supported")
        if normalized["model_impl"] != "gpt":
            raise ValueError("public training requires model_impl gpt")
        topology = build_canonical_topology(normalized)
        if topology.world_size != args.world_size:
            raise ValueError("canonical topology world size mismatch")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"config validation failed: {error}", file=sys.stderr)
        return 1

    print(
        "config validated: "
        f"model_impl={normalized['model_impl']} "
        f"world_size={topology.world_size} "
        f"grad_sync_mode={args.grad_sync_mode} "
        f"topology={topology.fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
