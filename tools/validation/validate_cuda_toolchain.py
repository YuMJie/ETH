#!/usr/bin/env python3
"""Validate PyTorch CUDA and the selected nvcc toolkit before launch."""

import argparse
import re
import subprocess
import sys


RELEASE_CUDA_VERSION = (12, 4)


def _version_pair(value, label):
    match = re.search(r"(\d+)\.(\d+)", value or "")
    if match is None:
        raise ValueError(f"unable to parse {label} version from {value!r}")
    return int(match.group(1)), int(match.group(2))


def validate_versions(torch_cuda_version, nvcc_output):
    torch_major, torch_minor = _version_pair(
        torch_cuda_version, "PyTorch CUDA"
    )
    release = re.search(r"release\s+(\d+)\.(\d+)", nvcc_output or "")
    if release is None:
        raise ValueError(f"unable to parse nvcc version from {nvcc_output!r}")
    nvcc_major, nvcc_minor = int(release.group(1)), int(release.group(2))
    if torch_major != nvcc_major:
        raise ValueError(
            "CUDA major version mismatch: "
            f"PyTorch CUDA {torch_major}.{torch_minor}, "
            f"nvcc {nvcc_major}.{nvcc_minor}"
        )
    if (torch_major, torch_minor) != RELEASE_CUDA_VERSION or \
            (nvcc_major, nvcc_minor) != RELEASE_CUDA_VERSION:
        raise ValueError(
            "release environment requires CUDA 12.4 for both PyTorch and nvcc; "
            f"got PyTorch CUDA {torch_major}.{torch_minor}, "
            f"nvcc {nvcc_major}.{nvcc_minor}"
        )
    return torch_major, torch_minor, nvcc_major, nvcc_minor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import torch

        if not torch.version.cuda:
            raise ValueError("PyTorch does not report a CUDA build version")
        nvcc_output = subprocess.check_output(
            [args.nvcc, "-V"], text=True, stderr=subprocess.STDOUT
        )
        versions = validate_versions(torch.version.cuda, nvcc_output)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"CUDA toolchain validation failed for nvcc {args.nvcc}: {error}", file=sys.stderr)
        return 1
    print(
        "CUDA toolchain validated: "
        f"PyTorch CUDA {versions[0]}.{versions[1]}, "
        f"nvcc {versions[2]}.{versions[3]} at {args.nvcc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
