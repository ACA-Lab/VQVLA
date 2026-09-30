#!/usr/bin/env python3
"""Verify GPTVQ archive metadata, complete coverage, and one reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vqvla.gptvq_archive import reconstruct_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-bits", required=True, choices=(3, 4), type=int)
    parser.add_argument("--expected-tensors", required=True, type=int)
    parser.add_argument("--expected-values", required=True, type=int)
    args = parser.parse_args()
    archive = torch.load(args.archive, map_location="cpu", weights_only=False)
    if archive.get("format") != "vqvla_gptvq_hdiag":
        raise ValueError(f"Unexpected archive format: {archive.get('format')!r}")
    config = archive["global_config"]
    coverage = archive["coverage"]
    if config["bits"] != args.expected_bits:
        raise ValueError(f"Expected {args.expected_bits} bits, got {config['bits']}")
    if len(archive["tensors"]) != args.expected_tensors or coverage["floating_tensor_count"] != args.expected_tensors:
        raise ValueError("Tensor coverage is incomplete")
    if coverage["floating_value_count"] != args.expected_values:
        raise ValueError("Value coverage is incomplete")
    name, record = next(iter(archive["tensors"].items()))
    restored = reconstruct_tensor(record)
    if restored.numel() != record["numel"]:
        raise ValueError(f"Reconstruction length mismatch for {name}")
    print(
        {
            "archive": str(args.archive),
            "config": config,
            "tensor_count": coverage["floating_tensor_count"],
            "value_count": coverage["floating_value_count"],
            "reconstruction_smoke_tensor": name,
            "reconstruction_smoke_shape": tuple(restored.shape),
        }
    )


if __name__ == "__main__":
    main()
