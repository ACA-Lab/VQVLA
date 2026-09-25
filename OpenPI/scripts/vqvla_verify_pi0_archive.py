#!/usr/bin/env python3
"""Verify that a complete Pi0 GPTVQ archive loads without source weights."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.quantization.gptvq_archive import load_gptvq_archive
from openpi.training import config as training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = training_config.get_config(args.config)
    model = PI0Pytorch(dataclasses.replace(config.model, pytorch_compile_mode=None)).to(args.device)
    coverage = load_gptvq_archive(model, args.archive)
    parameter_values = sum(parameter.numel() for parameter in model.parameters() if parameter.is_floating_point())
    if parameter_values != coverage["floating_value_count"]:
        raise RuntimeError(f"Archive has {coverage['floating_value_count']} values, model has {parameter_values}")
    print(f"verified {len(list(model.named_parameters()))} tensors / {parameter_values} values", flush=True)


if __name__ == "__main__":
    main()
