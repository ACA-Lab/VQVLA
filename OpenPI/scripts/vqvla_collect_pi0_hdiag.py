#!/usr/bin/env python3
"""Collect bounded input-Hessian-diagonal statistics for every Pi0 parameter owner.

The script receives raw observations collected by
``vqvla_collect_libero_calibration.py`` and runs normal policy inference.  It
stores only per-feature second moments (and embedding-token counts), never an
``X.T @ X`` matrix.  The resulting statistics are consumed lazily, block by
block, by the GPTVQ exporter.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.policies.policy_config import create_trained_policy
from openpi.training import config as training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--suite", default=None, help="Optional LIBERO suite name to select from observations")
    return parser.parse_args()


def _first_tensor(values: tuple[Any, ...]) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, list | tuple):
            nested = _first_tensor(tuple(value))
            if nested is not None:
                return nested
    return None


def _owner_map(model: torch.nn.Module) -> dict[str, str]:
    owners: dict[str, str] = {}
    for name, _ in model.named_parameters():
        module_name, separator, _ = name.rpartition(".")
        owners[name] = module_name if separator else ""
    return owners


def _make_hooks(model: torch.nn.Module) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Return bounded per-owner accumulators and their forward-pre hooks."""
    direct_parameter_modules = {
        name: module
        for name, module in model.named_modules()
        if any(parameter.is_floating_point() for parameter in module.parameters(recurse=False))
    }
    summaries: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []

    for module_name, module in direct_parameter_modules.items():
        if isinstance(module, torch.nn.Embedding):
            summaries[module_name] = {
                "kind": "embedding_rows",
                "sum": torch.zeros(module.num_embeddings, dtype=torch.float32),
                "count": 0,
            }

            def embedding_hook(_module: torch.nn.Module, values: tuple[Any, ...], name: str = module_name) -> None:
                token_ids = _first_tensor(values)
                if token_ids is None or token_ids.numel() == 0:
                    return
                token_ids = token_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
                token_ids = token_ids[(token_ids >= 0) & (token_ids < summaries[name]["sum"].numel())]
                if token_ids.numel():
                    summaries[name]["sum"].index_add_(
                        0,
                        token_ids,
                        torch.ones(token_ids.numel(), dtype=torch.float32),
                    )
                    summaries[name]["count"] += int(token_ids.numel())

            handles.append(module.register_forward_pre_hook(embedding_hook))
            continue

        summaries[module_name] = {"kind": "features", "sum": None, "count": 0, "axis": None}

        def feature_hook(current_module: torch.nn.Module, values: tuple[Any, ...], name: str = module_name) -> None:
            tensor = _first_tensor(values)
            if tensor is None or not tensor.is_floating_point() or tensor.numel() == 0:
                return
            tensor = tensor.detach().float()
            if isinstance(current_module, torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Conv3d):
                feature_axis = 1
            else:
                feature_axis = tensor.ndim - 1
            if tensor.ndim == 0:
                return
            tensor = tensor.movedim(feature_axis, -1).reshape(-1, tensor.shape[feature_axis])
            feature_sum = tensor.square().sum(dim=0).to(device="cpu")
            if summaries[name]["sum"] is None:
                summaries[name]["sum"] = feature_sum
                summaries[name]["axis"] = feature_axis
            elif summaries[name]["sum"].shape == feature_sum.shape:
                summaries[name]["sum"].add_(feature_sum)
            else:
                return
            summaries[name]["count"] += int(tensor.shape[0])

        handles.append(module.register_forward_pre_hook(feature_hook))

    return summaries, handles


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if not args.checkpoint.is_dir():
        raise NotADirectoryError(args.checkpoint)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    payload = np.load(args.observations, allow_pickle=True)
    observations = list(payload["samples"])
    if args.suite is not None:
        observations = [sample for sample in observations if sample["suite"] == args.suite]
    if not observations:
        raise ValueError("No calibration observations matched the requested suite")

    config = training_config.get_config(args.config)
    model_config = dataclasses.replace(config.model, pytorch_compile_mode=None)
    config = dataclasses.replace(config, model=model_config)
    policy = create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    model = policy._model  # noqa: SLF001
    model.eval()
    summaries, handles = _make_hooks(model)
    owners = _owner_map(model)
    try:
        with torch.inference_mode():
            for index, observation in enumerate(observations, start=1):
                policy.infer(
                    {
                        "observation/image": observation["image"],
                        "observation/wrist_image": observation["wrist_image"],
                        "observation/state": observation["state"],
                        "prompt": observation["prompt"],
                    }
                )
                print(f"Calibrated observation {index}/{len(observations)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    parameters = {}
    covered = 0
    fallback = 0
    for parameter_name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        module_name = owners[parameter_name]
        summary = summaries.get(module_name)
        observed = bool(summary and summary["count"] > 0 and summary["sum"] is not None)
        parameters[parameter_name] = {
            "owner": module_name,
            "shape": tuple(parameter.shape),
            "observed": observed,
        }
        covered += parameter.numel()
        fallback += 0 if observed else parameter.numel()

    output = {
        "version": 1,
        "method": "input_second_moment_diagonal",
        "checkpoint": str(args.checkpoint),
        "suite": args.suite,
        "observation_count": len(observations),
        "module_summaries": summaries,
        "parameters": parameters,
        "coverage": {
            "floating_parameter_tensors": len(parameters),
            "floating_parameter_values": covered,
            "uniform_fallback_values": fallback,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"Saved hdiag summaries for {len(parameters)} floating tensors "
        f"({covered} values; {fallback} uniform-fallback values) to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
