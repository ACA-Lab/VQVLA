#!/usr/bin/env python3
"""Collect bounded input-Hessian-diagonal proxies for every OpenVLA parameter.

For each direct parameter owner, forward-pre hooks accumulate only the input
feature-wise second moment.  It is an input-dependent diagonal proxy suitable
for weighted GPTVQ, while avoiding dense Hessians and unbounded activation
storage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

# Prefer this clean work tree over any editable OpenVLA installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def _first_tensor(values: tuple[Any, ...]) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            nested = _first_tensor(tuple(value))
            if nested is not None:
                return nested
    return None


def _make_hooks(model: torch.nn.Module) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not any(parameter.is_floating_point() for parameter in module.parameters(recurse=False)):
            continue
        if isinstance(module, torch.nn.Embedding):
            summaries[name] = {
                "kind": "embedding_rows",
                "sum": torch.zeros(module.num_embeddings, dtype=torch.float32),
                "count": 0,
            }

            def embedding_hook(_module: torch.nn.Module, values: tuple[Any, ...], owner: str = name) -> None:
                token_ids = _first_tensor(values)
                if token_ids is None or token_ids.numel() == 0:
                    return
                token_ids = token_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
                token_ids = token_ids[(token_ids >= 0) & (token_ids < summaries[owner]["sum"].numel())]
                if token_ids.numel():
                    summaries[owner]["sum"].index_add_(0, token_ids, torch.ones(token_ids.numel()))
                    summaries[owner]["count"] += int(token_ids.numel())

            handles.append(module.register_forward_pre_hook(embedding_hook))
            continue

        summaries[name] = {"kind": "features", "sum": None, "count": 0}

        def feature_hook(current: torch.nn.Module, values: tuple[Any, ...], owner: str = name) -> None:
            tensor = _first_tensor(values)
            if tensor is None or not tensor.is_floating_point() or tensor.numel() == 0 or tensor.ndim == 0:
                return
            feature_axis = 1 if isinstance(current, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)) else -1
            tensor = tensor.detach().float().movedim(feature_axis, -1).reshape(-1, tensor.shape[feature_axis])
            feature_sum = tensor.square().sum(dim=0).to(device="cpu")
            if summaries[owner]["sum"] is None:
                summaries[owner]["sum"] = feature_sum
            elif summaries[owner]["sum"].shape == feature_sum.shape:
                summaries[owner]["sum"].add_(feature_sum)
            else:
                return
            summaries[owner]["count"] += int(tensor.shape[0])

        handles.append(module.register_forward_pre_hook(feature_hook))
    return summaries, handles


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for OpenVLA calibration")
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    payload = np.load(args.observations, allow_pickle=True)
    samples = [sample for sample in payload["samples"].tolist() if sample["suite"] == args.suite]
    if not samples:
        raise ValueError(f"No {args.suite} samples in {args.observations}")

    cfg = SimpleNamespace(pretrained_checkpoint=args.checkpoint, load_in_8bit=False, load_in_4bit=False)
    model = get_vla(cfg).eval()
    processor = get_processor(cfg)
    summaries, handles = _make_hooks(model)
    owners = {name: module_name for name, _ in model.named_parameters() for module_name, _, _ in [name.rpartition(".")]}
    try:
        with torch.inference_mode():
            for index, sample in enumerate(samples, start=1):
                get_vla_action(
                    model,
                    processor,
                    args.checkpoint,
                    {"full_image": sample["image"]},
                    sample["task_description"],
                    args.suite,
                    center_crop=True,
                )
                print(f"Calibrated observation {index}/{len(samples)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    parameters: dict[str, dict[str, object]] = {}
    covered = fallback = 0
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        owner = owners[name]
        summary = summaries.get(owner)
        observed = bool(summary and summary["count"] and summary["sum"] is not None)
        parameters[name] = {"owner": owner, "shape": tuple(parameter.shape), "observed": observed}
        covered += parameter.numel()
        fallback += 0 if observed else parameter.numel()
    output = {
        "version": 1,
        "method": "input_second_moment_diagonal",
        "checkpoint": args.checkpoint,
        "suite": args.suite,
        "observation_count": len(samples),
        "module_summaries": summaries,
        "parameters": parameters,
        "coverage": {
            "floating_parameter_tensors": len(parameters),
            "floating_parameter_values": covered,
            "uniform_fallback_values": fallback,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved hdiag for {len(parameters)} tensors ({covered} values) to {output_path}", flush=True)


if __name__ == "__main__":
    main()
