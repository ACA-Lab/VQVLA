#!/usr/bin/env python3
"""Calibrate and export Hessian-diagonal-aware 4-bit VQ weights for LIBERO."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from safetensors import safe_open
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.vec_quant_hessian import (
    blockwise_hessian_diag_vq_quantize_weight_with_info_streaming,
)
from gr00t.model.gr00t_n1d7.vec_quant_save import (
    compute_quant_loss,
    should_quantize_linear_by_name,
)
from gr00t.policy.gr00t_policy import Gr00tPolicy


CHECKPOINTS = {
    "10": ("libero_10", "gr00t_10_gptvq_hdiag_4bit_b256_k256.pt"),
    "goal": ("libero_goal", "gr00t_goal_gptvq_hdiag_4bit_b256_k256.pt"),
    "object": ("libero_object", "gr00t_object_gptvq_hdiag_4bit_b256_k256.pt"),
    "spatial": ("libero_spatial", "gr00t_spatial_gptvq_hdiag_4bit_b256_k256.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=tuple(CHECKPOINTS), nargs="+", default=tuple(CHECKPOINTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--block-batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--dataset-path", type=Path, default=Path("demo_data/libero_demo"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/GR00T-N1.7-LIBERO")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("quant_weight"))
    return parser.parse_args()


def _parse_observation(data_point: Any, modality_configs: dict[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
    for key, value in data_point.images.items():
        observation["video"][key] = np.asarray(value)[None, :]
    for key, value in data_point.states.items():
        observation["state"][key] = np.asarray(value, dtype=np.float32)[None, :]
    for key in modality_configs["language"].modality_keys:
        observation["language"][key] = [[data_point.text]]
    return observation


def _calibration_positions(loader: LeRobotEpisodeLoader, count: int) -> list[tuple[int, int]]:
    episode_count = len(loader)
    per_episode = math.ceil(count / episode_count)
    positions = []
    for episode_index in range(episode_count):
        episode_length = len(loader[episode_index])
        for step in np.linspace(0, episode_length - 1, per_episode, dtype=int):
            positions.append((episode_index, int(step)))
    return positions[:count]


def collect_hessian_diagonals(
    checkpoint_path: Path,
    dataset_path: Path,
    device: str,
    sample_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(checkpoint_path),
        device=device,
    )
    policy.model.eval()
    target_modules = {
        name: module
        for name, module in policy.model.named_modules()
        if isinstance(module, torch.nn.Linear) and should_quantize_linear_by_name(name)
    }
    if len(target_modules) != 424:
        raise RuntimeError(f"Expected 424 target modules, found {len(target_modules)}")

    sums = {
        name: torch.zeros(module.in_features, dtype=torch.float32, device=device)
        for name, module in target_modules.items()
    }
    counts = {name: 0 for name in target_modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            sums[name].add_(values.square().sum(dim=0))
            counts[name] += values.shape[0]

        return hook

    for name, module in target_modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    modality_configs = dict(policy.get_modality_config())
    extraction_configs = dict(modality_configs)
    extraction_configs.pop("action")
    positions = _calibration_positions(loader, sample_count)
    try:
        for sample_index, (episode_index, step_index) in enumerate(positions, start=1):
            episode = loader[episode_index]
            data_point = extract_step_data(
                episode,
                step_index,
                extraction_configs,
                EmbodimentTag.LIBERO_PANDA,
            )
            observation = _parse_observation(data_point, modality_configs)
            policy.last_action = -1.0
            policy.get_action(observation)
            print(
                f"Calibration [{sample_index:02d}/{len(positions)}] "
                f"episode={episode_index} step={step_index}",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"No calibration activations were observed for {len(missing)} layers")
    diagonals = {(name): (sums[name] / counts[name]).cpu() for name in sums}
    del policy, loader, sums, target_modules
    gc.collect()
    torch.cuda.empty_cache()
    return diagonals, counts


def export_quantized(
    checkpoint_path: Path,
    output_path: Path,
    hessian_diagonals: dict[str, torch.Tensor],
    hessian_counts: dict[str, int],
    args: argparse.Namespace,
) -> None:
    index = json.loads((checkpoint_path / "model.safetensors.index.json").read_text())
    targets = [
        (tensor_name, shard_name, tensor_name.removesuffix(".weight"))
        for tensor_name, shard_name in index["weight_map"].items()
        if tensor_name.endswith(".weight")
        and should_quantize_linear_by_name(tensor_name.removesuffix(".weight"))
    ]
    if len(targets) != 424:
        raise RuntimeError(f"Expected 424 target weights, found {len(targets)}")

    export_data = {
        "version": 1,
        "global_config": {
            "block_rows": 256,
            "block_cols": 256,
            "codebook_size": 256,
            "vq_dim": 2,
            "kmeans_iters": args.kmeans_iters,
            "skip_incomplete_blocks": True,
            "hessian_type": "diagonal_input_gram",
            "hessian_damping": args.damping,
            "calibration_dataset": str(args.dataset_path),
            "calibration_samples": args.calibration_samples,
        },
        "layers": {},
    }
    shard_handles = {}
    try:
        for layer_index, (tensor_name, shard_name, module_name) in enumerate(targets, start=1):
            if shard_name not in shard_handles:
                shard_handles[shard_name] = safe_open(
                    checkpoint_path / shard_name, framework="pt", device="cpu"
                )
            original = shard_handles[shard_name].get_tensor(tensor_name)
            diagonal = hessian_diagonals[module_name]
            quantized, block_infos = blockwise_hessian_diag_vq_quantize_weight_with_info_streaming(
                original,
                diagonal,
                device=args.device,
                block_rows=256,
                block_cols=256,
                codebook_size=256,
                vq_dim=2,
                kmeans_iters=args.kmeans_iters,
                damping=args.damping,
                block_batch_size=args.block_batch_size,
            )
            quant_loss = float(compute_quant_loss(original, quantized).item())
            normalized_diagonal = diagonal.float() / diagonal.float().mean().clamp_min(1e-12)
            weighted_loss = float(
                (original.float() - quantized.float()).square().mul(normalized_diagonal).mean().item()
            )
            export_data["layers"][module_name] = {
                "original_shape": tuple(original.shape),
                "dtype": str(original.dtype),
                "quantized_weight": quantized.cpu(),
                "blocks": block_infos,
                "quant_loss": quant_loss,
                "hessian_weighted_quant_loss": weighted_loss,
                "hessian_diag": hessian_diagonals[module_name],
                "hessian_samples": hessian_counts[module_name],
            }
            del original, diagonal, quantized, block_infos
            print(
                f"[{layer_index:03d}/{len(targets)}] {module_name}: "
                f"mse={quant_loss:.8e} hdiag_mse={weighted_loss:.8e}",
                flush=True,
            )

        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=args.output_dir, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            torch.save(export_data, temporary_path)
            os.link(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        shard_handles.clear()
        del export_data
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device is available")
    if args.calibration_samples <= 0:
        raise ValueError("calibration-samples must be positive")
    if args.damping < 0:
        raise ValueError("damping must be non-negative")
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        checkpoint_dir_name, output_name = CHECKPOINTS[name]
        checkpoint_path = args.checkpoint_root / checkpoint_dir_name
        output_path = args.output_dir / output_name
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        free_bytes = shutil.disk_usage(args.output_dir).free
        required_bytes = int((args.minimum_free_gib + 16.0) * 2**30)
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"Not enough safe disk headroom: {free_bytes / 2**30:.1f} GiB free; "
                f"need at least {required_bytes / 2**30:.1f} GiB before export"
            )
        print(f"Collecting Hessian diagonals for {name}", flush=True)
        diagonals, counts = collect_hessian_diagonals(
            checkpoint_path, args.dataset_path, args.device, args.calibration_samples
        )
        print(f"Exporting Hessian-aware weights to {output_path}", flush=True)
        export_quantized(checkpoint_path, output_path, diagonals, counts, args)
        del diagonals, counts
        gc.collect()


if __name__ == "__main__":
    main()
