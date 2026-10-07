#!/usr/bin/env python3
"""Export complete RDT/SigLIP/T5 hdiag-GPTVQ floating-parameter archives."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
import yaml

from scripts.maniskill_model import create_model
from scripts.vqvla_rdt_tasks import TASK_INSTRUCTIONS
from scripts.vqvla_rdt_vq import dequantize_matrix, quantize_matrix


TASKS = (
    "PegInsertionSide-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PlugCharger-v1",
    "PushCube-v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdiag-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(3, 4), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--block-rows", type=int, default=None)
    parser.add_argument("--block-cols", type=int, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--vector-len", type=int, default=2)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--block-batch-size", type=int, default=4)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    return parser.parse_args()


def _configuration(args: argparse.Namespace) -> tuple[int, int, int]:
    block_default, codebook_default = (256, 256) if args.bits == 4 else (128, 64)
    rows = args.block_rows or block_default
    cols = args.block_cols or block_default
    codebook = args.codebook_size or codebook_default
    if codebook != 1 << (args.bits * args.vector_len):
        raise ValueError(
            f"bits={args.bits}, vector_len={args.vector_len} requires codebook_size="
            f"{1 << (args.bits * args.vector_len)}, got {codebook}"
        )
    return rows, cols, codebook


def _load_hdiag(hdiag_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, int], list[str]]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for task in TASKS:
        path = hdiag_dir / f"rdt_hdiag_{task}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required calibration file: {path}")
        payload = torch.load(path, map_location="cpu")
        if payload.get("method") != "input_second_moment_diagonal":
            raise ValueError(f"Unexpected hdiag method in {path}")
        for name, summary in payload["modules"].items():
            if summary.get("kind") == "embedding_tokens":
                continue
            count = int(summary["input_rows"])
            if not summary["observed"] or count <= 0:
                continue
            diagonal = summary["hdiag"].float()
            if name not in sums:
                sums[name] = diagonal * count
                counts[name] = count
            else:
                if sums[name].shape != diagonal.shape:
                    raise ValueError(f"Inconsistent hdiag width for {name}")
                sums[name].add_(diagonal * count)
                counts[name] += count
    means = {name: values / counts[name] for name, values in sums.items()}
    return means, counts, [f"rdt_hdiag_{task}.pt" for task in TASKS]


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0 or args.minimum_free_gib < 0:
        raise ValueError("CPU threads must be positive and memory reserve nonnegative")
    block_rows, block_cols, codebook_size = _configuration(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise FileExistsError("temporary output already exists; inspect it before retrying")
    try:
        import psutil

        available_gib = psutil.virtual_memory().available / (1024**3)
        if available_gib < args.minimum_free_gib:
            raise RuntimeError(
                f"Only {available_gib:.1f} GiB RAM available; reserve is "
                f"{args.minimum_free_gib:.1f} GiB"
            )
    except ImportError:
        available_gib = None
    disk_gib = shutil.disk_usage(Path.cwd()).free / (1024**3)
    if disk_gib < 10:
        raise RuntimeError(f"Only {disk_gib:.1f} GiB free disk space")

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for RDT GPTVQ export")
        free_gpu, total_gpu = torch.cuda.mem_get_info(torch.device(args.device))
        if free_gpu < 12 * 1024**3:
            raise RuntimeError(
                f"Only {free_gpu / 1024**3:.1f} GiB free on selected GPU; refusing export"
            )
        print(
            f"GPU memory: {free_gpu / 1024**3:.1f}/{total_gpu / 1024**3:.1f} GiB free; "
            f"host available RAM: {available_gib if available_gib is not None else 'unknown'} GiB",
            flush=True,
        )

    hdiag, hdiag_counts, calibration_files = _load_hdiag(args.hdiag_dir)
    config = yaml.safe_load(Path("configs/base.yaml").read_text())
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=str(args.checkpoint),
        pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
        pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
    )
    components = {"rdt": model.policy, "siglip": model.vision_model, "t5": model.text_model}
    target_tensors: dict[str, tuple[torch.Tensor, str]] = {}
    embedding_modules: dict[str, torch.nn.Embedding] = {}
    expected_parameter_names: set[str] = set()
    for prefix, component in components.items():
        module_weight_ids: set[int] = set()
        module_weight_names: dict[int, str] = {}
        for local_name, module in component.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                name = f"{prefix}.{local_name}" if local_name else prefix
                kind = (
                    "embedding" if isinstance(module, torch.nn.Embedding)
                    else "conv2d" if isinstance(module, torch.nn.Conv2d)
                    else "linear"
                )
                target_tensors[name] = (module.weight, kind)
                module_weight_ids.add(id(module.weight))
                module_weight_names[id(module.weight)] = name
                if isinstance(module, torch.nn.Embedding):
                    embedding_modules[name] = module
        for local_name, parameter in component.named_parameters():
            if torch.is_floating_point(parameter):
                name = f"{prefix}.{local_name}" if local_name else prefix
                if id(parameter) in module_weight_ids:
                    expected_parameter_names.add(module_weight_names[id(parameter)])
                    continue
                expected_parameter_names.add(name)
                target_tensors[name] = (parameter, "parameter")
    if not target_tensors:
        raise RuntimeError("No floating-point model parameters found")
    if set(target_tensors) != expected_parameter_names:
        missing = sorted(expected_parameter_names - set(target_tensors))
        extra = sorted(set(target_tensors) - expected_parameter_names)
        raise RuntimeError(
            "complete floating-parameter coverage failed: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    embedding_token_counts: dict[str, torch.Tensor] = {}
    if embedding_modules:
        for name, module in embedding_modules.items():
            if name.startswith("t5."):
                embedding_token_counts[name] = torch.zeros(module.num_embeddings, dtype=torch.int64)
            else:
                # Vision position tables use every position uniformly for each image.
                embedding_token_counts[name] = torch.ones(module.num_embeddings, dtype=torch.int64)
        for name in embedding_modules:
            if not name.startswith("t5."):
                continue
            for task in TASKS:
                token_ids = model.text_tokenizer(
                    TASK_INSTRUCTIONS[task],
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"].reshape(-1).cpu()
                module = embedding_modules[name]
                valid = token_ids[(token_ids >= 0) & (token_ids < module.num_embeddings)]
                embedding_token_counts[name].index_add_(
                    0, valid, torch.ones_like(valid, dtype=torch.int64)
                )

    layers: dict[str, Any] = {}
    fallback_modules: list[str] = []
    total_values = 0
    weighted_sq_error = 0.0
    for index, (name, (tensor, kind)) in enumerate(target_tensors.items(), start=1):
        original_shape = tuple(tensor.shape)
        if kind == "parameter":
            if tensor.ndim >= 2:
                matrix = tensor.detach().reshape(-1, original_shape[-1]).cpu()
            else:
                matrix = tensor.detach().reshape(1, -1).cpu()
                if matrix.shape[1] % args.vector_len:
                    padded_width = (
                        (matrix.shape[1] + args.vector_len - 1) // args.vector_len
                    ) * args.vector_len
                    matrix = torch.nn.functional.pad(matrix, (0, padded_width - matrix.shape[1]))
        else:
            matrix = tensor.detach().reshape(original_shape[0], -1).cpu()
        expected_width = matrix.shape[1]
        row_weights = embedding_token_counts.get(name)
        if kind == "embedding":
            diagonal = torch.ones(expected_width)
            hessian_samples = int(row_weights.sum()) if row_weights is not None else 0
            if row_weights is None or hessian_samples == 0:
                raise RuntimeError(f"No tokenizer-frequency calibration for embedding {name}")
        elif kind == "parameter":
            diagonal = torch.ones(expected_width)
            hessian_samples = 0
            fallback_modules.append(name)
        elif name not in hdiag:
            diagonal = torch.ones(expected_width)
            hessian_samples = 0
            fallback_modules.append(name)
        else:
            summary = hdiag[name]
            if summary.numel() != expected_width:
                raise ValueError(
                    f"Hdiag width mismatch for {name}: {summary.numel()} vs {expected_width}"
                )
            diagonal = summary
            hessian_samples = hdiag_counts[name]
        archive, _ = quantize_matrix(
            matrix,
            diagonal,
            device=args.device,
            block_rows=block_rows,
            block_cols=block_cols,
            codebook_size=codebook_size,
            vector_len=args.vector_len,
            kmeans_iters=args.kmeans_iters,
            damping=args.damping,
            block_batch_size=args.block_batch_size,
            seed=args.seed + index,
            row_weights=row_weights,
        )
        reconstructed = dequantize_matrix(archive, dtype=matrix.dtype, device="cpu")
        difference = (matrix.float() - reconstructed.float()).square()
        normalized_diagonal = diagonal.float().clamp_min(0)
        normalized_diagonal = (
            normalized_diagonal + args.damping * normalized_diagonal.mean().clamp_min(1e-12)
        ) / normalized_diagonal.mean().clamp_min(1e-12)
        if kind == "parameter" and tensor.ndim < 2:
            difference = difference[:, : tensor.numel()]
            normalized_diagonal = normalized_diagonal[: tensor.numel()]
        if row_weights is not None:
            normalized_rows = row_weights.float().clamp_min(0)
            normalized_rows = (
                normalized_rows + args.damping * normalized_rows.mean().clamp_min(1e-12)
            ) / normalized_rows.mean().clamp_min(1e-12)
            error_weights = normalized_rows[:, None] * normalized_diagonal[None, :]
        else:
            error_weights = normalized_diagonal[None, :]
        weighted_sq_error += float((difference * error_weights).sum().item())
        total_values += tensor.numel()
        archive["original_parameter_shape"] = original_shape
        if kind == "embedding":
            archive["tokenizer_observations"] = hessian_samples
            archive["token_row_counts"] = row_weights.clone()
        archive["module_type"] = kind
        archive["hessian_samples"] = hessian_samples
        layers[name] = archive
        del matrix, reconstructed, difference, archive
        if index % 25 == 0 or index == len(target_tensors):
            print(f"Quantized {index}/{len(target_tensors)} parameter tensors", flush=True)

    export = {
        "version": 1,
        "global_config": {
            "algorithm": "GPTVQ input-Hessian-diagonal weighted VQ",
            "bits": args.bits,
            "block_rows": block_rows,
            "block_cols": block_cols,
            "codebook_size": codebook_size,
            "vector_len": args.vector_len,
            "kmeans_iters": args.kmeans_iters,
            "damping": args.damping,
            "block_batch_size": args.block_batch_size,
            "checkpoint": args.checkpoint.name,
            "calibration_files": calibration_files,
            "calibration_tasks": list(TASKS),
            "embedding_calibration": (
                "T5 token-row frequencies over the five task instructions; uniform "
                "row usage for non-text embedding tables; uniform input-feature diagonal"
            ),
            "quantized_parameter_tensors": len(target_tensors),
            "quantized_parameter_values": total_values,
            "uniform_weighting_fallback_tensors": fallback_modules,
            "coverage": "all floating-point named parameters in RDT, SigLIP, and T5; "
            "scalar and vector parameters are padded to a complete VQ vector and cropped on load",
            "hessian_weighted_squared_error": weighted_sq_error,
        },
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        torch.save(export, temp_path)
        os.replace(temp_path, args.output)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    print(
        f"Saved {args.bits}-bit archive with {len(layers)} floating-point parameter tensors "
        f"({total_values} values) to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
