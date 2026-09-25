#!/usr/bin/env python3
"""Export non-destructive 4-bit-equivalent block-wise VQ weights for LIBERO.

This invokes the repository's current ``vec_quant_save`` implementation.  It is
called GPTVQ in the model API, but it is a weight-only K-means VQ implementation:
it does not collect calibration activations or use a Hessian.

The 256-entry codebook has an 8-bit index.  With ``vq_dim=2``, this is 4 bits
per scalar for the index stream.  Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open

from gr00t.model.gr00t_n1d7.vec_quant_save import (
    blockwise_vq_quantize_weight_with_info_streaming,
    compute_quant_loss,
    should_quantize_linear_by_name,
)


CHECKPOINTS = {
    "10": ("libero_10", "gr00t_10_gptvq_4bit_b256_k256.pt"),
    "goal": ("libero_goal", "gr00t_goal_gptvq_4bit_b256_k256.pt"),
    "object": ("libero_object", "gr00t_object_gptvq_4bit_b256_k256.pt"),
    "spatial": ("libero_spatial", "gr00t_spatial_gptvq_4bit_b256_k256.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        choices=tuple(CHECKPOINTS),
        nargs="+",
        default=tuple(CHECKPOINTS),
        help="Checkpoint variants to quantize (default: all four).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--block-batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/GR00T-N1.7-LIBERO")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("quant_weight"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-model VQ export, but no CUDA device is available.")
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
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        free_bytes = shutil.disk_usage(args.output_dir).free
        required_bytes = int((args.minimum_free_gib + 16.0) * 2**30)
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"Not enough safe disk headroom: {free_bytes / 2**30:.1f} GiB free; "
                f"need at least {required_bytes / 2**30:.1f} GiB before export"
            )

        print(
            f"Exporting {output_path} with block=256x256, codebook=256, "
            f"vq_dim=2, kmeans_iters={args.kmeans_iters}"
        )
        index = json.loads((checkpoint_path / "model.safetensors.index.json").read_text())
        targets = [
            (tensor_name, shard_name, tensor_name.removesuffix(".weight"))
            for tensor_name, shard_name in index["weight_map"].items()
            if tensor_name.endswith(".weight")
            and should_quantize_linear_by_name(tensor_name.removesuffix(".weight"))
        ]
        if len(targets) != 424:
            raise RuntimeError(f"Expected 424 target weights in {checkpoint_path}, found {len(targets)}")

        export_data = {
            "version": 1,
            "global_config": {
                "block_rows": 256,
                "block_cols": 256,
                "codebook_size": 256,
                "vq_dim": 2,
                "kmeans_iters": args.kmeans_iters,
                "skip_incomplete_blocks": True,
            },
            "layers": {},
        }
        shard_handles = {}
        total_loss = 0.0
        try:
            for layer_index, (tensor_name, shard_name, module_name) in enumerate(targets, start=1):
                if shard_name not in shard_handles:
                    shard_handles[shard_name] = safe_open(
                        checkpoint_path / shard_name, framework="pt", device="cpu"
                    )
                original_weight = shard_handles[shard_name].get_tensor(tensor_name)
                quantized_weight, block_infos = blockwise_vq_quantize_weight_with_info_streaming(
                    original_weight,
                    device=args.device,
                    block_rows=256,
                    block_cols=256,
                    codebook_size=256,
                    vq_dim=2,
                    kmeans_iters=args.kmeans_iters,
                    block_batch_size=args.block_batch_size,
                )
                quant_loss = float(compute_quant_loss(original_weight, quantized_weight).item())
                total_loss += quant_loss
                export_data["layers"][module_name] = {
                    "original_shape": tuple(original_weight.shape),
                    "dtype": str(original_weight.dtype),
                    "quantized_weight": quantized_weight.cpu(),
                    "blocks": block_infos,
                    "quant_loss": quant_loss,
                }
                del original_weight, quantized_weight, block_infos
                print(
                    f"[{layer_index:03d}/{len(targets)}] {module_name}: loss={quant_loss:.8e}",
                    flush=True,
                )

            # Publish only a fully written archive; os.link also refuses overwrites.
            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.name}.", suffix=".tmp", dir=args.output_dir, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            try:
                torch.save(export_data, temporary_path)
                os.link(temporary_path, output_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            print(
                f"Completed {output_path}: layers={len(targets)}, "
                f"average_layer_loss={total_loss / len(targets):.8e}",
                flush=True,
            )
        finally:
            shard_handles.clear()
            del export_data
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
