#!/usr/bin/env python3
"""Export every floating OpenVLA parameter as a complete hdiag-weighted GPTVQ archive.

The exporter streams individual safetensors tensors and small block batches.
It never copies original parameter tensors into the output and refuses to
overwrite an archive or start when free storage is below the requested floor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hdiag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=(3, 4))
    parser.add_argument("--block-rows", required=True, type=int)
    parser.add_argument("--block-cols", required=True, type=int)
    parser.add_argument("--codebook-size", required=True, type=int)
    parser.add_argument("--vector-length", type=int, default=2)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--block-batch-size", type=int, default=4)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    return parser.parse_args()


def _weighted_kmeans(
    vectors: torch.Tensor, weights: torch.Tensor, codebook_size: int, iterations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    block_count, vector_count, dimension = vectors.shape
    initial = torch.randint(0, vector_count, (block_count, codebook_size), device=vectors.device)
    centroids = vectors.gather(1, initial.unsqueeze(-1).expand(-1, -1, dimension)).clone()
    assignments = torch.full((block_count, vector_count), -1, dtype=torch.long, device=vectors.device)
    weighted_vectors = vectors * weights
    weighted_norm = (weighted_vectors * vectors).sum(dim=-1, keepdim=True)
    for _ in range(iterations):
        distances = (
            weighted_norm
            + torch.bmm(weights, centroids.square().transpose(1, 2))
            - 2 * torch.bmm(weighted_vectors, centroids.transpose(1, 2))
        )
        new_assignments = distances.argmin(dim=-1)
        if torch.equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        scatter = assignments.unsqueeze(-1).expand(-1, -1, dimension)
        numerator = torch.zeros_like(centroids)
        denominator = torch.zeros_like(centroids)
        numerator.scatter_add_(1, scatter, weighted_vectors)
        denominator.scatter_add_(1, scatter, weights)
        nonempty = denominator.sum(dim=-1) > 0
        replacements = vectors.gather(
            1,
            torch.randint(0, vector_count, (block_count, codebook_size), device=vectors.device)
            .unsqueeze(-1)
            .expand(-1, -1, dimension),
        )
        centroids = torch.where(nonempty.unsqueeze(-1), numerator / denominator.clamp_min(1e-12), replacements)
    return centroids, assignments


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    values = indices.to(dtype=torch.int32, device="cpu").flatten()
    values_per_group = math.lcm(8, bits) // bits
    padding = (-values.numel()) % values_per_group
    if padding:
        values = torch.cat((values, torch.zeros(padding, dtype=torch.int32)))
    groups = values.view(-1, values_per_group)
    packed = torch.zeros(groups.shape[0], dtype=torch.int64)
    for index in range(values_per_group):
        packed |= groups[:, index].to(torch.int64) << (index * bits)
    byte_count = values_per_group * bits // 8
    return torch.stack([(packed >> (8 * index)).to(torch.uint8) for index in range(byte_count)], dim=1).flatten()


def _summary_hdiag(
    shape: tuple[int, ...], summary: dict[str, Any] | None, start: int, count: int
) -> tuple[torch.Tensor, str]:
    if summary is None or summary["count"] <= 0 or summary["sum"] is None:
        return torch.ones(count, dtype=torch.float32), "uniform_fallback"
    values = summary["sum"].float() / max(int(summary["count"]), 1)
    positions = torch.arange(start, start + count, dtype=torch.long)
    if len(shape) == 2:
        rows, columns = shape
        if summary["kind"] == "features" and values.numel() == columns:
            return values[positions.remainder(columns)].clamp_min(0), "input_features"
        if summary["kind"] == "embedding_rows" and values.numel() == rows:
            return values[torch.div(positions, columns, rounding_mode="floor")].clamp_min(0), "embedding_rows"
    if len(shape) == 1 and summary["kind"] == "features" and values.numel() == shape[0]:
        return values[positions].clamp_min(0), "input_features"
    if len(shape) == 4 and summary["kind"] == "features":
        _, channels, height, width = shape
        if values.numel() == channels:
            channel_index = torch.div(
                positions.remainder(channels * height * width), height * width, rounding_mode="floor"
            )
            return values[channel_index].clamp_min(0), "input_channels"
    return torch.ones(count, dtype=torch.float32), "uniform_fallback"


def _normalise_weights(weights: torch.Tensor, damping: float) -> torch.Tensor:
    mean = weights.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return torch.ones_like(weights)
    return (weights + damping * mean) / mean


def _quantize_tensor(
    tensor: torch.Tensor, summary: dict[str, Any] | None, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, int]]:
    shape = tuple(tensor.shape)
    flat = tensor.detach().reshape(-1)
    block_elements = args.block_rows * args.block_cols
    vector_count = block_elements // args.vector_length
    full_blocks = flat.numel() // block_elements
    codebooks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    hdiag_sources: dict[str, int] = {}

    def consume(start_block: int, count_blocks: int) -> None:
        starts = [block_elements * (start_block + index) for index in range(count_blocks)]
        source = torch.stack([flat[start : start + block_elements].float() for start in starts]).to(args.device)
        hdiag_values = []
        for start in starts:
            values, source_name = _summary_hdiag(shape, summary, start, block_elements)
            hdiag_sources[source_name] = hdiag_sources.get(source_name, 0) + block_elements
            hdiag_values.append(_normalise_weights(values, args.damping))
        hdiag = torch.stack(hdiag_values).to(args.device)
        vectors = source.view(count_blocks, vector_count, args.vector_length)
        weights = hdiag.view_as(vectors)
        centroids, indices = _weighted_kmeans(vectors, weights, args.codebook_size, args.kmeans_iters)
        codebooks.append(centroids.to(dtype=tensor.dtype, device="cpu"))
        index_chunks.append(indices.to(dtype=torch.uint8, device="cpu").flatten())
        del source, hdiag, vectors, weights, centroids, indices

    for block_start in range(0, full_blocks, args.block_batch_size):
        consume(block_start, min(args.block_batch_size, full_blocks - block_start))
    remainder_start = full_blocks * block_elements
    remainder = flat.numel() - remainder_start
    if remainder:
        values = flat[remainder_start:].float()
        if values.numel() % args.vector_length:
            values = torch.cat((values, torch.zeros(args.vector_length - values.numel() % args.vector_length)))
        weights, source_name = _summary_hdiag(shape, summary, remainder_start, remainder)
        hdiag_sources[source_name] = hdiag_sources.get(source_name, 0) + remainder
        if weights.numel() % args.vector_length:
            weights = torch.cat((weights, torch.ones(args.vector_length - weights.numel() % args.vector_length)))
        vectors = values.view(1, -1, args.vector_length).to(args.device)
        vector_weights = _normalise_weights(weights, args.damping).view_as(vectors).to(args.device)
        centroids, indices = _weighted_kmeans(vectors, vector_weights, args.codebook_size, args.kmeans_iters)
        codebooks.append(centroids.to(dtype=tensor.dtype, device="cpu"))
        index_chunks.append(indices.to(dtype=torch.uint8, device="cpu").flatten())
        del values, weights, vectors, vector_weights, centroids, indices
    indices = torch.cat(index_chunks)
    return (
        {
            "shape": shape,
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "codebooks": torch.cat(codebooks),
            "indices": _pack_indices(indices, int(math.log2(args.codebook_size))),
            "index_bits": int(math.log2(args.codebook_size)),
            "vector_count": int(indices.numel()),
            "full_block_count": full_blocks,
            "vectors_per_full_block": vector_count,
            "hdiag_sources": hdiag_sources,
        },
        hdiag_sources,
    )


def _checkpoint_tensors(checkpoint: Path) -> list[tuple[str, list[str]]]:
    index_path = checkpoint / "model.safetensors.index.json"
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]
    per_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        per_shard.setdefault(shard, []).append(name)
    return [(shard, sorted(names)) for shard, names in sorted(per_shard.items())]


def main() -> None:
    args = parse_args()
    expected_codebook = 1 << (args.bits * args.vector_length)
    if args.codebook_size != expected_codebook:
        raise ValueError(f"Expected codebook size {expected_codebook}, received {args.codebook_size}")
    if min(args.block_rows, args.block_cols, args.vector_length, args.kmeans_iters, args.block_batch_size) <= 0:
        raise ValueError("block sizes, vector length, iteration count, and batch size must be positive")
    if args.block_rows * args.block_cols % args.vector_length:
        raise ValueError("block area must be divisible by vector length")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if shutil.disk_usage(args.output.parent).free < args.minimum_free_gib * 2**30:
        raise RuntimeError(f"Less than {args.minimum_free_gib} GiB free at {args.output.parent}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bounded GPU GPTVQ export")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    hdiag = torch.load(args.hdiag, map_location="cpu", weights_only=False)
    shard_tensors = _checkpoint_tensors(args.checkpoint)
    source_names = {name for _, names in shard_tensors for name in names}
    if source_names != set(hdiag["parameters"]):
        missing = set(hdiag["parameters"]) - source_names
        extra = source_names - set(hdiag["parameters"])
        raise RuntimeError(f"Checkpoint/hdiag parameter mismatch (missing={len(missing)}, extra={len(extra)})")

    archive: dict[str, Any] = {
        "version": 2,
        "format": "vqvla_gptvq_hdiag",
        "global_config": {
            "bits": args.bits,
            "block_rows": args.block_rows,
            "block_cols": args.block_cols,
            "codebook_size": args.codebook_size,
            "vector_length": args.vector_length,
            "kmeans_iters": args.kmeans_iters,
            "hessian_type": hdiag["method"],
            "hessian_damping": args.damping,
            "calibration_observations": hdiag["observation_count"],
            "layout": "flat_contiguous_blocks",
        },
        "tensors": {},
        "coverage": {"floating_tensor_count": 0, "floating_value_count": 0, "hdiag_sources": {}},
    }
    total = len(source_names)
    completed = 0
    for shard_name, names in shard_tensors:
        with safe_open(args.checkpoint / shard_name, framework="pt", device="cpu") as source:
            for name in names:
                tensor = source.get_tensor(name)
                if not tensor.is_floating_point():
                    raise RuntimeError(f"Unexpected non-floating source tensor: {name}")
                owner = hdiag["parameters"][name]["owner"]
                record, source_counts = _quantize_tensor(tensor, hdiag["module_summaries"].get(owner), args)
                archive["tensors"][name] = record
                archive["coverage"]["floating_tensor_count"] += 1
                archive["coverage"]["floating_value_count"] += tensor.numel()
                for key, value in source_counts.items():
                    archive["coverage"]["hdiag_sources"][key] = archive["coverage"]["hdiag_sources"].get(key, 0) + value
                completed += 1
                print(f"[{completed:03d}/{total}] {name}", flush=True)
                del tensor, record
                gc.collect()
                torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(archive, temporary)
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved complete {args.bits}-bit GPTVQ archive to {args.output}")


if __name__ == "__main__":
    main()
