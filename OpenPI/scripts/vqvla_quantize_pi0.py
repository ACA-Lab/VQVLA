#!/usr/bin/env python3
"""Export a complete Pi0 checkpoint as an input-Hessian-diagonal GPTVQ archive.

Every floating tensor in the source safetensors checkpoint is represented by
block-local codebooks and packed assignment indices.  No source weight tensor
is copied into the archive.  The companion loader reconstructs tensors lazily
when loading the policy.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from safetensors import safe_open
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdiag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", choices=(3, 4), type=int, required=True)
    parser.add_argument("--block-rows", type=int, required=True)
    parser.add_argument("--block-cols", type=int, required=True)
    parser.add_argument("--codebook-size", type=int, required=True)
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
    """Independent weighted K-means for ``[blocks, vectors, dimensions]``."""
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
    """Pack one-dimensional unsigned assignments without widening their storage."""
    values = indices.to(dtype=torch.int32, device="cpu").flatten()
    values_per_group = math.lcm(8, bits) // bits
    padded = (-values.numel()) % values_per_group
    if padded:
        values = torch.cat((values, torch.zeros(padded, dtype=torch.int32)))
    groups = values.view(-1, values_per_group)
    packed = torch.zeros(groups.shape[0], dtype=torch.int64)
    for index in range(values_per_group):
        packed |= groups[:, index].to(torch.int64) << (index * bits)
    byte_count = values_per_group * bits // 8
    return torch.stack([(packed >> (8 * index)).to(torch.uint8) for index in range(byte_count)], dim=1).flatten()


def _summary_hdiag(
    shape: tuple[int, ...], summary: dict[str, Any] | None, start: int, count: int
) -> tuple[torch.Tensor, str]:
    """Create exactly one flattened parameter block of diagonal weights on CPU."""
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
    elif len(shape) == 4 and summary["kind"] == "features":
        _, input_channels, height, width = shape
        if values.numel() == input_channels:
            channels = torch.div(
                positions.remainder(input_channels * height * width), height * width, rounding_mode="floor"
            )
            return values[channels].clamp_min(0), "input_channels"
    elif len(shape) == 1 and summary["kind"] == "features" and values.numel() == shape[0]:
        return values[positions].clamp_min(0), "input_features"
    return torch.ones(count, dtype=torch.float32), "uniform_fallback"


def _normalise_weights(weights: torch.Tensor, damping: float) -> torch.Tensor:
    mean = weights.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return torch.ones_like(weights)
    return (weights + damping * mean) / mean


def _resolve_hdiag_name(source_name: str, parameters: dict[str, Any]) -> str:
    """Resolve the one tied PaliGemma weight emitted under a loader alias."""
    if source_name in parameters:
        return source_name
    if source_name.endswith(".lm_head.weight"):
        candidate = source_name.removesuffix(".lm_head.weight") + ".model.language_model.embed_tokens.weight"
        if candidate in parameters:
            return candidate
    raise KeyError(f"No Hessian manifest entry for checkpoint tensor {source_name}")


def _quantize_tensor(
    tensor: torch.Tensor,
    summary: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    shape = tuple(tensor.shape)
    flat = tensor.detach().reshape(-1)
    block_elements = args.block_rows * args.block_cols
    vector_count = block_elements // args.vector_length
    codebook_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    hdiag_sources: dict[str, int] = {}
    full_blocks = flat.numel() // block_elements

    def consume(start_block: int, count_blocks: int) -> None:
        starts = [block_elements * (start_block + index) for index in range(count_blocks)]
        source = torch.stack([flat[start : start + block_elements].float() for start in starts]).to(
            args.device, non_blocking=True
        )
        hdiag_values = []
        for start in starts:
            weights, source_name = _summary_hdiag(shape, summary, start, block_elements)
            hdiag_sources[source_name] = hdiag_sources.get(source_name, 0) + block_elements
            hdiag_values.append(_normalise_weights(weights, args.damping))
        hdiag = torch.stack(hdiag_values).to(args.device, non_blocking=True)
        vectors = source.view(count_blocks, vector_count, args.vector_length)
        weights = hdiag.view_as(vectors)
        codebooks, indices = _weighted_kmeans(vectors, weights, args.codebook_size, args.kmeans_iters)
        codebook_chunks.append(codebooks.to(dtype=tensor.dtype, device="cpu"))
        index_chunks.append(indices.to(dtype=torch.uint8, device="cpu").flatten())
        del source, hdiag, vectors, weights, codebooks, indices

    for block_start in range(0, full_blocks, args.block_batch_size):
        consume(block_start, min(args.block_batch_size, full_blocks - block_start))

    remainder_start = full_blocks * block_elements
    remainder = flat.numel() - remainder_start
    if remainder:
        vector_values = flat[remainder_start:].float()
        if vector_values.numel() % args.vector_length:
            vector_values = torch.cat(
                (vector_values, torch.zeros(args.vector_length - vector_values.numel() % args.vector_length))
            )
        weights, source_name = _summary_hdiag(shape, summary, remainder_start, remainder)
        hdiag_sources[source_name] = hdiag_sources.get(source_name, 0) + remainder
        if weights.numel() % args.vector_length:
            weights = torch.cat((weights, torch.ones(args.vector_length - weights.numel() % args.vector_length)))
        vectors = vector_values.view(1, -1, args.vector_length).to(args.device)
        vector_weights = _normalise_weights(weights, args.damping).view_as(vectors).to(args.device)
        codebooks, indices = _weighted_kmeans(vectors, vector_weights, args.codebook_size, args.kmeans_iters)
        codebook_chunks.append(codebooks.to(dtype=tensor.dtype, device="cpu"))
        index_chunks.append(indices.to(dtype=torch.uint8, device="cpu").flatten())
        del vector_values, weights, vectors, vector_weights, codebooks, indices

    indices = torch.cat(index_chunks)
    index_bits = int(math.log2(args.codebook_size))
    return (
        {
            "shape": shape,
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "codebooks": torch.cat(codebook_chunks),
            "indices": _pack_indices(indices, index_bits),
            "index_bits": index_bits,
            "vector_count": int(indices.numel()),
            "full_block_count": full_blocks,
            "vectors_per_full_block": vector_count,
            "hdiag_sources": hdiag_sources,
        },
        hdiag_sources,
    )


def main() -> None:
    args = parse_args()
    expected_codebook = 1 << (args.bits * args.vector_length)
    if args.codebook_size != expected_codebook:
        raise ValueError(
            f"{args.bits}-bit scalar VQ with vector length {args.vector_length} requires "
            f"codebook size {expected_codebook}, got {args.codebook_size}"
        )
    if min(args.block_rows, args.block_cols, args.vector_length, args.kmeans_iters, args.block_batch_size) <= 0:
        raise ValueError("block sizes, vector length, iterations, and batch size must be positive")
    if args.block_rows * args.block_cols % args.vector_length:
        raise ValueError("block area must be divisible by vector length")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < args.minimum_free_gib * 2**30:
        raise RuntimeError(f"Less than {args.minimum_free_gib} GiB free at {args.output.parent}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bounded GPU GPTVQ export")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    hdiag = torch.load(args.hdiag, map_location="cpu", weights_only=False)
    source_path = args.checkpoint / "model.safetensors"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with safe_open(source_path, framework="pt", device="cpu") as source:
        names = list(source.keys())
        hdiag_names = set(hdiag["parameters"])
        resolved_names = {_resolve_hdiag_name(name, hdiag["parameters"]) for name in names}
        if resolved_names != hdiag_names:
            raise RuntimeError("Hessian manifest and source safetensors do not contain compatible tensor keys")
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
            "parameter_aliases": {},
            "coverage": {"floating_tensor_count": 0, "floating_value_count": 0, "hdiag_sources": {}},
        }
        for number, name in enumerate(names, start=1):
            tensor = source.get_tensor(name)
            if not tensor.is_floating_point():
                raise RuntimeError(f"Unexpected non-floating checkpoint tensor: {name}")
            hdiag_name = _resolve_hdiag_name(name, hdiag["parameters"])
            if hdiag_name != name:
                archive["parameter_aliases"][name] = hdiag_name
            owner = hdiag["parameters"][hdiag_name]["owner"]
            record, source_counts = _quantize_tensor(tensor, hdiag["module_summaries"].get(owner), args)
            archive["tensors"][name] = record
            archive["coverage"]["floating_tensor_count"] += 1
            archive["coverage"]["floating_value_count"] += tensor.numel()
            for key, value in source_counts.items():
                archive["coverage"]["hdiag_sources"][key] = archive["coverage"]["hdiag_sources"].get(key, 0) + value
            print(f"[{number:03d}/{len(names)}] {name}", flush=True)
            del tensor, record
            gc.collect()
            torch.cuda.empty_cache()

    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(archive, temporary)
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved complete {args.bits}-bit GPTVQ archive to {args.output}", flush=True)


if __name__ == "__main__":
    main()
