#!/usr/bin/env python3
"""Export every floating OpenVLA-OFT parameter as a complete hdiag-weighted GPTVQ archive.

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
    parser.add_argument("--action-head", required=True, type=Path)
    parser.add_argument("--proprio-projector", required=True, type=Path)
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
    shape: tuple[int, ...],
    values: torch.Tensor | None,
    kind: str | None,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> tuple[torch.Tensor, str]:
    """Expand one input-Hessian summary over a row/column tile."""
    rows = shape[0] if len(shape) > 1 else 1
    columns = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    row_ids = torch.arange(row_start, row_end, dtype=torch.long)
    col_ids = torch.arange(col_start, col_end, dtype=torch.long)
    if values is not None and kind == "embedding_rows" and values.numel() == rows:
        return (
            values[row_ids, None].expand(-1, col_ids.numel()).reshape(-1).clamp_min(0),
            "embedding_rows",
        )
    if values is not None and kind == "features":
        if values.numel() == columns:
            return (
                values[col_ids][None, :].expand(row_ids.numel(), -1).reshape(-1).clamp_min(0),
                "input_features",
            )
        if len(shape) >= 3 and values.numel() == shape[1]:
            spatial_size = math.prod(shape[2:])
            channel_ids = torch.div(col_ids, spatial_size, rounding_mode="floor")
            return (
                values[channel_ids][None, :].expand(row_ids.numel(), -1).reshape(-1).clamp_min(0),
                "input_channels",
            )
    return torch.ones((row_end - row_start) * (col_end - col_start), dtype=torch.float32), "uniform_fallback"


def _normalise_weights(
    weights: torch.Tensor, damping: float, reference_mean: torch.Tensor
) -> torch.Tensor:
    mean = reference_mean
    if not torch.isfinite(mean) or mean <= 0:
        return torch.ones_like(weights)
    return (weights + damping * mean) / mean


def _quantize_tensor(
    tensor: torch.Tensor, summary: dict[str, Any] | None, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, int]]:
    shape = tuple(tensor.shape)
    if not shape or tensor.numel() == 0:
        raise ValueError("Cannot quantize a scalar or empty parameter tensor")
    rows = shape[0] if len(shape) > 1 else 1
    columns = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    matrix = tensor.detach().reshape(rows, columns)
    summary_values = None
    summary_kind = None
    reference_mean = torch.tensor(1.0)
    if summary is not None and summary["count"] > 0 and summary["sum"] is not None:
        summary_values = summary["sum"].float() / max(int(summary["count"]), 1)
        summary_kind = summary["kind"]
        reference_mean = summary_values.mean()

    codebooks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    block_coordinates: list[tuple[int, int, int, int]] = []
    block_shapes: list[tuple[int, int]] = []
    block_vector_counts: list[int] = []
    hdiag_sources: dict[str, int] = {}

    coordinates_by_shape: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for row_start in range(0, rows, args.block_rows):
        row_end = min(row_start + args.block_rows, rows)
        for col_start in range(0, columns, args.block_cols):
            col_end = min(col_start + args.block_cols, columns)
            coordinates_by_shape.setdefault((row_end - row_start, col_end - col_start), []).append(
                (row_start, row_end, col_start, col_end)
            )

    for (tile_rows, tile_columns), coordinates in coordinates_by_shape.items():
        for batch_start in range(0, len(coordinates), args.block_batch_size):
            batch_coordinates = coordinates[batch_start : batch_start + args.block_batch_size]
            source_tiles = []
            weight_tiles = []
            counts = []
            for row_start, row_end, col_start, col_end in batch_coordinates:
                tile = matrix[row_start:row_end, col_start:col_end].float().reshape(-1)
                hdiag, source_name = _summary_hdiag(
                    shape,
                    summary_values,
                    summary_kind,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                )
                hdiag_sources[source_name] = hdiag_sources.get(source_name, 0) + tile.numel()
                normalized_hdiag = _normalise_weights(hdiag, args.damping, reference_mean)
                padding = (-tile.numel()) % args.vector_length
                if padding:
                    tile = torch.cat((tile, torch.zeros(padding, dtype=tile.dtype)))
                    normalized_hdiag = torch.cat(
                        (normalized_hdiag, torch.zeros(padding, dtype=normalized_hdiag.dtype))
                    )
                source_tiles.append(tile)
                weight_tiles.append(normalized_hdiag)
                counts.append(tile.numel() // args.vector_length)

            batch_count = len(batch_coordinates)
            source = torch.stack(source_tiles).to(args.device)
            hdiag = torch.stack(weight_tiles).to(args.device)
            vector_count = source.shape[1] // args.vector_length
            vectors = source.view(batch_count, vector_count, args.vector_length)
            weights = hdiag.view_as(vectors)
            centroids, indices = _weighted_kmeans(
                vectors, weights, args.codebook_size, args.kmeans_iters
            )
            codebooks.append(centroids.to(dtype=tensor.dtype, device="cpu"))
            index_chunks.append(indices.to(dtype=torch.uint8, device="cpu").flatten())
            block_coordinates.extend(batch_coordinates)
            block_shapes.extend((tile_rows, tile_columns) for _ in batch_coordinates)
            block_vector_counts.extend(counts)
            del source, hdiag, vectors, weights, centroids, indices, source_tiles, weight_tiles

    indices = torch.cat(index_chunks)
    return (
        {
            "shape": shape,
            "matrix_shape": (rows, columns),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "codebooks": torch.cat(codebooks),
            "indices": _pack_indices(indices, int(math.log2(args.codebook_size))),
            "index_bits": int(math.log2(args.codebook_size)),
            "vector_count": int(indices.numel()),
            "block_coordinates": block_coordinates,
            "block_shapes": block_shapes,
            "block_vector_counts": block_vector_counts,
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


def _load_component_tensors(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {path}")
    return {
        (name[7:] if name.startswith("module.") else name): tensor
        for name, tensor in state.items()
    }


def main() -> None:
    args = parse_args()
    expected_codebook = 1 << (args.bits * args.vector_length)
    if args.codebook_size != expected_codebook:
        raise ValueError(f"Expected codebook size {expected_codebook}, received {args.codebook_size}")
    if min(args.block_rows, args.block_cols, args.vector_length, args.kmeans_iters, args.block_batch_size) <= 0:
        raise ValueError("block sizes, vector length, iteration count, and batch size must be positive")
    if args.cpu_threads <= 0 or args.damping < 0:
        raise ValueError("cpu_threads must be positive and damping must be non-negative")
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
    shard_tensors = _checkpoint_tensors(args.checkpoint)
    source_names = {
        f"model.{name}" for _, names in shard_tensors for name in names
    }
    component_sources = {
        "action_head": args.action_head,
        "proprio_projector": args.proprio_projector,
    }
    component_names: dict[str, list[str]] = {}
    for prefix, path in component_sources.items():
        state = _load_component_tensors(path)
        component_names[prefix] = sorted(state)
        source_names.update(f"{prefix}.{name}" for name in state)
        del state
    if source_names != set(hdiag["parameters"]):
        missing = set(hdiag["parameters"]) - source_names
        extra = source_names - set(hdiag["parameters"])
        raise RuntimeError(f"Checkpoint/hdiag parameter mismatch (missing={len(missing)}, extra={len(extra)})")
    queries_per_task = hdiag.get("queries_per_task", [])
    max_per_task = int(hdiag.get("max_observations_per_task", 16))
    calibration_observations = sum(min(int(count), max_per_task) for count in queries_per_task)
    if not queries_per_task:
        calibration_observations = int(hdiag["observation_count"])
    policy_query_count = int(hdiag.get("policy_query_count", sum(queries_per_task)))

    archive: dict[str, Any] = {
        "version": 3,
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
            "calibration_observations": calibration_observations,
            "policy_query_count": policy_query_count,
            "layout": "row_col_tiles_flattened_after_first_dimension",
            "components": ["model", "action_head", "proprio_projector"],
        },
        "checkpoint_components": {
            "model": str(args.checkpoint),
            "action_head": str(args.action_head),
            "proprio_projector": str(args.proprio_projector),
        },
        "tensors": {},
        "coverage": {"floating_tensor_count": 0, "floating_value_count": 0, "hdiag_sources": {}},
    }
    total = len(source_names)
    completed = 0

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal completed
        if not tensor.is_floating_point():
            raise RuntimeError(f"Unexpected non-floating source tensor: {name}")
        parameter_info = hdiag["parameters"][name]
        if tuple(tensor.shape) != tuple(parameter_info["shape"]):
            raise RuntimeError(
                f"Checkpoint/hdiag shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(parameter_info['shape'])}"
            )
        owner = parameter_info["owner"]
        summary = hdiag["module_summaries"].get(owner)
        # A Linear bias is an output offset, not a weight multiplying the
        # module input. Its output-side Hessian is unavailable here, so don't
        # misapply the input-feature diagonal even when dimensions coincide.
        if tensor.ndim == 1 and name.endswith(".bias"):
            summary = None
        record, source_counts = _quantize_tensor(
            tensor, summary, args
        )
        archive["tensors"][name] = record
        archive["coverage"]["floating_tensor_count"] += 1
        archive["coverage"]["floating_value_count"] += tensor.numel()
        for key, value in source_counts.items():
            counts = archive["coverage"]["hdiag_sources"]
            counts[key] = counts.get(key, 0) + value
        completed += 1
        print(f"[{completed:03d}/{total}] {name}", flush=True)
        del record
        gc.collect()
        torch.cuda.empty_cache()

    for shard_name, names in shard_tensors:
        with safe_open(args.checkpoint / shard_name, framework="pt", device="cpu") as source:
            for name in names:
                tensor = source.get_tensor(name)
                add_tensor(f"model.{name}", tensor)
                del tensor

    for prefix, path in component_sources.items():
        state = _load_component_tensors(path)
        if set(state) != set(component_names[prefix]):
            raise RuntimeError(f"Component tensor list changed while reading {path}")
        for name in component_names[prefix]:
            tensor = state[name]
            add_tensor(f"{prefix}.{name}", tensor)
            del tensor
        del state
    expected_values = sum(math.prod(info["shape"]) for info in hdiag["parameters"].values())
    if archive["coverage"]["floating_tensor_count"] != len(hdiag["parameters"]):
        raise RuntimeError("Quantizer did not process every hdiag parameter")
    if archive["coverage"]["floating_value_count"] != expected_values:
        raise RuntimeError("Quantizer did not process every floating-point value")
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
