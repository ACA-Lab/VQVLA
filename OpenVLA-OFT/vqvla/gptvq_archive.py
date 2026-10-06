"""Loader for complete hdiag GPTVQ archives."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch


def _unpack_indices(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    values_per_group = math.lcm(8, bits) // bits
    byte_count = values_per_group * bits // 8
    groups = packed.to(dtype=torch.int64, device="cpu").view(-1, byte_count)
    integers = torch.zeros(groups.shape[0], dtype=torch.int64)
    for byte_index in range(byte_count):
        integers |= groups[:, byte_index] << (8 * byte_index)
    mask = (1 << bits) - 1
    values = torch.stack(
        [(integers >> (value_index * bits)) & mask for value_index in range(values_per_group)], dim=1
    ).flatten()
    return values[:count].to(dtype=torch.long)


def reconstruct_tensor(record: dict[str, Any]) -> torch.Tensor:
    """Reconstruct one parameter tensor on CPU from block-local codebooks."""
    numel = int(record["numel"])
    codebooks = record["codebooks"].float()
    assignments = _unpack_indices(record["indices"], int(record["index_bits"]), record["vector_count"])
    if "block_coordinates" in record:
        rows, columns = tuple(record["matrix_shape"])
        matrix = torch.empty((rows, columns), dtype=torch.float32)
        offset = 0
        for block_index, (row_start, row_end, col_start, col_end) in enumerate(
            record["block_coordinates"]
        ):
            count = int(record["block_vector_counts"][block_index])
            tile_rows, tile_columns = record["block_shapes"][block_index]
            tile = codebooks[block_index][assignments[offset : offset + count]].reshape(-1)
            offset += count
            matrix[row_start:row_end, col_start:col_end] = tile[: tile_rows * tile_columns].reshape(
                tile_rows, tile_columns
            )
        if offset != assignments.numel():
            raise ValueError("Archive block assignment counts do not match packed indices")
        dtype = getattr(torch, record["dtype"].removeprefix("torch."))
        return matrix.reshape(tuple(record["shape"]))[:].to(dtype=dtype).reshape(tuple(record["shape"]))

    vectors_per_full_block = int(record["vectors_per_full_block"])
    full_block_count = int(record["full_block_count"])
    values: list[torch.Tensor] = []
    offset = 0
    for block_index, codebook in enumerate(codebooks):
        count = vectors_per_full_block if block_index < full_block_count else assignments.numel() - offset
        values.append(codebook[assignments[offset : offset + count]].reshape(-1))
        offset += count
    dtype = getattr(torch, record["dtype"].removeprefix("torch."))
    return torch.cat(values)[:numel].to(dtype=dtype).reshape(tuple(record["shape"]))


def load_complete_archive(
    model: torch.nn.Module | Mapping[str, torch.nn.Module], archive_path: str | Path
) -> dict[str, Any]:
    """Replace every learned parameter, optionally across named model components."""
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    if archive.get("format") != "vqvla_gptvq_hdiag":
        raise ValueError(f"Unsupported archive format: {archive.get('format')!r}")

    components = {"": model} if isinstance(model, torch.nn.Module) else dict(model)
    if not components or any(not isinstance(module, torch.nn.Module) for module in components.values()):
        raise TypeError("model must be a torch module or a non-empty mapping of component modules")
    component_parameters = {
        component: dict(module.named_parameters()) for component, module in components.items()
    }
    parameters = {
        f"{component}.{name}" if component else name: parameter
        for component, values in component_parameters.items()
        for name, parameter in values.items()
    }
    aliases = archive.get("parameter_aliases", {})
    loaded: set[str] = set()
    for source_name, record in archive["tensors"].items():
        target_name = aliases.get(source_name, source_name)
        parameter = parameters.get(target_name)
        canonical_name = target_name if parameter is not None else None
        if parameter is None:
            for component, values in component_parameters.items():
                candidates = [target_name]
                if component and target_name.startswith(f"{component}."):
                    candidates.append(target_name[len(component) + 1 :])
                elif not component and target_name.startswith("model."):
                    candidates.append(target_name[len("model.") :])
                for candidate in candidates:
                    if candidate in values:
                        parameter = values[candidate]
                        canonical_name = f"{component}.{candidate}" if component else candidate
                        break
                if parameter is not None:
                    break
        if parameter is None:
            raise KeyError(f"Archive parameter is absent from model: {target_name}")
        reconstructed = reconstruct_tensor(record)
        if tuple(parameter.shape) != tuple(reconstructed.shape):
            raise ValueError(f"Shape mismatch for {target_name}")
        parameter.data.copy_(reconstructed.to(device=parameter.device, dtype=parameter.dtype))
        loaded.add(canonical_name)
    missing = set(parameters) - loaded
    if missing:
        raise RuntimeError(f"Archive omitted model parameters: {sorted(missing)[:10]}")
    return archive["coverage"]
