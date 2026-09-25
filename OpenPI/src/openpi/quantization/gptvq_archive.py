"""Minimal loader for complete VQVLA GPTVQ archives.

Archives contain codebooks and packed vector assignments for every floating
model tensor.  Reconstruction happens tensor by tensor, so loading does not
require an unquantized checkpoint beside the archive.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def _unpack_indices(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    values_per_group = math.lcm(8, bits) // bits
    bytes_per_group = values_per_group * bits // 8
    groups = packed.to(dtype=torch.int64, device="cpu").view(-1, bytes_per_group)
    integers = torch.zeros(groups.shape[0], dtype=torch.int64)
    for byte_index in range(bytes_per_group):
        integers |= groups[:, byte_index] << (8 * byte_index)
    mask = (1 << bits) - 1
    values = torch.stack(
        [(integers >> (value_index * bits)) & mask for value_index in range(values_per_group)], dim=1
    ).flatten()
    return values[:count].to(dtype=torch.long)


def reconstruct_tensor(record: dict[str, Any]) -> torch.Tensor:
    """Reconstruct one tensor on CPU from block-local VQ data."""
    shape = tuple(record["shape"])
    numel = int(record["numel"])
    codebooks = record["codebooks"].float()
    block_count = codebooks.shape[0]
    if block_count == 0:
        raise ValueError("GPTVQ tensor has no codebooks")
    assignments = _unpack_indices(record["indices"], int(record["index_bits"]), record["vector_count"])
    values: list[torch.Tensor] = []
    offset = 0
    full_block_count = int(record["full_block_count"])
    vectors_per_full_block = int(record["vectors_per_full_block"])
    for block_index in range(block_count):
        count = vectors_per_full_block if block_index < full_block_count else assignments.numel() - offset
        values.append(codebooks[block_index][assignments[offset : offset + count]].reshape(-1))
        offset += count
    flat = torch.cat(values)[:numel]
    dtype_name = record["dtype"].removeprefix("torch.")
    dtype = getattr(torch, dtype_name)
    return flat.to(dtype=dtype).reshape(shape)


def load_gptvq_archive(model: torch.nn.Module, archive_path: str | Path) -> dict[str, Any]:
    """Replace every floating parameter in ``model`` from a complete archive."""
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    if archive.get("format") != "vqvla_gptvq_hdiag":
        raise ValueError(f"Unsupported GPTVQ archive format: {archive.get('format')!r}")
    named_parameters = dict(model.named_parameters())
    aliases = archive.get("parameter_aliases", {})
    loaded = 0
    for source_name, record in archive["tensors"].items():
        target_name = aliases.get(source_name, source_name)
        parameter = named_parameters.get(target_name)
        if parameter is None:
            raise KeyError(f"Model does not expose archive parameter {target_name}")
        reconstructed = reconstruct_tensor(record)
        if tuple(parameter.shape) != tuple(reconstructed.shape):
            raise ValueError(
                f"Shape mismatch for {target_name}: model {tuple(parameter.shape)}, "
                f"archive {tuple(reconstructed.shape)}"
            )
        parameter.data.copy_(reconstructed.to(device=parameter.device, dtype=parameter.dtype))
        loaded += 1
    if loaded != len(named_parameters):
        missing = sorted(set(named_parameters) - {aliases.get(name, name) for name in archive["tensors"]})
        raise RuntimeError(f"Archive did not cover all model parameters; missing {missing[:10]}")
    return archive["coverage"]
