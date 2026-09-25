# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Input-aware block VQ using a diagonal Hessian approximation.

For a linear layer, calibration inputs give the Gauss-Newton/Hessian proxy
``H = X.T @ X``.  Keeping its diagonal makes calibration streaming and bounded
in memory while retaining per-input-channel sensitivity.  The quantization
objective is ``sum((W - Wq) ** 2 * diag(H))``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _weighted_kmeans_vq(
    vectors: torch.Tensor,
    vector_weights: torch.Tensor,
    k: int,
    iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2 or vector_weights.shape != vectors.shape:
        raise ValueError("vectors and vector_weights must have the same [N, D] shape")
    if k <= 0 or iters <= 0:
        raise ValueError("k and iters must be positive")

    n = vectors.shape[0]
    if n == 0:
        raise ValueError("vectors must be non-empty")
    if n >= k:
        indices = torch.randperm(n, device=vectors.device)[:k]
    else:
        indices = torch.randint(0, n, (k,), device=vectors.device)
    centroids = vectors[indices].clone()
    assignments = torch.full((n,), -1, dtype=torch.long, device=vectors.device)

    weighted_vectors = vector_weights * vectors
    weighted_x2 = (weighted_vectors * vectors).sum(dim=1, keepdim=True)
    for _ in range(iters):
        distances = (
            weighted_x2
            + vector_weights @ centroids.square().t()
            - 2.0 * weighted_vectors @ centroids.t()
        )
        new_assignments = distances.argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

        numerator = torch.zeros_like(centroids)
        denominator = torch.zeros_like(centroids)
        numerator.index_add_(0, assignments, weighted_vectors)
        denominator.index_add_(0, assignments, vector_weights)
        non_empty = denominator.sum(dim=1) > 0
        new_centroids = torch.empty_like(centroids)
        new_centroids[non_empty] = numerator[non_empty] / denominator[non_empty].clamp_min(1e-12)
        if (~non_empty).any():
            replacement = torch.randint(
                0, n, (int((~non_empty).sum().item()),), device=vectors.device
            )
            new_centroids[~non_empty] = vectors[replacement]
        centroids = new_centroids

    return centroids, assignments


def _weighted_kmeans_vq_batched(
    vectors: torch.Tensor,
    vector_weights: torch.Tensor,
    k: int,
    iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent weighted K-means for a batch of equally shaped blocks."""
    if vectors.ndim != 3 or vector_weights.shape != vectors.shape:
        raise ValueError("vectors and vector_weights must have the same [B, N, D] shape")
    batch_size, vector_count, vector_dim = vectors.shape
    initial_indices = torch.randint(
        0, vector_count, (batch_size, k), device=vectors.device
    )
    centroids = vectors.gather(
        1, initial_indices.unsqueeze(-1).expand(-1, -1, vector_dim)
    ).clone()
    assignments = torch.full(
        (batch_size, vector_count), -1, dtype=torch.long, device=vectors.device
    )
    weighted_vectors = vector_weights * vectors
    weighted_x2 = (weighted_vectors * vectors).sum(dim=2, keepdim=True)

    for _ in range(iters):
        distances = (
            weighted_x2
            + torch.bmm(vector_weights, centroids.square().transpose(1, 2))
            - 2.0 * torch.bmm(weighted_vectors, centroids.transpose(1, 2))
        )
        new_assignments = distances.argmin(dim=2)
        if torch.equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

        scatter_indices = assignments.unsqueeze(-1).expand(-1, -1, vector_dim)
        numerator = torch.zeros_like(centroids)
        denominator = torch.zeros_like(centroids)
        numerator.scatter_add_(1, scatter_indices, weighted_vectors)
        denominator.scatter_add_(1, scatter_indices, vector_weights)
        non_empty = denominator.sum(dim=2) > 0
        new_centroids = numerator / denominator.clamp_min(1e-12)
        if (~non_empty).any():
            replacement_indices = torch.randint(
                0, vector_count, (batch_size, k), device=vectors.device
            )
            replacements = vectors.gather(
                1, replacement_indices.unsqueeze(-1).expand(-1, -1, vector_dim)
            )
            new_centroids = torch.where(non_empty.unsqueeze(-1), new_centroids, replacements)
        centroids = new_centroids

    return centroids, assignments


def blockwise_hessian_diag_vq_quantize_weight_with_info_batched(
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    damping: float = 0.01,
    block_batch_size: int = 8,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Batched diagonal-Hessian VQ for complete, equally shaped blocks."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    if hessian_diag.shape != (columns,):
        raise ValueError(
            f"Expected hessian_diag shape ({columns},), got {tuple(hessian_diag.shape)}"
        )
    values = (block_rows, block_cols, codebook_size, vq_dim, kmeans_iters, block_batch_size)
    if min(values) <= 0:
        raise ValueError("all size and iteration parameters must be positive")
    if rows % block_rows or columns % block_cols:
        raise ValueError("batched Hessian VQ requires complete blocks")
    if block_rows * block_cols % vq_dim:
        raise ValueError("block size must be divisible by vq_dim")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    original_dtype = weight.dtype
    source = weight.detach().float()
    diagonal = (
        hessian_diag.detach().to(device=weight.device, dtype=torch.float32).clamp_min(0)
    )
    mean_diagonal = diagonal.mean().clamp_min(1e-12)
    diagonal = (diagonal + damping * mean_diagonal) / mean_diagonal
    quantized = source.clone()
    coordinates = [
        (row_start, column_start)
        for row_start in range(0, rows, block_rows)
        for column_start in range(0, columns, block_cols)
    ]
    block_infos: List[Dict] = []
    vector_count = block_rows * block_cols // vq_dim

    for batch_start in range(0, len(coordinates), block_batch_size):
        batch_coordinates = coordinates[batch_start : batch_start + block_batch_size]
        blocks = torch.stack(
            [
                source[
                    row_start : row_start + block_rows,
                    column_start : column_start + block_cols,
                ]
                for row_start, column_start in batch_coordinates
            ]
        )
        scalar_weights = torch.stack(
            [
                diagonal[column_start : column_start + block_cols].expand(block_rows, -1)
                for _, column_start in batch_coordinates
            ]
        )
        vectors = blocks.contiguous().view(len(batch_coordinates), vector_count, vq_dim)
        vector_weights = scalar_weights.contiguous().view_as(vectors)
        codebooks, assignments = _weighted_kmeans_vq_batched(
            vectors, vector_weights, codebook_size, kmeans_iters
        )
        reconstructed = codebooks.gather(
            1, assignments.unsqueeze(-1).expand(-1, -1, vq_dim)
        ).view_as(blocks)
        codebooks_cpu = codebooks.detach().cpu()
        assignments_cpu = assignments.detach().cpu()

        for index, (row_start, column_start) in enumerate(batch_coordinates):
            quantized[
                row_start : row_start + block_rows,
                column_start : column_start + block_cols,
            ] = reconstructed[index]
            block_infos.append(
                {
                    "block_index": (row_start // block_rows, column_start // block_cols),
                    "row_range": (row_start, row_start + block_rows),
                    "col_range": (column_start, column_start + block_cols),
                    "shape": (block_rows, block_cols),
                    "codebook": codebooks_cpu[index],
                    "indices": assignments_cpu[index],
                }
            )

    return quantized.to(device=weight.device, dtype=original_dtype), block_infos


def blockwise_hessian_diag_vq_quantize_weight_with_info_streaming(
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    device: torch.device | str,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    damping: float = 0.01,
    block_batch_size: int = 1,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Diagonal-Hessian VQ with only the current block batch resident on GPU."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    if hessian_diag.shape != (columns,):
        raise ValueError(
            f"Expected hessian_diag shape ({columns},), got {tuple(hessian_diag.shape)}"
        )
    values = (block_rows, block_cols, codebook_size, vq_dim, kmeans_iters, block_batch_size)
    if min(values) <= 0:
        raise ValueError("all size and iteration parameters must be positive")
    if rows % block_rows or columns % block_cols:
        raise ValueError("streaming Hessian VQ requires complete blocks")
    if block_rows * block_cols % vq_dim:
        raise ValueError("block size must be divisible by vq_dim")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    source = weight.detach().cpu()
    diagonal = hessian_diag.detach().cpu().float().clamp_min(0)
    mean_diagonal = diagonal.mean().clamp_min(1e-12)
    diagonal = (diagonal + damping * mean_diagonal) / mean_diagonal
    quantized = source.clone()
    coordinates = [
        (row_start, column_start)
        for row_start in range(0, rows, block_rows)
        for column_start in range(0, columns, block_cols)
    ]
    block_infos: List[Dict] = []
    vector_count = block_rows * block_cols // vq_dim

    for batch_start in range(0, len(coordinates), block_batch_size):
        batch_coordinates = coordinates[batch_start : batch_start + block_batch_size]
        blocks = torch.stack(
            [
                source[
                    row_start : row_start + block_rows,
                    column_start : column_start + block_cols,
                ]
                for row_start, column_start in batch_coordinates
            ]
        ).to(device=device, dtype=torch.float32)
        scalar_weights = torch.stack(
            [
                diagonal[column_start : column_start + block_cols].expand(block_rows, -1)
                for _, column_start in batch_coordinates
            ]
        ).to(device=device)
        vectors = blocks.contiguous().view(len(batch_coordinates), vector_count, vq_dim)
        vector_weights = scalar_weights.contiguous().view_as(vectors)
        codebooks, assignments = _weighted_kmeans_vq_batched(
            vectors, vector_weights, codebook_size, kmeans_iters
        )
        reconstructed = codebooks.gather(
            1, assignments.unsqueeze(-1).expand(-1, -1, vq_dim)
        ).view_as(blocks)
        reconstructed_cpu = reconstructed.to(device="cpu", dtype=source.dtype)
        codebooks_cpu = codebooks.detach().cpu()
        assignments_cpu = assignments.detach().cpu()

        for index, (row_start, column_start) in enumerate(batch_coordinates):
            quantized[
                row_start : row_start + block_rows,
                column_start : column_start + block_cols,
            ] = reconstructed_cpu[index]
            block_infos.append(
                {
                    "block_index": (row_start // block_rows, column_start // block_cols),
                    "row_range": (row_start, row_start + block_rows),
                    "col_range": (column_start, column_start + block_cols),
                    "shape": (block_rows, block_cols),
                    "codebook": codebooks_cpu[index],
                    "indices": assignments_cpu[index],
                }
            )
        del (
            blocks,
            scalar_weights,
            vectors,
            vector_weights,
            codebooks,
            assignments,
            reconstructed,
            reconstructed_cpu,
            codebooks_cpu,
            assignments_cpu,
        )

    return quantized, block_infos


def blockwise_hessian_diag_vq_quantize_weight_with_info(
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    damping: float = 0.01,
    skip_incomplete_blocks: bool = True,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Quantize a 2-D weight using input-channel Hessian-diagonal weights."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got {tuple(weight.shape)}")
    if hessian_diag.ndim != 1 or hessian_diag.numel() != weight.shape[1]:
        raise ValueError(
            f"Expected hessian_diag shape ({weight.shape[1]},), got {tuple(hessian_diag.shape)}"
        )
    if block_rows <= 0 or block_cols <= 0 or codebook_size <= 0 or vq_dim <= 0:
        raise ValueError("block dimensions, codebook_size, and vq_dim must be positive")
    if block_rows * block_cols % vq_dim:
        raise ValueError("block size must be divisible by vq_dim")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    original_dtype = weight.dtype
    original_device = weight.device
    source = weight.detach().float()
    diagonal = hessian_diag.detach().to(device=weight.device, dtype=torch.float32)
    diagonal = diagonal.clamp_min(0)
    mean_diagonal = diagonal.mean().clamp_min(1e-12)
    diagonal = (diagonal + damping * mean_diagonal) / mean_diagonal

    quantized = source.clone()
    block_infos: List[Dict] = []
    rows, columns = source.shape
    for row_start in range(0, rows, block_rows):
        row_end = min(row_start + block_rows, rows)
        for column_start in range(0, columns, block_cols):
            column_end = min(column_start + block_cols, columns)
            block = source[row_start:row_end, column_start:column_end]
            full = block.shape == (block_rows, block_cols)
            if not full and skip_incomplete_blocks:
                continue
            if block.numel() % vq_dim:
                if skip_incomplete_blocks:
                    continue
                raise ValueError(f"Block {tuple(block.shape)} is not divisible by vq_dim={vq_dim}")

            scalar_weights = diagonal[column_start:column_end].expand(block.shape[0], -1)
            vectors = block.contiguous().view(-1, vq_dim)
            vector_weights = scalar_weights.contiguous().view(-1, vq_dim)
            codebook, indices = _weighted_kmeans_vq(
                vectors, vector_weights, codebook_size, kmeans_iters
            )
            reconstructed = codebook[indices].view_as(block)
            quantized[row_start:row_end, column_start:column_end] = reconstructed
            block_infos.append(
                {
                    "block_index": (row_start // block_rows, column_start // block_cols),
                    "row_range": (row_start, row_end),
                    "col_range": (column_start, column_end),
                    "shape": tuple(block.shape),
                    "codebook": codebook.detach().cpu(),
                    "indices": indices.detach().cpu(),
                }
            )

    return quantized.to(device=original_device, dtype=original_dtype), block_infos
