# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List, Tuple

import torch
from torch import nn


logger = logging.getLogger(__name__)


def compute_quant_loss(
    weight: torch.Tensor,
    quantized_weight: torch.Tensor,
) -> torch.Tensor:
    return (weight - quantized_weight).abs().pow(2).mean()


def should_quantize_linear_by_name(full_name: str) -> bool:
    """
    根据模块完整名字判断某个 nn.Linear 是否属于目标量化范围。
    """

    # 1) Gr00tN1d7.backbone.model.model.visual
    if full_name.startswith("backbone.model.model.visual."):
        return any(
            k in full_name
            for k in [
                ".attn.qkv",
                ".attn.proj",
                ".mlp.linear_fc1",
                ".mlp.linear_fc2",
            ]
        )

    # 2) Gr00tN1d7.backbone.model.model.language_model
    if full_name.startswith("backbone.model.model.language_model."):
        return any(
            k in full_name
            for k in [
                ".self_attn.q_proj",
                ".self_attn.k_proj",
                ".self_attn.v_proj",
                ".self_attn.o_proj",
                ".mlp.gate_proj",
                ".mlp.up_proj",
                ".mlp.down_proj",
            ]
        )

    # 3) Gr00tN1d7.action_head.model
    if full_name.startswith("action_head.model."):
        return any(
            k in full_name
            for k in [
                ".attn1.to_q",
                ".attn1.to_k",
                ".attn1.to_v",
                ".attn1.to_out.0",
                ".ff.net.0.proj",
                ".ff.net.2",
            ]
        )

    # 4) Gr00tN1d7.action_head.vl_self_attention
    if full_name.startswith("action_head.vl_self_attention."):
        return any(
            k in full_name
            for k in [
                ".attn1.to_q",
                ".attn1.to_k",
                ".attn1.to_v",
                ".attn1.to_out.0",
                ".ff.net.0.proj",
                ".ff.net.2",
            ]
        )

    return False


def _pairwise_squared_distance(
    x: torch.Tensor,          # [N, D]
    centroids: torch.Tensor,  # [K, D]
) -> torch.Tensor:
    """
    返回 [N, K] 的平方欧氏距离矩阵
    """
    x2 = (x * x).sum(dim=1, keepdim=True)  # [N, 1]
    c2 = (centroids * centroids).sum(dim=1).unsqueeze(0)  # [1, K]
    xc = x @ centroids.t()  # [N, K]
    dist = x2 + c2 - 2.0 * xc
    return dist


def _kmeans_init_random_samples(
    vectors: torch.Tensor,  # [N, D]
    k: int,
) -> torch.Tensor:
    """
    从 vectors 中随机抽样初始化 centroids。
    若 N < k，则允许重复采样。
    """
    n = vectors.shape[0]
    device = vectors.device

    if n <= 0:
        raise ValueError("vectors must be non-empty")

    if n >= k:
        perm = torch.randperm(n, device=device)[:k]
        centroids = vectors[perm].clone()
    else:
        idx = torch.randint(0, n, (k,), device=device)
        centroids = vectors[idx].clone()
    return centroids


def _kmeans_vq(
    vectors: torch.Tensor,  # [N, D]
    k: int = 256,
    iters: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    最简单稳定版 kmeans:
      - 输入 [N, D]
      - 输出:
          centroids: [K, D]
          assignments: [N]
    """
    if vectors.dim() != 2:
        raise ValueError(f"Expected vectors with dim=2, got shape={tuple(vectors.shape)}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")

    n, d = vectors.shape
    device = vectors.device

    if n <= 0:
        raise ValueError("vectors must be non-empty")

    centroids = _kmeans_init_random_samples(vectors, k)
    assignments = torch.zeros(n, dtype=torch.long, device=device)

    for _ in range(iters):
        dist = _pairwise_squared_distance(vectors, centroids)  # [N, K]
        new_assignments = dist.argmin(dim=1)

        if torch.equal(new_assignments, assignments):
            assignments = new_assignments
            break

        assignments = new_assignments

        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assignments, minlength=k).to(vectors.dtype)  # [K]
        new_centroids.index_add_(0, assignments, vectors)

        non_empty = counts > 0
        if non_empty.any():
            new_centroids[non_empty] = (
                new_centroids[non_empty] / counts[non_empty].unsqueeze(1)
            )

        empty = ~non_empty
        if empty.any():
            reinit = _kmeans_init_random_samples(vectors, int(empty.sum().item()))
            new_centroids[empty] = reinit

        centroids = new_centroids

    return centroids, assignments


def _quantize_single_block_with_vq(
    block: torch.Tensor,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对单个 2D block 做独立 codebook 的向量量化。

    返回:
      quantized_block: [H, W]
      centroids: [K, D]
      assignments: [num_vectors]
    """
    if block.dim() != 2:
        raise ValueError(f"Expected 2D block, got shape={tuple(block.shape)}")
    if codebook_size <= 0:
        raise ValueError(f"codebook_size must be positive, got {codebook_size}")
    if vq_dim <= 0:
        raise ValueError(f"vq_dim must be positive, got {vq_dim}")
    if kmeans_iters <= 0:
        raise ValueError(f"kmeans_iters must be positive, got {kmeans_iters}")

    block_h, block_w = block.shape
    numel = block.numel()

    if numel % vq_dim != 0:
        raise ValueError(
            f"block shape {tuple(block.shape)} has numel={numel}, "
            f"which is not divisible by vq_dim={vq_dim}"
        )

    orig_dtype = block.dtype
    device = block.device

    x = block.detach().to(torch.float32).contiguous()
    vectors = x.view(-1, vq_dim)  # [num_vectors, vq_dim]

    centroids, assignments = _kmeans_vq(
        vectors=vectors,
        k=codebook_size,
        iters=kmeans_iters,
    )

    quantized_vectors = centroids[assignments]  # [num_vectors, vq_dim]
    quantized_block = quantized_vectors.view(block_h, block_w)

    return (
        quantized_block.to(device=device, dtype=orig_dtype),
        centroids,
        assignments,
    )


def _kmeans_vq_batched(
    vectors: torch.Tensor,  # [B, N, D]
    k: int,
    iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent K-means for several equally shaped blocks at once."""
    if vectors.ndim != 3:
        raise ValueError(f"Expected [B, N, D] vectors, got {tuple(vectors.shape)}")
    batch_size, vector_count, vector_dim = vectors.shape
    if vector_count <= 0 or k <= 0 or iters <= 0:
        raise ValueError("vector_count, k, and iters must be positive")

    # Sampling with replacement avoids constructing B full random permutations and
    # has negligible impact when N (32768) is much larger than K (256).
    initial_indices = torch.randint(
        0, vector_count, (batch_size, k), device=vectors.device
    )
    centroids = vectors.gather(
        1, initial_indices.unsqueeze(-1).expand(-1, -1, vector_dim)
    ).clone()
    assignments = torch.full(
        (batch_size, vector_count), -1, dtype=torch.long, device=vectors.device
    )
    vector_squared_norm = vectors.square().sum(dim=2, keepdim=True)

    for _ in range(iters):
        distances = (
            vector_squared_norm
            + centroids.square().sum(dim=2).unsqueeze(1)
            - 2.0 * torch.bmm(vectors, centroids.transpose(1, 2))
        )
        new_assignments = distances.argmin(dim=2)
        if torch.equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

        new_centroids = torch.zeros_like(centroids)
        scatter_indices = assignments.unsqueeze(-1).expand(-1, -1, vector_dim)
        new_centroids.scatter_add_(1, scatter_indices, vectors)
        counts = torch.zeros(
            batch_size, k, dtype=vectors.dtype, device=vectors.device
        )
        counts.scatter_add_(1, assignments, torch.ones_like(assignments, dtype=vectors.dtype))
        non_empty = counts > 0
        new_centroids = new_centroids / counts.clamp_min(1).unsqueeze(-1)
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


def blockwise_vq_quantize_weight_with_info_batched(
    weight: torch.Tensor,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    block_batch_size: int = 8,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Batched fast path for weights exactly divisible by the block dimensions."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {tuple(weight.shape)}")
    values = (block_rows, block_cols, codebook_size, vq_dim, kmeans_iters, block_batch_size)
    if min(values) <= 0:
        raise ValueError(
            "block dimensions, codebook size, VQ dimension, iterations, and batch size "
            "must be positive"
        )
    rows, columns = weight.shape
    if rows % block_rows or columns % block_cols:
        raise ValueError(
            f"Batched fast path requires complete blocks, got weight={tuple(weight.shape)} "
            f"and block=({block_rows}, {block_cols})"
        )
    if block_rows * block_cols % vq_dim:
        raise ValueError("block size must be divisible by vq_dim")

    original_dtype = weight.dtype
    original_device = weight.device
    source = weight.detach().float()
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
        vectors = blocks.contiguous().view(len(batch_coordinates), vector_count, vq_dim)
        codebooks, assignments = _kmeans_vq_batched(vectors, codebook_size, kmeans_iters)
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

    return quantized.to(device=original_device, dtype=original_dtype), block_infos


def blockwise_vq_quantize_weight_with_info_streaming(
    weight: torch.Tensor,
    device: torch.device | str,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    block_batch_size: int = 1,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Quantize CPU weights while placing only the current block batch on ``device``.

    This has the same independent block-wise K-means objective as the batched fast
    path, but avoids keeping a full layer and its float32 working copies on the GPU.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {tuple(weight.shape)}")
    values = (block_rows, block_cols, codebook_size, vq_dim, kmeans_iters, block_batch_size)
    if min(values) <= 0:
        raise ValueError(
            "block dimensions, codebook size, VQ dimension, iterations, and batch size "
            "must be positive"
        )
    rows, columns = weight.shape
    if rows % block_rows or columns % block_cols:
        raise ValueError(
            f"Streaming path requires complete blocks, got weight={tuple(weight.shape)} "
            f"and block=({block_rows}, {block_cols})"
        )
    if block_rows * block_cols % vq_dim:
        raise ValueError("block size must be divisible by vq_dim")

    source = weight.detach().cpu()
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
        vectors = blocks.contiguous().view(len(batch_coordinates), vector_count, vq_dim)
        codebooks, assignments = _kmeans_vq_batched(vectors, codebook_size, kmeans_iters)
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
            vectors,
            codebooks,
            assignments,
            reconstructed,
            reconstructed_cpu,
            codebooks_cpu,
            assignments_cpu,
        )

    return quantized, block_infos


def blockwise_vq_quantize_weight_with_info(
    weight: torch.Tensor,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    skip_incomplete_blocks: bool = True,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    对 2D Linear weight 做 block-wise vector quantization，并返回详细 block 信息。

    返回:
      quantized_weight
      block_infos: list[dict]
        每个元素包含:
          - block_index: (br, bc)
          - row_range: (r0, r1)
          - col_range: (c0, c1)
          - shape: (h, w)
          - codebook: [K, D] (cpu tensor)
          - indices: [num_vectors] (cpu tensor)
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got {tuple(weight.shape)}")
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(
            f"block_rows and block_cols must be positive, got "
            f"block_rows={block_rows}, block_cols={block_cols}"
        )
    if codebook_size <= 0:
        raise ValueError(f"codebook_size must be positive, got {codebook_size}")
    if vq_dim <= 0:
        raise ValueError(f"vq_dim must be positive, got {vq_dim}")
    if kmeans_iters <= 0:
        raise ValueError(f"kmeans_iters must be positive, got {kmeans_iters}")

    full_block_numel = block_rows * block_cols
    if full_block_numel % vq_dim != 0:
        raise ValueError(
            f"full block ({block_rows}x{block_cols}) has numel={full_block_numel}, "
            f"which is not divisible by vq_dim={vq_dim}"
        )

    orig_dtype = weight.dtype
    device = weight.device

    w = weight.detach().to(torch.float32).clone()
    out_features, in_features = w.shape

    q = w.clone()
    block_infos: List[Dict] = []

    n_block_rows = (out_features + block_rows - 1) // block_rows
    n_block_cols = (in_features + block_cols - 1) // block_cols

    for br in range(n_block_rows):
        r0 = br * block_rows
        r1 = min((br + 1) * block_rows, out_features)

        for bc in range(n_block_cols):
            c0 = bc * block_cols
            c1 = min((bc + 1) * block_cols, in_features)

            block = w[r0:r1, c0:c1]
            is_full_block = block.shape == (block_rows, block_cols)

            if not is_full_block and skip_incomplete_blocks:
                continue

            if block.numel() % vq_dim != 0:
                if skip_incomplete_blocks:
                    continue
                raise ValueError(
                    f"Incomplete block shape {tuple(block.shape)} has numel={block.numel()}, "
                    f"which is not divisible by vq_dim={vq_dim}"
                )

            q_block, centroids, assignments = _quantize_single_block_with_vq(
                block,
                codebook_size=codebook_size,
                vq_dim=vq_dim,
                kmeans_iters=kmeans_iters,
            )

            q[r0:r1, c0:c1] = q_block

            block_infos.append(
                {
                    "block_index": (br, bc),
                    "row_range": (r0, r1),
                    "col_range": (c0, c1),
                    "shape": tuple(block.shape),
                    "codebook": centroids.detach().cpu(),
                    "indices": assignments.detach().cpu(),
                }
            )

    return q.to(device=device, dtype=orig_dtype), block_infos


def blockwise_vq_quant_dequant_weight(
    weight: torch.Tensor,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    skip_incomplete_blocks: bool = True,
) -> torch.Tensor:
    """
    对 2D Linear weight 做 block-wise vector quantization，只返回量化后的权重。
    """
    quantized_weight, _ = blockwise_vq_quantize_weight_with_info(
        weight=weight,
        block_rows=block_rows,
        block_cols=block_cols,
        codebook_size=codebook_size,
        vq_dim=vq_dim,
        kmeans_iters=kmeans_iters,
        skip_incomplete_blocks=skip_incomplete_blocks,
    )
    return quantized_weight


def apply_blockwise_gptvq_to_named_linears(
    root_module: nn.Module,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    skip_incomplete_blocks: bool = True,
    verbose: bool = True,
) -> int:
    """
    对命中的 nn.Linear.weight 执行 block-wise VQ quant-dequant，并统计量化损失。
    """
    count = 0
    total_quant_loss = 0.0

    for full_name, module in root_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_linear_by_name(full_name):
            continue

        weight = module.weight
        if getattr(weight, "is_meta", False) or weight.device.type == "meta":
            if verbose:
                logger.warning(f"[GPTVQ][apply] skip meta tensor layer: {full_name}")
            continue

        with torch.no_grad():
            original_weight = weight.data.detach().clone()

            quantized_weight = blockwise_vq_quant_dequant_weight(
                original_weight,
                block_rows=block_rows,
                block_cols=block_cols,
                codebook_size=codebook_size,
                vq_dim=vq_dim,
                kmeans_iters=kmeans_iters,
                skip_incomplete_blocks=skip_incomplete_blocks,
            )

            quant_loss = compute_quant_loss(original_weight, quantized_weight)
            quant_loss_value = float(quant_loss.item())

            module.weight.data.copy_(quantized_weight)

        count += 1
        total_quant_loss += quant_loss_value

        if verbose:
            logger.warning(
                f"[GPTVQ][apply] quantized {full_name}, "
                f"shape={tuple(module.weight.shape)}, "
                f"block=({block_rows}x{block_cols}), "
                f"codebook_size={codebook_size}, "
                f"vq_dim={vq_dim}, "
                f"kmeans_iters={kmeans_iters}, "
                f"quant_loss={quant_loss_value:.8e}"
            )

    if verbose:
        avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
        logger.warning(f"[GPTVQ][apply] total quantized linear layers: {count}")
        logger.warning(f"[GPTVQ][apply] average quant_loss: {avg_quant_loss:.8e}")

    return count


def export_blockwise_gptvq_to_file(
    root_module: nn.Module,
    save_path: str,
    block_rows: int = 256,
    block_cols: int = 256,
    codebook_size: int = 256,
    vq_dim: int = 2,
    kmeans_iters: int = 20,
    skip_incomplete_blocks: bool = True,
    verbose: bool = True,
) -> int:
    """
    对命中的 nn.Linear.weight 执行 block-wise VQ，并把结果保存到文件。

    保存内容包括：
      - 每层 quantized_weight
      - 每个 block 的 codebook
      - 每个 block 的 indices
      - 全局量化配置
    """
    export_data = {
        "version": 1,
        "global_config": {
            "block_rows": block_rows,
            "block_cols": block_cols,
            "codebook_size": codebook_size,
            "vq_dim": vq_dim,
            "kmeans_iters": kmeans_iters,
            "skip_incomplete_blocks": skip_incomplete_blocks,
        },
        "layers": {},
    }

    count = 0
    total_quant_loss = 0.0

    for full_name, module in root_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_linear_by_name(full_name):
            continue

        weight = module.weight
        if getattr(weight, "is_meta", False) or weight.device.type == "meta":
            if verbose:
                logger.warning(f"[GPTVQ][export] skip meta tensor layer: {full_name}")
            continue

        with torch.no_grad():
            original_weight = weight.data.detach().clone()

            quantized_weight, block_infos = blockwise_vq_quantize_weight_with_info(
                original_weight,
                block_rows=block_rows,
                block_cols=block_cols,
                codebook_size=codebook_size,
                vq_dim=vq_dim,
                kmeans_iters=kmeans_iters,
                skip_incomplete_blocks=skip_incomplete_blocks,
            )

            quant_loss = compute_quant_loss(original_weight, quantized_weight)
            quant_loss_value = float(quant_loss.item())

            export_data["layers"][full_name] = {
                "original_shape": tuple(original_weight.shape),
                "dtype": str(original_weight.dtype),
                "quantized_weight": quantized_weight.detach().cpu(),
                "blocks": block_infos,
                "quant_loss": quant_loss_value,
            }

        count += 1
        total_quant_loss += quant_loss_value

        if verbose:
            logger.warning(
                f"[GPTVQ][export] quantized {full_name}, "
                f"shape={tuple(original_weight.shape)}, "
                f"block=({block_rows}x{block_cols}), "
                f"codebook_size={codebook_size}, "
                f"vq_dim={vq_dim}, "
                f"kmeans_iters={kmeans_iters}, "
                f"quant_loss={quant_loss_value:.8e}"
            )

    torch.save(export_data, save_path)

    if verbose:
        avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
        logger.warning(f"[GPTVQ][export] saved to: {save_path}")
        logger.warning(f"[GPTVQ][export] total quantized linear layers: {count}")
        logger.warning(f"[GPTVQ][export] average quant_loss: {avg_quant_loss:.8e}")

    return count


def load_blockwise_quantized_weights_from_file(
    root_module: nn.Module,
    save_path: str,
    strict: bool = True,
    verbose: bool = True,
) -> int:
    """
    从 export_blockwise_gptvq_to_file 保存的文件中读取量化后的权重，
    并覆盖写回模型对应的 nn.Linear.weight。

    注意：
      - 这里只加载 quantized_weight 覆盖模型权重
      - codebook / indices 会保留在文件中，但不会参与替换流程
    """
    data = torch.load(save_path, map_location="cpu")
    layers = data["layers"]

    module_dict = dict(root_module.named_modules())
    count = 0

    for full_name, layer_data in layers.items():
        if full_name not in module_dict:
            msg = f"[GPTVQ][load] target module not found: {full_name}"
            if strict:
                raise KeyError(msg)
            if verbose:
                logger.warning(msg)
            continue

        module = module_dict[full_name]
        if not isinstance(module, nn.Linear):
            msg = f"[GPTVQ][load] target module is not nn.Linear: {full_name}"
            if strict:
                raise TypeError(msg)
            if verbose:
                logger.warning(msg)
            continue

        q_weight = layer_data["quantized_weight"]

        if tuple(module.weight.shape) != tuple(q_weight.shape):
            msg = (
                f"[GPTVQ][load] shape mismatch for {full_name}: "
                f"model={tuple(module.weight.shape)} vs saved={tuple(q_weight.shape)}"
            )
            if strict:
                raise ValueError(msg)
            if verbose:
                logger.warning(msg)
            continue

        with torch.no_grad():
            module.weight.data.copy_(
                q_weight.to(device=module.weight.device, dtype=module.weight.dtype)
            )

        count += 1

        if verbose:
            logger.warning(
                f"[GPTVQ][load] loaded quantized weight for {full_name}, "
                f"shape={tuple(module.weight.shape)}"
            )

    if verbose:
        logger.warning(f"[GPTVQ][load] total loaded quantized linear layers: {count}")

    return count
