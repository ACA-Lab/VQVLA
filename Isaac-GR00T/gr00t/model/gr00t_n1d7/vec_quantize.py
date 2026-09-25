# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

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
    x: torch.Tensor,      # [N, D]
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
    vectors: torch.Tensor,   # [N, D]
    k: int,
) -> torch.Tensor:
    """
    从 vectors 中随机抽样初始化 centroids。
    若 N < k，则允许重复采样。
    """
    n = vectors.shape[0]
    device = vectors.device

    if n >= k:
        perm = torch.randperm(n, device=device)[:k]
        centroids = vectors[perm].clone()
    else:
        idx = torch.randint(0, n, (k,), device=device)
        centroids = vectors[idx].clone()
    return centroids


def _kmeans_vq(
    vectors: torch.Tensor,   # [N, D]
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
    assert vectors.dim() == 2
    n, d = vectors.shape
    device = vectors.device

    if n == 0:
        raise ValueError("vectors must be non-empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

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
) -> torch.Tensor:
    """
    对单个 block 做独立 codebook 的向量量化。

    规则：
      - 将 block reshape 为 [num_vectors, vq_dim]
      - 用 kmeans 得到 codebook_size 个 centroid
      - 每个向量替换为最近 centroid
      - reshape 回原始 block 形状
    """
    assert block.dim() == 2, f"Expected 2D block, got shape={tuple(block.shape)}"
    if vq_dim <= 0:
        raise ValueError(f"vq_dim must be positive, got {vq_dim}")

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

    return quantized_block.to(device=device, dtype=orig_dtype)


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
    对 2D Linear weight 做 block-wise vector quantization。

    支持任意 block_rows x block_cols，只要：
      - 完整 block 的 numel 能被 vq_dim 整除
      - 若处理不完整边缘 block，则边缘 block 的 numel 也能被 vq_dim 整除

    默认策略：
      - 对完整 block 做 VQ
      - 对边缘不足 block_rows x block_cols 的 block：
          skip_incomplete_blocks=True 时跳过，保留原值
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {tuple(weight.shape)}"
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(
            f"block_rows and block_cols must be positive, got "
            f"block_rows={block_rows}, block_cols={block_cols}"
        )
    if vq_dim <= 0:
        raise ValueError(f"vq_dim must be positive, got {vq_dim}")

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

            q_block = _quantize_single_block_with_vq(
                block,
                codebook_size=codebook_size,
                vq_dim=vq_dim,
                kmeans_iters=kmeans_iters,
            )
            q[r0:r1, c0:c1] = q_block

    return q.to(device=device, dtype=orig_dtype)


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
                logger.warning(f"[GPTVQ][blockwise] skip meta tensor layer: {full_name}")
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
                f"[GPTVQ][blockwise] quantized {full_name}, "
                f"shape={tuple(module.weight.shape)}, "
                f"block=({block_rows}x{block_cols}), "
                f"codebook_size={codebook_size}, "
                f"vq_dim={vq_dim}, "
                f"kmeans_iters={kmeans_iters}, "
                f"quant_loss={quant_loss_value:.8e}"
            )

    if verbose:
        avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
        logger.warning(f"[GPTVQ][blockwise] total quantized linear layers: {count}")
        logger.warning(f"[GPTVQ][blockwise] average quant_loss: {avg_quant_loss:.8e}")

    return count


# # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0

# import logging
# from typing import Tuple

# import torch
# from torch import nn


# logger = logging.getLogger(__name__)


# def compute_quant_loss(
#     weight: torch.Tensor,
#     quantized_weight: torch.Tensor,
# ) -> torch.Tensor:
#     return (weight - quantized_weight).abs().pow(2).mean()


# def should_quantize_linear_by_name(full_name: str) -> bool:
#     """
#     根据模块完整名字判断某个 nn.Linear 是否属于目标量化范围。
#     """

#     # 1) Gr00tN1d7.backbone.model.model.visual
#     if full_name.startswith("backbone.model.model.visual."):
#         return any(
#             k in full_name
#             for k in [
#                 ".attn.qkv",
#                 ".attn.proj",
#                 ".mlp.linear_fc1",
#                 ".mlp.linear_fc2",
#             ]
#         )

#     # 2) Gr00tN1d7.backbone.model.model.language_model
#     if full_name.startswith("backbone.model.model.language_model."):
#         return any(
#             k in full_name
#             for k in [
#                 ".self_attn.q_proj",
#                 ".self_attn.k_proj",
#                 ".self_attn.v_proj",
#                 ".self_attn.o_proj",
#                 ".mlp.gate_proj",
#                 ".mlp.up_proj",
#                 ".mlp.down_proj",
#             ]
#         )

#     # 3) Gr00tN1d7.action_head.model
#     if full_name.startswith("action_head.model."):
#         return any(
#             k in full_name
#             for k in [
#                 ".attn1.to_q",
#                 ".attn1.to_k",
#                 ".attn1.to_v",
#                 ".attn1.to_out.0",
#                 ".ff.net.0.proj",
#                 ".ff.net.2",
#             ]
#         )

#     # 4) Gr00tN1d7.action_head.vl_self_attention
#     if full_name.startswith("action_head.vl_self_attention."):
#         return any(
#             k in full_name
#             for k in [
#                 ".attn1.to_q",
#                 ".attn1.to_k",
#                 ".attn1.to_v",
#                 ".attn1.to_out.0",
#                 ".ff.net.0.proj",
#                 ".ff.net.2",
#             ]
#         )

#     return False


# def _pairwise_squared_distance(
#     x: torch.Tensor,      # [N, D]
#     centroids: torch.Tensor,  # [K, D]
# ) -> torch.Tensor:
#     """
#     返回 [N, K] 的平方欧氏距离矩阵
#     """
#     # (x - c)^2 = x^2 + c^2 - 2xc
#     x2 = (x * x).sum(dim=1, keepdim=True)         # [N, 1]
#     c2 = (centroids * centroids).sum(dim=1).unsqueeze(0)  # [1, K]
#     xc = x @ centroids.t()                        # [N, K]
#     dist = x2 + c2 - 2.0 * xc
#     return dist


# def _kmeans_init_random_samples(
#     vectors: torch.Tensor,   # [N, D]
#     k: int,
# ) -> torch.Tensor:
#     """
#     从 vectors 中随机抽样初始化 centroids。
#     若 N < k，则允许重复采样。
#     """
#     n = vectors.shape[0]
#     device = vectors.device

#     if n >= k:
#         perm = torch.randperm(n, device=device)[:k]
#         centroids = vectors[perm].clone()
#     else:
#         idx = torch.randint(0, n, (k,), device=device)
#         centroids = vectors[idx].clone()
#     return centroids


# def _kmeans_vq(
#     vectors: torch.Tensor,   # [N, D]
#     k: int = 256,
#     iters: int = 20,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     最简单稳定版 kmeans:
#       - 输入 [N, D]
#       - 输出:
#           centroids: [K, D]
#           assignments: [N]
#     """
#     assert vectors.dim() == 2
#     n, d = vectors.shape
#     device = vectors.device

#     centroids = _kmeans_init_random_samples(vectors, k)

#     assignments = torch.zeros(n, dtype=torch.long, device=device)

#     for _ in range(iters):
#         dist = _pairwise_squared_distance(vectors, centroids)  # [N, K]
#         new_assignments = dist.argmin(dim=1)

#         if torch.equal(new_assignments, assignments):
#             assignments = new_assignments
#             break
#         assignments = new_assignments

#         # M-step
#         new_centroids = torch.zeros_like(centroids)
#         counts = torch.bincount(assignments, minlength=k).to(vectors.dtype)  # [K]

#         new_centroids.index_add_(0, assignments, vectors)

#         non_empty = counts > 0
#         if non_empty.any():
#             new_centroids[non_empty] = new_centroids[non_empty] / counts[non_empty].unsqueeze(1)

#         # 对空簇重新随机初始化
#         empty = ~non_empty
#         if empty.any():
#             reinit = _kmeans_init_random_samples(vectors, int(empty.sum().item()))
#             new_centroids[empty] = reinit

#         centroids = new_centroids

#     return centroids, assignments


# def _quantize_single_256x256_block_with_vq(
#     block: torch.Tensor,   # [256, 256]
#     codebook_size: int = 256,
#     vq_dim: int = 2,
#     kmeans_iters: int = 20,
# ) -> torch.Tensor:
#     """
#     对单个 256x256 block 做独立 codebook 的向量量化。

#     规则：
#       - reshape 为 [256*256/2, 2]
#       - 用 kmeans 得到 256 个 centroid
#       - 每个向量替换为最近 centroid
#       - reshape 回 [256, 256]
#     """
#     assert block.shape == (256, 256)
#     assert vq_dim == 2

#     orig_dtype = block.dtype
#     device = block.device

#     x = block.detach().to(torch.float32).contiguous()
#     vectors = x.view(-1, vq_dim)  # [32768, 2]

#     centroids, assignments = _kmeans_vq(
#         vectors=vectors,
#         k=codebook_size,
#         iters=kmeans_iters,
#     )

#     quantized_vectors = centroids[assignments]  # [32768, 2]
#     quantized_block = quantized_vectors.view(256, 256)

#     return quantized_block.to(device=device, dtype=orig_dtype)


# def blockwise_vq_quant_dequant_weight(
#     weight: torch.Tensor,
#     block_rows: int = 256,
#     block_cols: int = 256,
#     codebook_size: int = 256,
#     vq_dim: int = 2,
#     kmeans_iters: int = 20,
#     skip_incomplete_blocks: bool = True,
# ) -> torch.Tensor:
#     """
#     对 2D Linear weight 做 block-wise vector quantization。

#     你的目标配置：
#       - block_rows = 256
#       - block_cols = 256
#       - codebook_size = 256
#       - vq_dim = 2
#       - 每个 block 独立 codebook

#     默认策略：
#       - 对完整的 256x256 block 做 VQ
#       - 对边缘不足 256x256 的 block：
#           skip_incomplete_blocks=True 时跳过，保留原值
#     """
#     assert weight.dim() == 2, f"Expected 2D weight, got {tuple(weight.shape)}"
#     assert block_rows == 256 and block_cols == 256, "当前实现按你的要求固定支持 256x256 block"
#     assert vq_dim == 2, "当前实现按你的要求固定支持向量长度 2"

#     orig_dtype = weight.dtype
#     device = weight.device

#     w = weight.detach().to(torch.float32).clone()
#     out_features, in_features = w.shape

#     q = w.clone()

#     n_block_rows = (out_features + block_rows - 1) // block_rows
#     n_block_cols = (in_features + block_cols - 1) // block_cols

#     for br in range(n_block_rows):
#         r0 = br * block_rows
#         r1 = min((br + 1) * block_rows, out_features)

#         for bc in range(n_block_cols):
#             c0 = bc * block_cols
#             c1 = min((bc + 1) * block_cols, in_features)

#             block = w[r0:r1, c0:c1]

#             if block.shape != (block_rows, block_cols):
#                 if skip_incomplete_blocks:
#                     continue
#                 else:
#                     # 如需 padding，可后续再扩展
#                     continue

#             q_block = _quantize_single_256x256_block_with_vq(
#                 block,
#                 codebook_size=codebook_size,
#                 vq_dim=vq_dim,
#                 kmeans_iters=kmeans_iters,
#             )
#             q[r0:r1, c0:c1] = q_block

#     return q.to(device=device, dtype=orig_dtype)


# def apply_blockwise_gptvq_to_named_linears(
#     root_module: nn.Module,
#     block_rows: int = 256,
#     block_cols: int = 256,
#     codebook_size: int = 256,
#     vq_dim: int = 2,
#     kmeans_iters: int = 20,
#     skip_incomplete_blocks: bool = True,
#     verbose: bool = True,
# ) -> int:
#     """
#     对命中的 nn.Linear.weight 执行 block-wise VQ quant-dequant，并统计量化损失。
#     """
#     count = 0
#     total_quant_loss = 0.0

#     for full_name, module in root_module.named_modules():
#         if not isinstance(module, nn.Linear):
#             continue
#         if not should_quantize_linear_by_name(full_name):
#             continue

#         with torch.no_grad():
#             original_weight = module.weight.data.detach().clone()

#             quantized_weight = blockwise_vq_quant_dequant_weight(
#                 original_weight,
#                 block_rows=block_rows,
#                 block_cols=block_cols,
#                 codebook_size=codebook_size,
#                 vq_dim=vq_dim,
#                 kmeans_iters=kmeans_iters,
#                 skip_incomplete_blocks=skip_incomplete_blocks,
#             )

#             quant_loss = compute_quant_loss(original_weight, quantized_weight)
#             quant_loss_value = float(quant_loss.item())

#             module.weight.data.copy_(quantized_weight)

#         count += 1
#         total_quant_loss += quant_loss_value

#         if verbose:
#             logger.warning(
#                 f"[GPTVQ][blockwise] quantized {full_name}, "
#                 f"shape={tuple(module.weight.shape)}, "
#                 f"block=({block_rows}x{block_cols}), "
#                 f"codebook_size={codebook_size}, "
#                 f"vq_dim={vq_dim}, "
#                 f"kmeans_iters={kmeans_iters}, "
#                 f"quant_loss={quant_loss_value:.8e}"
#             )

#     if verbose:
#         avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
#         logger.warning(f"[GPTVQ][blockwise] total quantized linear layers: {count}")
#         logger.warning(f"[GPTVQ][blockwise] average quant_loss: {avg_quant_loss:.8e}")

#     return count