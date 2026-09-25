# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


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


def compute_quant_loss(
    weight: torch.Tensor,
    quantized_weight: torch.Tensor,
) -> torch.Tensor:
    """
    计算量化损失：
        (weight - quantized_weight).abs().pow(2).mean()
    """
    return (weight - quantized_weight).abs().pow(2).mean()


def blockwise_int3_quant_dequant_weight(
    weight: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对 2D weight 做 block-wise 对称 int3 量化再反量化。
    默认按最后一维分块，即对每一行的列方向按 block_size 切分。
    对于 shape = [out_features, in_features] 的权重：
      - 每一行独立处理
      - 每个 block 共享一个 scale
      - int3 范围使用 [-4, 3]
    Args:
        weight: 2D tensor, usually [out_features, in_features]
        block_size: 每个 block 的长度，建议 64 或 128
        eps: 防止 scale 为 0
    Returns:
        反量化后的 tensor，shape/dtype/device 与输入一致
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {tuple(weight.shape)}"
    orig_dtype = weight.dtype
    device = weight.device
    w = weight.detach().to(torch.float32)
    out_features, in_features = w.shape
    # 最后一维补齐到 block_size 的整数倍
    pad_size = (block_size - (in_features % block_size)) % block_size
    if pad_size > 0:
        w_padded = F.pad(w, (0, pad_size), mode="constant", value=0.0)
    else:
        w_padded = w
    padded_in_features = w_padded.shape[1]
    num_blocks = padded_in_features // block_size
    # [out_features, num_blocks, block_size]
    w_blocks = w_padded.view(out_features, num_blocks, block_size)
    # 每个 block 一个 scale
    max_abs = w_blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 3.0, min=eps)
    q = torch.round(w_blocks / scale).clamp(-4, 3)
    w_dequant_blocks = q * scale
    # reshape 回去
    w_dequant_padded = w_dequant_blocks.view(out_features, padded_in_features)
    # 去掉 padding
    if pad_size > 0:
        w_dequant = w_dequant_padded[:, :in_features]
    else:
        w_dequant = w_dequant_padded
    return w_dequant.to(device=device, dtype=orig_dtype)


def apply_blockwise_int3_quant_to_named_linears(
    root_module: nn.Module,
    block_size: int = 128,
    verbose: bool = True,
) -> int:
    """
    遍历 root_module 下所有 named_modules，对命中的 nn.Linear.weight 执行：
        float -> block-wise int3 quant -> dequant float
    并原地写回参数。
    同时输出每层量化损失：
        (weight - quantized_weight).abs().pow(2).mean()
    Args:
        root_module: 根模块
        block_size: block-wise 量化块大小，建议 64 或 128
        verbose: 是否打印日志
    Returns:
        被量化的 linear 层数
    """
    count = 0
    total_quant_loss = 0.0
    for full_name, module in root_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_linear_by_name(full_name):
            continue
        with torch.no_grad():
            original_weight = module.weight.data.detach().clone()
            quantized_weight = blockwise_int3_quant_dequant_weight(
                original_weight,
                block_size=block_size,
            )
            quant_loss = compute_quant_loss(original_weight, quantized_weight)
            quant_loss_value = float(quant_loss.item())
            module.weight.data.copy_(quantized_weight)
        count += 1
        total_quant_loss += quant_loss_value
        if verbose:
            logger.warning(
                f"[INT3-QDQ][blockwise] quantized {full_name}, "
                f"shape={tuple(module.weight.shape)}, "
                f"block_size={block_size}, "
                f"quant_loss={quant_loss_value:.8e}"
            )
    if verbose:
        avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
        logger.warning(f"[INT3-QDQ][blockwise] total quantized linear layers: {count}")
        logger.warning(f"[INT3-QDQ][blockwise] average quant_loss: {avg_quant_loss:.8e}")
    return count


def blockwise_int4_quant_dequant_weight(
    weight: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对 2D weight 做 block-wise 对称 int4 量化再反量化。
    默认按最后一维分块，即对每一行的列方向按 block_size 切分。
    对于 shape = [out_features, in_features] 的权重：
      - 每一行独立处理
      - 每个 block 共享一个 scale
      - int4 范围使用 [-8, 7]
    Args:
        weight: 2D tensor, usually [out_features, in_features]
        block_size: 每个 block 的长度，建议 64 或 128
        eps: 防止 scale 为 0
    Returns:
        反量化后的 tensor，shape/dtype/device 与输入一致
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {tuple(weight.shape)}"
    orig_dtype = weight.dtype
    device = weight.device
    w = weight.detach().to(torch.float32)
    out_features, in_features = w.shape
    # 最后一维补齐到 block_size 的整数倍
    pad_size = (block_size - (in_features % block_size)) % block_size
    if pad_size > 0:
        w_padded = F.pad(w, (0, pad_size), mode="constant", value=0.0)
    else:
        w_padded = w
    padded_in_features = w_padded.shape[1]
    num_blocks = padded_in_features // block_size
    # [out_features, num_blocks, block_size]
    w_blocks = w_padded.view(out_features, num_blocks, block_size)
    # 每个 block 一个 scale
    max_abs = w_blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 7.0, min=eps)
    q = torch.round(w_blocks / scale).clamp(-8, 7)
    w_dequant_blocks = q * scale
    # reshape 回去
    w_dequant_padded = w_dequant_blocks.view(out_features, padded_in_features)
    # 去掉 padding
    if pad_size > 0:
        w_dequant = w_dequant_padded[:, :in_features]
    else:
        w_dequant = w_dequant_padded
    return w_dequant.to(device=device, dtype=orig_dtype)


def apply_blockwise_int4_quant_to_named_linears(
    root_module: nn.Module,
    block_size: int = 128,
    verbose: bool = True,
) -> int:
    """
    遍历 root_module 下所有 named_modules，对命中的 nn.Linear.weight 执行：
        float -> block-wise int4 quant -> dequant float
    并原地写回参数。
    同时输出每层量化损失：
        (weight - quantized_weight).abs().pow(2).mean()
    Args:
        root_module: 根模块
        block_size: block-wise 量化块大小，建议 64 或 128
        verbose: 是否打印日志
    Returns:
        被量化的 linear 层数
    """
    count = 0
    total_quant_loss = 0.0
    for full_name, module in root_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_linear_by_name(full_name):
            continue
        with torch.no_grad():
            original_weight = module.weight.data.detach().clone()
            quantized_weight = blockwise_int4_quant_dequant_weight(
                original_weight,
                block_size=block_size,
            )
            quant_loss = compute_quant_loss(original_weight, quantized_weight)
            quant_loss_value = float(quant_loss.item())
            module.weight.data.copy_(quantized_weight)
        count += 1
        total_quant_loss += quant_loss_value
        if verbose:
            logger.warning(
                f"[INT4-QDQ][blockwise] quantized {full_name}, "
                f"shape={tuple(module.weight.shape)}, "
                f"block_size={block_size}, "
                f"quant_loss={quant_loss_value:.8e}"
            )
    if verbose:
        avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
        logger.warning(f"[INT4-QDQ][blockwise] total quantized linear layers: {count}")
        logger.warning(f"[INT4-QDQ][blockwise] average quant_loss: {avg_quant_loss:.8e}")
    return count


def matrixwise_int4_quant_dequant_weight(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对 2D weight 做 matrix-wise 对称 int4 量化再反量化。
    整个矩阵共享一个 scale。

    Args:
        weight: 2D tensor, usually [out_features, in_features]
        eps: minimum scale to avoid divide-by-zero

    Returns:
        Dequantized tensor with same shape / dtype / device as input
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {tuple(weight.shape)}"

    orig_dtype = weight.dtype
    device = weight.device

    w = weight.detach().to(torch.float32)
    max_abs = w.abs().max()
    scale = torch.clamp(max_abs / 7.0, min=eps)

    q = torch.round(w / scale).clamp(-8, 7)
    w_dequant = q * scale

    return w_dequant.to(device=device, dtype=orig_dtype)


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


def apply_matrixwise_int4_quant_to_named_linears(
    root_module: nn.Module,
    verbose: bool = True,
) -> int:
    """
    遍历 root_module 下所有 named_modules，对命中的 nn.Linear.weight 执行：
        float -> int4 quant -> dequant float
    并原地写回参数。
    同时输出每层量化损失：
        (weight - quantized_weight).abs().pow(2).mean()
    Returns:
        被量化的 linear 层数
    """
    count = 0
    total_quant_loss = 0.0
    for full_name, module in root_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_linear_by_name(full_name):
            continue
        with torch.no_grad():
            original_weight = module.weight.data.detach().clone()
            quantized_weight = matrixwise_int4_quant_dequant_weight(original_weight)
            quant_loss = compute_quant_loss(original_weight, quantized_weight)
            quant_loss_value = float(quant_loss.item())
            module.weight.data.copy_(quantized_weight)
        count += 1
        total_quant_loss += quant_loss_value
        if verbose:
            logger.warning(
                f"[INT4-QDQ] quantized {full_name}, "
                f"shape={tuple(module.weight.shape)}, "
                f"mode=matrixwise, "
                f"quant_loss={quant_loss_value:.8e}"
            )
    if verbose:
        avg_quant_loss = total_quant_loss / count if count > 0 else 0.0
        logger.warning(f"[INT4-QDQ] total quantized linear layers: {count}")
        logger.warning(f"[INT4-QDQ] average quant_loss: {avg_quant_loss:.8e}")
    return count