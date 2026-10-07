"""Blockwise input-Hessian-diagonal VQ helpers for RDT model parameters."""

from __future__ import annotations

from typing import Any

import torch


def _weighted_kmeans_batched(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    codebook_size: int,
    iterations: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent weighted k-means for [blocks, vectors, vector_len]."""
    batch, vector_count, vector_len = vectors.shape
    initial = torch.randint(
        vector_count,
        (batch, codebook_size),
        device=vectors.device,
        generator=generator,
    )
    centers = vectors.gather(
        1, initial.unsqueeze(-1).expand(-1, -1, vector_len)
    ).clone()
    assignments = torch.full(
        (batch, vector_count), -1, dtype=torch.long, device=vectors.device
    )
    weighted_vectors = weights * vectors
    weighted_norm = (weighted_vectors * vectors).sum(dim=-1, keepdim=True)

    for _ in range(iterations):
        distances = (
            weighted_norm
            + torch.bmm(weights, centers.square().transpose(1, 2))
            - 2.0 * torch.bmm(weighted_vectors, centers.transpose(1, 2))
        )
        new_assignments = distances.argmin(dim=-1)
        if torch.equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        scatter_indices = assignments.unsqueeze(-1).expand(-1, -1, vector_len)
        numerator = torch.zeros_like(centers)
        denominator = torch.zeros_like(centers)
        numerator.scatter_add_(1, scatter_indices, weighted_vectors)
        denominator.scatter_add_(1, scatter_indices, weights)
        non_empty = denominator.sum(dim=-1, keepdim=True) > 0
        centers = numerator / denominator.clamp_min(1e-12)
        if (~non_empty.squeeze(-1)).any():
            replacement = torch.randint(
                vector_count,
                (batch, codebook_size),
                device=vectors.device,
                generator=generator,
            )
            replacement_vectors = vectors.gather(
                1, replacement.unsqueeze(-1).expand(-1, -1, vector_len)
            )
            centers = torch.where(non_empty, centers, replacement_vectors)
    return centers, assignments


def pack_indices(indices: torch.Tensor, bits_per_index: int) -> torch.Tensor:
    """Pack fixed-width nonnegative indices into a CPU uint8 byte stream."""
    if bits_per_index <= 0 or bits_per_index > 8:
        raise ValueError("bits_per_index must be in [1, 8]")
    values = indices.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
    if values.numel() and (values.min() < 0 or values.max() >= 1 << bits_per_index):
        raise ValueError("index does not fit the requested bit width")
    if bits_per_index == 8:
        return values.to(torch.uint8)
    bit = torch.arange(bits_per_index, dtype=torch.int64)
    positions = torch.arange(values.numel(), dtype=torch.int64).unsqueeze(1) * bits_per_index + bit
    contributions = ((values.unsqueeze(1) >> bit) & 1) << (positions % 8)
    packed = torch.zeros((values.numel() * bits_per_index + 7) // 8, dtype=torch.int32)
    packed.index_add_(0, (positions // 8).reshape(-1), contributions.reshape(-1).to(torch.int32))
    return packed.to(torch.uint8)


def unpack_indices(packed: torch.Tensor, count: int, bits_per_index: int) -> torch.Tensor:
    """Decode the little-endian bitstream emitted by :func:`pack_indices`."""
    if count < 0 or bits_per_index <= 0 or bits_per_index > 8:
        raise ValueError("invalid count or bit width")
    packed = packed.detach().reshape(-1).to(device="cpu", dtype=torch.uint8)
    if bits_per_index == 8:
        if packed.numel() != count:
            raise ValueError("packed byte count does not match index count")
        return packed.to(torch.long)
    positions = torch.arange(count * bits_per_index, dtype=torch.int64)
    bytes_ = packed.to(torch.int64)[positions // 8]
    bits = (bytes_ >> (positions % 8)) & 1
    bit = torch.arange(bits_per_index, dtype=torch.int64)
    return (bits.reshape(count, bits_per_index) << bit).sum(dim=1)


def quantize_matrix(
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    *,
    device: torch.device | str,
    block_rows: int,
    block_cols: int,
    codebook_size: int,
    vector_len: int = 2,
    kmeans_iters: int = 20,
    damping: float = 0.01,
    block_batch_size: int = 4,
    seed: int = 42,
    row_weights: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Quantize a 2-D matrix, including ragged edge blocks.

    The returned archive stores only packed assignments and codebooks. The
    second result is a CPU reconstruction for verification.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, received {tuple(weight.shape)}")
    rows, columns = weight.shape
    if hessian_diag.shape != (columns,):
        raise ValueError(f"expected hdiag shape ({columns},), got {tuple(hessian_diag.shape)}")
    params = (block_rows, block_cols, codebook_size, vector_len, kmeans_iters, block_batch_size)
    if min(params) <= 0 or damping < 0:
        raise ValueError("quantization dimensions/iterations must be positive and damping nonnegative")
    if block_rows * block_cols % vector_len:
        raise ValueError("full block element count must be divisible by vector_len")
    if codebook_size & (codebook_size - 1):
        raise ValueError("codebook_size must be a power of two for packed assignments")

    code_bits = codebook_size.bit_length() - 1
    source = weight.detach().to(device="cpu").float()
    diagonal = hessian_diag.detach().to(device="cpu", dtype=torch.float32).clamp_min(0)
    diagonal_mean = diagonal.mean().clamp_min(1e-12)
    diagonal = (diagonal + damping * diagonal_mean) / diagonal_mean
    if row_weights is not None:
        row_weights = row_weights.detach().to(device="cpu", dtype=torch.float32).clamp_min(0)
        if row_weights.shape != (rows,):
            raise ValueError(f"expected row_weights shape ({rows},), got {tuple(row_weights.shape)}")
        row_mean = row_weights.mean().clamp_min(1e-12)
        row_weights = (row_weights + damping * row_mean) / row_mean
    reconstructed = source.clone()
    block_shapes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row_start in range(0, rows, block_rows):
        height = min(block_rows, rows - row_start)
        for col_start in range(0, columns, block_cols):
            width = min(block_cols, columns - col_start)
            if height * width % vector_len:
                raise ValueError(
                    f"ragged block {(height, width)} is not divisible by vector_len={vector_len}"
                )
            block_shapes.setdefault((height, width), []).append((row_start, col_start))

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    archive_blocks = []
    for (height, width), coordinates in block_shapes.items():
        vector_count = height * width // vector_len
        for batch_start in range(0, len(coordinates), block_batch_size):
            batch_coords = coordinates[batch_start : batch_start + block_batch_size]
            blocks = torch.stack(
                [source[r : r + height, c : c + width] for r, c in batch_coords]
            ).to(device=device, dtype=torch.float32)
            scalar_weights = torch.stack(
                [
                    diagonal[c : c + width].expand(height, -1)
                    if row_weights is None
                    else row_weights[r : r + height, None]
                    * diagonal[c : c + width][None, :]
                    for r, c in batch_coords
                ]
            ).to(device=device)
            vectors = blocks.contiguous().view(len(batch_coords), vector_count, vector_len)
            vector_weights = scalar_weights.contiguous().view_as(vectors)
            codebooks, assignments = _weighted_kmeans_batched(
                vectors, vector_weights, codebook_size, kmeans_iters, generator
            )
            quantized = codebooks.gather(
                1, assignments.unsqueeze(-1).expand(-1, -1, vector_len)
            ).view_as(blocks)
            for index, (row_start, col_start) in enumerate(batch_coords):
                reconstructed[row_start : row_start + height, col_start : col_start + width] = (
                    quantized[index].detach().cpu()
                )
                archive_blocks.append(
                    {
                        "row_start": row_start,
                        "col_start": col_start,
                        "height": height,
                        "width": width,
                        "vector_count": vector_count,
                        "codebook": codebooks[index].detach().to(device="cpu", dtype=torch.float16),
                        "packed_indices": pack_indices(assignments[index], code_bits),
                    }
                )
            del blocks, scalar_weights, vectors, vector_weights, codebooks, assignments, quantized

    archive = {
        "version": 1,
        "shape": tuple(weight.shape),
        "dtype": str(weight.dtype),
        "block_rows": block_rows,
        "block_cols": block_cols,
        "vector_len": vector_len,
        "codebook_size": codebook_size,
        "bits_per_index": code_bits,
        "blocks": archive_blocks,
    }
    return archive, reconstructed.to(dtype=weight.dtype)


def dequantize_matrix(
    archive: dict[str, Any], *, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """Reconstruct a matrix from its block codebooks and packed indices."""
    rows, columns = archive["shape"]
    output = torch.empty((rows, columns), dtype=dtype, device=device)
    vector_len = int(archive["vector_len"])
    bits = int(archive["bits_per_index"])
    for block in archive["blocks"]:
        count = int(block["vector_count"])
        indices = unpack_indices(block["packed_indices"], count, bits).to(device=device)
        codebook = block["codebook"].to(device=device, dtype=dtype)
        values = codebook.index_select(0, indices).reshape(block["height"], block["width"])
        r, c = int(block["row_start"]), int(block["col_start"])
        output[r : r + block["height"], c : c + block["width"]] = values
    return output


class RdtMixedWeightRouter:
    """Swap 4-bit execution and 3-bit transition matrices without weight copies."""

    def __init__(
        self,
        routed_modules: list[torch.nn.Module],
        routed_parameters: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]],
    ):
        self.routed_modules = routed_modules
        self.routed_parameters = routed_parameters
        self.execution_state = True

    @classmethod
    def load(
        cls,
        components: dict[str, torch.nn.Module],
        archive_4bit_path: str,
        archive_3bit_path: str,
    ) -> "RdtMixedWeightRouter":
        archive_4bit = torch.load(archive_4bit_path, map_location="cpu")
        archive_3bit = torch.load(archive_3bit_path, map_location="cpu")
        if archive_4bit.get("version") != 1 or archive_3bit.get("version") != 1:
            raise ValueError("unsupported RDT VQ archive version")
        layers_4 = archive_4bit.get("layers", {})
        layers_3 = archive_3bit.get("layers", {})
        modules: dict[str, torch.nn.Module] = {}
        parameters: dict[str, torch.nn.Parameter] = {}
        module_weight_ids: set[int] = set()
        for prefix, component in components.items():
            for local_name, module in component.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                    name = f"{prefix}.{local_name}" if local_name else prefix
                    modules[name] = module
                    module_weight_ids.add(id(module.weight))
            for local_name, parameter in component.named_parameters():
                if torch.is_floating_point(parameter) and id(parameter) not in module_weight_ids:
                    name = f"{prefix}.{local_name}" if local_name else prefix
                    parameters[name] = parameter
        expected_names = set(modules) | set(parameters)
        missing_4 = sorted(expected_names - set(layers_4))
        missing_3 = sorted(expected_names - set(layers_3))
        extra_4 = sorted(set(layers_4) - expected_names)
        extra_3 = sorted(set(layers_3) - expected_names)
        if missing_4 or missing_3 or extra_4 or extra_3:
            raise ValueError(
                "archive/module coverage mismatch: "
                f"missing4={missing_4[:5]}, missing3={missing_3[:5]}, "
                f"extra4={extra_4[:5]}, extra3={extra_3[:5]}"
            )

        routed = []
        for index, (name, module) in enumerate(modules.items(), start=1):
            target_shape = tuple(module.weight.shape)
            matrix_shape = (target_shape[0], int(module.weight[0].numel()))
            q4_archive = layers_4.pop(name)
            q3_archive = layers_3.pop(name)
            if tuple(q4_archive["shape"]) != matrix_shape or tuple(q3_archive["shape"]) != matrix_shape:
                raise ValueError(
                    f"archive shape mismatch for {name}: expected {matrix_shape}, "
                    f"got {q4_archive['shape']} and {q3_archive['shape']}"
                )
            q4 = dequantize_matrix(
                q4_archive, dtype=module.weight.dtype, device=module.weight.device
            ).reshape(target_shape)
            q3 = dequantize_matrix(
                q3_archive, dtype=module.weight.dtype, device=module.weight.device
            ).reshape(target_shape)
            module._vq_weight_4bit = q4
            module._vq_weight_3bit = q3
            module.weight.data = module._vq_weight_4bit
            routed.append(module)
            if index % 100 == 0:
                print(f"Loaded mixed VQ weights for {index}/{len(modules)} modules", flush=True)
            del q4_archive, q3_archive, q4, q3

        routed_parameters = []
        for index, (name, parameter) in enumerate(parameters.items(), start=1):
            parameter_shape = tuple(parameter.shape)
            q4_archive = layers_4.pop(name)
            q3_archive = layers_3.pop(name)
            if tuple(q4_archive.get("original_parameter_shape", ())) != parameter_shape or tuple(
                q3_archive.get("original_parameter_shape", ())
            ) != parameter_shape:
                raise ValueError(f"original parameter shape mismatch for {name}")
            if parameter.ndim >= 2:
                matrix_shape = (parameter.numel() // parameter_shape[-1], parameter_shape[-1])
            else:
                padded_width = (
                    (parameter.numel() + int(q4_archive["vector_len"]) - 1)
                    // int(q4_archive["vector_len"])
                ) * int(q4_archive["vector_len"])
                matrix_shape = (1, padded_width)
            if tuple(q4_archive["shape"]) != matrix_shape or tuple(q3_archive["shape"]) != matrix_shape:
                raise ValueError(f"archive shape mismatch for parameter {name}")
            q4_flat = dequantize_matrix(
                q4_archive, dtype=parameter.dtype, device=parameter.device
            ).reshape(-1)
            q3_flat = dequantize_matrix(
                q3_archive, dtype=parameter.dtype, device=parameter.device
            ).reshape(-1)
            q4 = q4_flat[: parameter.numel()].reshape(parameter_shape)
            q3 = q3_flat[: parameter.numel()].reshape(parameter_shape)
            parameter.data = q4
            routed_parameters.append((parameter, q4, q3))
            if index % 100 == 0:
                print(
                    f"Loaded mixed VQ weights for {len(modules) + index}/"
                    f"{len(expected_names)} tensors",
                    flush=True,
                )
            del q4_archive, q3_archive, q4, q3

        if layers_4 or layers_3:
            raise RuntimeError("unconsumed archive entries remain after coverage validation")
        return cls(routed, routed_parameters)

    def set_execution_state(self, execution_state: bool) -> None:
        self.execution_state = bool(execution_state)
        attribute = "_vq_weight_4bit" if self.execution_state else "_vq_weight_3bit"
        for module in self.routed_modules:
            module.weight.data = getattr(module, attribute)
        for parameter, weight_4, weight_3 in self.routed_parameters:
            parameter.data = weight_4 if self.execution_state else weight_3

    @property
    def active_bits(self) -> int:
        return 4 if self.execution_state else 3
