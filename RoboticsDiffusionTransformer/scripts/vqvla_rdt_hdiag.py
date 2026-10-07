"""Bounded input-Hessian diagonal collection for RDT linear layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn


class RdtHdiagCollector:
    """Accumulate per-input-feature second moments without retaining activations."""

    def __init__(self, components: dict[str, nn.Module]):
        self.summaries: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []
        for prefix, component in components.items():
            for module_name, module in component.named_modules():
                if isinstance(module, nn.Linear):
                    input_width = module.in_features
                    kind = "linear"
                elif isinstance(module, nn.Conv2d) and module.groups == 1:
                    input_width = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                    kind = "conv2d"
                elif isinstance(module, nn.Embedding):
                    name = f"{prefix}.{module_name}" if module_name else prefix
                    self.summaries[name] = {
                        "kind": "embedding_tokens",
                        "token_counts": torch.zeros(module.num_embeddings, dtype=torch.int64),
                        "count": 0,
                        "input_width": module.embedding_dim,
                    }
                    self.handles.append(
                        module.register_forward_pre_hook(self._make_embedding_hook(name))
                    )
                    continue
                else:
                    continue
                name = f"{prefix}.{module_name}" if module_name else prefix
                self.summaries[name] = {
                    "sum": torch.zeros(input_width, dtype=torch.float32),
                    "count": 0,
                    "input_width": input_width,
                    "kind": kind,
                }
                self.handles.append(
                    module.register_forward_pre_hook(self._make_hook(name))
                )
        if not self.summaries:
            raise ValueError("No nn.Linear modules found in the supplied components")

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            values = inputs[0].detach()
            summary = self.summaries[name]
            if summary["kind"] == "conv2d":
                module = _module
                if not isinstance(module, nn.Conv2d) or values.ndim != 4:
                    return
                padding = module.padding
                if isinstance(padding, str):
                    if padding == "valid":
                        padding = (0, 0)
                    elif padding == "same":
                        height, width = values.shape[-2:]
                        kernel_h, kernel_w = module.kernel_size
                        stride_h, stride_w = module.stride
                        dilation_h, dilation_w = module.dilation
                        pad_h = max(
                            (int((height + stride_h - 1) / stride_h) - 1) * stride_h
                            + (kernel_h - 1) * dilation_h + 1 - height,
                            0,
                        )
                        pad_w = max(
                            (int((width + stride_w - 1) / stride_w) - 1) * stride_w
                            + (kernel_w - 1) * dilation_w + 1 - width,
                            0,
                        )
                        values = functional.pad(
                            values,
                            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                        )
                        padding = (0, 0)
                    else:
                        raise ValueError(f"unsupported Conv2d padding mode: {padding}")
                values = functional.unfold(
                    values,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=padding,
                    stride=module.stride,
                ).transpose(1, 2)
                values = values.reshape(-1, summary["input_width"])
            if values.ndim == 0 or values.shape[-1] != self.summaries[name]["input_width"]:
                return
            rows = values.reshape(-1, values.shape[-1]).float()
            if rows.numel() == 0:
                return
            row_sum = rows.square().sum(dim=0).cpu()
            self.summaries[name]["sum"].add_(row_sum)
            self.summaries[name]["count"] += int(rows.shape[0])

        return hook

    def _make_embedding_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            token_ids = inputs[0].detach().reshape(-1).to(device="cpu", dtype=torch.long)
            token_ids = token_ids[
                (token_ids >= 0) & (token_ids < self.summaries[name]["token_counts"].numel())
            ]
            if token_ids.numel():
                self.summaries[name]["token_counts"].index_add_(
                    0, token_ids, torch.ones_like(token_ids, dtype=torch.int64)
                )
                self.summaries[name]["count"] += int(token_ids.numel())

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def payload(self, *, checkpoint: str, observations: int, suite: str) -> dict[str, Any]:
        modules: dict[str, dict[str, Any]] = {}
        for name, summary in self.summaries.items():
            count = int(summary["count"])
            if summary["kind"] == "embedding_tokens":
                modules[name] = {
                    "kind": summary["kind"],
                    "token_counts": summary["token_counts"].clone(),
                    "input_rows": count,
                    "observed": count > 0,
                    "input_width": summary["input_width"],
                }
                continue
            modules[name] = {
                "hdiag": (summary["sum"] / max(count, 1)).float(),
                "input_rows": count,
                "observed": count > 0,
                "input_width": summary["input_width"],
                "kind": summary["kind"],
            }
        return {
            "version": 1,
            "method": "input_second_moment_diagonal",
            "checkpoint": checkpoint,
            "suite": suite,
            "observation_count": observations,
            "modules": modules,
            "coverage": {
                "matrix_modules": sum(item["kind"] != "embedding_tokens" for item in modules.values()),
                "embedding_modules": sum(item["kind"] == "embedding_tokens" for item in modules.values()),
                "observed_modules": sum(item["observed"] for item in modules.values()),
            },
        }

    def save(self, path: str | Path, **metadata: Any) -> None:
        output_path = Path(path)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload(**metadata), output_path)
