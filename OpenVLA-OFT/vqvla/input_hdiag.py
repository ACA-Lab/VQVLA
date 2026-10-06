"""Bounded input second-moment summaries for GPTVQ weight archives."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _first_tensor(values: tuple[Any, ...]) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            tensor = _first_tensor(tuple(value))
            if tensor is not None:
                return tensor
    return None


class InputHdiagCollector:
    """Accumulate only feature-wise input second moments, never full activations."""

    def __init__(self, modules: dict[str, torch.nn.Module], max_observations: int = 16) -> None:
        if max_observations <= 0:
            raise ValueError("max_observations must be positive")
        self.max_observations = max_observations
        self.total_observations = 0
        self.total_policy_queries = 0
        self.task_observation_counts: list[int] = []
        self.module_summaries: dict[str, dict[str, Any]] = {}
        self.parameters: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

        for root_name, root in modules.items():
            for module_name, module in root.named_modules():
                owner = f"{root_name}.{module_name}" if module_name else root_name
                direct_parameters = tuple(module.parameters(recurse=False))
                if not any(parameter.is_floating_point() for parameter in direct_parameters):
                    continue

                if isinstance(module, torch.nn.Embedding):
                    summary = {
                        "kind": "embedding_rows",
                        "sum": torch.zeros(module.num_embeddings, dtype=torch.float32),
                        "count": 0,
                    }
                    self.module_summaries[owner] = summary

                    def embedding_hook(
                        _module: torch.nn.Module,
                        inputs: tuple[Any, ...],
                        key: str = owner,
                    ) -> None:
                        if self._task_limit_reached():
                            return
                        token_ids = _first_tensor(inputs)
                        if token_ids is None or token_ids.numel() == 0:
                            return
                        rows = token_ids.detach().reshape(-1).to(device="cpu", dtype=torch.long)
                        rows = rows[(rows >= 0) & (rows < self.module_summaries[key]["sum"].numel())]
                        if rows.numel():
                            ones = torch.ones(rows.numel(), dtype=torch.float32)
                            self.module_summaries[key]["sum"].index_add_(0, rows, ones)
                            self.module_summaries[key]["count"] += int(rows.numel())

                    self.handles.append(module.register_forward_pre_hook(embedding_hook))
                else:
                    self.module_summaries[owner] = {"kind": "features", "sum": None, "count": 0}

                    def feature_hook(
                        current: torch.nn.Module,
                        inputs: tuple[Any, ...],
                        key: str = owner,
                    ) -> None:
                        if self._task_limit_reached():
                            return
                        tensor = _first_tensor(inputs)
                        if tensor is None or not tensor.is_floating_point() or tensor.numel() == 0:
                            return
                        if tensor.ndim == 0:
                            return

                        axis = (
                            1
                            if isinstance(
                                current,
                                (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
                            )
                            else -1
                        )
                        rows = tensor.detach().movedim(axis, -1).reshape(-1, tensor.shape[axis])
                        feature_sum = rows.float().square().sum(dim=0).to(device="cpu")
                        summary = self.module_summaries[key]
                        if summary["sum"] is None:
                            summary["sum"] = feature_sum
                        elif summary["sum"].shape == feature_sum.shape:
                            summary["sum"].add_(feature_sum)
                        else:
                            return
                        summary["count"] += int(rows.shape[0])

                    self.handles.append(module.register_forward_pre_hook(feature_hook))

            for parameter_name, parameter in root.named_parameters():
                if not parameter.is_floating_point():
                    continue
                full_name = f"{root_name}.{parameter_name}"
                owner_name = parameter_name.rpartition(".")[0]
                owner = f"{root_name}.{owner_name}" if owner_name else root_name
                self.parameters[full_name] = {
                    "owner": owner,
                    "shape": tuple(parameter.shape),
                }

    def begin_task(self) -> None:
        """Start collecting up to the configured number of queries for one task."""
        self.task_observation_counts.append(0)

    def _task_limit_reached(self) -> bool:
        if not self.task_observation_counts:
            self.begin_task()
        return self.task_observation_counts[-1] >= self.max_observations

    def record_query(self) -> None:
        """Mark one complete policy query for the current task."""
        if not self.task_observation_counts:
            self.begin_task()
        if self.task_observation_counts[-1] < self.max_observations:
            self.total_observations += 1
        self.total_policy_queries += 1
        self.task_observation_counts[-1] += 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, output: str | Path, checkpoint: str, suite: str) -> dict[str, int]:
        self.close()
        if self.total_policy_queries == 0:
            raise RuntimeError("Cannot save hdiag because no policy queries were collected")
        output_path = Path(output)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")

        covered = sum(
            torch.Size(value["shape"]).numel() for value in self.parameters.values()
        )
        fallback = 0
        for value in self.parameters.values():
            summary = self.module_summaries.get(value["owner"])
            if summary is None or summary["count"] <= 0 or summary["sum"] is None:
                fallback += int(torch.Size(value["shape"]).numel())

        payload = {
            "version": 1,
            "method": "input_second_moment_diagonal",
            "model_family": "openvla-oft",
            "checkpoint": checkpoint,
            "suite": suite,
            "observation_count": self.total_observations,
            "policy_query_count": self.total_policy_queries,
            "queries_per_task": self.task_observation_counts,
            "observations_per_task": [
                min(count, self.max_observations) for count in self.task_observation_counts
            ],
            "max_observations_per_task": self.max_observations,
            "module_summaries": self.module_summaries,
            "parameters": self.parameters,
            "coverage": {
                "floating_parameter_tensors": len(self.parameters),
                "floating_parameter_values": covered,
                "uniform_fallback_values": fallback,
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.link(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return payload["coverage"]
