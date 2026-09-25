#!/usr/bin/env python3
"""Collect a small, deterministic set of raw LIBERO observations for GPTVQ calibration.

Run this script with ``examples/libero/.venv/bin/python``.  It intentionally
does not load a policy or step the environment: the selected task initial
states provide representative images, robot state, and language instructions
without an evaluation rollout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a scalar-last quaternion without modifying the simulator array."""
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator


def create_environment(task: object, resolution: int, seed: int) -> OffScreenRenderEnv:
    task_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    environment = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    environment.seed(seed)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-suite", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_suite <= 0:
        raise ValueError("--samples-per-suite must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    samples: list[dict[str, object]] = []
    benchmark_map = benchmark.get_benchmark_dict()
    for suite_name in SUITES:
        suite = benchmark_map[suite_name]()
        task_ids = np.linspace(0, suite.n_tasks - 1, num=min(args.samples_per_suite, suite.n_tasks), dtype=int)
        for task_id in task_ids:
            task = suite.get_task(int(task_id))
            initial_states = suite.get_task_init_states(int(task_id))
            environment = create_environment(task, 256, args.seed)
            try:
                environment.reset()
                observation = environment.set_init_state(initial_states[0])
                base_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                wrist_image = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
                samples.append(
                    {
                        "suite": suite_name,
                        "task_id": int(task_id),
                        "image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(base_image, args.resolution, args.resolution)
                        ),
                        "wrist_image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_image, args.resolution, args.resolution)
                        ),
                        "state": np.concatenate(
                            (
                                observation["robot0_eef_pos"],
                                quat_to_axisangle(observation["robot0_eef_quat"]),
                                observation["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32),
                        "prompt": str(task.language),
                    }
                )
                print(f"Collected {suite_name} task {task_id}", flush=True)
            finally:
                environment.close()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, samples=np.asarray(samples, dtype=object))
    print(f"Saved {len(samples)} calibration observations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
