#!/usr/bin/env python3
"""Save a small, deterministic set of representative LIBERO observations.

This is calibration input only: it writes RGB images and task prompts, never
model activations or a dense Hessian.  The companion hdiag script reduces
these samples to bounded per-input-feature second moments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from libero.libero import benchmark

# Prefer this clean work tree over any editable OpenVLA installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    samples: list[dict[str, object]] = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        try:
            for episode_id, initial_state in enumerate(initial_states):
                env.reset()
                observation = env.set_init_state(initial_state)
                samples.append(
                    {
                        "suite": args.suite,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "task_description": task_description,
                        "image": get_libero_image(observation, 224),
                    }
                )
                if len(samples) >= args.max_samples:
                    break
        finally:
            env.close()
        if len(samples) >= args.max_samples:
            break

    if not samples:
        raise RuntimeError(f"No calibration samples were produced for {args.suite}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, samples=np.asarray(samples, dtype=object))
    print(f"Saved {len(samples)} {args.suite} calibration samples to {args.output}")


if __name__ == "__main__":
    main()
