# OpenPI VQVLA evaluation record

All results use π0.5-LIBERO, seed 42, ten rollouts per LIBERO task, and the
same simulator protocol for baseline and routed models. A candidate was
stopped early only when its accumulated failures made the requested 3
percentage-point constraint mathematically impossible.

| Suite | Original baseline | Selected threshold | Mixed accuracy | Transition-state ratio | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| `libero_spatial` | 99% | 0.55 | 97% | 77.99% | valid |
| `libero_object` | 97% | — | — | — | no tested complete-archive route met the 94% minimum |
| `libero_goal` | 94% | — | — | — | no tested complete-archive route met the 91% minimum |
| `libero_10` | 93% | 0.40 | 91% | 65.06% | valid |

For spatial, lowering the threshold from 0.55 to 0.50 reached five failures
at episode 73, so it could not meet the required 96% minimum. For object and
goal, the pure 4bit candidates and the 0.55 mixed candidates were early-stopped
below their respective minimums; no threshold is recommended for those suites
with these archives. The result is specific to this checkpoint, calibration,
seed, and benchmark version; tune `--routing-threshold` for another setting.

Each listed route uses a complete execution-state 4bit archive and a complete
transition-state 3bit archive. The quantization configuration is described in
the top-level README.
