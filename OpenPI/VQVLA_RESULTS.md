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

## Additional threshold probes (2026-09-28)

The evaluator was rerun without replay-video output and with a seven-failure
early-stop guard. These probes use the same seed, checkpoint, and complete
archives as the table above:

| Suite | Threshold | Episodes / successes | Transition-state ratio | Outcome |
| --- | ---: | ---: | ---: | --- |
| `libero_object` | 0.80 | 20 / 13 | 31.04% | rejected (early stop) |
| `libero_object` | 1.20 | 20 / 13 | 0.00% | rejected (early stop) |
| `libero_goal` | 0.80 | 57 / 50 | 43.53% | rejected (early stop) |
| `libero_goal` | 1.20 | 54 / 47 | 2.67% | rejected (early stop) |

Both Object probes reached seven failures before 100 episodes, so increasing
the threshold cannot recover the requested accuracy under this protocol. Both
Goal probes likewise reached seven failures before 100 episodes; the higher
threshold only reduced transition-state use and did not recover accuracy.

Each listed route uses a complete execution-state 4bit archive and a complete
transition-state 3bit archive. The quantization configuration is described in
the top-level README.
