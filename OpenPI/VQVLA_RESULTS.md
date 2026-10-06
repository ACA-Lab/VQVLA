# OpenPI mixed-inference results

All reported runs use the π0.5-LIBERO checkpoint, seed 42, ten rollouts per
task, and the same evaluation protocol for each suite. Results are for mixed
execution-state 4bit / transition-state 3bit inference only.

| LIBERO suite | Threshold | Accuracy | Transition-state ratio |
| --- | ---: | ---: | ---: |
| Spatial | 0.55 | 97% | 77.99% |
| Object | — | No selected result | — |
| Goal | — | No selected result | — |
| LIBERO-10 | 0.40 | 91% | 65.06% |

Thresholds are adjustable through `--routing-threshold`. Results can vary with
the checkpoint, simulator version, seed, and evaluation protocol. See the
[OpenPI README](README.md) for setup and evaluation commands.
