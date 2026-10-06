# LIBERO inference notes

This project supports evaluation on LIBERO Spatial, Object, Goal, and LIBERO-10.
The benchmark package is installed separately from this repository. Follow the
commands in [SETUP.md](SETUP.md), then verify the CLI with
`python experiments/robot/libero/run_libero_eval.py --help`.

## Assets and checkpoints

The first run downloads the fine-tuned checkpoint selected by
`--pretrained_checkpoint`, for example
`moojink/openvla-7b-oft-finetuned-libero-spatial`. The four model IDs are
listed in [README.md](README.md). Make sure you have accepted any model-gating
terms on Hugging Face before starting a long evaluation.

LIBERO also needs its benchmark task assets. If the installed LIBERO version
does not provide them automatically, follow the dataset/assets setup documented
by the [LIBERO project](https://github.com/Lifelong-Robot-Learning/LIBERO).
The RLDS training datasets are not required for inference.

## Evaluation

Run the mixed execution-state 4bit / transition-state 3bit example in the main
[README](README.md#run-a-mixed-libero-evaluation). Change the checkpoint,
`--task_suite_name`, two archive paths, and `--routing_threshold` together so
they all refer to the same suite. The script defaults to 50 trials per task;
the published results use 10 trials per task and seed 42.

For a bounded smoke test, use `--num_trials_per_task 1` and
`--early_stop_failures 1`. This is only an installation check, not a meaningful
accuracy measurement. Evaluation creates local logs; video recording and
Weights & Biases logging are disabled in the documented command.

## Collect input-Hessian calibration data and re-quantize

Calibration is optional for inference and is not needed to download or load the
published archives. To collect an input-Hessian diagonal, run the raw policy
(omit both GPTVQ archive flags) and choose a new output path:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 --seed 42 \
  --hdiag_output /path/to/new/spatial-input-hdiag.pt \
  --hdiag_observations 16 --save_video false --use_wandb false
```

The collector saves in a `finally` block, so it can save collected observations
even if evaluation exits with an error. Do not point it at an existing file; the
program refuses to overwrite. Use the resulting suite/checkpoint-specific
calibration with `scripts/vqvla_quantize_oft.py`. The exporter documents its
arguments via `--help`, limits CPU threads, validates full parameter coverage,
and refuses to overwrite output archives. Quantization can require substantial
host RAM and disk space; inspect free resources before starting it.
