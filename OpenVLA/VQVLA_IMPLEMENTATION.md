# OpenVLA VQVLA implementation notes

This clean work tree is an isolated implementation area for the OpenVLA part
of VQVLA.  It was made from the upstream `HEAD` using `git archive`, so it has
no original checkpoint, old experimental weight directory, or local source
edits from `/home/haozhe.jiang/openvla`.

## Safety contract

Every model operation must use one GPU and bounded host parallelism:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONPATH=/home/haozhe.jiang/openvla-vqvla-work:/home/haozhe.jiang/openvla/LIBERO
```

Use `nice -n 15`.  Do not run a second evaluation, hdiag pass, or quantizer
while a quantizer is active.  Check `nvidia-smi`, `free -h`, and `df -h` before
each heavy invocation.  Quantization writes atomically only after all tensors
are processed and refuses to start below the configured free-space floor.

## Method and coverage

The calibration collector saves at most 16 deterministic LIBERO initial-state
images and task prompts for one suite.  The hdiag collector runs normal OpenVLA
action inference and records only feature-wise input second moments and
embedding token counts.  It does not keep dense Hessians or activation batches.

`vqvla_quantize_openvla.py` traverses every safetensors shard listed in
`model.safetensors.index.json`; it refuses a hdiag/source-name mismatch.  Each
floating tensor is encoded into codebooks and packed assignments.  No original
floating tensor is put in the archive.  The spatial smoke test established
coverage of 982 tensors / 7,541,237,184 values.

The mandatory configurations are:

| archive | blocks | codebook | vector length |
| --- | --- | --- | --- |
| execution-state 4bit | 256 x 256 | 256 | 2 |
| transition-state 3bit | 128 x 128 | 64 | 2 |

## Portable inference backend

The local environment has neither FlashAttention2 nor a Transformers/PyTorch
combination that permits SDPA or BF16 eager masking.  `openvla_utils.py`
therefore reads `OPENVLA_ATTN_IMPLEMENTATION` (default `eager`) and
`OPENVLA_DTYPE` (default `float16`).  A compatible deployment can opt into
FlashAttention2 and BF16 through those environment variables.

## Per-suite command template

Set `CHECKPOINT` to the local snapshot directory for the selected suite, then:

```bash
python scripts/vqvla_collect_libero_calibration.py \
  --suite libero_spatial --max-samples 16 \
  --output quant_weight/intermediate/libero_spatial_calibration.npz

python scripts/vqvla_collect_openvla_hdiag.py \
  --checkpoint "$CHECKPOINT" --suite libero_spatial \
  --observations quant_weight/intermediate/libero_spatial_calibration.npz \
  --output quant_weight/intermediate/libero_spatial_hdiag.pt --cpu-threads 2

python scripts/vqvla_quantize_openvla.py \
  --checkpoint "$CHECKPOINT" --hdiag quant_weight/intermediate/libero_spatial_hdiag.pt \
  --output quant_weight/openvla_libero_spatial_gptvq_hdiag_4bit_b256_k256_v2.pt \
  --bits 4 --block-rows 256 --block-cols 256 --codebook-size 256 \
  --vector-length 2 --kmeans-iters 20 --block-batch-size 2 --cpu-threads 2
```

For 3bit, change the output suffix, `--bits 3`, both block dimensions to 128,
and `--codebook-size 64`.

After each export, run `scripts/vqvla_verify_archive.py` with expected count
`982` and expected values `7541237184`.  It checks format, bit width, exact
coverage, and a packed-index reconstruction without loading a policy model.

## Evaluation routing

`run_libero_eval.py` accepts `--gptvq_archive_4bit`,
`--gptvq_archive_3bit`, and `--routing_threshold`.  Supplying only the 4bit
archive evaluates a complete execution-state 4bit model.  Supplying both
archives constructs two independent complete models.  At the start of each
episode the router chooses 4bit; afterward it chooses transition-state 3bit
when the previous unnormalized action's XYZ L2 magnitude is at least the
threshold.  It writes query counts, the transition-state ratio, and metric
percentiles to the evaluation log.  The threshold is deliberately exposed to
users; it must be selected separately from measured baseline and mixed-policy
success rates.

The initial-action rule is explicit in code: an absent previous action cannot
meet the threshold.  This avoids treating its bookkeeping value (`inf`) as a
real robot state and ensures every episode is bootstrapped by the complete
execution-state 4bit model.

Evaluation videos are disabled by default through `--save_rollout_videos
False`, which keeps repeated threshold sweeps from consuming system storage.

## Current state

Read the cross-model handoff log at `/home/haozhe.jiang/VQVLA_HANDOFF.md`
before resuming work.  It contains the active process/output, measured resource
limits, OpenPI completion state, and GitHub/Hugging Face credential blocker.

As of 2026-09-25 UTC, the spatial original-policy baseline is the only active
heavy operation.  It was a deterministic seed-42, 10-rollout-per-task run with
no video capture; its console log is `/tmp/openvla_spatial_original_baseline.log`.
It completed at 82/100 (82%), so a spatial routed candidate must reach at least
79%.  Its post-completion `EGL_NOT_INITIALIZED` destructor warnings appeared
after the final score and do not affect rollout validity.  A pure complete-4bit
evaluation was started next (log:
`/tmp/openvla_spatial_pure4bit_retry.log`) before any routed evaluation.  Its
loader confirmed exact complete coverage of 982 tensors / 7,541,237,184 values;
the hdiag source counts were embedding rows 131,334,144, input features
7,392,366,464, input channels 1,279,488, and uniform fallback 16,257,088.
That evaluation completed at **79/100 = 79%**.  Against the original-policy
baseline of 82%, this exactly meets (but does not exceed) the maximum
three-percentage-point-loss floor.  The evaluator released GPU0 after the
score; its subsequent EGL destructor messages are shutdown-only warnings.
The next experiment is a serialized two-complete-model routed evaluation that
will report both success and transition-state 3bit usage.

Before committing to a 100-rollout threshold measurement, a one-rollout-per-
task routing probe may be used to observe the actual action-magnitude p50/p90
and the resulting transition-state ratio.  Such a probe is only a threshold-
range diagnostic; it is never used as an accuracy result.

For the spatial suite, the completed seed-42 probe at threshold 0.55 selected
transition-state 3bit for 903 of 1,697 queries (53.21%), with prior-action
translation metric p50=0.6003 and p90=0.9928.  It establishes 0.55 as the
first formal mixed-evaluation threshold, not a performance result.

The evaluator writes a routing checkpoint after every completed task as well
as the final summary.  This makes the observed transition-state ratio
recoverable if a candidate is safely stopped once it can no longer meet its
predeclared success floor.

Use `--max_failures 22` for the spatial suite when enforcing its 79/100 floor:
the evaluator stops after the 22nd failure, writes the partial task result,
routing checkpoint, and final routing summary, then releases the model.  This
is preferable to externally signalling a simulator process.

For an evaluator started before that option was available,
`scripts/vqvla_monitor_early_stop.sh <pid> <log> 22` polls only the flushed
text log every 30 seconds and sends `SIGTERM` to that exact PID once its
failure budget is exhausted.  It consumes no GPU and negligible host resources.
