# VQVLA OpenPI

This component adds complete-model GPTVQ archives and execution-state /
transition-state routing to the upstream π0.5 LIBERO inference path. All
learned floating-point parameters are reconstructed from the selected archive;
the release does not leave the VLM or action expert in full precision.

## Setup

Use Linux, CUDA, Python 3.11 or newer, and one GPU with at least 20 GiB free
memory. From the umbrella repository:

```bash
git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
cd VQVLA/OpenPI
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Install LIBERO as described in [`examples/libero/README.md`](examples/libero/README.md).
Obtain the base π0.5-LIBERO checkpoint using the upstream OpenPI checkpoint
instructions, then download the matching complete archives from
[LeoJiang123/VQVLA](https://huggingface.co/LeoJiang123/VQVLA). Archive files
are stored under `OpenPI/quant_weight/` in the Hugging Face repository. For
example, from the umbrella checkout:

```bash
python -m pip install --upgrade huggingface_hub
hf download LeoJiang123/VQVLA \
  --include 'OpenPI/quant_weight/pi05_libero_spatial_gptvq_hdiag_*' \
  --local-dir .
```

The published execution-state archive is 4bit (vector length 2, 256×256
blocks, codebook size 256); the transition-state archive is 3bit (vector
length 2, 128×128 blocks, codebook size 64).

## Evaluation

Follow the original-policy server/client instructions in
[`examples/libero/README.md`](examples/libero/README.md) to verify the base
checkpoint. For mixed inference, start the policy server with both archives
and a routing threshold:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run scripts/serve_policy.py --env LIBERO --port 8002 \
  --gptvq-archive quant_weight/pi05_libero_spatial_gptvq_hdiag_4bit_b256_k256_v2.pt \
  --gptvq-archive-3bit quant_weight/pi05_libero_spatial_gptvq_hdiag_3bit_b128_k64_v2.pt \
  --routing-threshold 0.55 \
  policy:checkpoint --policy.config=pi05_libero --policy.dir /path/to/pi05_libero
```

In another terminal, run the LIBERO client command from the example README
with `--args.task-suite-name libero_spatial`, `--args.num-trials-per-task 10`,
and `--args.seed 42`. Replace the suite and archive pair for other suites.
The first query of each episode uses execution-state 4bit; subsequent queries
use transition-state 3bit when the previous action chunk's mean XYZ magnitude
meets the threshold. The threshold is a user-adjustable parameter.

With seed 42 and ten rollouts per task, spatial achieved 97% accuracy and a
77.99% transition-state ratio at threshold `0.55` (baseline 99%); LIBERO-10
achieved 91% and 65.06% at `0.40` (baseline 93%). See
[`VQVLA_RESULTS.md`](VQVLA_RESULTS.md) for the full evaluation record and
limits of the tested routes.

To regenerate archives, use `scripts/vqvla_collect_pi0_hdiag.py` followed by
`scripts/vqvla_quantize_pi0.py`; validate outputs with
`scripts/vqvla_verify_pi0_archive.py`. Upstream OpenPI source and its license
notices are retained; this release focuses on inference and quantization.
