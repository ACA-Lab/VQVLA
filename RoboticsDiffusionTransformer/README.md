# VQVLA RoboticsDiffusionTransformer

This is the RDT component of [VQVLA](https://github.com/ACA-Lab/VQVLA). It
provides complete-model GPTVQ-Hdiag quantization and mixed-precision inference
for the ManiSkill tasks PegInsertionSide, PickCube, StackCube, PlugCharger,
and PushCube. The quantized archive covers every Linear, Conv2d, and Embedding
weight in the RDT policy, SigLIP vision encoder, and T5 text encoder, plus
standalone rank-2-or-higher learned weight/embedding parameters. Biases and
one-dimensional normalization parameters remain at their original precision.

The execution-state archive uses 4bit VQ (vector length 2, 256×256 blocks,
codebook size 256). The transition-state archive uses 3bit VQ (vector length
2, 128×128 blocks, codebook size 64). Input-Hessian-diagonal statistics are
collected from the five benchmark tasks; unobserved parameters use the
documented uniform-weight fallback.

For methodological precision, these archives use diagonal-input-Hessian-
weighted block K-means (the hdiag block-VQ workflow used for GR00T). They do
not use GPTQ's sequential full-Hessian error-compensation path.

## Install

Use Linux, Python 3.10, an NVIDIA GPU, and enough free storage for the RDT
checkpoint, T5/SigLIP encoders, and quantized archives. The tested environment
uses PyTorch 2.0.1 / torchvision 0.15.2 with CUDA 11.7 and ManiSkill 3.0.0b21.
Install the matching PyTorch wheels for your driver and CUDA version using the
[official previous-version instructions](https://pytorch.org/get-started/previous-versions/),
then install the VQVLA dependencies:

```bash
git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
cd VQVLA/RoboticsDiffusionTransformer
conda create -n vqvla-rdt python=3.10 -y
conda activate vqvla-rdt
python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
  --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements-vqvla.txt
```

Configure Vulkan as described in the
[ManiSkill installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan).
The first model load downloads the external T5 and SigLIP encoders from Hugging
Face. To use local copies, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` after placing them in the standard Hugging Face cache.

## Download checkpoints and quantized weights

Download the ManiSkill RDT checkpoint and the VQVLA archives. The RDT files
are stored under this model's directory in the VQVLA Hugging Face repository.
Run the archive download from `VQVLA/RoboticsDiffusionTransformer`; the
`--local-dir ..` destination recreates the paths expected by the commands.

```bash
python -m pip install --upgrade huggingface_hub
hf download robotics-diffusion-transformer/maniskill-model \
  rdt/mp_rank_00_model_states.pt --local-dir checkpoints/base

hf download LeoJiang123/VQVLA \
  RoboticsDiffusionTransformer/quant_weight/rdt_maniskill_4bit_b256_k256_kmeanspp_iter100_v5_weights-only.pt \
  RoboticsDiffusionTransformer/quant_weight/rdt_maniskill_3bit_b128_k64_kmeanspp_iter100_v5_weights-only.pt \
  --local-dir ..
```

The model checkpoint is not included in this source tree. Keep it and all
downloaded archives outside version control.

## Run an evaluation

Start with the original, unquantized checkpoint to establish a baseline. This
example runs 100 episodes on one GPU and caps CPU thread use:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m eval_sim.eval_rdt_maniskill \
  --pretrained_path checkpoints/base/rdt/mp_rank_00_model_states.pt \
  --env-id StackCube-v1 --obs-mode rgb --num-traj 100 --random_seed 0 \
  --sim-backend gpu
```

Run mixed inference with the matching archive pair. The threshold controls the
existing TCP-height state-selection rule; it is a command-line option, so you
can tune it for your checkpoint and evaluation setup.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m eval_sim.eval_rdt_maniskill \
  --pretrained_path checkpoints/base/rdt/mp_rank_00_model_states.pt \
  --env-id StackCube-v1 --obs-mode rgb --num-traj 100 --random_seed 0 \
  --sim-backend gpu \
  --vq-4bit-archive quant_weight/rdt_maniskill_4bit_b256_k256_kmeanspp_iter100_v5_weights-only.pt \
  --vq-3bit-archive quant_weight/rdt_maniskill_3bit_b128_k64_kmeanspp_iter100_v5_weights-only.pt \
  --transition-height-threshold 0.12 --vq-mode mixed
```

The evaluator prints task success rate and the share of policy queries routed
to transition-state 3bit. `--vq-mode 4bit` and `--vq-mode 3bit` are available
for debugging; the reported VQVLA benchmark results will cover mixed inference
only. Available task IDs are `PegInsertionSide-v1`, `PickCube-v1`,
`StackCube-v1`, `PlugCharger-v1`, and `PushCube-v1`.
Threshold candidates can be bounded with `--stop-after-failures N`; the
reported accuracy uses the number of completed episodes if evaluation stops
early.

## Recreate the archives

Collect bounded hdiag observations on all five tasks, then export separate
4bit and 3bit archives. Use the commands from the repository root; change the
task ID for each calibration run. The exporter requires all five files and
refuses to overwrite an existing output.

```bash
mkdir -p quant_weight/intermediate
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m eval_sim.eval_rdt_maniskill \
  --pretrained_path checkpoints/base/rdt/mp_rank_00_model_states.pt \
  --env-id StackCube-v1 --num-traj 1 --sim-backend gpu \
  --collect-hdiag-output quant_weight/intermediate/rdt_hdiag_StackCube-v1.pt \
  --hdiag-max-queries 8

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.vqvla_quantize_rdt \
  --checkpoint checkpoints/base/rdt/mp_rank_00_model_states.pt \
  --hdiag-dir quant_weight/intermediate --bits 4 \
  --output quant_weight/rdt_maniskill_4bit_b256_k256_kmeanspp_iter100_v5_weights-only.pt

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.vqvla_quantize_rdt \
  --checkpoint checkpoints/base/rdt/mp_rank_00_model_states.pt \
  --hdiag-dir quant_weight/intermediate --bits 3 \
  --output quant_weight/rdt_maniskill_3bit_b128_k64_kmeanspp_iter100_v5_weights-only.pt
```

The collector is task-specific: repeat it for the other four task IDs before
exporting. Keep CPU threads bounded and use one visible GPU for calibration,
quantization, and evaluation.

Run the CPU-only quantization/router checks with:

```bash
python -m unittest discover -s tests -v
```

These tests cover packed assignments, ragged edge blocks, routing for module
weights and standalone embedding matrices, and preservation of bias,
normalization, vector, and scalar parameters.

## Results

The results below use seed 0 and 100 episodes where the mixed run completed.
“Transition-state ratio” is the share of policy queries routed to 3bit.

| ManiSkill task | Raw baseline | Threshold | Mixed success | Transition-state ratio | Finding |
|---|---:|---:|---:|---:|---|
| PushCube-v1 | 95% | 0.12 | 98% | 15.46% | Meets the 3pp loss limit; thresholds 0.10, 0.05, and -1.0 produced the same observed result and ratio. |
| PlugCharger-v1 | 1% | 0.12 | 0% | 2.00% | 1pp below baseline; threshold -1.0 was also unchanged. |
| PickCube-v1 | 79% | 0.12 | 45% | 3.18% | Does not meet the accuracy limit. |
| StackCube-v1 | 75% | 0.00 / 0.12 / 0.20 | Early-stopped screening only | Not recorded | No candidate could still reach the required 73/100 success rate: 61/90, 63/93, and 63/92 respectively. These are not complete accuracy scores. |

No single threshold currently meets the accuracy requirement across all tasks.
The StackCube candidates were early-stopped as soon as the remaining episodes
could no longer raise the result to 73/100. Those logs are documented in the
project handoff; they must not be interpreted as 100-episode scores. Threshold
values are user-adjustable via `--transition-height-threshold`.
