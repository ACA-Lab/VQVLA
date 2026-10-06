# VQVLA OpenVLA LIBERO inference

This directory contains the OpenVLA code and complete-model GPTVQ inference
support. The quantized archives are published separately in the
[`LeoJiang123/VQVLA` Hugging Face repository](https://huggingface.co/LeoJiang123/VQVLA),
under `OpenVLA/quant_weight/`.

## Download the archives

Install the Hugging Face CLI, then run from the VQVLA repository root:

```bash
hf download LeoJiang123/VQVLA \
  OpenVLA/quant_weight/openvla_libero_spatial_gptvq_hdiag_4bit_b256_k256_v2.pt \
  OpenVLA/quant_weight/openvla_libero_spatial_gptvq_hdiag_3bit_b128_k64_v2.pt \
  OpenVLA/quant_weight/openvla_libero_object_gptvq_hdiag_4bit_b256_k256_v2.pt \
  OpenVLA/quant_weight/openvla_libero_object_gptvq_hdiag_3bit_b128_k64_v2.pt \
  OpenVLA/quant_weight/openvla_libero_goal_gptvq_hdiag_4bit_b256_k256_v2.pt \
  OpenVLA/quant_weight/openvla_libero_goal_gptvq_hdiag_3bit_b128_k64_v2.pt \
  OpenVLA/quant_weight/openvla_libero_10_gptvq_hdiag_4bit_b256_k256_v2.pt \
  OpenVLA/quant_weight/openvla_libero_10_gptvq_hdiag_3bit_b128_k64_v2.pt \
  --repo-type model --local-dir .
```

The command stores the files under `OpenVLA/quant_weight/` in the local
checkout. Each archive covers all 982 floating-point parameter tensors (about
7.54 billion values). The execution-state 4bit archives use vector length 2,
256×256 blocks, and codebook size 256. The transition-state 3bit archives use
vector length 2, 128×128 blocks, and codebook size 64.

## Run a LIBERO evaluation

Install from the VQVLA umbrella checkout. The example below assumes you have
already installed CUDA-matched PyTorch 2.2.x in a Python 3.10 environment:

```bash
git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
cd VQVLA
python -m pip install -e OpenVLA
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
python -m pip install -e external/LIBERO
python -m pip install -r OpenVLA/experiments/robot/libero/libero_requirements.txt
python -m pip install --upgrade huggingface_hub
```

Do not clone the upstream OpenVLA repository in place of this VQVLA checkout.
Then, from `VQVLA/OpenVLA`, run for example:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --task_suite_name libero_spatial \
  --gptvq_archive_4bit quant_weight/openvla_libero_spatial_gptvq_hdiag_4bit_b256_k256_v2.pt \
  --gptvq_archive_3bit quant_weight/openvla_libero_spatial_gptvq_hdiag_3bit_b128_k64_v2.pt \
  --routing_threshold 1.15
```

Change `--task_suite_name`, both archive paths, and `--routing_threshold` for
another suite. The first action in each episode uses execution-state 4bit;
subsequent actions use transition-state 3bit when the previous action's XYZ
translation magnitude reaches the selected threshold. The default attention
backend in this fork is PyTorch eager, and the default inference dtype is
float16; compatible installations can select another backend or dtype through
`OPENVLA_ATTN_IMPLEMENTATION` and `OPENVLA_DTYPE`.

## Recorded results

Results below are from seed 42 with ten rollouts per task. The transition-state
ratio counts routed model queries, not episodes.

| LIBERO suite | Threshold | Success | Transition-state ratio |
| --- | ---: | ---: | ---: |
| Spatial | 1.15 | 81/100 | 1.6933% |
| Object | 0.65 | 85/100 | 51.4784% |
| Goal | 1.00 | 77/100 | 8.5965% |
| LIBERO-10 | no passing threshold confirmed | 41/90 before early stop | 1.3106% before early stop |

The LIBERO-10 mixed run stopped early and is not a complete-suite accuracy
result. Its original-policy reference was 54/100; the pure execution-state
4bit run also stopped early at 42/91. See the experiment logs for full
methodology and rejected candidates.
