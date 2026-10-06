# VQVLA OpenVLA-OFT

This directory provides OpenVLA-OFT inference with complete-model GPTVQ
archives and execution-state / transition-state routing. The archives cover
all 1,002 floating-point parameter tensors, including the VLM, action head,
and proprio projector. Checkpoints and quantized weights are hosted separately
from the code.

## Tested configuration and results

The recommended thresholds below were evaluated with seed 42 and 10 trials per
task. The transition-state ratio is the share of policy queries routed to 3bit.

| LIBERO suite | Threshold | Success | Transition-state 3bit queries |
| --- | ---: | ---: | ---: |
| Spatial | 0.79 | 98/100 (98.0%) | 54.13% |
| Object | 0.00 | 98/100 (98.0%) | 94.52% |
| Goal | 0.55 | 96/100 (96.0%) | 60.47% |
| LIBERO-10 | 0.00 | 92/100 (92.0%) | 97.31% |

Thresholds are adjustable with `--routing_threshold`; a different checkpoint,
simulator version, or seed can change the result. The first policy query of
each episode always uses execution-state 4bit. Subsequent queries use
transition-state 3bit when the previous action chunk's XYZ translation
magnitude reaches the threshold.

## Requirements and installation

Use Linux, Python 3.10, and one CUDA-capable NVIDIA GPU. Pure inference uses
one model instance; mixed inference keeps both quantized model instances
resident, so use a GPU with sufficient memory (40 GiB or more recommended).
Install CUDA-matched PyTorch wheels using the [official selector](https://pytorch.org/get-started/locally/).

```bash
git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
cd VQVLA/OpenVLA-OFT
conda create -n vqvla-oft python=3.10 -y
conda activate vqvla-oft
python -m pip install --upgrade pip
python -m pip install -e .
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
python -m pip install -e /tmp/LIBERO
python -m pip install -r experiments/robot/libero/libero_requirements.txt
python -m pip install -U huggingface_hub
```

The first evaluation downloads the corresponding fine-tuned checkpoint from
Hugging Face (`moojink/openvla-7b-oft-finetuned-libero-*`). If you prefer a
local copy, pass its path to `--pretrained_checkpoint`. See
[`SETUP.md`](SETUP.md) and [`LIBERO.md`](LIBERO.md) for environment details.

## Download weights

The VQVLA model repository is
[LeoJiang123/VQVLA](https://huggingface.co/LeoJiang123/VQVLA). Download only
the suite you plan to evaluate; each suite has one execution-state 4bit and
one transition-state 3bit archive.

```bash
# Spatial
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-Spatial/openvla_oft_libero_spatial_4bit_b256x256_k256_v3.pt --local-dir .vqvla-weights
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-Spatial/openvla_oft_libero_spatial_3bit_b128x128_k64_v3.pt --local-dir .vqvla-weights

# Object
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-Object/openvla_oft_libero_object_4bit_b256x256_k256_v3.pt --local-dir .vqvla-weights
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-Object/openvla_oft_libero_object_3bit_b128x128_k64_v3.pt --local-dir .vqvla-weights

# Goal
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-Goal/openvla_oft_libero_goal_4bit_b256x256_k256_v3.pt --local-dir .vqvla-weights
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-Goal/openvla_oft_libero_goal_3bit_b128x128_k64_v3.pt --local-dir .vqvla-weights

# LIBERO-10
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-10/openvla_oft_libero_10_4bit_b256x256_k256_v3.pt --local-dir .vqvla-weights
hf download LeoJiang123/VQVLA OpenVLA-OFT/LIBERO-10/openvla_oft_libero_10_3bit_b128x128_k64_v3.pt --local-dir .vqvla-weights
```

Each archive pair is about 6.7 GiB total. The downloads are placed below
`.vqvla-weights/OpenVLA-OFT/<suite>/`.

## Run a mixed LIBERO evaluation

From this directory, select the matching suite, archive paths, and tested
threshold from the table above. This example runs Spatial with bounded CPU
threads, one visible GPU, videos disabled, and 10 trials per task:

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 10 --seed 42 \
  --gptvq_archive_4bit .vqvla-weights/OpenVLA-OFT/LIBERO-Spatial/openvla_oft_libero_spatial_4bit_b256x256_k256_v3.pt \
  --gptvq_archive_3bit .vqvla-weights/OpenVLA-OFT/LIBERO-Spatial/openvla_oft_libero_spatial_3bit_b128x128_k64_v3.pt \
  --routing_threshold 0.79 --save_video false --use_wandb false
```

Replace the checkpoint, `--task_suite_name`, both archive paths, and threshold
for another suite. Use `--gptvq_archive_4bit` alone to test execution-state
4bit or `--gptvq_archive_3bit` alone to test pure transition-state 3bit.

## Re-quantize or validate archives

The input-hdiag calibration is suite- and checkpoint-specific. Follow the
collection instructions in [`LIBERO.md`](LIBERO.md), then use
`scripts/vqvla_quantize_oft.py` to export the complete model. The published
4bit setting is vector length 2, 256×256 blocks, codebook size 256; the 3bit
setting is vector length 2, 128×128 blocks, codebook size 64. The exporter
checks parameter coverage and refuses to overwrite existing archives.
