# OpenVLA-OFT inference setup

These instructions install the VQVLA OpenVLA-OFT inference code from the
multi-model repository. They do not install the original training pipeline.

## System requirements

- Linux x86_64, Python 3.10, and a CUDA-capable NVIDIA GPU.
- CUDA-compatible PyTorch. Install the wheel matching the host driver/toolkit
  using the [PyTorch install selector](https://pytorch.org/get-started/locally/).
- For mixed 4bit/3bit inference, allow roughly 40 GiB of GPU memory for both
  model instances. CPU and system-memory needs depend on checkpoint loading.

## Install

```bash
git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
cd VQVLA/OpenVLA-OFT
conda create -n vqvla-oft python=3.10 -y
conda activate vqvla-oft
python -m pip install --upgrade pip
# Install the CUDA-matched torch/torchvision first (see link above).
python -m pip install -e .
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
python -m pip install -e external/LIBERO
python -m pip install -r experiments/robot/libero/libero_requirements.txt
python -m pip install --upgrade huggingface_hub
```

The evaluation imports LIBERO from the Python environment; the optional
`external/LIBERO` checkout is used to install that package. If `external/LIBERO`
already exists, reuse it rather than cloning over it. LIBERO may also require
its task assets; see [LIBERO setup](LIBERO.md).

The public VQVLA archives are fetched separately from
[LeoJiang123/VQVLA](https://huggingface.co/LeoJiang123/VQVLA). Base and
fine-tuned OpenVLA-OFT checkpoints are fetched from the model IDs listed in
the main README. Hugging Face authentication may be required for gated models.

## Quick environment check

From `VQVLA/OpenVLA-OFT`:

```bash
python -c 'import torch, transformers, libero; print("torch", torch.__version__, "CUDA", torch.cuda.is_available())'
python experiments/robot/libero/run_libero_eval.py --help
```

For lower CPU load, set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and
`OPENBLAS_NUM_THREADS=1` when running evaluation. Set `CUDA_VISIBLE_DEVICES`
to a single GPU if the machine has multiple GPUs.
