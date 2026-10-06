# VQVLA OpenVLA

This component adds complete-model GPTVQ archives and execution-state /
transition-state routing to OpenVLA's LIBERO inference pipeline. Both archive
types cover all 982 floating-point parameter tensors; checkpoints and VQVLA
archives are hosted separately from the source code.

## Start here

1. Clone the umbrella repository with its submodules:

   ```bash
   git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
   cd VQVLA
   ```

2. Install CUDA-matched PyTorch 2.2.x in a Python 3.10 environment, then
   install OpenVLA and LIBERO:

   ```bash
   python -m pip install -e OpenVLA
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
   python -m pip install -e external/LIBERO
   python -m pip install -r OpenVLA/experiments/robot/libero/libero_requirements.txt
   python -m pip install --upgrade huggingface_hub
   ```

3. Download the suite's complete execution-state 4bit and transition-state
   3bit archives from
   [LeoJiang123/VQVLA](https://huggingface.co/LeoJiang123/VQVLA). They are
   stored under `OpenVLA/quant_weight/` in the model repository.

4. Follow [`VQVLA.md`](VQVLA.md) for exact download commands, evaluation
   options, thresholds, and measured results. The guide also explains how to
   run the base model and compare it with mixed inference.

Execution-state archives use vector length 2, 256×256 blocks, and codebook
size 256. Transition-state archives use vector length 2, 128×128 blocks, and
codebook size 64. In mixed evaluation, the first action of each episode uses
execution-state 4bit; later actions select transition-state 3bit according to
the previous action's XYZ translation magnitude. Thresholds are configurable
for other checkpoints and evaluation settings.

The source retains upstream OpenVLA/Prismatic components needed by model
loading and evaluation, along with their license notices. Training code is not
part of the VQVLA workflow; see the upstream project for its training guides.
