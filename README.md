# VQVLA

This repository collects multiple vision-language-action models. Each model is
kept in its own top-level directory.

## Models

- [Isaac-GR00T](Isaac-GR00T/README.md) — NVIDIA Isaac GR00T N1.7, including
  quantized LIBERO inference support.
- [OpenPI](OpenPI/README.md) — OpenPI, including GPTVQ LIBERO inference support.
- [OpenVLA](OpenVLA/README.md) — OpenVLA, including GPTVQ LIBERO inference support.
- [OpenVLA-OFT](OpenVLA-OFT/README.md) — OpenVLA-OFT, including complete-model
  GPTVQ archives and LIBERO execution-state / transition-state inference.

Clone with submodules before following a model-specific README:

```bash
git clone --recurse-submodules https://github.com/ACA-Lab/VQVLA.git
cd VQVLA
```

Then follow the README in the model directory you want to use.
