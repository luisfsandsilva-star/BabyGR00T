# BabyGR00T: Distilling GR00T N1.5 with TRM + nanoLLaVA-1B

**BabyGR00T** is a compact, deployable variant of the **GR00T N1.5** foundational robotic model.  
It integrates **Samsung SAIT’s Tiny Recursive Model (TRM)** for efficient reasoning and **nanoLLaVA-1B** for lightweight multimodal perception, enabling real-time operation on constrained robotic hardware.

---

## Overview

Foundational robotic models such as GR00T N1.5, RT-X, and OpenVLA unify perception, reasoning, and control, but their high computational cost prevents on-device use.  
BabyGR00T achieves comparable generalization with a fraction of the parameters through:

- **Structured knowledge distillation** from the GR00T N1.5 teacher  
- **Recursive reasoning core (TRM)** that iteratively refines internal states  
- **Compact vision-language grounding** using nanoLLaVA-1B  

The system targets a balance between performance, speed, and deployability.

---

## Architecture

![BabyGR00T Architecture](docs/figs/babygr00t_arch.png)

### Components

**nanoLLaVA-1B (Perception front-end)**  
A ~1B-parameter vision-language model based on a compact Qwen-style LLM and a SigLIP vision encoder.  
It provides efficient image–text grounding for scene understanding, captioning, and robotic instruction following with minimal memory footprint.

**TRM (Tiny Recursive Model)**  
A ~140M-parameter recursive reasoning model developed by Samsung SAIT Montréal.  
It updates a latent state *z* and candidate output *y* through multiple refinement steps given input *x*, achieving high reasoning quality without large-scale architectures.

**Encoder–Decoder Interface**  
Encodes state and action tokens into TRM latents, then decodes refined latents back into next-state or action predictions, enabling closed-loop policy reasoning.

---

## Methodology

1. **Foundational Distillation**  
   Align TRM latent representations to GR00T’s multimodal encodings via teacher–student objectives.  
   Use “dream data” synthetic rollouts to expand data diversity efficiently.

2. **Encoder–Decoder Imitation Learning**  
   Train the compact TRM-based student model to emulate GR00T’s action–state dynamics with joint reconstruction and distillation losses.

---

## Hypothesis

A strategically distilled GR00T N1.5 derivative can:

1. Retain teacher-level perception, reasoning, and control accuracy  
2. Operate efficiently under strict compute and energy constraints  
3. Preserve generalization through recursive refinement and multimodal distillation  

This supports the premise that *efficiency and generality in robotic intelligence can co-exist.*

---

## Setup

### Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
# Optional: Weights & Biases logging
wandb login
```

### Example — Distillation Training

```bash
python train.py   --teacher_path checkpoints/gr00t_n15.pth   --data_path data/dream_rollouts   --vlm nanollava_1b   --reasoner trm_tiny   --epochs 200   --save_dir runs/babygr00t_trm_nanollava
```

Metrics include latent-alignment loss, imitation accuracy, latency, and energy efficiency.

---

## Model Summary

| Component | Role | Key Traits |
|------------|------|------------|
| **TRM** | Recursive reasoning | 140 M parameters · latent refinement · minimal compute |
| **nanoLLaVA-1B** | Visual–language grounding | 1 B parameters · SigLIP encoder · edge-friendly |
| **Distillation Pipeline** | Teacher–student training | Representation + behavioral transfer |

---

## Roadmap

- Reintroduce **Depth Fusion** with token-efficient integration  
- Add **quantization-aware training** and fused-kernel inference paths  
- Expand benchmarks to manipulation and navigation tasks  
- Provide deployment scripts for Jetson Orin and other embedded boards  

---

## Data Ethics and Transparency

BabyGR00T follows a **Data Ethics Canvas** framework:

- Clear data provenance and bias assessment  
- Controlled sharing and anonymization  
- Compliance with AI ethics and robotics data governance  

See `docs/data_ethics_canvas.md` for details.

---

## Citation

If you use BabyGR00T in research or development, please cite:

```bibtex
@article{munoz2025babygr00t,
  title        = {BabyGR00T: Making Foundational Robotic Models Small, Fast, and Scalable},
  author       = {Munoz, L. A.},
  year         = {2025},
  institution  = {Tec de Monterrey},
  address      = {Monterrey, N.L., México}
}
```

**Related Works**

- Jolicoeur-Martineau, A. (2025). *Less is More: Recursive Reasoning with Tiny Networks (TRM).*  
- Liu et al. (2024). *nanoLLaVA-1B: Compact Vision-Language Models for Edge AI.*
- Bjorck et al. (2025). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*

---

## Acknowledgments

Developed as part of the **BabyGR00T** research effort on efficient embodied intelligence.  
Thanks to contributors from open-source projects on recursive reasoning (Samsung SAIT) and compact VLMs (nanoLLaVA team).  
GR00T N1.5 model and datasets are referenced solely for academic distillation and benchmarking purposes.
