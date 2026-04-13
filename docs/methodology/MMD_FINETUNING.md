# MMD Fine-tuning for Model Theft Simulation

This document describes the Maximum Mean Discrepancy (MMD) fine-tuning approach used to simulate model theft scenarios in our ownership verification framework.

---

## 1. Motivation

Model theft is a realistic threat where an adversary obtains a copy of the owner's diffusion model and attempts to create a derivative model. To evaluate our ownership verification system against such attacks, we simulate model theft using MMD fine-tuning, which:

1. Preserves generation quality
2. Maintains distributional properties
3. Retains memorization signals (low t-error on training data)

---

## 2. Method

### 2.1 Overview

Starting from Model A (owner's model), we fine-tune to create Model B by minimizing the MMD between generated samples and real samples in CLIP feature space.

### 2.2 Sampling: 10-Step DDIM

We use a **differentiable 10-step DDIM** ($\eta = 0$) chain for efficient gradient-based optimization:

**Single Step Update**:
$$x_{t_{prev}} = \sqrt{\bar{\alpha}_{t_{prev}}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t_{prev}}} \cdot \epsilon$$

where:
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**Configuration**:
- $K = 10$ steps with linear timestep spacing
- $t_{start} = 900$ (avoiding numerical instability near $t = T$)
- Deterministic ($\eta = 0$)

### 2.3 Feature Space: CLIP

We compute MMD in the CLIP ViT-B/32 feature space:

1. **Preprocessing** (differentiable):
   - Unnormalize from DDPM space to $[0, 1]$
   - Resize to 224×224 (bicubic interpolation)
   - Normalize with CLIP mean/std

2. **Feature Extraction**:
   - Frozen CLIP ViT-B/32 encoder
   - Output: 512-dimensional feature vectors

### 2.4 MMD² Loss

**Kernel**: Cubic polynomial
$$k(u, v) = \left(\frac{u \cdot v}{d} + 1\right)^3$$

**Unbiased Estimator**:
$$\text{MMD}^2 = E_{xx} + E_{yy} - 2E_{xy}$$

where:
$$E_{xx} = \frac{\sum K_{xx} - \text{tr}(K_{xx})}{n(n-1)}, \quad E_{yy} = \frac{\sum K_{yy} - \text{tr}(K_{yy})}{m(m-1)}, \quad E_{xy} = \text{mean}(K_{xy})$$

---

## 3. Training Configuration

### 3.1 Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Iterations** | 500 | Fine-tuning steps |
| **Learning Rate** | $5 \times 10^{-6}$ | Adam optimizer |
| **Batch Size** | 128 | Per-iteration samples |
| **EMA Decay** | 0.9999 | Exponential moving average |
| **DDIM Steps** | 10 | Sampling chain length |
| **$t_{start}$** | 900 | Starting timestep |
| **AMP** | Enabled | Mixed precision training |
| **Gradient Checkpointing** | Optional | Memory optimization |

### 3.2 Initialization

- **Weights**: Initialize from Model A's EMA checkpoint
- **EMA**: Maintain fresh EMA during fine-tuning (EMA_ft)
- **Optimizer**: Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$

---

## 4. Multi-Dataset Configuration

Each dataset uses adapted configurations:

| Dataset | Config File | Base Checkpoint |
|---------|-------------|-----------------|
| CIFAR-10 | `configs/mmd_finetune_cifar10.yaml` | `runs/ddim_cifar10/main/best_for_mia.ckpt` |
| CIFAR-100 | `configs/mmd_finetune_cifar100.yaml` | `runs/ddim_cifar100/main/best_for_mia.ckpt` |
| STL-10 | `configs/mmd_finetune_stl10.yaml` | `runs/ddim_stl10/main/best_for_mia.ckpt` |
| CelebA | `configs/mmd_finetune_celeba.yaml` | `runs/ddim_celeba/main/best_for_mia.ckpt` |

---

## 5. Usage

### 5.1 Fine-tuning

```bash
python scripts/finetune_mmd_ddm.py \
    --config configs/mmd_finetune_{dataset}.yaml \
    --out runs/mmd_finetune/{dataset}/model_b \
    --device cuda:0 \
    --iters 500 \
    --grad-checkpoint
```

### 5.2 Sampling from Model B

```bash
python scripts/sample_ddim10.py \
    --checkpoint runs/mmd_finetune/{dataset}/model_b/ckpt_0500_ema.pt \
    --num-samples 256 \
    --seed 123 \
    --t-start 900 \
    --out-dir samples/
```

---

## 6. Output Files

```
runs/mmd_finetune/{dataset}/model_b/
├── ckpt_0500_raw.pt          # Raw model weights
├── ckpt_0500_ema.pt          # EMA model weights (used for evaluation)
└── configs/
    └── mmd_finetune.yaml     # Configuration snapshot
```

---

## 7. Numerical Stability

### 7.1 Key Considerations

1. **Timestep Selection**: Start from $t = 900$ to avoid near-zero $\bar{\alpha}_t$ values
2. **Precision**: DDIM chain computed in FP32 for stability
3. **Clamping**: Denominator clamp $\geq 10^{-4}$ in DDIM update

### 7.2 Verified Behavior

- $x_0$ reconstruction range: approximately $[-2, 2]$
- MMD² values: positive and decreasing during training
- EMA checkpoint produces identical samples with fixed seed

---

## 8. Expected Outcomes

After MMD fine-tuning:

1. **Model B generation quality**: Comparable to Model A
2. **T-error on watermark data**: Model B ≈ Model A (low)
3. **T-error on public reference models**: Much higher than Model A/B
4. **Ownership verification**: Both criteria satisfied

This demonstrates that MMD fine-tuning, while modifying model weights, preserves the memorization signal that enables ownership verification.

---

## 9. Related Files

| File | Description |
|------|-------------|
| `scripts/finetune_mmd_ddm.py` | Main fine-tuning script |
| `src/ddpm_ddim/mmd_loss.py` | MMD loss implementation |
| `src/ddpm_ddim/samplers/ddim10.py` | 10-step DDIM sampler |
| `src/ddpm_ddim/clip_features.py` | CLIP feature extraction |
| `configs/mmd_finetune_*.yaml` | Per-dataset configurations |

---

**Last Updated**: January 2026
