# Source Code Module Reference

This document provides a reference for the key modules in the codebase.

---

## 1. Primary Modules (Active)

### 1.1 Attack QR Engine (`src/attack_qr/`)

The main QR-MIA implementation:

```
src/attack_qr/
├── __init__.py
├── engine/
│   ├── build_pairs.py          # Data pair construction
│   ├── cli_eval.py             # Evaluation CLI
│   ├── cli_train.py            # Training CLI
│   ├── eval_attack.py          # Attack evaluation
│   └── train_qr_bagging.py     # Bagging ensemble training
├── features/
│   └── t_error.py              # T-error computation
├── models/
│   └── qr_resnet18.py          # QR models (ResNet18QR, ResNet18GaussianQR)
└── utils/
    ├── losses.py               # Pinball and Gaussian NLL losses
    ├── metrics.py              # Bootstrap metrics
    └── seeding.py              # Random seed utilities
```

**Key Classes/Functions**:

| Module | Class/Function | Description |
|--------|----------------|-------------|
| `qr_resnet18.py` | `ResNet18QR` | Direct quantile prediction model |
| `qr_resnet18.py` | `ResNet18GaussianQR` | Gaussian parameterization model |
| `train_qr_bagging.py` | `train_bagging_ensemble_gaussian_scores()` | Train Gaussian ensemble |
| `eval_attack.py` | `evaluate_attack_scores_gaussian()` | Evaluate Gaussian ensemble |
| `eval_attack.py` | `EvalScoresDataset` | Evaluation dataset |
| `t_error.py` | `t_error_aggregate()` | Multi-timestep aggregation |
| `t_error.py` | `uniform_timesteps()` | Uniform timestep sampling |

### 1.2 Baseline Loaders (`src/attacks/baselines/`)

Public reference model loading:

```
src/attacks/baselines/
├── __init__.py                 # Registry and convenience functions
├── huggingface_loader.py       # HuggingFace DDPM loading
├── ldm_loader.py               # Latent Diffusion Model loading
└── t_error_hf.py               # T-error for HF models
```

**Key Functions**:

| Module | Function | Description |
|--------|----------|-------------|
| `__init__.py` | `load_baseline_from_registry()` | Load baseline by name |
| `__init__.py` | `compute_baseline_scores()` | Compute t-error for baseline |
| `huggingface_loader.py` | `load_hf_baseline()` | Load HuggingFace DDPM |
| `ldm_loader.py` | `compute_ldm_t_error()` | T-error for LDM models |

### 1.3 Evaluation Metrics (`src/attacks/eval/`)

Shared metric computation:

```
src/attacks/eval/
├── __init__.py
└── metrics.py                  # ROC-AUC, TPR@FPR
```

**Key Functions**:

| Function | Description |
|----------|-------------|
| `roc_auc(scores_in, scores_out)` | Compute ROC-AUC |
| `tpr_precision_at_fpr(scores_in, scores_out, target_fpr)` | TPR and precision at FPR |

### 1.4 DDIM Training (`src/ddpm_ddim/`)

Diffusion model training:

```
src/ddpm_ddim/
├── __init__.py
├── ddim/
│   └── forward_reverse.py      # DDIM forward/reverse
├── models/
│   └── unet.py                 # UNet architecture
├── samplers/
│   └── ddim10.py               # 10-step DDIM sampler
├── schedulers/
│   └── betas.py                # Beta schedule (cosine)
├── clip_features.py            # CLIP feature extraction
├── mmd_loss.py                 # MMD loss for fine-tuning
├── select_checkpoints.py       # Best checkpoint selection
└── train_ddim.py               # Main training script
```

---

## 2. Scripts (`scripts/`)

### 2.1 Main Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `generate_splits.py` | Generate data splits with manifests |
| `finetune_mmd_ddm.py` | MMD fine-tuning for Model B |
| `eval_ownership.py` | Ownership verification evaluation |
| `generate_cross_dataset_summary.py` | Cross-dataset result aggregation |

### 2.2 Utility Scripts

| Script | Purpose |
|--------|---------|
| `compute_baseline_metrics.py` | Baseline (Yeom) metrics |
| `sample_ddim10.py` | Generate samples with DDIM |
| `eval_fid.py` | FID score computation |

---

## 3. Configuration Files (`configs/`)

### 3.1 Per-Dataset Configs

| Config | Description |
|--------|-------------|
| `data_{dataset}.yaml` | Dataset paths and normalization |
| `model_ddim_{dataset}.yaml` | Model architecture |
| `mmd_finetune_{dataset}.yaml` | MMD fine-tuning params |

### 3.2 Global Configs

| Config | Description |
|--------|-------------|
| `attack_qr.yaml` | QR attack configuration |
| `baselines_by_dataset.yaml` | Public baseline registry |

---

## 4. Tools (`tools/`)

| Tool | Purpose |
|------|---------|
| `compute_scores.py` | Pre-compute t-error scores |

---

## 5. Module Dependencies

### 5.1 Evaluation Pipeline

```
scripts/eval_ownership.py
    → src/attacks/baselines/__init__.py
        → huggingface_loader.py
        → ldm_loader.py
    → src/ddpm_ddim/models/unet.py
    → src/ddpm_ddim/schedulers/betas.py
```

### 5.2 QR Training Pipeline

```
src/attack_qr/engine/cli_train.py
    → train_qr_bagging.py
        → models/qr_resnet18.py
        → utils/losses.py
    → features/t_error.py
```

### 5.3 QR Evaluation Pipeline

```
src/attack_qr/engine/cli_eval.py
    → eval_attack.py
        → models/qr_resnet18.py
        → src/attacks/eval/metrics.py
```

---

## 6. Import Examples

### 6.1 Loading Reference Models

```python
from src.attacks.baselines import (
    load_baseline_from_registry,
    compute_baseline_scores,
)

model, alphas_bar = load_baseline_from_registry(
    "ddpm-cifar10", "cifar10", "cuda"
)
scores = compute_baseline_scores(loader, model, alphas_bar)
```

### 6.2 Using QR Models

```python
from src.attack_qr.models.qr_resnet18 import ResNet18GaussianQR
from src.attack_qr.engine.eval_attack import gaussian_quantile_from_params

model = ResNet18GaussianQR(stats_dim=3)
mu, log_sigma = model(images, stats)
q_log = gaussian_quantile_from_params(mu, log_sigma, tau=0.999)
```

### 6.3 Computing Metrics

```python
from src.attacks.eval.metrics import roc_auc, tpr_precision_at_fpr

auc = roc_auc(margins_in, margins_out)
tpr_result = tpr_precision_at_fpr(margins_in, margins_out, target_fpr=0.001)
```

---

## 7. Directory Summary

| Directory | Contents |
|-----------|----------|
| `src/attack_qr/` | QR-MIA implementation (primary) |
| `src/attacks/baselines/` | Public baseline loaders |
| `src/attacks/eval/` | Shared metrics |
| `src/ddpm_ddim/` | DDIM training and sampling |
| `src/ddpm/` | Legacy DDPM code |
| `scripts/` | Pipeline scripts |
| `tools/` | Utility tools |
| `configs/` | Configuration files |

---

**Last Updated**: January 2026
