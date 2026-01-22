# Documentation

Technical documentation for the Membership Inference Attack (MIA) system for diffusion model ownership verification.

---

## Overview

This project implements a **t-error based ownership verification** system for diffusion models, supporting multiple datasets (CIFAR-10, CIFAR-100, STL-10, CelebA) with a unified pipeline.

**Key Approach**: Models trained on specific data exhibit lower reconstruction errors (t-error) on that data. By comparing t-error scores between owner models and public baselines, we can verify model ownership with statistical significance.

---

## Documentation Structure

```
docs/
├── README.md              # This file
├── guides/                # Implementation guides
│   ├── CODE_STRUCTURE.md              # System architecture
│   ├── QUANTILE_REGRESSION_IMPLEMENTATION.md  # QR algorithm
│   ├── GAUSSIAN_QR_IMPLEMENTATION.md  # Gaussian QR model
│   ├── ATTACK_SCORE_ANALYSIS.md       # T-error score definition
│   ├── GAUSSIAN_QR_MARGIN_BASED_METRICS.md    # Evaluation metrics
│   ├── COMPUTE_SCORES_CLARIFICATION.md        # Score utilities
│   └── REQUIRED_ATTACKS_FILES.md      # Module reference
└── methodology/           # Research methodology (for paper writing)
    ├── METHODOLOGY_OVERVIEW.md        # Complete methodology overview
    ├── T_ERROR_OWNERSHIP.md           # T-error and ownership verification
    ├── GAUSSIAN_QUANTILE_REGRESSION.md # Gaussian QR mathematical details
    └── MMD_FINETUNING.md              # Model theft simulation
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

### 2. Generate Data Splits

```bash
python scripts/generate_splits.py --dataset all
```

### 3. Train Models

```bash
# Train Model A (owner model)
python src/ddpm_ddim/train_ddim.py \
    --config configs/model_ddim_cifar10.yaml \
    --data configs/data_cifar10.yaml \
    --mode main --select-best

# Create Model B (simulated theft via MMD fine-tuning)
python scripts/finetune_mmd_ddm.py \
    --config configs/mmd_finetune_cifar10.yaml
```

### 4. Ownership Verification

```bash
python scripts/eval_ownership.py \
    --dataset cifar10 \
    --split watermark_private \
    --model-a runs/ddim_cifar10/main/best_for_mia.ckpt \
    --model-b runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt \
    --save-pdf
```

### 5. Full Pipeline

```bash
bash run_all.sh 2>&1 | tee run_all.log
```

---

## Documentation Index

### System Architecture

- **[Code Structure](guides/CODE_STRUCTURE.md)**: Complete system overview, five-step pipeline, configuration files, and multi-dataset support.

### Algorithm Details

- **[Quantile Regression Implementation](guides/QUANTILE_REGRESSION_IMPLEMENTATION.md)**: Mathematical foundation of QR-MIA, including Gaussian parameterization, loss functions, and ensemble training.

- **[Gaussian QR Implementation](guides/GAUSSIAN_QR_IMPLEMENTATION.md)**: Detailed implementation of the Gaussian head model, including architecture, training, and inference.

### Attack Analysis

- **[Attack Score Analysis](guides/ATTACK_SCORE_ANALYSIS.md)**: T-error score definition, multi-timestep aggregation, and statistical analysis for ownership verification.

- **[Margin-Based Metrics](guides/GAUSSIAN_QR_MARGIN_BASED_METRICS.md)**: Evaluation metrics (ROC-AUC, TPR@FPR), margin computation, and bootstrap confidence intervals.

### Reference

- **[Compute Scores Clarification](guides/COMPUTE_SCORES_CLARIFICATION.md)**: Quick reference for score computation utilities.

- **[Module Reference](guides/REQUIRED_ATTACKS_FILES.md)**: Source code module documentation and import examples.

### Methodology (Paper Reference)

- **[Methodology Overview](methodology/METHODOLOGY_OVERVIEW.md)**: Complete methodology overview suitable for paper introduction/methodology sections.

- **[T-Error and Ownership](methodology/T_ERROR_OWNERSHIP.md)**: Detailed t-error formulation and ownership verification criteria.

- **[Gaussian Quantile Regression](methodology/GAUSSIAN_QUANTILE_REGRESSION.md)**: Mathematical foundations of Gaussian QR for membership inference.

- **[MMD Fine-tuning](methodology/MMD_FINETUNING.md)**: Model theft simulation via MMD fine-tuning.

---

## Key Concepts

### T-Error Score

Measures reconstruction quality after forward-reverse diffusion:

$$s(x) = \text{q25}\left(\left\{\frac{\|x_0 - \hat{x}_0(x_t, t)\|_2^2}{HWC}\right\}_{t \in \mathcal{T}}\right)$$

**Lower t-error** indicates the model has memorized the data (likely member).

### Ownership Verification

Compare t-error scores between:
- **Owner models** (Model A, Model B): Low t-error on watermark data
- **Public baselines**: High t-error on watermark data

### Verification Criteria

| Criterion | Condition | Interpretation |
|-----------|-----------|----------------|
| Consistency | Model A ≈ Model B | Same origin |
| Separation | Owner ≪ Baseline | Ownership proved |
| Ratio | Baseline/Owner > 5× | Strong discrimination |

### Gaussian QR-MIA

Margin-based membership inference:

$$m(x) = \hat{q}_\tau^{\text{ens}}(x) - \log(1 + s(x))$$

**Positive margin** → likely member

---

## Supported Datasets

| Dataset | Resolution | Watermark Size | Baseline |
|---------|------------|----------------|----------|
| CIFAR-10 | 32×32 | 5,000 | `google/ddpm-cifar10-32` |
| CIFAR-100 | 32×32 | 5,000 | `google/ddpm-cifar10-32` |
| STL-10 | 96×96 | 1,000 | `google/ddpm-ema-bedroom-256` |
| CelebA | 64×64 | 5,000 | `google/ddpm-celebahq-256` |

---

## Output Locations

| Output | Location |
|--------|----------|
| Data splits | `data/splits/{dataset}/` |
| Model A | `runs/ddim_{dataset}/main/best_for_mia.ckpt` |
| Model B | `runs/mmd_finetune/{dataset}/model_b/ckpt_0500_ema.pt` |
| Reports | `runs/attack_qr/reports/{dataset}/` |
| Summary | `runs/attack_qr/reports/summary_all_datasets.*` |

---

## Citation

If you use this code in your research, please cite our work.

---

**Last Updated**: January 2026
**Version**: Multi-dataset support with Gaussian QR
