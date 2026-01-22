# Code Structure and Pipeline Architecture

This document provides a comprehensive overview of the Membership Inference Attack (MIA) system for diffusion models, covering the multi-dataset ownership verification pipeline.

---

## 1. System Overview

### 1.1 Problem Definition

Given a diffusion model $M$, we aim to verify whether $M$ was trained on a specific private dataset $\mathcal{D}_{private}$. Our approach compares **t-error scores** (reconstruction errors) between:

1. **Owner models** (Model A, Model B) - trained on $\mathcal{D}_{private}$
2. **Public baselines** - models that never saw $\mathcal{D}_{private}$

**Key Insight**: Models trained on specific data exhibit lower reconstruction errors on that data due to memorization.

### 1.2 Supported Datasets

| Dataset | Resolution | Train Size | Watermark Size | Baseline Model |
|---------|------------|------------|----------------|----------------|
| **CIFAR-10** | 32×32 | 50,000 | 5,000 | `google/ddpm-cifar10-32` |
| **CIFAR-100** | 32×32 | 50,000 | 5,000 | `google/ddpm-cifar10-32` |
| **STL-10** | 96×96 | 5,000 | 1,000 | `google/ddpm-ema-bedroom-256` |
| **CelebA** | 64×64 | 162,770 | 5,000 | `google/ddpm-celebahq-256` |

---

## 2. Project Structure

```
mia_ddpm_qr/
├── configs/                    # Configuration files
│   ├── data_{dataset}.yaml     # Dataset configs (cifar10, cifar100, stl10, celeba)
│   ├── model_ddim_{dataset}.yaml # Model architecture configs
│   ├── mmd_finetune_{dataset}.yaml # MMD fine-tuning configs
│   ├── attack_qr.yaml          # QR attack config
│   └── baselines_by_dataset.yaml # Public baseline registry
├── scripts/                    # Executable scripts
│   ├── generate_splits.py      # Step 1: Generate data splits
│   ├── finetune_mmd_ddm.py     # Step 3: MMD fine-tuning
│   ├── eval_ownership.py       # Step 4: Ownership evaluation
│   └── generate_cross_dataset_summary.py # Step 5: Cross-dataset summary
├── src/
│   ├── attack_qr/              # Quantile Regression attack
│   │   ├── engine/             # Training and evaluation
│   │   ├── models/             # QR models (ResNet18QR, ResNet18GaussianQR)
│   │   └── utils/              # Losses, metrics, seeding
│   ├── attacks/                # Attack utilities
│   │   ├── baselines/          # Public baseline loaders
│   │   ├── eval/               # Evaluation metrics
│   │   └── scores/             # T-error computation
│   ├── ddpm/                   # DDPM training
│   └── ddpm_ddim/              # DDIM training and fine-tuning
├── tools/
│   └── compute_scores.py       # Score computation utility
├── data/splits/                # Generated data splits
│   └── {dataset}/              # Per-dataset splits
│       ├── watermark_private.json  # Private watermark set W_D
│       ├── eval_nonmember.json     # Non-member evaluation set
│       ├── member_train.json       # Full training set
│       └── manifest.json           # Metadata and SHA256 hashes
├── runs/                       # Training outputs
│   ├── ddim_{dataset}/main/    # Model A checkpoints
│   ├── mmd_finetune/{dataset}/ # Model B checkpoints
│   └── attack_qr/reports/      # Evaluation reports
└── run_all.sh                  # Full pipeline automation
```

---

## 3. Five-Step Pipeline

The complete ownership verification pipeline consists of five steps:

### 3.1 Step 1: Generate Splits

**Script**: `scripts/generate_splits.py`

Creates deterministic data splits with cryptographic verification:

```bash
python scripts/generate_splits.py \
    --dataset all \
    --output-dir data/splits \
    --seed 20251030
```

**Outputs** (per dataset):
- `watermark_private.json`: K indices for private watermark set $\mathcal{W}_D$
- `eval_nonmember.json`: K indices disjoint from $\mathcal{W}_D$
- `member_train.json`: Full training indices (includes $\mathcal{W}_D$)
- `manifest.json`: SHA256 hashes for reproducibility

**Split Properties**:
- Watermark set $\mathcal{W}_D \subset \mathcal{D}_{train}$
- Evaluation set $\mathcal{E} \cap \mathcal{W}_D = \emptyset$
- Deterministic given seed

### 3.2 Step 2: Train Model A

**Script**: `src/ddpm_ddim/train_ddim.py`

Trains the owner's diffusion model on the full training set:

```bash
python src/ddpm_ddim/train_ddim.py \
    --config configs/model_ddim_{dataset}.yaml \
    --data configs/data_{dataset}.yaml \
    --mode main \
    --select-best
```

**Configuration** (`configs/model_ddim_{dataset}.yaml`):
```yaml
model:
  channels: 128
  channel_multipliers: [1, 2, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [16]
  dropout: 0.1

diffusion:
  timesteps: 1000
  beta_schedule: cosine

training:
  total_iterations: 400000  # 200k for STL-10
  batch_size: 128
  lr: 2.0e-4
  ema_decay: 0.9999
```

**Outputs**:
- `runs/ddim_{dataset}/main/best_for_mia.ckpt`: Best checkpoint
- `runs/ddim_{dataset}/main/run.json`: Training metadata
- `runs/ddim_{dataset}/main/watermark_exposure.json`: Exposure tracking

### 3.3 Step 3: MMD Fine-tune → Model B

**Script**: `scripts/finetune_mmd_ddm.py`

Simulates model theft via MMD fine-tuning in CLIP feature space:

```bash
python scripts/finetune_mmd_ddm.py \
    --config configs/mmd_finetune_{dataset}.yaml \
    --out runs/mmd_finetune/{dataset}/model_b \
    --iters 500
```

**Configuration** (`configs/mmd_finetune_{dataset}.yaml`):
```yaml
base:
  model_config: configs/model_ddim_{dataset}.yaml
  data_config: configs/data_{dataset}.yaml
  checkpoint: runs/ddim_{dataset}/main/best_for_mia.ckpt

finetune:
  iterations: 500
  batch_size: 128
  lr: 5.0e-6
  ema_decay: 0.9999
  amp: true

sampler:
  steps: 10
  t_start: 900
```

**Outputs**:
- `runs/mmd_finetune/{dataset}/model_b/ckpt_0500_ema.pt`: Fine-tuned Model B

### 3.4 Step 4: Ownership Evaluation

**Script**: `scripts/eval_ownership.py`

Compares t-error scores across models on both watermark and non-member data:

```bash
python scripts/eval_ownership.py \
    --dataset {dataset} \
    --split watermark_private \
    --model-a runs/ddim_{dataset}/main/best_for_mia.ckpt \
    --model-b runs/mmd_finetune/{dataset}/model_b/ckpt_0500_ema.pt \
    --baselines-config configs/baselines_by_dataset.yaml \
    --output runs/attack_qr/reports/{dataset}/ \
    --k-timesteps 50 \
    --agg q25 \
    --save-pdf
```

**Evaluated Models**:
1. **Model A**: Owner's original model
2. **Model B**: Fine-tuned (stolen) model
3. **Public Baselines**: HuggingFace pretrained models

**Outputs**:
- `baseline_comparison_{dataset}_{split}.json`: Full statistical report
- `t_error_distributions_{split}.npz`: Raw score distributions
- `report_{dataset}_{split}.pdf`: Visualization report

**Ownership Criteria**:
- **Consistency**: Model A ≈ Model B on watermark data (p > 0.05)
- **Separation**: Owner models vs baselines (p < 1e-6, |Cohen's d| > 2.0)
- **Ratio**: Baseline t-error / Owner t-error > 5.0

### 3.5 Step 5: Cross-Dataset Summary

**Script**: `scripts/generate_cross_dataset_summary.py`

Aggregates results across all datasets:

```bash
python scripts/generate_cross_dataset_summary.py \
    --reports-dir runs/attack_qr/reports \
    --output runs/attack_qr/reports/summary_all_datasets.csv \
    --datasets cifar10 cifar100 stl10 celeba
```

**Outputs**:
- `summary_all_datasets.csv`: Tabular results
- `summary_all_datasets.json`: Structured results
- `summary_report.md`: Markdown report

---

## 4. T-Error Score Computation

### 4.1 Mathematical Definition

For a sample $x_0$ and timestep $t$:

$$\text{t-error}(x_0, t) = \|x_0 - \hat{x}_0(x_t, t)\|_2^2$$

where:
1. **Forward diffusion**: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
2. **Model prediction**: $\hat{\epsilon} = M(x_t, t)$
3. **Reconstruction**: $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$

### 4.2 Aggregation

For $K$ uniformly sampled timesteps:

$$s(x_0) = \text{quantile}_{0.25}\left(\{\text{t-error}(x_0, t_k)\}_{k=1}^K\right)$$

**Default Configuration**:
- $K = 50$ timesteps
- Aggregation: 25th percentile (q25)
- This lower quantile is robust to outlier timesteps

### 4.3 Code Location

| Component | File |
|-----------|------|
| T-error computation | `src/attacks/baselines/__init__.py::compute_baseline_scores()` |
| Uniform timesteps | `src/attack_qr/features/t_error.py::uniform_timesteps()` |
| Aggregation | `src/attack_qr/features/t_error.py::t_error_aggregate()` |

---

## 5. Configuration Files

### 5.1 Dataset Configuration

**`configs/data_{dataset}.yaml`**:

```yaml
dataset:
  name: CIFAR10  # or CIFAR100, STL10, CelebA
  root: data/{dataset}
  image_shape: [3, 32, 32]  # [3, 96, 96] for STL-10, [3, 64, 64] for CelebA
  normalization:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]

splits:
  seed: 20251030
  paths:
    watermark_private: data/splits/{dataset}/watermark_private.json
    eval_nonmember: data/splits/{dataset}/eval_nonmember.json
    member_train: data/splits/{dataset}/member_train.json
```

### 5.2 Baseline Registry

**`configs/baselines_by_dataset.yaml`**:

```yaml
cifar10:
  - name: ddpm-cifar10
    model_id: google/ddpm-cifar10-32
    resolution: 32
    type: ddpm

cifar100:
  - name: ddpm-cifar10
    model_id: google/ddpm-cifar10-32
    resolution: 32
    type: ddpm

stl10:
  - name: ddpm-bedroom
    model_id: google/ddpm-ema-bedroom-256
    resolution: 256
    type: ddpm

celeba:
  - name: ddpm-celebahq
    model_id: google/ddpm-celebahq-256
    resolution: 256
    type: ddpm
  - name: ldm-celebahq
    model_id: CompVis/ldm-celebahq-256
    resolution: 256
    type: ldm
```

### 5.3 Attack Configuration

**`configs/attack_qr.yaml`**:

```yaml
seed: 20251030

target_fprs:
  - 0.001
  - 0.0001

t_error:
  T: 1000
  k_uniform: 50
  aggregate: q25
  cache_dir: scores

qr:
  mode: gaussian  # Gaussian head for quantile prediction
  target_space: log1p

bagging:
  B: 50
  bootstrap_ratio: 0.8

train:
  epochs: 50
  batch_size: 256
  lr: 0.001
  log1p: true
```

---

## 6. Key Modules

### 6.1 Baseline Loaders

| Module | File | Function |
|--------|------|----------|
| HuggingFace DDPM | `src/attacks/baselines/huggingface_loader.py` | Load pretrained DDPM |
| LDM (CelebA) | `src/attacks/baselines/ldm_loader.py` | Load Latent Diffusion |
| Registry lookup | `src/attacks/baselines/__init__.py` | `load_baseline_from_registry()` |

### 6.2 Quantile Regression

| Module | File | Description |
|--------|------|-------------|
| Model (Pinball) | `src/attack_qr/models/qr_resnet18.py::ResNet18QR` | Direct quantile prediction |
| Model (Gaussian) | `src/attack_qr/models/qr_resnet18.py::ResNet18GaussianQR` | Gaussian parameterization |
| Training | `src/attack_qr/engine/train_qr_bagging.py` | Bagging ensemble training |
| Evaluation | `src/attack_qr/engine/eval_attack.py` | Attack evaluation |
| CLI | `src/attack_qr/engine/cli_train.py`, `cli_eval.py` | Command-line interface |

### 6.3 Evaluation Metrics

| Metric | File | Description |
|--------|------|-------------|
| ROC-AUC | `src/attacks/eval/metrics.py::roc_auc()` | Area under ROC curve |
| TPR@FPR | `src/attacks/eval/metrics.py::tpr_precision_at_fpr()` | True positive rate at fixed FPR |
| Cohen's d | `scripts/eval_ownership.py::perform_statistical_tests()` | Effect size |

---

## 7. Full Pipeline Automation

**`run_all.sh`** executes all five steps:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASETS=("cifar10" "cifar100" "stl10" "celeba")
SEED=20251030

# Step 1: Generate splits
python scripts/generate_splits.py --dataset all

# Step 2: Train Model A (all datasets)
for ds in "${DATASETS[@]}"; do
    python src/ddpm_ddim/train_ddim.py \
        --config configs/model_ddim_${ds}.yaml \
        --data configs/data_${ds}.yaml \
        --mode main --select-best
done

# Step 3: MMD fine-tune → Model B
for ds in "${DATASETS[@]}"; do
    python scripts/finetune_mmd_ddm.py \
        --config configs/mmd_finetune_${ds}.yaml \
        --out runs/mmd_finetune/${ds}/model_b
done

# Step 4: Ownership evaluation
for ds in "${DATASETS[@]}"; do
    for split in watermark_private eval_nonmember; do
        python scripts/eval_ownership.py \
            --dataset $ds --split $split \
            --model-a runs/ddim_${ds}/main/best_for_mia.ckpt \
            --model-b runs/mmd_finetune/${ds}/model_b/ckpt_0500_ema.pt
    done
done

# Step 5: Cross-dataset summary
python scripts/generate_cross_dataset_summary.py \
    --reports-dir runs/attack_qr/reports
```

---

## 8. Expected Results

### 8.1 Ownership Verification Criteria

For successful ownership verification:

| Criterion | Condition | Interpretation |
|-----------|-----------|----------------|
| **Consistency** | T-test p-value > 0.05 | Model A ≈ Model B |
| **Separation** | T-test p-value < 1e-6, \|Cohen's d\| > 2.0 | Owner ≠ Baseline |
| **Ratio** | Baseline / Owner > 5.0 | Strong discrimination |

### 8.2 Typical Results

| Dataset | Owner t-error | Baseline t-error | Ratio | Cohen's d |
|---------|---------------|------------------|-------|-----------|
| CIFAR-10 | ~28.7 | ~697.2 | 24.3× | -23.95 |
| CIFAR-100 | ~30.5 | ~685.4 | 22.5× | -22.10 |
| STL-10 | ~45.2 | ~520.3 | 11.5× | -15.80 |
| CelebA | ~35.8 | ~480.6 | 13.4× | -18.20 |

---

## 9. Quick Reference

### 9.1 Key File Locations

| Purpose | Location |
|---------|----------|
| Data splits | `data/splits/{dataset}/` |
| Model A checkpoint | `runs/ddim_{dataset}/main/best_for_mia.ckpt` |
| Model B checkpoint | `runs/mmd_finetune/{dataset}/model_b/ckpt_0500_ema.pt` |
| Evaluation reports | `runs/attack_qr/reports/{dataset}/` |
| Cross-dataset summary | `runs/attack_qr/reports/summary_all_datasets.*` |

### 9.2 Common Commands

```bash
# Generate all splits
python scripts/generate_splits.py --dataset all

# Evaluate ownership (single dataset)
python scripts/eval_ownership.py --dataset cifar10 --split watermark_private

# Full pipeline
bash run_all.sh 2>&1 | tee run_all_$(date +%Y%m%d).log
```

---

**Last Updated**: January 2026
**Codebase Version**: Multi-dataset support with Gaussian QR
