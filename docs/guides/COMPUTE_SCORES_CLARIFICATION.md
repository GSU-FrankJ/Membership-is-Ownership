# Score Computation Reference

This document clarifies the score computation utilities in the codebase.

---

## 1. Primary Score Computation

### 1.1 Ownership Evaluation (Recommended)

For ownership verification with baseline comparison, use:

```bash
python scripts/eval_ownership.py \
    --dataset cifar10 \
    --split watermark_private \
    --model-a runs/ddim_cifar10/main/best_for_mia.ckpt \
    --model-b runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt \
    --k-timesteps 50 \
    --agg q25
```

This script:
- Loads Model A, Model B, and public baselines
- Computes t-error scores for all models
- Performs statistical tests (t-test, Mann-Whitney, Cohen's d)
- Generates comprehensive reports

### 1.2 Score Pre-computation (QR Training)

For pre-computing scores for QR model training:

```bash
python tools/compute_scores.py \
    --config configs/attack_qr.yaml \
    --data-config configs/data_cifar10.yaml \
    --tag q25
```

This generates:
- `scores/q25_{split}.pt`: Aggregated scores and statistics
- Used as input for QR model training

---

## 2. File Locations

| File | Purpose | Usage |
|------|---------|-------|
| `scripts/eval_ownership.py` | Ownership verification | Primary evaluation script |
| `tools/compute_scores.py` | Pre-compute scores | QR training preparation |
| `src/attack_qr/features/t_error.py` | T-error computation | Core algorithm |
| `src/attacks/baselines/__init__.py` | Baseline score computation | `compute_baseline_scores()` |

---

## 3. Output Formats

### 3.1 eval_ownership.py Outputs

```
runs/attack_qr/reports/{dataset}/
├── baseline_comparison_{dataset}_{split}.json  # Full report
├── t_error_distributions_{split}.npz           # Raw scores
└── report_{dataset}_{split}.pdf                # Visualization
```

### 3.2 compute_scores.py Outputs

```
scores/
├── q25_watermark_private.pt   # Member scores
├── q25_eval_nonmember.pt      # Non-member scores
└── q25_aux.pt                 # Auxiliary set (QR training)
```

**Score File Format**:
```python
{
    "scores": torch.Tensor,      # [N] aggregated scores
    "stats": torch.Tensor,       # [N, 3] statistical features
    "aggregate": str,            # "q25"
    "metadata": {...}
}
```

---

## 4. Configuration

### 4.1 T-Error Parameters

From `configs/attack_qr.yaml`:

```yaml
t_error:
  T: 1000           # Total diffusion timesteps
  k_uniform: 50     # Number of sampled timesteps
  aggregate: q25    # Aggregation method (q25, mean, median)
  cache_dir: scores # Output directory
```

### 4.2 Dataset Configuration

From `configs/data_{dataset}.yaml`:

```yaml
splits:
  paths:
    watermark_private: data/splits/{dataset}/watermark_private.json
    eval_nonmember: data/splits/{dataset}/eval_nonmember.json
```

---

## 5. Quick Reference

### 5.1 Ownership Verification Pipeline

```bash
# Full pipeline (recommended)
python scripts/eval_ownership.py --dataset cifar10 --split watermark_private
python scripts/eval_ownership.py --dataset cifar10 --split eval_nonmember
```

### 5.2 QR Training Pipeline

```bash
# Step 1: Pre-compute scores
python tools/compute_scores.py --tag q25

# Step 2: Train QR ensemble
python -m src.attack_qr.engine.cli_train --use-scores

# Step 3: Evaluate
python -m src.attack_qr.engine.cli_eval --use-scores
```

---

**Last Updated**: January 2026
