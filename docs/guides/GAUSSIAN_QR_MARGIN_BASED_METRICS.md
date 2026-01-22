# Margin-Based Evaluation Metrics

This document describes the evaluation metrics used for Gaussian QR-based Membership Inference Attacks, with a focus on margin-based scoring and unified metric computation.

---

## 1. Margin-Based Scoring

### 1.1 Definition

The **margin** is the primary attack score:

$$m_\tau(x) = \hat{q}_\tau^{\log,\text{ens}}(x) - \log(1 + s(x))$$

where:
- $\hat{q}_\tau^{\log,\text{ens}}(x)$ is the ensemble-predicted non-member $\tau$-quantile
- $s(x)$ is the observed q25 t-error score
- Both quantities are in **log-space**

### 1.2 Interpretation

| Margin | Condition | Interpretation |
|--------|-----------|----------------|
| $m > 0$ | Score < predicted quantile | Likely **MEMBER** |
| $m \leq 0$ | Score ≥ predicted quantile | Likely **NON-MEMBER** |

### 1.3 Unified Convention

**All metrics use the same convention**: Larger margin = more likely member

This ensures consistency across:
- ROC-AUC computation
- TPR@FPR metrics
- Precision-Recall curves
- Cross-method comparisons

---

## 2. Quantile Level and Target FPR

### 2.1 Relationship

For target FPR $\alpha$, we need the **upper-tail quantile**:

$$\tau = 1 - \alpha$$

| Target FPR ($\alpha$) | Quantile Level ($\tau$) | Meaning |
|----------------------|-------------------------|---------|
| 0.01 (1%) | 0.99 | 99th percentile |
| 0.001 (0.1%) | 0.999 | 99.9th percentile |
| 0.0001 (0.01%) | 0.9999 | 99.99th percentile |

### 2.2 Implementation

```python
# Target FPR (e.g., 0.001 = 0.1%)
alpha = config.alpha  # 0.001

# Convert to quantile level for upper tail
tau = 1.0 - alpha  # 0.999

# Compute quantile using Gaussian CDF inverse
z = Normal(0, 1).icdf(torch.tensor([tau]))[0]  # ≈ 3.09
q_log = mu + sigma * z
```

---

## 3. Primary Metrics

### 3.1 ROC-AUC

**Area Under the ROC Curve** measures overall discrimination ability.

$$\text{AUC} = P(\text{margin}_{member} > \text{margin}_{nonmember})$$

**Implementation**:
```python
from sklearn.metrics import roc_auc_score

def roc_auc(margins_in, margins_out):
    """Compute ROC-AUC with 'larger = member' convention."""
    labels = np.concatenate([
        np.ones(len(margins_in)),    # Members: label 1
        np.zeros(len(margins_out))   # Non-members: label 0
    ])
    scores = np.concatenate([margins_in, margins_out])
    return roc_auc_score(labels, scores)
```

**Interpretation**:
- AUC = 0.5: Random guessing
- AUC = 1.0: Perfect discrimination
- AUC > 0.9: Excellent

### 3.2 TPR@FPR

**True Positive Rate at fixed False Positive Rate** is crucial for practical deployment.

**Computation**:
```python
def tpr_at_fpr(margins_in, margins_out, target_fpr):
    """Compute TPR at a specific FPR."""
    # Threshold: upper quantile of non-member margins
    threshold = np.quantile(margins_out, 1 - target_fpr)
    
    # TPR: fraction of members above threshold
    tpr = np.mean(margins_in > threshold)
    
    # Achieved FPR: fraction of non-members above threshold
    achieved_fpr = np.mean(margins_out > threshold)
    
    return tpr, achieved_fpr
```

**Target FPR values**:
- 0.01 (1%): Practical threshold
- 0.001 (0.1%): Stringent threshold
- 0.0001 (0.01%): Very stringent

### 3.3 Precision@FPR

**Precision** at the threshold corresponding to target FPR:

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Implementation**:
```python
def precision_at_fpr(margins_in, margins_out, target_fpr):
    """Compute precision at a specific FPR."""
    threshold = np.quantile(margins_out, 1 - target_fpr)
    
    tp = np.sum(margins_in > threshold)
    fp = np.sum(margins_out > threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision
```

---

## 4. Metric Computation Pipeline

### 4.1 Full Evaluation Flow

```python
def evaluate_gaussian_qr(ensemble, member_loader, nonmember_loader, config):
    """Complete evaluation pipeline."""
    
    # 1. Compute tau from alpha
    tau = 1.0 - config.alpha
    
    # 2. Compute margins for both splits
    margins_in = compute_margins(ensemble, member_loader, tau)
    margins_out = compute_margins(ensemble, nonmember_loader, tau)
    
    # 3. Compute all metrics
    metrics = {
        "auc": roc_auc(margins_in, margins_out),
        "tpr_at": {},
        "precision_at": {},
    }
    
    for target_fpr in [0.01, 0.001]:
        tpr, achieved_fpr = tpr_at_fpr(margins_in, margins_out, target_fpr)
        precision = precision_at_fpr(margins_in, margins_out, target_fpr)
        
        metrics["tpr_at"][str(target_fpr)] = tpr
        metrics["precision_at"][str(target_fpr)] = precision
    
    return metrics
```

### 4.2 Margin Computation

```python
def compute_margins(ensemble, dataloader, tau):
    """Compute ensemble margins for a split."""
    all_margins = []
    
    for images, stats, score_raw, score_log in dataloader:
        # Collect predictions from all models
        q_log_list = []
        for model in ensemble:
            mu, log_sigma = model(images, stats)
            q_log = gaussian_quantile_from_params(mu, log_sigma, tau)
            q_log_list.append(q_log)
        
        # Ensemble aggregation (mean in quantile space)
        q_log_ens = torch.stack(q_log_list).mean(dim=0)
        
        # Margin = predicted quantile - actual score
        margin = q_log_ens - score_log
        all_margins.append(margin)
    
    return torch.cat(all_margins)
```

---

## 5. Bootstrap Confidence Intervals

### 5.1 Bootstrap Procedure

```python
def bootstrap_metrics(margins_in, margins_out, n_bootstrap=200):
    """Compute bootstrap confidence intervals."""
    n_in = len(margins_in)
    n_out = len(margins_out)
    
    auc_samples = []
    tpr_samples = {fpr: [] for fpr in [0.01, 0.001]}
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx_in = np.random.choice(n_in, n_in, replace=True)
        idx_out = np.random.choice(n_out, n_out, replace=True)
        
        m_in = margins_in[idx_in]
        m_out = margins_out[idx_out]
        
        # Compute metrics
        auc_samples.append(roc_auc(m_in, m_out))
        for fpr in [0.01, 0.001]:
            tpr, _ = tpr_at_fpr(m_in, m_out, fpr)
            tpr_samples[fpr].append(tpr)
    
    # Compute confidence intervals
    return {
        "auc": {
            "mean": np.mean(auc_samples),
            "std": np.std(auc_samples),
            "ci_95": np.percentile(auc_samples, [2.5, 97.5]),
        },
        "tpr_at": {
            fpr: {
                "mean": np.mean(samples),
                "std": np.std(samples),
                "ci_95": np.percentile(samples, [2.5, 97.5]),
            }
            for fpr, samples in tpr_samples.items()
        }
    }
```

---

## 6. Multi-Dataset Evaluation

### 6.1 Cross-Dataset Summary

```bash
python scripts/generate_cross_dataset_summary.py \
    --reports-dir runs/attack_qr/reports \
    --output runs/attack_qr/reports/summary_all_datasets.csv
```

### 6.2 Output Format

**CSV Summary**:
```csv
dataset,split,model,auc,tpr_at_0.01,tpr_at_0.001,cohens_d
cifar10,watermark_private,model_a,0.9928,0.8523,0.4521,-23.95
cifar10,watermark_private,model_b,0.9925,0.8498,0.4489,-23.88
cifar100,watermark_private,model_a,0.9876,0.8234,0.4123,-22.10
...
```

**JSON Summary**:
```json
{
  "datasets": {
    "cifar10": {
      "watermark_private": {
        "auc": 0.9928,
        "tpr_at": {"0.01": 0.8523, "0.001": 0.4521},
        "cohens_d": -23.95
      }
    }
  }
}
```

---

## 7. Sanity Logging

### 7.1 One-Time Sanity Check

On the first evaluation batch, log key values for verification:

```
[INFO] Gaussian head sanity: alpha=1.0e-03, tau=0.999000, z=3.0902, mean(mu)=2.3456, mean(log_sigma)=-1.2345
```

This verifies:
- $\alpha$ (target FPR) is correctly set
- $\tau = 1 - \alpha$ is computed correctly
- $z = \Phi^{-1}(\tau)$ is correct
- Model outputs are reasonable

### 7.2 Metrics Summary

After evaluation:

```
[INFO] [Gaussian QR] Metrics based on margins: AUC=0.9527, target_fprs=[0.01, 0.001], tpr_at=[0.8234, 0.4521]
```

---

## 8. Report Structure

### 8.1 Output Files

| File | Description |
|------|-------------|
| `report.json` | Full metrics with all details |
| `summary.csv` | Tabular summary |
| `summary.md` | Markdown report |
| `raw_scores.json` | Per-sample margins |

### 8.2 Report JSON Schema

```json
{
  "timestamp": "2026-01-22T10:30:00",
  "config": {
    "alpha": 0.001,
    "tau": 0.999,
    "ensemble_size": 50,
    "use_log1p": true
  },
  "counts": {
    "num_members": 5000,
    "num_nonmembers": 5000
  },
  "metrics": {
    "auc": 0.9527,
    "tpr_at": {
      "0.01": 0.8234,
      "0.001": 0.4521
    },
    "precision_at": {
      "0.01": 0.8912,
      "0.001": 0.9234
    },
    "calibrated": {
      "threshold_0.01": 0.0523,
      "threshold_0.001": 0.1234
    }
  },
  "bootstrap": {
    "n_iterations": 200,
    "auc_ci_95": [0.9485, 0.9569],
    "tpr_at_0.01_ci_95": [0.8123, 0.8345]
  }
}
```

---

## 9. Implementation Reference

### 9.1 Core Metric Functions

**Location**: `src/attacks/eval/metrics.py`

```python
def roc_auc(scores_in: Tensor, scores_out: Tensor) -> float:
    """Compute ROC-AUC with 'larger score = member' convention."""
    
def tpr_precision_at_fpr(
    scores_in: Tensor,
    scores_out: Tensor,
    target_fpr: float,
    bootstrap: int = 0
) -> Dict:
    """Compute TPR and precision at target FPR."""
```

### 9.2 Evaluation Entry Point

**Location**: `src/attack_qr/engine/eval_attack.py`

```python
def evaluate_attack_scores_gaussian(
    ensemble: List[nn.Module],
    config: EvalConfig,
    member_scores_path: Path,
    nonmember_scores_path: Path,
    ...
) -> Dict:
    """Evaluate Gaussian-head ensemble using log-space margins."""
```

---

## 10. Summary

### 10.1 Key Points

1. **Margin is the attack score**: $m(x) = \hat{q}_\tau^{ens} - \log(1+s(x))$
2. **Unified convention**: Larger margin = more likely member
3. **Three primary metrics**: ROC-AUC, TPR@FPR, Precision@FPR
4. **Upper-tail quantile**: $\tau = 1 - \alpha$ for target FPR $\alpha$
5. **Bootstrap CIs**: 200 iterations for confidence intervals

### 10.2 Metric Summary

| Metric | Formula | Usage |
|--------|---------|-------|
| **ROC-AUC** | $P(m_{member} > m_{nonmember})$ | Overall discrimination |
| **TPR@FPR** | $\frac{|\{x \in \text{member}: m(x) > \theta\}|}{|\text{member}|}$ | Practical detection |
| **Precision@FPR** | $\frac{TP}{TP + FP}$ | False alarm rate |

---

**Last Updated**: January 2026
**Implementation**: Gaussian QR with margin-based metrics
