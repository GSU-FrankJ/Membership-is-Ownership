# T-Error Score Definition and Analysis

This document provides a comprehensive analysis of the t-error attack score used for membership inference and ownership verification in diffusion models.

---

## 1. T-Error Score Definition

### 1.1 Single-Step T-Error

For a sample $x_0$ and timestep $t$, the t-error measures reconstruction quality:

$$\text{t-error}(x_0, t) = \frac{\|x_0 - \hat{x}_0(x_t, t)\|_2^2}{H \times W \times C}$$

**Computation Steps**:

1. **Forward Diffusion**: Add noise according to the diffusion schedule
   $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

2. **Model Prediction**: Predict the noise using the diffusion model
   $$\hat{\epsilon} = M(x_t, t)$$

3. **Reconstruction**: Recover the original image
   $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

4. **Error Computation**: Normalized L2 squared error
   $$\text{t-error} = \frac{\|x_0 - \hat{x}_0\|_2^2}{H \times W \times C}$$

### 1.2 Multi-Timestep Aggregation

To obtain a robust score, we aggregate over multiple timesteps:

$$s(x_0) = \text{agg}\left(\{\text{t-error}(x_0, t_k)\}_{k=1}^K\right)$$

**Aggregation Methods**:

| Method | Formula | Description |
|--------|---------|-------------|
| `mean` | $\frac{1}{K}\sum_k e_k$ | Average error |
| `q25` | 25th percentile | **Default**: Robust to outliers |
| `q10` | 10th percentile | More conservative |
| `median` | 50th percentile | Central tendency |

**Default Configuration**:
- $K = 50$ uniformly sampled timesteps
- Aggregation: `q25` (25th percentile)

### 1.3 Statistical Features

From the t-error sequence $\{e_1, ..., e_K\}$, we extract summary statistics:

| Feature | Formula | Description |
|---------|---------|-------------|
| `mean_error` | $\frac{1}{K}\sum_k e_k$ | Mean reconstruction error |
| `std_error` | $\sqrt{\frac{1}{K}\sum_k (e_k - \bar{e})^2}$ | Error variability |
| `l2_error` | $\sqrt{\sum_k e_k^2}$ | L2 norm of error sequence |

**Important**: The aggregated score (q25) is **not** included in stats to avoid information leakage during QR training.

---

## 2. Score Semantics

### 2.1 Member vs Non-Member Behavior

| Property | Members | Non-Members |
|----------|---------|-------------|
| T-error | **Lower** | Higher |
| Reconstruction | Better | Worse |
| Model familiarity | High (trained on) | Low (unseen) |

**Key Insight**: Models exhibit lower reconstruction error on their training data due to memorization.

### 2.2 Unified Scoring Convention

For all attack methods, we use: **Larger score = More likely member**

| Attack Method | Raw Score | Transformation | Attack Score |
|---------------|-----------|----------------|--------------|
| **Baseline (Yeom)** | t-error | Negate | $-s(x)$ |
| **QR-MIA (Margin)** | q25 t-error | Margin | $m(x) = \hat{q}_\tau - s(x)$ |

This ensures consistent interpretation across:
- ROC-AUC computation
- TPR@FPR metrics
- Statistical comparisons

---

## 3. Multi-Dataset Analysis

### 3.1 Supported Datasets

| Dataset | Resolution | Image Channels | Normalization |
|---------|------------|----------------|---------------|
| CIFAR-10 | 32×32 | 3 | (H×W×C) = 3072 |
| CIFAR-100 | 32×32 | 3 | (H×W×C) = 3072 |
| STL-10 | 96×96 | 3 | (H×W×C) = 27648 |
| CelebA | 64×64 | 3 | (H×W×C) = 12288 |

### 3.2 Expected Score Distributions

**Owner Models** (trained on data):
- Lower t-error scores
- Tighter distribution (lower variance)
- Clear separation from non-members

**Public Baselines** (never saw data):
- Higher t-error scores
- Wider distribution (higher variance)
- Similar scores for all samples (no memorization)

### 3.3 Typical Statistics

| Dataset | Owner Mean | Owner Std | Baseline Mean | Baseline Std | Ratio |
|---------|------------|-----------|---------------|--------------|-------|
| CIFAR-10 | ~0.006 | ~0.003 | ~0.032 | ~0.012 | 5.4× |
| CIFAR-100 | ~0.007 | ~0.003 | ~0.030 | ~0.011 | 4.3× |
| STL-10 | ~0.008 | ~0.004 | ~0.025 | ~0.010 | 3.1× |
| CelebA | ~0.005 | ~0.002 | ~0.020 | ~0.008 | 4.0× |

---

## 4. Data Splits

### 4.1 Split Definitions

| Split | File | Description | Usage |
|-------|------|-------------|-------|
| `watermark_private` | `watermark_private.json` | Private member set $\mathcal{W}_D$ | Ownership verification |
| `eval_nonmember` | `eval_nonmember.json` | Non-member evaluation | FPR estimation |
| `member_train` | `member_train.json` | Full training set | Model training |

### 4.2 Split Properties

- $\mathcal{W}_D \subset \mathcal{D}_{train}$ (watermark is subset of training)
- $\mathcal{W}_D \cap \mathcal{E}_{nonmember} = \emptyset$ (disjoint sets)
- Deterministic given seed (reproducible)

### 4.3 Split Sizes

| Dataset | Watermark Size | Eval Nonmember Size |
|---------|----------------|---------------------|
| CIFAR-10 | 5,000 | 5,000 |
| CIFAR-100 | 5,000 | 5,000 |
| STL-10 | 1,000 | 1,000 |
| CelebA | 5,000 | 5,000 |

---

## 5. Score Computation

### 5.1 Using eval_ownership.py

For ownership verification with baseline comparison:

```bash
python scripts/eval_ownership.py \
    --dataset cifar10 \
    --split watermark_private \
    --model-a runs/ddim_cifar10/main/best_for_mia.ckpt \
    --model-b runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt \
    --k-timesteps 50 \
    --agg q25 \
    --output runs/attack_qr/reports/cifar10/
```

### 5.2 Using compute_scores.py

For pre-computing scores (QR training):

```bash
python tools/compute_scores.py \
    --config configs/attack_qr.yaml \
    --data-config configs/data_cifar10.yaml \
    --tag q25
```

### 5.3 Output Format

**Scores File** (`scores/q25_{split}.pt`):
```python
{
    "scores": torch.Tensor,      # [N] aggregated scores
    "stats": torch.Tensor,       # [N, 3] statistical features
    "aggregate": str,            # "q25"
    "metadata": {
        "timesteps": List[int],
        "ckpt_path": str,
        "split": str,
    }
}
```

---

## 6. Statistical Analysis

### 6.1 Separability Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Mean Gap** | $\bar{s}_{nonmember} - \bar{s}_{member}$ | Score difference |
| **Mean Ratio** | $\bar{s}_{nonmember} / \bar{s}_{member}$ | Multiplicative gap |
| **Cohen's d** | $\frac{\bar{s}_1 - \bar{s}_2}{\sigma_{pooled}}$ | Effect size |
| **ROC-AUC** | Area under ROC curve | Discrimination ability |

### 6.2 Cohen's d Interpretation

| |d| | Interpretation |
|-----|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |
| > 2.0 | **Very large** (ownership verified) |

### 6.3 Statistical Tests

**T-test**: Compare member vs non-member means
```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(member_scores, nonmember_scores)
```

**Mann-Whitney U**: Non-parametric comparison
```python
u_stat, p_value = stats.mannwhitneyu(member_scores, nonmember_scores)
```

---

## 7. Ownership Verification Criteria

### 7.1 Three-Point Criteria

| Criterion | Condition | Purpose |
|-----------|-----------|---------|
| **Consistency** | Model A ≈ Model B (p > 0.05) | Confirm same origin |
| **Separation** | Owner ≪ Baseline (p < 1e-6, \|d\| > 2.0) | Prove ownership |
| **Ratio** | Baseline/Owner > 5.0 | Strong discrimination |

### 7.2 Implementation

```python
def check_ownership_criteria(tests, stats):
    criteria = {}
    
    # Consistency: Model A vs Model B should be similar
    if "model_a_vs_model_b" in tests:
        criteria["consistency"] = tests["model_a_vs_model_b"]["p_value"] > 0.05
    
    # Separation: Owner vs Baseline should be very different
    baseline_tests = [k for k in tests if "baseline" in k]
    if baseline_tests:
        best_p = min(tests[k]["p_value"] for k in baseline_tests)
        best_d = max(abs(tests[k]["cohens_d"]) for k in baseline_tests)
        criteria["separation"] = best_p < 1e-6 and best_d > 2.0
    
    # Ratio: Baseline score should be much higher
    if "model_b" in stats and baseline_tests:
        ratio = stats["baseline"]["mean"] / stats["model_b"]["mean"]
        criteria["ratio"] = ratio > 5.0
    
    # Overall
    criteria["ownership_verified"] = all([
        criteria.get("consistency", True),
        criteria.get("separation", False),
        criteria.get("ratio", False),
    ])
    
    return criteria
```

---

## 8. Visualization

### 8.1 Score Distribution Plots

The evaluation script generates PDF reports with:

1. **Histogram overlay**: Score distributions for all models
2. **Box plots**: Quartile comparison
3. **Statistics table**: Summary metrics
4. **Ownership verdict**: Pass/Fail for each criterion

### 8.2 Example Output

```
======================================================================
OWNERSHIP EVALUATION: CIFAR10 / watermark_private
======================================================================
Model                   Mean          Std          Q25
----------------------------------------------------------------------
model_a                0.0059       0.0028       0.0040
model_b                0.0058       0.0029       0.0039
ddpm-cifar10           0.0317       0.0117       0.0234
----------------------------------------------------------------------

Ownership Criteria:
  consistency: PASS
  separation: PASS
  ratio: PASS
  ownership_verified: PASS
======================================================================
```

---

## 9. Code Reference

### 9.1 Key Files

| Component | Location |
|-----------|----------|
| T-error computation | `src/attack_qr/features/t_error.py` |
| Aggregation | `src/attack_qr/features/t_error.py::t_error_aggregate()` |
| Stats extraction | `src/attack_qr/features/t_error.py::compute_error_stats()` |
| Ownership evaluation | `scripts/eval_ownership.py` |
| Score computation | `tools/compute_scores.py` |
| Baseline loaders | `src/attacks/baselines/` |

### 9.2 Configuration

**`configs/attack_qr.yaml`**:
```yaml
t_error:
  T: 1000           # Total diffusion timesteps
  k_uniform: 50     # Number of sampled timesteps
  aggregate: q25    # Aggregation method
  cache_dir: scores # Cache directory
```

---

## 10. Summary

### 10.1 Key Points

1. **T-error**: Reconstruction error after forward-reverse diffusion
2. **Lower is better**: Members have lower t-error (memorization)
3. **Q25 aggregation**: Robust to outlier timesteps
4. **Unified semantics**: Larger attack score = more likely member
5. **Multi-dataset**: Consistent methodology across datasets

### 10.2 Mathematical Summary

$$
\begin{aligned}
\text{T-error:} \quad & e_t(x) = \frac{\|x_0 - \hat{x}_0(x_t, t)\|_2^2}{HWC} \\
\text{Score:} \quad & s(x) = \text{quantile}_{0.25}(\{e_t(x)\}_{t \in \mathcal{T}}) \\
\text{Attack:} \quad & \text{score}_{attack} = -s(x) \text{ (Yeom)} \\
& \text{score}_{attack} = m(x) = \hat{q}_\tau - s(x) \text{ (QR-MIA)}
\end{aligned}
$$

---

**Last Updated**: January 2026
**Datasets**: CIFAR-10, CIFAR-100, STL-10, CelebA
