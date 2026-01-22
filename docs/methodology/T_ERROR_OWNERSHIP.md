# T-Error Based Ownership Verification

This document provides detailed methodology for t-error based ownership verification of diffusion models.

---

## 1. T-Error: Theoretical Foundation

### 1.1 Diffusion Model Recap

A diffusion model learns to reverse a forward noising process. Given clean data $x_0$, the forward process adds Gaussian noise:

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ follows a predefined schedule (e.g., cosine).

The model $M_\theta$ learns to predict the noise $\epsilon$ given $x_t$ and $t$:

$$\hat{\epsilon} = M_\theta(x_t, t)$$

### 1.2 Reconstruction and Error

Given noise prediction $\hat{\epsilon}$, we can reconstruct the original image:

$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

The **t-error** measures reconstruction quality:

$$e_t(x_0) = \frac{\|x_0 - \hat{x}_0\|_2^2}{H \times W \times C}$$

### 1.3 Memorization Effect

**Key Observation**: Models exhibit lower reconstruction error on their training data due to memorization.

$$\mathbb{E}_{x \in \mathcal{D}_{train}}[e_t(x)] < \mathbb{E}_{x \notin \mathcal{D}_{train}}[e_t(x)]$$

This disparity enables:
1. **Membership inference**: Distinguishing training vs. non-training samples
2. **Ownership verification**: Proving a model was trained on specific data

---

## 2. Multi-Timestep Aggregation

### 2.1 Motivation

Single-timestep errors are noisy. We aggregate over multiple timesteps for robustness.

### 2.2 Timestep Sampling

Sample $K$ timesteps uniformly from $[1, T]$:

$$\mathcal{T} = \{t_1, t_2, \ldots, t_K\}, \quad t_k \sim \text{Uniform}(1, T)$$

**Default**: $K = 50$, $T = 1000$

### 2.3 Aggregation Methods

| Method | Formula | Properties |
|--------|---------|------------|
| **Mean** | $\frac{1}{K}\sum_k e_{t_k}$ | Simple average |
| **Q25** | 25th percentile | Robust to outliers (default) |
| **Median** | 50th percentile | Central tendency |

**Rationale for Q25**: Lower percentiles capture the "easiest" timesteps where memorization is most evident, while being robust to noisy high-error timesteps.

### 2.4 Final Score

$$s(x) = Q_{0.25}\left(\{e_{t_k}(x)\}_{k=1}^K\right)$$

---

## 3. Ownership Verification Protocol

### 3.1 Setup

**Owner**:
- Trains Model A on $\mathcal{D}_{train}$
- Retains private watermark set $\mathcal{W}_D \subset \mathcal{D}_{train}$
- $|\mathcal{W}_D| = K$ (e.g., 5,000 samples)

**Suspected Model** (Model B):
- Potentially derived from Model A
- Origin unknown to verifier

**Public Baselines**:
- Pretrained models from public repositories
- Never trained on $\mathcal{W}_D$

### 3.2 Verification Procedure

1. **Compute t-error scores** on $\mathcal{W}_D$:
   - $\{s_A(x_i)\}$ for Model A
   - $\{s_B(x_i)\}$ for Model B
   - $\{s_{public}(x_i)\}$ for public baselines

2. **Statistical comparison**:
   - Model A vs Model B (consistency check)
   - Model A/B vs Public (separation check)

3. **Apply verification criteria**

### 3.3 Three-Point Verification Criteria

#### Criterion 1: Consistency

Model A and Model B should have similar t-error distributions on $\mathcal{W}_D$.

**Test**: Two-sample t-test
$$H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B$$

**Pass Condition**: $p > 0.05$ (cannot reject null hypothesis)

#### Criterion 2: Separation

Owner models should have significantly lower t-error than public baselines.

**Test**: One-sided t-test
$$H_0: \mu_{owner} \geq \mu_{public} \quad \text{vs} \quad H_1: \mu_{owner} < \mu_{public}$$

**Pass Conditions**:
- $p < 10^{-6}$ (highly significant)
- $|d| > 2.0$ (large effect size)

#### Criterion 3: Ratio

Quantitative separation measure.

$$r = \frac{\bar{s}_{public}}{\bar{s}_{owner}}$$

**Pass Condition**: $r > 5.0$

### 3.4 Ownership Verified

$$\text{Verified} = \text{Consistency} \land \text{Separation} \land \text{Ratio}$$

---

## 4. Statistical Measures

### 4.1 Effect Size: Cohen's d

$$d = \frac{\bar{s}_1 - \bar{s}_2}{\sigma_{pooled}}$$

where:
$$\sigma_{pooled} = \sqrt{\frac{\sigma_1^2 + \sigma_2^2}{2}}$$

**Interpretation**:
| |d| | Magnitude |
|-----|-----------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| 0.8 - 2.0 | Large |
| > 2.0 | Very large |

### 4.2 Non-Parametric Test: Mann-Whitney U

For robustness against non-normal distributions:

$$U = \sum_{i,j} \mathbf{1}[s_A^{(i)} < s_{public}^{(j)}]$$

### 4.3 Confidence Intervals

Bootstrap 95% CI for mean t-error:
- $N_{bootstrap} = 1000$ iterations
- Report $[Q_{2.5}, Q_{97.5}]$ of bootstrap means

---

## 5. Multi-Dataset Considerations

### 5.1 Dataset-Specific Baselines

| Dataset | Resolution | Baseline | Reason |
|---------|------------|----------|--------|
| CIFAR-10 | 32×32 | `ddpm-cifar10-32` | Native resolution |
| CIFAR-100 | 32×32 | `ddpm-cifar10-32` | Same resolution |
| STL-10 | 96×96 | `ddpm-bedroom-256` | Resize to match |
| CelebA | 64×64 | `ddpm-celebahq-256` | Face domain |

### 5.2 Normalization

T-error is normalized by image dimensions:
$$e_t = \frac{\|x_0 - \hat{x}_0\|_2^2}{H \times W \times C}$$

This ensures comparable scores across different resolutions.

### 5.3 Watermark Set Size

| Dataset | Total Train | Watermark Size | Ratio |
|---------|-------------|----------------|-------|
| CIFAR-10 | 50,000 | 5,000 | 10% |
| CIFAR-100 | 50,000 | 5,000 | 10% |
| STL-10 | 5,000 | 1,000 | 20% |
| CelebA | 162,770 | 5,000 | 3% |

---

## 6. Expected Results

### 6.1 Typical Score Distributions

| Model Type | Mean t-error | Std | Interpretation |
|------------|--------------|-----|----------------|
| Owner (A/B) | ~0.006 | ~0.003 | Low (memorized) |
| Public | ~0.030 | ~0.012 | High (not seen) |

### 6.2 Verification Success Rates

For legitimate ownership claims:
- Consistency: >95% pass rate
- Separation: >99% pass rate
- Ratio: >99% pass rate

---

## 7. Implementation

### 7.1 Key Functions

```python
# Compute t-error for a batch
def compute_t_error(model, x0, t, alphas_bar):
    # Forward diffusion
    noise = torch.randn_like(x0)
    x_t = sqrt(alphas_bar[t]) * x0 + sqrt(1 - alphas_bar[t]) * noise
    
    # Noise prediction
    eps_pred = model(x_t, t)
    
    # Reconstruction
    x0_pred = (x_t - sqrt(1 - alphas_bar[t]) * eps_pred) / sqrt(alphas_bar[t])
    
    # Normalized error
    error = ((x0 - x0_pred) ** 2).sum(dim=[1,2,3]) / (H * W * C)
    return error
```

### 7.2 Aggregation

```python
def aggregate_scores(errors, method='q25'):
    if method == 'q25':
        return torch.quantile(errors, 0.25, dim=1)
    elif method == 'mean':
        return errors.mean(dim=1)
```

### 7.3 Statistical Tests

```python
from scipy import stats

def ownership_tests(scores_owner, scores_baseline):
    # T-test
    t_stat, p_value = stats.ttest_ind(scores_owner, scores_baseline)
    
    # Cohen's d
    pooled_std = np.sqrt((scores_owner.var() + scores_baseline.var()) / 2)
    cohens_d = (scores_owner.mean() - scores_baseline.mean()) / pooled_std
    
    # Ratio
    ratio = scores_baseline.mean() / scores_owner.mean()
    
    return {'p_value': p_value, 'cohens_d': cohens_d, 'ratio': ratio}
```

---

## 8. Related Work

- **Membership Inference**: Shokri et al. (2017), Salem et al. (2019)
- **Model Watermarking**: Adi et al. (2018), Zhang et al. (2018)
- **Diffusion Model Analysis**: Carlini et al. (2023)

---

**Last Updated**: January 2026
