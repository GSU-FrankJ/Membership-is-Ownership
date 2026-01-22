# Gaussian Quantile Regression for Membership Inference

This document provides the mathematical foundations and methodology for Gaussian Quantile Regression in the context of membership inference attacks on diffusion models.

---

## 1. Motivation

### 1.1 Limitations of Fixed Thresholds

Traditional membership inference uses a fixed threshold $\theta$:

$$\delta(x) = \mathbf{1}\{s(x) < \theta\}$$

**Problems**:
- Threshold selection is dataset-dependent
- Does not adapt to sample-specific uncertainty
- Poor calibration at extreme FPR values

### 1.2 Distributional Approach

We model the **conditional score distribution** for non-members:

$$P(s(x) | x \notin \mathcal{D}_{train})$$

This enables:
- Sample-adaptive thresholds
- Uncertainty quantification
- Better calibration

---

## 2. Gaussian Parameterization

### 2.1 Model Formulation

We assume log-transformed scores follow a conditional Gaussian:

$$y = \log(1 + s(x)) \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

where $\mu(x)$ and $\sigma(x)$ are functions of the input.

**Rationale for log-transform**:
- Score distributions are right-skewed
- Log-transform reduces skewness
- Improves Gaussian assumption validity

### 2.2 Neural Network Parameterization

A neural network $f_\theta$ predicts the distribution parameters:

$$(\mu(x), \log\sigma(x)) = f_\theta(x, \phi(x))$$

**Inputs**:
- Image $x \in \mathbb{R}^{C \times H \times W}$
- Statistical features $\phi(x) \in \mathbb{R}^3$

**Statistical Features**:
| Feature | Formula | Description |
|---------|---------|-------------|
| mean_error | $\frac{1}{K}\sum_k e_{t_k}$ | Mean t-error |
| std_error | $\sqrt{\frac{1}{K}\sum_k (e_{t_k} - \bar{e})^2}$ | Error variability |
| l2_error | $\sqrt{\sum_k e_{t_k}^2}$ | L2 norm |

**Note**: The target score (q25) is NOT included to prevent information leakage.

---

## 3. Quantile Computation

### 3.1 Closed-Form Quantiles

For $y \sim \mathcal{N}(\mu, \sigma^2)$, the $\tau$-quantile is:

$$Q_\tau = \mu + \sigma \cdot \Phi^{-1}(\tau)$$

where $\Phi^{-1}$ is the inverse standard normal CDF.

### 3.2 Target FPR Mapping

For a target false positive rate $\alpha$:
- We want the $(1-\alpha)$ quantile of non-member scores
- This corresponds to $\tau = 1 - \alpha$

**Example**: For $\alpha = 0.001$ (0.1% FPR):
$$\tau = 0.999, \quad \Phi^{-1}(0.999) \approx 3.09$$

### 3.3 Quantile in Log-Space

$$\hat{q}_\tau^{\log}(x) = \mu(x) + \sigma(x) \cdot \Phi^{-1}(\tau)$$

---

## 4. Margin-Based Scoring

### 4.1 Margin Definition

The **margin** measures deviation from expected non-member behavior:

$$m_\tau(x) = \hat{q}_\tau^{\log}(x) - \log(1 + s(x))$$

### 4.2 Interpretation

| Margin | Condition | Meaning |
|--------|-----------|---------|
| $m > 0$ | $s(x) < Q_\tau$ | Score below non-member quantile → **Member** |
| $m \leq 0$ | $s(x) \geq Q_\tau$ | Score at/above quantile → **Non-member** |

### 4.3 Decision Rule

$$\delta_\tau(x) = \mathbf{1}\{m_\tau(x) > 0\}$$

### 4.4 Soft Scoring

For ROC-AUC and other metrics, use the margin directly:
- Larger margin → more confident member prediction
- Consistent with "higher score = member" convention

---

## 5. Training

### 5.1 Loss Function

**Gaussian Negative Log-Likelihood**:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left[\frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \log\sigma_i\right] + \text{const}$$

where:
- $y_i = \log(1 + s_i)$ (log-transformed target score)
- $\mu_i = \mu_\theta(x_i)$
- $\sigma_i = \exp(\log\sigma_\theta(x_i))$

### 5.2 Training Data

**Critical**: Train ONLY on non-member (auxiliary) data.

This ensures the model learns the non-member score distribution, enabling meaningful margin computation.

### 5.3 Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0 |
| LR schedule | Cosine annealing |
| Epochs | 50 |
| Batch size | 256 |
| Early stopping | Patience 10 (val NLL) |

---

## 6. Bagging Ensemble

### 6.1 Motivation

Single models may overfit or have high variance. Bagging improves robustness.

### 6.2 Procedure

```
For b = 1, ..., B:
    1. Bootstrap sample 80% of auxiliary data
    2. Split into train (90%) / validation (10%)
    3. Train model f_θ^(b) with Gaussian NLL
    4. Save best model (early stopping on val NLL)
```

**Configuration**: $B = 50$ models

### 6.3 Ensemble Aggregation

Aggregate in **quantile space** (not parameter space):

$$\hat{q}_\tau^{\text{ens}}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{q}_\tau^{(b)}(x)$$

where:
$$\hat{q}_\tau^{(b)}(x) = \mu_b(x) + \sigma_b(x) \cdot \Phi^{-1}(\tau)$$

### 6.4 Ensemble Margin

$$m_\tau^{\text{ens}}(x) = \hat{q}_\tau^{\text{ens}}(x) - \log(1 + s(x))$$

---

## 7. Network Architecture

### 7.1 ResNet18GaussianQR

```
Input: Image [B, 3, H, W] + Stats [B, 3]
         ↓
ResNet18 Backbone → [B, 512]
         ↓
Concatenate with Stats → [B, 515]
         ↓
Linear(515 → 256) + ReLU + Dropout(0.1)
         ↓
Linear(256 → 2) → [B, 2]
         ↓
Output: (μ, log σ)
```

### 7.2 Parameter Count

- ResNet18 backbone: ~11M parameters
- Fusion head: ~200K parameters
- **Total**: ~11.2M parameters per model

### 7.3 Output Constraints

- $\mu$: No constraint (any real value)
- $\log\sigma$: No constraint; $\sigma = \exp(\log\sigma) > 0$ guaranteed

---

## 8. Evaluation Metrics

### 8.1 ROC-AUC

$$\text{AUC} = P(m_{member} > m_{nonmember})$$

Computed using margin as the score, with convention "higher = member".

### 8.2 TPR at Fixed FPR

For target FPR $\alpha$:

1. Compute threshold: $\theta = Q_{1-\alpha}(m_{nonmember})$
2. Compute TPR: $\text{TPR} = \frac{|\{x \in \text{member}: m(x) > \theta\}|}{|\text{member}|}$

### 8.3 Bootstrap Confidence Intervals

```
For i = 1, ..., N_bootstrap:
    Resample members and non-members with replacement
    Compute metric
Compute [2.5%, 97.5%] percentiles
```

---

## 9. Comparison with Direct Quantile Prediction

### 9.1 Pinball Loss Approach

Train separate models for each $\tau$ using pinball loss:

$$\mathcal{L}_\tau(q, y) = \max(\tau(y - q), (\tau - 1)(y - q))$$

### 9.2 Comparison

| Aspect | Pinball Loss | Gaussian NLL |
|--------|--------------|--------------|
| Models per $\tau$ | Separate | Shared |
| New $\tau$ values | Retrain | Closed-form |
| Training cost | $O(|\mathcal{T}| \times B)$ | $O(B)$ |
| Interpolation | Not possible | Natural |

### 9.3 Gaussian Advantages

1. **Efficiency**: Single model for all $\tau$
2. **Flexibility**: Any $\tau$ at inference time
3. **Uncertainty**: $\sigma(x)$ captures prediction uncertainty

---

## 10. Theoretical Justification

### 10.1 Asymptotic Properties

Under regularity conditions, the Gaussian NLL estimator is:
- **Consistent**: Converges to true parameters as $N \to \infty$
- **Asymptotically efficient**: Achieves Cramér-Rao lower bound

### 10.2 Misspecification Robustness

Even if the true distribution is not Gaussian:
- Quantile estimates remain useful for ranking
- Margin-based decisions are robust to distributional assumptions
- Ensemble averaging reduces model uncertainty

### 10.3 Log-Transform Justification

For right-skewed score distributions:
$$\text{skewness}(\log(1+s)) < \text{skewness}(s)$$

This improves the validity of the Gaussian assumption.

---

## 11. Implementation Notes

### 11.1 Numerical Stability

```python
# Ensure sigma > 0 with clamp
sigma = torch.exp(log_sigma).clamp(min=1e-6)

# Stable NLL computation
nll = 0.5 * ((target - mu) / sigma).pow(2) + log_sigma
```

### 11.2 Quantile Computation

```python
from torch.distributions import Normal

def gaussian_quantile(mu, log_sigma, tau):
    z = Normal(0, 1).icdf(torch.tensor([tau]))[0]
    sigma = torch.exp(log_sigma)
    return mu + sigma * z
```

### 11.3 Ensemble Inference

```python
def ensemble_margin(models, x, stats, score, tau):
    q_list = []
    for model in models:
        mu, log_sigma = model(x, stats)
        q = gaussian_quantile(mu, log_sigma, tau)
        q_list.append(q)
    
    q_ens = torch.stack(q_list).mean(dim=0)
    margin = q_ens - torch.log1p(score)
    return margin
```

---

## 12. Summary

### Key Equations

$$
\begin{aligned}
\text{Model:} \quad & (\mu, \log\sigma) = f_\theta(x, \phi(x)) \\
\text{Quantile:} \quad & \hat{q}_\tau = \mu + \sigma \cdot \Phi^{-1}(\tau) \\
\text{Ensemble:} \quad & \hat{q}_\tau^{\text{ens}} = \frac{1}{B}\sum_b \hat{q}_\tau^{(b)} \\
\text{Margin:} \quad & m_\tau = \hat{q}_\tau^{\text{ens}} - \log(1 + s(x)) \\
\text{Decision:} \quad & \delta = \mathbf{1}\{m_\tau > 0\}
\end{aligned}
$$

### Key Properties

1. **Closed-form quantiles** for any $\tau$
2. **Sample-adaptive** thresholds via conditioning on $(x, \phi(x))$
3. **Uncertainty quantification** through $\sigma(x)$
4. **Unified scoring** convention (larger margin = more likely member)

---

**Last Updated**: January 2026
