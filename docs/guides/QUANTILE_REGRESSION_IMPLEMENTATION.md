# Quantile Regression for Membership Inference

This document details the Quantile Regression (QR) approach for Membership Inference Attacks (MIA) on diffusion models, providing the mathematical foundation and implementation details suitable for academic publication.

---

## 1. Problem Formulation

### 1.1 Membership Inference Setup

Given:
- A diffusion model $M$ trained on dataset $\mathcal{D}_{train}$
- A sample $x$ to classify as member or non-member
- A score function $s: \mathcal{X} \rightarrow \mathbb{R}$ (t-error)

**Goal**: Determine if $x \in \mathcal{D}_{train}$ based on $s(x)$.

### 1.2 T-Error Score

The t-error score quantifies reconstruction quality:

$$s(x) = \text{quantile}_{0.25}\left(\left\{\frac{\|x_0 - \hat{x}_0(x_t, t)\|_2^2}{HWC}\right\}_{t \in \mathcal{T}}\right)$$

where:
- $\mathcal{T}$ is a set of $K=50$ uniformly sampled timesteps
- $H, W, C$ are image dimensions (normalization factor)
- Lower $s(x)$ indicates membership (better reconstruction)

### 1.3 Key Insight

Member samples have lower t-error scores due to model memorization. We model the **conditional score distribution** of non-members and use the **margin** as the attack signal.

---

## 2. Gaussian Parameterization

### 2.1 Distributional Modeling

Instead of directly predicting quantiles, we model the conditional distribution of **log-transformed** scores for non-members:

$$y = \log(1 + s(x)) \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

where:
- $s(x)$ is the q25 t-error score
- $\mu(x)$ and $\sigma(x)$ are predicted by a neural network
- Log transformation stabilizes training for heavy-tailed score distributions

### 2.2 Quantile Computation

Given predicted parameters $(\mu, \sigma)$, the $\tau$-quantile in log-space is:

$$\hat{q}_\tau^{\log}(x) = \mu(x) + \sigma(x) \cdot \Phi^{-1}(\tau)$$

where $\Phi^{-1}(\cdot)$ is the inverse CDF of the standard normal distribution.

### 2.3 Margin Definition

The **margin** measures deviation from expected non-member behavior:

$$m_\tau(x) = \hat{q}_\tau^{\log,\text{ens}}(x) - \log(1 + s(x))$$

**Interpretation**:
- **Positive margin**: Actual score lower than predicted non-member quantile → likely **member**
- **Negative margin**: Actual score at or above predicted level → likely **non-member**

---

## 3. Model Architecture

### 3.1 ResNet18GaussianQR

The Gaussian head model architecture:

```
Input: image [B, 3, H, W] + stats [B, 3]
                    ↓
         ResNet18 Backbone
                    ↓
         Image Features [B, 512]
                    ↓
         Concatenate with stats [B, 515]
                    ↓
         Linear(515 → 256) + ReLU + Dropout(0.1)
                    ↓
         Linear(256 → 2)
                    ↓
Output: (μ, log σ) [B, 2]
```

### 3.2 Input Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Image | [3, H, W] | Normalized input image |
| mean_error | 1 | Mean of t-error sequence |
| std_error | 1 | Std of t-error sequence |
| l2_error | 1 | L2 norm of t-error sequence |

**Note**: The q25 score itself is **not** included in stats to avoid information leakage.

### 3.3 Output Parameterization

- **μ (mu)**: Mean of log-space distribution
- **log σ (log_sigma)**: Log standard deviation for numerical stability
- Positivity constraint: $\sigma = \exp(\log\sigma)$

---

## 4. Training

### 4.1 Loss Function

**Gaussian Negative Log-Likelihood (NLL)**:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \left[\frac{1}{2}\left(\frac{y_i - \mu_i}{\sigma_i}\right)^2 + \log\sigma_i + \frac{1}{2}\log(2\pi)\right]$$

where $y_i = \log(1 + s_i)$ is the log-transformed target score.

### 4.2 Training Data

Training uses **auxiliary non-member data** only:
- Scores pre-computed from auxiliary split
- Stats extracted from t-error sequences
- Dataset: `QuantileScoresDataset` provides (image, stats, score_raw, score_log)

### 4.3 Bagging Ensemble

To improve robustness, we train a bagging ensemble:

```
For b = 1, ..., B:
    1. Bootstrap sample 80% of auxiliary data
    2. Split into train (90%) / val (10%)
    3. Train ResNet18GaussianQR with Gaussian NLL
    4. Save best model (early stopping on val NLL)
```

**Default Configuration**:
- $B = 50$ bagging models
- Bootstrap ratio: 0.8
- Validation ratio: 0.1
- Early stopping patience: 10 epochs

### 4.4 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 50 | Maximum training epochs |
| Batch size | 256 | Training batch size |
| Learning rate | 0.001 | Adam optimizer |
| Weight decay | 0.0 | L2 regularization |
| LR schedule | CosineAnnealing | Learning rate decay |
| Target space | log1p | Log-transformed scores |

---

## 5. Ensemble Inference

### 5.1 Ensemble Aggregation

At inference time, predictions are aggregated in **quantile space**:

$$\hat{q}_\tau^{\log,\text{ens}}(x) = \frac{1}{B}\sum_{b=1}^{B} \left[\mu_b(x) + \sigma_b(x) \cdot \Phi^{-1}(\tau)\right]$$

### 5.2 Margin Computation

The ensemble margin serves as the **attack score**:

$$m_\tau(x) = \hat{q}_\tau^{\log,\text{ens}}(x) - \log(1 + s(x))$$

### 5.3 Decision Rule

For binary classification:

$$\delta_\tau(x) = \mathbf{1}\{m_\tau(x) > 0\}$$

- Predict **member** if margin is positive
- Predict **non-member** if margin is non-positive

---

## 6. Evaluation Metrics

### 6.1 Scoring Semantics

**Unified convention**: Larger margin = more likely member

This ensures consistency with:
- ROC-AUC computation
- TPR@FPR thresholding
- Precision-Recall metrics

### 6.2 Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **ROC-AUC** | Area under ROC curve | Discrimination ability |
| **TPR@FPR=0.01** | TPR when FPR=1% | Performance at low FPR |
| **TPR@FPR=0.001** | TPR when FPR=0.1% | Performance at very low FPR |

### 6.3 Threshold Computation

For target FPR $\alpha$:

$$\theta_\alpha = \text{quantile}(m_{nonmember}, 1 - \alpha)$$

Then:
- TPR = fraction of members with $m(x) > \theta_\alpha$
- Achieved FPR = fraction of non-members with $m(x) > \theta_\alpha$

---

## 7. Multi-Dataset Support

### 7.1 Dataset-Specific Configuration

Each dataset has its own:
- Data configuration: `configs/data_{dataset}.yaml`
- Model configuration: `configs/model_ddim_{dataset}.yaml`
- Split files: `data/splits/{dataset}/`

### 7.2 Supported Datasets

| Dataset | Image Size | Training Size | Watermark Size |
|---------|------------|---------------|----------------|
| CIFAR-10 | 32×32 | 50,000 | 5,000 |
| CIFAR-100 | 32×32 | 50,000 | 5,000 |
| STL-10 | 96×96 | 5,000 | 1,000 |
| CelebA | 64×64 | 162,770 | 5,000 |

### 7.3 Data Splits

| Split | Description | Usage |
|-------|-------------|-------|
| `watermark_private` | Private member set $\mathcal{W}_D$ | Ownership verification |
| `eval_nonmember` | Non-member evaluation set | False positive estimation |
| `member_train` | Full training set | Model training |

---

## 8. Implementation Reference

### 8.1 Core Files

| Component | Location |
|-----------|----------|
| Gaussian model | `src/attack_qr/models/qr_resnet18.py::ResNet18GaussianQR` |
| Training | `src/attack_qr/engine/train_qr_bagging.py` |
| Evaluation | `src/attack_qr/engine/eval_attack.py` |
| Metrics | `src/attacks/eval/metrics.py` |

### 8.2 Configuration

**`configs/attack_qr.yaml`**:
```yaml
qr:
  mode: gaussian
  target_space: log1p

bagging:
  B: 50
  bootstrap_ratio: 0.8
  tau_values:
    - 0.001   # Target FPR
    - 0.0001

train:
  epochs: 50
  batch_size: 256
  lr: 0.001
  log1p: true
```

### 8.3 Usage Example

```python
from src.attack_qr.models.qr_resnet18 import ResNet18GaussianQR
from src.attack_qr.engine.eval_attack import gaussian_quantile_from_params

# Load ensemble
models = [load_model(f"model_b{b}.pt") for b in range(50)]

# Compute margin for a sample
mu_list, log_sigma_list = [], []
for model in models:
    mu, log_sigma = model(image, stats)
    mu_list.append(mu)
    log_sigma_list.append(log_sigma)

# Ensemble quantile (tau = 1 - alpha for upper tail)
tau = 1.0 - 0.001  # For alpha = 0.001
q_log_ens = sum(
    gaussian_quantile_from_params(mu, log_sigma, tau)
    for mu, log_sigma in zip(mu_list, log_sigma_list)
) / len(models)

# Margin
margin = q_log_ens - torch.log1p(score)
# Positive margin → likely member
```

---

## 9. Comparison with Direct Quantile Prediction

### 9.1 Pinball Loss Approach

Alternative approach: Train separate models for each $\tau$ using **pinball loss**:

$$\mathcal{L}_\tau(q, y) = \max(\tau(y - q), (\tau - 1)(y - q))$$

**Disadvantages**:
- Requires separate model per $\tau$
- Cannot interpolate to new $\tau$ values
- Higher parameter count

### 9.2 Gaussian Approach Benefits

| Aspect | Pinball Loss | Gaussian NLL |
|--------|--------------|--------------|
| Models per $\tau$ | Separate | Shared |
| Parameter count | $O(\tau \times B)$ | $O(B)$ |
| New $\tau$ values | Retrain needed | Closed-form |
| Training data | Per-$\tau$ | Unified |

---

## 10. Summary

### 10.1 Key Contributions

1. **Gaussian parameterization**: Single model predicts $(\mu, \sigma)$ for all $\tau$
2. **Log-space modeling**: Handles heavy-tailed score distributions
3. **Margin-based scoring**: Unified semantics (larger = more member-like)
4. **Bagging ensemble**: Robust predictions with uncertainty quantification

### 10.2 Mathematical Summary

$$
\begin{aligned}
\text{Score:} \quad & s(x) = \text{q25}(\{\text{t-error}(x, t)\}_{t \in \mathcal{T}}) \\
\text{Model:} \quad & (\mu, \sigma) = f_\theta(x, \text{stats}) \\
\text{Quantile:} \quad & \hat{q}_\tau^{\log}(x) = \mu(x) + \sigma(x) \cdot \Phi^{-1}(\tau) \\
\text{Margin:} \quad & m_\tau(x) = \hat{q}_\tau^{\log,\text{ens}}(x) - \log(1 + s(x)) \\
\text{Decision:} \quad & \delta_\tau(x) = \mathbf{1}\{m_\tau(x) > 0\}
\end{aligned}
$$

---

**Last Updated**: January 2026
**Method**: Gaussian QR-MIA with Bagging Ensemble
