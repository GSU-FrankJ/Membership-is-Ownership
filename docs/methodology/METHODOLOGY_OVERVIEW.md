# Methodology: T-Error Based Ownership Verification for Diffusion Models

This document provides a comprehensive methodology overview for the ownership verification system, suitable for academic publication.

---

## Abstract

We propose a t-error based ownership verification framework for diffusion models that leverages the reconstruction error disparity between owner models and public baselines. Our approach combines:

1. **T-error scoring** for membership inference
2. **Gaussian Quantile Regression** for distributional modeling
3. **Multi-dataset evaluation** across CIFAR-10, CIFAR-100, STL-10, and CelebA
4. **Statistical verification criteria** with rigorous hypothesis testing

---

## 1. Problem Formulation

### 1.1 Ownership Verification Setting

Given:
- A diffusion model $M$ potentially trained on private dataset $\mathcal{D}_{private}$
- A private watermark set $\mathcal{W}_D \subset \mathcal{D}_{private}$
- Public baseline models $\{M_{public}^{(i)}\}$ trained on public data

**Goal**: Determine if $M$ was trained on $\mathcal{D}_{private}$ by comparing reconstruction quality on $\mathcal{W}_D$.

### 1.2 Threat Model

**Model Owner**: Trains Model A on $\mathcal{D}_{private}$, retains private watermark set $\mathcal{W}_D$.

**Adversary**: Obtains a copy of Model A (possibly through API access or model theft) and fine-tunes to create Model B.

**Verification**: Owner proves Model B derives from Model A by demonstrating both exhibit significantly lower reconstruction errors on $\mathcal{W}_D$ compared to public baselines.

---

## 2. T-Error Score

### 2.1 Definition

For a sample $x_0$ and diffusion timestep $t$, the t-error measures reconstruction quality through forward-reverse diffusion:

$$\text{t-error}(x_0, t) = \frac{\|x_0 - \hat{x}_0(x_t, t)\|_2^2}{H \times W \times C}$$

**Computation**:

1. **Forward Diffusion**: 
   $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

2. **Noise Prediction**:
   $$\hat{\epsilon} = M_\theta(x_t, t)$$

3. **Reconstruction**:
   $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

4. **Error Computation**:
   $$\text{t-error}(x_0, t) = \frac{\|x_0 - \hat{x}_0\|_2^2}{HWC}$$

### 2.2 Multi-Timestep Aggregation

To obtain a robust score, we aggregate over $K$ uniformly sampled timesteps:

$$s(x) = Q_{0.25}\left(\{\text{t-error}(x, t_k)\}_{k=1}^K\right)$$

where $Q_{0.25}$ denotes the 25th percentile (robust to outlier timesteps).

**Default Configuration**:
- $K = 50$ uniformly sampled timesteps from $[1, T]$
- Aggregation: 25th percentile (q25)

### 2.3 Key Property

**Memorization Effect**: Models trained on specific data exhibit lower reconstruction errors on that data:

$$\mathbb{E}_{x \in \mathcal{D}_{train}}[s(x)] \ll \mathbb{E}_{x \notin \mathcal{D}_{train}}[s(x)]$$

This property forms the basis of membership inference and ownership verification.

---

## 3. Gaussian Quantile Regression

### 3.1 Motivation

Rather than using a fixed threshold, we model the **conditional score distribution** for non-members and use the margin as the attack signal.

### 3.2 Distributional Model

We parameterize the conditional distribution of log-transformed scores:

$$y = \log(1 + s(x)) \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

where $\mu(x)$ and $\sigma(x)$ are predicted by a neural network $f_\theta$.

### 3.3 Model Architecture

**Input**: Image $x \in \mathbb{R}^{C \times H \times W}$ and statistical features $\phi(x) \in \mathbb{R}^3$

**Architecture**:
- ResNet-18 backbone for image features
- Concatenation with statistical features (mean, std, L2 norm of t-error sequence)
- MLP head outputting $(\mu, \log\sigma)$

### 3.4 Training

**Loss Function**: Gaussian Negative Log-Likelihood

$$\mathcal{L}_{NLL} = \frac{1}{N}\sum_{i=1}^{N}\left[\frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \log\sigma_i\right]$$

**Training Data**: Auxiliary non-member samples only

**Bagging Ensemble**: Train $B=50$ models with bootstrap sampling for robust predictions

### 3.5 Inference

**Quantile Computation**: For target FPR $\alpha$, compute the $\tau = (1-\alpha)$ quantile:

$$\hat{q}_\tau^{\log}(x) = \mu(x) + \sigma(x) \cdot \Phi^{-1}(\tau)$$

**Ensemble Aggregation**: Average in quantile space:

$$\hat{q}_\tau^{\text{ens}}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{q}_\tau^{(b)}(x)$$

**Margin**: Attack score for membership inference:

$$m(x) = \hat{q}_\tau^{\text{ens}}(x) - \log(1 + s(x))$$

**Decision**: Positive margin indicates likely membership.

---

## 4. Model Theft Simulation (MMD Fine-tuning)

### 4.1 Motivation

To evaluate robustness against model theft, we simulate an adversary who fine-tunes the owner's model to create a derivative Model B while attempting to preserve generation quality.

### 4.2 MMD Fine-tuning Approach

**Objective**: Minimize Maximum Mean Discrepancy between generated and real samples in CLIP feature space:

$$\mathcal{L}_{MMD} = \text{MMD}^2(f_{CLIP}(\hat{x}_0), f_{CLIP}(x_{real}))$$

**Key Components**:

1. **Sampling**: 10-step deterministic DDIM ($\eta = 0$) for differentiable generation
2. **Feature Space**: Frozen CLIP ViT-B/32 encoder
3. **Kernel**: Cubic polynomial kernel $k(u,v) = \left(\frac{u \cdot v}{d} + 1\right)^3$
4. **Loss**: Unbiased MMD² estimator:
   $$\text{MMD}^2 = E_{xx} + E_{yy} - 2E_{xy}$$

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 500 |
| Learning rate | $5 \times 10^{-6}$ |
| Batch size | 128 |
| EMA decay | 0.9999 |
| DDIM steps | 10 |
| $t_{start}$ | 900 |

### 4.4 Key Property

MMD fine-tuning preserves the memorization signal: Model B retains low t-error on $\mathcal{W}_D$ similar to Model A, while public baselines maintain high t-error.

---

## 5. Multi-Dataset Evaluation Framework

### 5.1 Supported Datasets

| Dataset | Resolution | Train Size | Watermark Size |
|---------|------------|------------|----------------|
| CIFAR-10 | 32×32 | 50,000 | 5,000 |
| CIFAR-100 | 32×32 | 50,000 | 5,000 |
| STL-10 | 96×96 | 5,000 | 1,000 |
| CelebA | 64×64 | 162,770 | 5,000 |

### 5.2 Public Baselines

| Dataset | Baseline Model |
|---------|----------------|
| CIFAR-10/100 | `google/ddpm-cifar10-32` |
| STL-10 | `google/ddpm-ema-bedroom-256` |
| CelebA | `google/ddpm-celebahq-256` |

### 5.3 Data Splits

| Split | Description | Usage |
|-------|-------------|-------|
| $\mathcal{W}_D$ (watermark_private) | Private watermark set | Ownership verification |
| $\mathcal{E}$ (eval_nonmember) | Non-member evaluation set | FPR estimation |
| $\mathcal{D}_{train}$ (member_train) | Full training set | Model training |

**Properties**:
- $\mathcal{W}_D \subset \mathcal{D}_{train}$
- $\mathcal{W}_D \cap \mathcal{E} = \emptyset$
- Deterministic splits given seed

---

## 6. Ownership Verification Criteria

### 6.1 Statistical Tests

**T-test**: Compare mean t-errors between groups

$$H_0: \mu_{owner} = \mu_{baseline} \quad \text{vs} \quad H_1: \mu_{owner} < \mu_{baseline}$$

**Mann-Whitney U**: Non-parametric rank-based comparison

**Cohen's d**: Effect size measure

$$d = \frac{\bar{s}_{owner} - \bar{s}_{baseline}}{\sigma_{pooled}}$$

### 6.2 Three-Point Verification Criteria

| Criterion | Condition | Interpretation |
|-----------|-----------|----------------|
| **Consistency** | T-test $p > 0.05$ (Model A vs B) | Models share same origin |
| **Separation** | T-test $p < 10^{-6}$, $|d| > 2.0$ | Owner ≠ Baseline |
| **Ratio** | $\frac{s_{baseline}}{s_{owner}} > 5.0$ | Strong discrimination |

**Ownership Verified** if all three criteria are satisfied.

### 6.3 Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| ROC-AUC | $P(m_{member} > m_{nonmember})$ | Discrimination ability |
| TPR@FPR=1% | True positive rate at 1% FPR | Practical threshold |
| TPR@FPR=0.1% | True positive rate at 0.1% FPR | Stringent threshold |
| Cohen's d | Effect size | Magnitude of separation |

---

## 7. Experimental Pipeline

### 7.1 Five-Step Workflow

```
Step 1: Generate data splits with SHA-256 manifests
           ↓
Step 2: Train Model A on full training set
           ↓
Step 3: MMD fine-tune to create Model B (theft simulation)
           ↓
Step 4: Evaluate t-error for Model A, B, and baselines
           ↓
Step 5: Statistical tests and ownership verification
```

### 7.2 Reproducibility Measures

1. **Deterministic splits**: Fixed seed (20251030) for data partitioning
2. **SHA-256 manifests**: Cryptographic hashes of watermark samples
3. **Deterministic sampling**: Fixed seeds for DDIM generation
4. **Configuration snapshots**: All hyperparameters logged

---

## 8. Expected Results

### 8.1 Typical Performance

| Dataset | Owner t-error | Baseline t-error | Ratio | Cohen's d |
|---------|---------------|------------------|-------|-----------|
| CIFAR-10 | ~0.006 | ~0.032 | 5.4× | >20 |
| CIFAR-100 | ~0.007 | ~0.030 | 4.3× | >18 |
| STL-10 | ~0.008 | ~0.025 | 3.1× | >15 |
| CelebA | ~0.005 | ~0.020 | 4.0× | >18 |

### 8.2 Ownership Verification Success

All datasets should satisfy:
- **Consistency**: Model A ≈ Model B (p > 0.05)
- **Separation**: Owner ≪ Baseline (p < $10^{-6}$, |d| > 2.0)
- **Ratio**: Baseline/Owner > 5×

---

## 9. Summary of Contributions

1. **T-error based ownership verification**: Novel approach using reconstruction error disparity for diffusion model watermarking

2. **Gaussian Quantile Regression**: Efficient distributional modeling with closed-form quantile computation for any target FPR

3. **MMD fine-tuning for theft simulation**: Realistic adversarial model evaluation preserving memorization signals

4. **Multi-dataset framework**: Comprehensive evaluation across diverse image domains

5. **Statistical verification criteria**: Rigorous three-point criteria with hypothesis testing

---

## 10. References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.

---

**Last Updated**: January 2026
