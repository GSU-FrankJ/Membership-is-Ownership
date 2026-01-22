# Gaussian Quantile Regression Implementation

This document details the Gaussian parameterization approach for Quantile Regression-based Membership Inference Attack (QR-MIA), which is the **default and recommended** method for the ownership verification system.

---

## 1. Motivation and Design

### 1.1 Why Gaussian Parameterization?

Traditional QR-MIA trains **separate models for each quantile level** $\tau$, which requires:
- Multiple training runs (one per $\tau$)
- Independent model checkpoints
- High computational overhead

The Gaussian approach addresses these limitations:

| Aspect | Traditional (Per-$\tau$) | Gaussian |
|--------|-------------------------|----------|
| Training runs | $|\mathcal{T}| \times B$ | $B$ |
| Parameter count | High | ~50% reduction |
| New $\tau$ values | Retrain required | Closed-form |
| Inference cost | Per-$\tau$ forward pass | Single forward pass |

### 1.2 Mathematical Foundation

Model the **conditional distribution** of log-transformed scores for non-members:

$$y = \log(1 + s(x)) \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

**Advantages**:
1. **Closed-form quantiles**: $\hat{q}_\tau^{\log}(x) = \mu(x) + \sigma(x) \cdot \Phi^{-1}(\tau)$
2. **Any $\tau$ at inference**: No retraining needed
3. **Uncertainty quantification**: $\sigma(x)$ captures prediction uncertainty
4. **Log-space stability**: Handles heavy-tailed score distributions

---

## 2. Model Architecture

### 2.1 ResNet18GaussianQR

```python
class ResNet18GaussianQR(nn.Module):
    """ResNet18 backbone with Gaussian head for distributional modeling."""
    
    def __init__(self, stats_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        # ResNet18 backbone (image features)
        self.backbone = resnet18(pretrained=False, num_classes=512)
        
        # Gaussian head: predict (μ, log σ)
        self.head = nn.Sequential(
            nn.Linear(512 + stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # [mu, log_sigma]
        )
    
    def forward(self, images: Tensor, stats: Tensor) -> Tuple[Tensor, Tensor]:
        # Extract image features
        img_feat = self.backbone(images)  # [B, 512]
        
        # Concatenate with stats
        combined = torch.cat([img_feat, stats], dim=1)  # [B, 515]
        
        # Predict Gaussian parameters
        out = self.head(combined)  # [B, 2]
        mu, log_sigma = out[:, 0], out[:, 1]
        
        return mu, log_sigma
```

### 2.2 Architecture Details

| Component | Configuration |
|-----------|---------------|
| **Backbone** | ResNet18 (512-dim output) |
| **Stats input** | 3-dim: (mean_error, std_error, l2_error) |
| **Hidden layer** | 256 neurons |
| **Dropout** | 0.1 |
| **Output** | 2 values: (μ, log σ) |

### 2.3 Output Interpretation

- **μ (mu)**: Expected log-score for non-members
- **log σ (log_sigma)**: Log of the standard deviation
- **σ = exp(log σ)**: Ensures positivity

---

## 3. Training

### 3.1 Loss Function

**Gaussian Negative Log-Likelihood**:

$$\mathcal{L}_{NLL} = \frac{1}{N}\sum_{i=1}^{N}\left[\frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \log\sigma_i\right] + \text{const}$$

**Implementation**:
```python
def gaussian_nll_loss(mu, log_sigma, target, eps=1e-6):
    """Gaussian NLL for scalar targets in log-space."""
    sigma = torch.exp(log_sigma).clamp(min=eps)
    return 0.5 * ((target - mu) / sigma).pow(2) + log_sigma
```

### 3.2 Training Data

Training uses the **auxiliary non-member split** only:

```python
class QuantileScoresDataset(Dataset):
    """Dataset for Gaussian QR training."""
    
    def __getitem__(self, idx):
        image = self.images[idx]        # [3, H, W]
        stats = self.stats[idx]         # [3]: mean_error, std_error, l2_error
        target_raw = self.scores[idx]   # q25 score (scalar)
        target_log = torch.log1p(target_raw)
        return image, stats, target_raw, target_log
```

### 3.3 Bagging Ensemble Training

```python
def train_bagging_ensemble_gaussian_scores(dataset, cfg, logger):
    """Train B Gaussian QR models with bagging."""
    models = []
    
    for b in range(cfg.B):
        # Bootstrap sample (80% of data)
        indices = bootstrap_sample(len(dataset), ratio=0.8)
        subset = Subset(dataset, indices)
        
        # Train/val split (90/10)
        train_set, val_set = random_split(subset, [0.9, 0.1])
        
        # Train single model
        model = ResNet18GaussianQR(stats_dim=3)
        state_dict, val_loss = train_single_model_gaussian(
            model, train_set, val_set, cfg
        )
        
        models.append({"state_dict": state_dict, "val_loss": val_loss})
    
    return models
```

### 3.4 Training Configuration

```yaml
# configs/attack_qr.yaml
qr:
  mode: gaussian
  target_space: log1p

bagging:
  B: 50
  bootstrap_ratio: 0.8

train:
  epochs: 50
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0
  cosine_anneal: true
  val_ratio: 0.1
  early_stop_patience: 10
```

---

## 4. Inference

### 4.1 Quantile Computation

Given $(\mu, \log\sigma)$ from a model, compute any $\tau$-quantile:

```python
def gaussian_quantile_from_params(mu, log_sigma, tau):
    """Compute q_tau in log-space for y ~ N(mu, sigma^2)."""
    z = Normal(0, 1).icdf(torch.tensor([tau]))[0]  # Φ^{-1}(τ)
    sigma = torch.exp(log_sigma)
    return mu + sigma * z
```

### 4.2 Ensemble Aggregation

Aggregate predictions in **quantile space** (not parameter space):

```python
def compute_ensemble_margin(models, image, stats, score, tau):
    """Compute ensemble margin for a sample."""
    q_log_list = []
    
    for model in models:
        mu, log_sigma = model(image, stats)
        q_log = gaussian_quantile_from_params(mu, log_sigma, tau)
        q_log_list.append(q_log)
    
    # Ensemble quantile (mean in quantile space)
    q_log_ens = torch.stack(q_log_list).mean(dim=0)
    
    # Margin
    score_log = torch.log1p(score)
    margin = q_log_ens - score_log
    
    return margin
```

### 4.3 Target FPR and Tau Relationship

For a target FPR of $\alpha$ (e.g., 0.001):
- We want the **upper-tail quantile** of the non-member distribution
- $\tau = 1 - \alpha$ (e.g., 0.999 for $\alpha = 0.001$)

```python
# Target FPR = 0.001 (0.1%)
alpha = 0.001
tau = 1.0 - alpha  # = 0.999

# This gives the 99.9th percentile of non-member scores
q_99_9 = gaussian_quantile_from_params(mu, log_sigma, tau)
```

---

## 5. Margin Semantics

### 5.1 Definition

$$m_\tau(x) = \hat{q}_\tau^{\log,\text{ens}}(x) - \log(1 + s(x))$$

### 5.2 Interpretation

| Margin | Meaning | Prediction |
|--------|---------|------------|
| $m > 0$ | Score below non-member quantile | **MEMBER** |
| $m \leq 0$ | Score at/above non-member quantile | **NON-MEMBER** |

### 5.3 Unified Scoring Convention

**Larger margin = more likely member**

This convention is consistent across:
- ROC-AUC computation
- TPR@FPR metrics
- Baseline comparison

---

## 6. Evaluation

### 6.1 Evaluation Pipeline

```python
def evaluate_attack_scores_gaussian(ensemble, config, ...):
    """Evaluate Gaussian QR ensemble on member/non-member data."""
    
    # Compute tau from alpha
    tau = 1.0 - config.alpha
    
    # Compute margins for members
    margins_in = compute_margins_for_split(
        models=ensemble,
        dataloader=member_loader,
        tau=tau
    )
    
    # Compute margins for non-members
    margins_out = compute_margins_for_split(
        models=ensemble,
        dataloader=nonmember_loader,
        tau=tau
    )
    
    # Compute metrics
    auc = roc_auc(margins_in, margins_out)
    tpr_results = {
        fpr: tpr_precision_at_fpr(margins_in, margins_out, fpr)
        for fpr in [0.01, 0.001]
    }
    
    return {"auc": auc, "tpr_at": tpr_results}
```

### 6.2 Output Report

```json
{
  "alpha": 0.001,
  "tau": 0.999,
  "M": 50,
  "metrics": {
    "auc": 0.9527,
    "tpr_at": {
      "0.01": 0.8234,
      "0.001": 0.4521
    },
    "precision_at": {
      "0.01": 0.8912,
      "0.001": 0.9234
    }
  }
}
```

---

## 7. Multi-Dataset Support

### 7.1 Dataset Configuration

Each dataset uses its own configuration files:

```bash
configs/
├── data_cifar10.yaml
├── data_cifar100.yaml
├── data_stl10.yaml
└── data_celeba.yaml
```

### 7.2 Score Files

Pre-computed scores are stored per dataset:

```
scores/{dataset}/
├── q25_watermark_private.pt   # Member scores + stats
├── q25_eval_nonmember.pt      # Non-member scores + stats
└── q25_aux.pt                 # Auxiliary set for QR training
```

### 7.3 Ensemble Checkpoints

Trained ensembles are stored per dataset:

```
runs/attack_qr/ensembles/{dataset}/{timestamp}/
├── manifest.json
├── model_b0.pt
├── model_b1.pt
└── ...
```

---

## 8. CLI Usage

### 8.1 Training

```bash
python -m src.attack_qr.engine.cli_train \
    --config configs/attack_qr.yaml \
    --data-config configs/data_cifar10.yaml \
    --mode gaussian \
    --use-scores \
    --scores-path scores/cifar10/q25_aux.pt \
    --out runs/attack_qr/ensembles/cifar10/
```

### 8.2 Evaluation

```bash
python -m src.attack_qr.engine.cli_eval \
    --config configs/attack_qr.yaml \
    --data-config configs/data_cifar10.yaml \
    --mode gaussian \
    --ensemble runs/attack_qr/ensembles/cifar10/ \
    --alpha 0.001 \
    --report-dir runs/attack_qr/reports/cifar10/
```

---

## 9. Comparison with Pinball Loss

### 9.1 Parameter Efficiency

**Pinball Loss** (2 $\tau$ values, 50 bagging):
```
Per model: ~11.2M parameters
Total: 2 × 50 × 11.2M = 1,120M parameters
```

**Gaussian** (any $\tau$, 50 bagging):
```
Per model: ~11.4M parameters (slightly larger head)
Total: 50 × 11.4M = 570M parameters
Savings: ~50% reduction
```

### 9.2 Training Efficiency

| Aspect | Pinball Loss | Gaussian |
|--------|--------------|----------|
| Training runs | 100 (2τ × 50) | 50 |
| Training time | ~2× longer | Baseline |
| Hyperparameters | Per-τ tuning | Single config |

### 9.3 Inference Flexibility

| Aspect | Pinball Loss | Gaussian |
|--------|--------------|----------|
| New τ values | Retrain required | Closed-form |
| Continuous τ | Not supported | Supported |
| Interpolation | Not possible | Natural |

---

## 10. Implementation Files

| Component | File |
|-----------|------|
| Model definition | `src/attack_qr/models/qr_resnet18.py` |
| Training | `src/attack_qr/engine/train_qr_bagging.py` |
| Evaluation | `src/attack_qr/engine/eval_attack.py` |
| CLI (train) | `src/attack_qr/engine/cli_train.py` |
| CLI (eval) | `src/attack_qr/engine/cli_eval.py` |
| Metrics | `src/attacks/eval/metrics.py` |
| Configuration | `configs/attack_qr.yaml` |

---

## 11. Summary

### Key Features

1. **Single model for all τ**: Gaussian parameterization enables closed-form quantile computation
2. **Log-space modeling**: Handles heavy-tailed score distributions
3. **Bagging ensemble**: 50 models for robust predictions
4. **Margin-based scoring**: Unified semantics (larger = more member-like)
5. **Multi-dataset support**: CIFAR-10, CIFAR-100, STL-10, CelebA

### Mathematical Summary

$$
\begin{aligned}
\text{Model:} \quad & (\mu, \sigma) = f_\theta(x, \text{stats}) \\
\text{Training:} \quad & \min_\theta \mathbb{E}\left[\frac{(y - \mu)^2}{2\sigma^2} + \log\sigma\right] \\
\text{Quantile:} \quad & \hat{q}_\tau^{\log}(x) = \mu(x) + \sigma(x) \cdot \Phi^{-1}(\tau) \\
\text{Margin:} \quad & m_\tau(x) = \hat{q}_\tau^{\log,\text{ens}}(x) - \log(1 + s(x))
\end{aligned}
$$

---

**Last Updated**: January 2026
**Status**: Production implementation
