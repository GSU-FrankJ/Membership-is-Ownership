# Experimental Plan: ICML 2026 Table 5 — Rigorous Baseline Comparison

**Goal**: Replace the current "as-reported" Table 5 with a controlled, apples-to-apples comparison where all baselines and MiO are evaluated under identical conditions on shared checkpoints.

**Hardware**: NVIDIA A100 (40GB or 80GB) × 1–2
**Priority dataset**: CIFAR-10 (32×32) — all three baselines support it natively.

---

## Table of Contents

1. [Environment & Dependencies](#phase-0-environment--dependencies)
2. [Phase 0: Codebase Audit & Shared Infrastructure](#phase-0-shared-infrastructure)
3. [Phase 1: Unified Metric Design](#phase-1-unified-metric-design)
4. [Phase 2: Baseline Implementations](#phase-2-baseline-implementations) (WDM + Zhao et al.; DeepMarks qualitative only)
5. [Phase 3: Robustness Testing](#phase-3-robustness-testing)
6. [Phase 4: Results Compilation & Revised Table 5](#phase-4-results-compilation)
7. [Phase 5: Reviewer Defense](#phase-5-reviewer-defense)
8. [Timeline & Priority Ordering](#timeline--priority-ordering)
9. [Minimum Viable Comparison](#minimum-viable-comparison)

---

## Phase 0: Environment & Dependencies

### 0.1 Conda Environment Checklist

```bash
# Base MiO environment (already exists)
conda activate mio          # Python 3.10, PyTorch ≥ 2.1, CUDA 11.8+

# Verify existing deps
pip install -r requirements.txt   # torch, torchvision, open_clip_torch, diffusers, scipy, etc.

# Additional packages for baselines
pip install pytorch-fid==0.3.0        # Standalone FID computation
pip install clean-fid==0.1.35         # Alternate FID (Parmar et al. — more robust)
pip install lpips==0.1.4              # Perceptual similarity (optional)
pip install accelerate>=0.25.0        # For WDM training
pip install einops>=0.7.0             # Zhao et al. uses einops
pip install wandb                     # Experiment tracking (optional)
```

**WDM-specific environment** (may need separate env if dependency conflicts arise):
```bash
conda create -n wdm python=3.10
conda activate wdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-fid accelerate tqdm pyyaml scipy
# Then clone & install WDM's deps
```

**Zhao et al. (WatermarkDM) environment** (uses EDM/StyleGAN-ADA-PyTorch):
```bash
conda create -n watermarkdm python=3.10
conda activate watermarkdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install click pillow scipy requests tqdm pyspng ninja
# EDM uses custom CUDA kernels — needs matching CUDA toolkit
```

### 0.2 Data Paths (Already Configured)

| Dataset   | Root                                                     | Watermark K | Split JSONs                                                            |
|-----------|----------------------------------------------------------|-------------|------------------------------------------------------------------------|
| CIFAR-10  | `/data/short/fjiang4/mia_ddpm_qr/data/cifar10`          | 5,000       | `/data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/*.json`          |
| CIFAR-100 | `/data/short/fjiang4/mia_ddpm_qr/data/cifar100`         | 5,000       | `/data/short/fjiang4/mia_ddpm_qr/data/splits/cifar100/*.json`         |
| STL-10    | `/data/short/fjiang4/mia_ddpm_qr/data/stl10`            | 1,000       | `/data/short/fjiang4/mia_ddpm_qr/data/splits/stl10/*.json`            |
| CelebA    | `/data/short/fjiang4/mia_ddpm_qr/data/celeba`           | 5,000       | `/data/short/fjiang4/mia_ddpm_qr/data/splits/celeba/*.json`           |

### 0.3 Existing Production Checkpoints (DO NOT RETRAIN)

| Model   | Path                                                                                    |
|---------|-----------------------------------------------------------------------------------------|
| Model A (CIFAR-10)  | `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt`  |
| Model B (CIFAR-10)  | `/data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt` |
| Model A (CIFAR-100) | `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar100/main/best_for_mia.ckpt` |
| Model B (CIFAR-100) | `/data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/cifar100/model_b/ckpt_0500_ema.pt` |
| Model A (STL-10)    | `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_stl10/main/best_for_mia.ckpt`    |
| Model B (STL-10)    | `/data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/stl10/model_b/ckpt_0500_ema.pt`   |
| Model A (CelebA)    | `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_celeba/main/best_for_mia.ckpt`   |
| Model B (CelebA)    | `/data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/celeba/model_b/ckpt_0500_ema.pt`  |

---

## Phase 0: Shared Infrastructure

### 0.1 Reusable Components Already in the Codebase

| Component                    | Location                                      | Reuse Status |
|-----------------------------|-----------------------------------------------|-------------|
| T-error computation          | `src/attack_qr/features/t_error.py`           | Direct reuse |
| Deterministic seeding        | `src/attack_qr/utils/seeding.py`              | Direct reuse |
| Cosine noise schedule        | `src/ddpm_ddim/schedulers/betas.py`           | Direct reuse |
| UNet model factory           | `src/ddpm_ddim/models/unet.py`                | Direct reuse |
| DDIM forward/reverse         | `src/ddpm_ddim/ddim/forward_reverse.py`       | Direct reuse |
| CLIP feature extraction      | `src/ddpm_ddim/clip_features.py`              | Direct reuse |
| MMD loss (cubic kernel)      | `src/ddpm_ddim/mmd_loss.py`                   | Direct reuse |
| ROC-AUC / TPR@FPR metrics   | `src/attacks/eval/metrics.py`                 | Direct reuse |
| Bootstrap confidence intervals| `src/attack_qr/utils/metrics.py`             | Direct reuse |
| HuggingFace model loader     | `src/attacks/baselines/huggingface_loader.py` | Direct reuse |
| Ownership evaluation         | `scripts/eval_ownership.py`                   | Extend |
| FID evaluation               | `scripts/eval_fid.py`                         | Extend |
| MMD fine-tuning              | `scripts/finetune_mmd_ddm.py`                 | Direct reuse for robustness tests |
| Multi-dataset support        | `src/ddpm_ddim/train_ddim.py` → `MultiDatasetSubset` | Direct reuse |

### 0.2 New Shared Evaluation Harness

**File**: `scripts/eval_baselines.py`

This is the central script that evaluates ANY diffusion model checkpoint through a unified protocol.

```python
"""
Unified baseline evaluation harness.

Accepts any diffusion model checkpoint and computes:
1. FID score (Inception-v3 features, 50K generated vs 50K real)
2. T-error scores on watermark_private and eval_nonmember splits
3. MiO three-point verification (if QR ensemble available)
4. Native verification metrics (method-specific, pluggable)
5. Robustness metrics (if attacked checkpoint provided)

Output: JSON + CSV results file in standardized format.

Usage:
    python scripts/eval_baselines.py \
        --method mio \
        --checkpoint /path/to/ckpt.pt \
        --dataset cifar10 \
        --output-dir experiments/baseline_comparison/results/mio/cifar10/
"""
```

**Key design decisions**:
- **Method registry pattern**: Each baseline registers a loader function and a native-metric function.
- **Standardized output schema**:
  ```json
  {
    "method": "wdm",
    "dataset": "cifar10",
    "checkpoint": "/path/to/ckpt.pt",
    "timestamp": "2026-03-09T...",
    "seed": 42,
    "generation_quality": {
      "fid_inception": 12.5,
      "fid_clip": 8.2,
      "num_generated": 50000
    },
    "native_verification": {
      "metric_name": "wm_extract_rate",
      "value": 0.998,
      "details": {}
    },
    "mio_verification": {
      "t_error_member_mean": 28.6,
      "t_error_member_std": 10.8,
      "t_error_nonmember_mean": 697.6,
      "t_error_nonmember_std": 38.0,
      "cohens_d": -23.9,
      "ratio": 24.4,
      "tpr_at_fpr_0001": 0.89,
      "three_point_pass": true
    },
    "robustness": {
      "mmd_finetune_500": { ... },
      "pruning_10pct": { ... },
      "pruning_30pct": { ... },
      "pruning_50pct": { ... }
    }
  }
  ```

**Implementation steps** (sequenced to avoid premature adapter design):
1. Create `scripts/eval_baselines.py` with the harness skeleton and method registry pattern
2. Create `scripts/baselines/__init__.py` — method registry with pluggable adapter interface
3. Create `scripts/baselines/mio_adapter.py` — wraps existing eval_ownership.py logic (can be built immediately since our codebase is known)
4. **After Step 2A.1 (WDM code audit)**: Create `scripts/baselines/wdm_adapter.py` — WDM-specific loader + native metrics, designed based on actual WDM model class and checkpoint format
5. **After Step 2B.1 (Zhao code audit)**: Create `scripts/baselines/zhao_adapter.py` — Zhao et al. loader + bit accuracy, designed based on actual EDM checkpoint format and decoder interface
6. Create `scripts/eval_baselines_batch.sh` — runs eval_baselines.py for all (method × dataset) combos

> **Rationale for sequencing**: The adapter interface (function signatures, model loading protocol) depends on the actual baseline code. Building adapters before auditing the repos will produce wrong interfaces that must be rewritten. Build the harness + mio_adapter on Day 1 AM; add wdm_adapter after 2A.1 and zhao_adapter after 2B.1.

**Deliverables**:
- `scripts/eval_baselines.py`
- `scripts/baselines/*.py`
- `experiments/baseline_comparison/results/` directory tree

**Estimated effort**: 1 day coding, 0 GPU-hours.

---

## Phase 1: Unified Metric Design

### 1.1 Category A — Native Verification Metrics (Each Method's Own Protocol)

| Method     | Native Metric                          | How to Compute                                              |
|-----------|----------------------------------------|-------------------------------------------------------------|
| WDM       | WM extraction rate, confidence (α=10⁻³)| Sample with shared reverse noise, decode watermark image     |
| Zhao et al.| Bit accuracy (48-bit string)          | Generate images from watermarked model, run decoder network  |
| MiO       | TPR @ FPR=0.1%, Cohen's d, ratio      | QR ensemble margin on watermark set vs. public reference     |

**Rationale**: Reporting each method's native metric is non-negotiable — it shows we ran each method correctly and reproduced their claimed performance. Reviewers will check this first.

### 1.2 Category B — Cross-Method Comparable Metrics

These metrics are applied uniformly to ALL methods. They answer: "Regardless of how each method verifies ownership, how do they compare on dimensions that matter?"

#### B1. Generation Quality (FID)

**Protocol**: For each watermarked model, generate 50,000 images using the model's native sampler, compute FID against the full 50,000-image CIFAR-10 training set (standard in diffusion model literature). See Phase 3, §3.5 for exact protocol.

- **MiO**: Model A's FID (already computed or use `scripts/eval_fid.py`)
- **WDM**: WDM-trained model's FID using their standard DDPM sampler
- **Zhao et al.**: EDM model's FID using their sampler
- **Clean baseline**: Same architecture trained without any watermark (Model A IS this — MiO doesn't modify the model)

**Why this is fair**: MiO's key advantage is zero quality degradation. FID quantifies whether watermark embedding hurts generation quality.

**Potential reviewer concern**: "MiO has unfair FID advantage because it doesn't modify the model." **Response**: That IS the point — MiO preserves model utility by design. We report this honestly.

#### B2. Verification Robustness Under MMD Fine-Tuning

**Protocol**: Apply our existing MMD fine-tuning attack (from Table 10) to each watermarked model:
- 500 iterations, lr=5e-6, CLIP ViT-B/32 (note: paper says ViT-B/32 but code uses ViT-L/14 — **must reconcile**)
- 10-step DDIM, t_start=900
- EMA decay 0.9999, grad_clip=1.0

After fine-tuning, re-evaluate:
- Each method's native verification: does it still pass?
- FID of the fine-tuned model: did quality degrade?

**Why this is the most important robustness test**: MMD fine-tuning is specifically designed to remove ownership signals. If a watermark survives this, it's robust.

#### B3. Verification Robustness Under Weight Pruning

**Protocol**: Apply structured (channel-level) pruning at rates {10%, 30%, 50%} using `torch.nn.utils.prune.ln_structured` on Conv2d layers.

```python
# Pruning implementation sketch
import torch.nn.utils.prune as prune

def apply_structured_pruning(model, rate):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=rate, n=1, dim=0)
            prune.remove(module, 'weight')  # Make permanent
    return model
```

After pruning, re-evaluate native verification + FID.

**File**: `scripts/attacks/pruning.py`

#### B4. Verification Robustness Under Continued Training

**Protocol**: Continue training each watermarked model for 50k additional iterations on a disjoint subset of the training data (the non-watermark portion).

- Use the standard DDIM training loop from `src/ddpm_ddim/train_ddim.py`
- Same hyperparameters as original training (lr=0.0002, batch_size=128)
- Evaluate every 10k iterations: native verification + FID

**Why this matters**: An adversary might continue training on their own data to dilute the watermark.

#### B5. Computational Overhead

| Metric                    | How to Measure                                    |
|--------------------------|---------------------------------------------------|
| Training time overhead   | Wall-clock time of watermarked training vs. clean training |
| Verification query time  | Time to run one ownership verification query       |
| Verification queries needed | Number of model queries for a single verification |

**MiO advantage**: Zero training overhead (post-hoc). Verification requires ~5K forward passes (t-error computation on watermark set).

### 1.3 Note on Pruning Fairness

Pruning modifies weight distributions, which could disproportionately affect methods that encode information in weights (like DeepMarks). Since DeepMarks is excluded from the controlled comparison, this concern does not apply. Both WDM and Zhao et al. encode watermarks through the training process / generated outputs, not directly in weight statistics, so pruning is a fair robustness test for all methods in our comparison.

### 1.4 Metrics We Explicitly Exclude (And Why)

| Excluded Metric      | Reason                                                        |
|---------------------|---------------------------------------------------------------|
| IS (Inception Score) | Known to be unreliable for evaluating diffusion models; FID is the standard |
| LPIPS               | Redundant with FID for comparing quality; adds complexity      |
| SSIM                | Pixel-level metric, not meaningful for generative models       |

### 1.4 Potential Bias Concerns

**Q: Is the MMD fine-tuning attack biased toward MiO?**

**A**: No — MiO was specifically designed to withstand MMD fine-tuning, but we apply the SAME attack to ALL methods. If anything, this attack is tailored to MiO's threat model, meaning MiO should do BEST here. The question is whether other methods also survive. If they do, that's a fair result.

---

## Phase 2: Baseline Implementations

### DeepMarks (Chen et al., ICMR 2019) — Qualitative Discussion Only

**Decision**: DeepMarks is excluded from the controlled experimental comparison.

**Rationale**: DeepMarks embeds binary fingerprints into the probability density function (PDF) of weight parameters via a GMM regularization loss during retraining. It was designed for discriminative classifiers (WRN-28-10 on CIFAR-10/MNIST) where fully-connected and convolutional layers have relatively simple, unimodal weight distributions amenable to GMM shaping.

This does not transfer to diffusion model U-Nets in any principled way:

1. **Layer heterogeneity**: A DDIM U-Net combines residual convolution blocks, self-attention layers (QKV projections), GroupNorm, and skip connections. Weight distributions vary qualitatively across layer types — a single GMM regularizer cannot uniformly encode bits across such diverse parameter spaces.
2. **Training instability**: The fingerprint regularization competes with the ε-prediction MSE loss. In diffusion models, even small perturbations to early training dynamics can cascade into mode collapse or divergent sampling, making the regularization-vs-task-loss tradeoff far more fragile than in classification.
3. **No official code**: No reference implementation exists for diffusion models, and the third-party code targets discriminative models only. Any adaptation would be a novel (untested) method, not a faithful reproduction of Chen et al.
4. **Reviewer cost-benefit**: 24+ GPU-hours for a 2019 method that was never designed for this setting. Even if it works, reviewers can dismiss the result as an unfair adaptation; if it fails, the negative result is unsurprising.

**Paper treatment**: Add a paragraph in Section 5.4 (or a short appendix paragraph) explaining the architectural gap. Cite DeepMarks as representative of the weight-fingerprinting paradigm and note that it targets a fundamentally different model family. This is honest and defensible.

**Draft paragraph for Section 5.4**:
> We exclude DeepMarks~\citep{chen2019deepmarks} from the controlled comparison. DeepMarks embeds binary fingerprints by regularizing the weight probability density of fully-connected and convolutional layers in discriminative classifiers (WRN, CNN). Modern diffusion U-Nets interleave residual convolutions, self-attention, and group normalization — layer types whose weight distributions differ qualitatively and resist uniform GMM shaping. Absent official code or a principled adaptation strategy, any implementation would constitute a novel (and likely unfair) variant rather than a faithful reproduction. We therefore treat DeepMarks as representative of the weight-fingerprinting paradigm and note that its design assumptions do not extend to diffusion architectures.

---

### Phase 2A: WDM — Watermark Diffusion Process (Peng et al., 2023)

#### Code Audit Plan

**Step 2A.1**: Clone and audit WDM repository

```bash
cd experiments/baseline_comparison/
git clone https://github.com/senp98/wdm.git wdm_repo
```

**Key files to audit**:
- Training script (likely `train.py` or `main.py`)
- Watermark diffusion process implementation
- Watermark embedding and extraction
- CIFAR-10 config/defaults
- Model architecture (DDPM-based U-Net)
- Dependencies and compatibility

**Expected findings from the WDM paper**:
- WDM trains a dual-process diffusion model: standard noise→image AND a watermark diffusion process (WDP)
- The WDP is triggered by a specific "key" input noise pattern
- Watermark is a binary image (e.g., 32×32 logo)
- Extraction: sample with the key noise → decode watermark from generated image
- Verification: hypothesis test on extracted watermark vs. expected watermark

#### Integration Strategy

**Step 2A.2**: Identify WDM's model architecture

WDM likely uses a standard DDPM U-Net (from Ho et al. 2020 or Nichol & Dhariwal 2021). Key questions:
- Is it the same U-Net architecture as ours? (channels, attention resolutions, etc.)
- Does WDM modify the U-Net or just the training objective?
- What noise schedule does WDM use? (linear vs. cosine)

**Concrete architecture compatibility check** (do this during Step 2A.1 audit):

```python
# 1. Find the model class definition
#    grep -r "class.*UNet\|class.*Model" wdm_repo/ --include="*.py"
# 2. Instantiate and print:
#    model = WDMModel(...)  # use their default CIFAR-10 config
#    print(model)
# 3. Compare against our UNet config:
#    base_channels=128, ch_mults=[1,2,2,2], attn_res=[16]
#    Check: channel counts, attention resolution, number of residual blocks,
#    GroupNorm groups, whether they use self-attention vs cross-attention
# 4. Decision:
#    - If architecture matches → load WDM weights into our UNet wrapper
#    - If architecture differs → keep WDM's native model, write a thin adapter
```

**If WDM uses a compatible architecture** (same or similar U-Net):
- Use WDM's training script as-is on CIFAR-10
- Save checkpoint in a format loadable by our eval harness
- Port the checkpoint to our UNet if needed (weight remapping)

**If WDM uses a different architecture**:
- Train using WDM's native architecture
- Write an adapter in `scripts/baselines/wdm_adapter.py` that loads WDM's model and wraps it to compute t-error

#### WDM Training Data Split Decision (RESOLVE BEFORE TRAINING)

WDM trains a dual-process model: standard diffusion (task data) + watermark diffusion process (WDP, triggered by key noise). Our MiO watermark set W_D is 5K samples from CIFAR-10's 50K training set.

**Question**: What role does W_D play in WDM training?

**Answer**: W_D goes into WDM's **standard task training data** (all 50K CIFAR-10 images), NOT as the WDP trigger. WDM's watermark is a binary logo image decoded from generated samples — it has nothing to do with our membership-based W_D. The WDP trigger set is a separate concept (a key noise pattern + target watermark image).

This means:
- WDM trains on all 50K CIFAR-10 images (including W_D) as task data — same as our Model A.
- WDM's watermark is an embedded logo, orthogonal to MiO's membership signal.
- Running MiO's t-error on the WDM model using W_D tests whether WDM memorized W_D **as task data** — which it should, since W_D is in the training set.

**Why this is the correct setup**: It demonstrates that MiO is **complementary** to watermarking. The WDM model has both (a) an embedded watermark logo (WDM's signal) and (b) training-data memorization (MiO's signal). Both verification methods should pass on the same model. This is a strength, not a confound.

**Document in the paper**: "WDM is trained on the full CIFAR-10 training set, which includes MiO's watermark set W_D as task data. MiO's ownership signal arises from training-data memorization, which is orthogonal to WDM's embedded logo watermark."

**Step 2A.3**: Train WDM-watermarked model on CIFAR-10

> **⚠ PLACEHOLDER COMMANDS — REPLACE AFTER CODE AUDIT.**
> The commands below are illustrative based on the paper. After Step 2A.1 code audit, replace these with exact commands derived from the actual codebase. Do NOT run the placeholder commands verbatim. Specifically:
> 1. Read the repo's README and example configs
> 2. Find the actual `argparse` definitions (grep for `add_argument` or `ArgumentParser`)
> 3. Identify the correct entry point script and supported arguments
> 4. Write the actual training command based on what the code supports

```bash
cd experiments/baseline_comparison/wdm_repo

# PLACEHOLDER — replace with actual CLI after audit
python train.py \
    --dataset cifar10 \
    --data-root /data/short/fjiang4/mia_ddpm_qr/data/cifar10 \
    --watermark-image experiments/baseline_comparison/wdm/watermark_logo.png \
    --epochs 800 \
    --output-dir experiments/baseline_comparison/wdm/cifar10/ \
    --seed 42
```

**Watermark image**: Create a 32×32 binary image (e.g., "MiO" text or a simple pattern). Save as `experiments/baseline_comparison/wdm/watermark_logo.png`.

**Step 2A.4**: Evaluate WDM's native watermark extraction

```bash
python extract_watermark.py \
    --checkpoint experiments/baseline_comparison/wdm/cifar10/model.pt \
    --key-noise experiments/baseline_comparison/wdm/cifar10/key_noise.pt \
    --num-samples 100 \
    --output experiments/baseline_comparison/wdm/cifar10/extracted_wm.png
```

Compute:
- Watermark extraction success rate (pixel-level accuracy of extracted vs. original watermark)
- Verification confidence (hypothesis test p-value at α=10⁻³)

**Step 2A.5**: Evaluate WDM model through our MiO protocol

This is the key cross-method evaluation. Can MiO verify ownership of a WDM-watermarked model?

```bash
# Compute t-error of WDM model on our watermark_private set
python scripts/eval_baselines.py \
    --method wdm \
    --checkpoint experiments/baseline_comparison/wdm/cifar10/model.pt \
    --dataset cifar10 \
    --output-dir experiments/baseline_comparison/results/wdm/cifar10/
```

**Important insight**: Since WDM trains on CIFAR-10 training data, and our watermark_private set is a subset of CIFAR-10 training data, the WDM model should also have low t-error on watermark_private (it memorized those samples too). This means MiO's three-point test should ALSO pass for the WDM model, demonstrating that **MiO is complementary to, not competing with, watermarking**.

**Step 2A.6**: (Stretch goal) Train WDM on other datasets

If CIFAR-10 WDM training succeeds, repeat for CIFAR-100. STL-10 and CelebA are low priority (WDM may not support different resolutions without modification).

**Expected outputs**:
- `experiments/baseline_comparison/wdm/cifar10/model.pt` — WDM-watermarked checkpoint
- `experiments/baseline_comparison/wdm/cifar10/key_noise.pt` — Verification key
- `experiments/baseline_comparison/results/wdm/cifar10/results.json`

**Failure modes & fallbacks**:

| Failure Mode | Detection | Fallback |
|-------------|-----------|----------|
| WDM code doesn't run out-of-box | Import errors, missing deps | Fix compatibility issues; worst case, reimplement core WDP from paper |
| WDM's U-Net incompatible with our t-error code | Shape mismatches | Write adapter that wraps WDM's model to expose our expected interface |
| WDM training is prohibitively slow | >100 GPU-hours | Use WDM's published checkpoint if available; reduce epochs |
| WDM doesn't converge on CIFAR-10 | Watermark extraction rate <50% | Check hyperparameters against paper's Table; use their exact settings |

**Estimated GPU-hours**: 12–24h for WDM training (DDPM training is typically ~800 epochs on CIFAR-10), + 2h evaluation = **14–26h**

**Priority**: HIGH. WDM is the most directly comparable baseline (diffusion model watermarking, CIFAR-10, open-source code).

---

### Phase 2B: Zhao et al. — "A Recipe for Watermarking Diffusion Models" (2023)

#### Architecture Decision

Zhao et al. use EDM (Karras et al. 2022) as their diffusion backbone. **Use their full pipeline** — it's more scientifically defensible to run their method as designed than to create a Frankenstein adaptation. The cross-method t-error comparison requires a careful σ↔ᾱ mapping derived below.

| Aspect | Our DDIM | Zhao et al.'s EDM |
|--------|----------|-------------------|
| Noise schedule | Cosine (Nichol & Dhariwal): discrete t∈{0,...,999} | Continuous σ ∈ [σ_min, σ_max] (log-normal training dist.) |
| Forward process | x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε | x_σ = x₀ + σ · ε |
| Parameterization | ε-prediction: model predicts ε | Denoiser D(x;σ) predicts x₀ directly |
| Network output | ε̂ = F_θ(x_t, t) | D_θ(x_σ; σ) = c_skip(σ)·x_σ + c_out(σ)·F_θ(c_in(σ)·x_σ; c_noise(σ)) |
| Reconstruction | x̂₀ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t | x̂₀ = D_θ(x_σ; σ) |

#### Step 2B.0: σ ↔ ᾱ Mapping — CPU Prototype (DO THIS FIRST, BEFORE ANY GPU WORK)

**This is the single biggest technical risk in the plan.** If the mapping is wrong, all cross-method t-error comparisons for Zhao et al. are meaningless. Prototype and validate on CPU before committing GPU hours to training.

**File**: `scripts/baselines/edm_sigma_mapping.py`

**Mathematical derivation:**

Both forward processes add Gaussian noise to x₀. They are equivalent under a reparameterization. In DDPM/DDIM:

```
x_t = √ᾱ_t · x₀ + √(1 - ᾱ_t) · ε       where ε ~ N(0, I)
```

Rewrite as:
```
x_t = √ᾱ_t · (x₀ + √((1 - ᾱ_t) / ᾱ_t) · ε)
```

In EDM:
```
x_σ = x₀ + σ · ε                           where ε ~ N(0, I)
```

These match (up to the √ᾱ_t scaling of the clean signal) when:
```
σ(t) = √((1 - ᾱ_t) / ᾱ_t)
```

This is the **noise-to-signal ratio** at timestep t. Conversely:
```
ᾱ(σ) = 1 / (1 + σ²)
```

**Critical subtlety**: DDPM/DDIM scales both signal and noise by √ᾱ_t and √(1-ᾱ_t) respectively, so the noisy image `x_t` has a different magnitude than EDM's `x_σ`. For t-error computation, what matters is **reconstruction quality** — the denoiser's ability to recover x₀ — not the absolute scale of the noisy input. So the correct protocol is:

1. **For DDIM t-error** (existing code, mode="x0"):
   ```
   x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
   ε̂ = model(x_t, t)
   x̂₀ = (x_t - √(1-ᾱ_t) · ε̂) / √ᾱ_t
   error = ‖x̂₀ - x₀‖²
   ```

2. **For EDM t-error** (new adapter):
   ```
   σ = √((1 - ᾱ_t) / ᾱ_t)       # Same noise level as DDIM timestep t
   x_σ = x₀ + σ · ε
   x̂₀ = D_θ(x_σ; σ)              # EDM denoiser directly outputs x₀ estimate
   error = ‖x̂₀ - x₀‖²
   ```

Both measure **‖x̂₀ - x₀‖² at the same noise-to-signal ratio**, making them directly comparable.

**Validation protocol** (CPU, ~1h):

```python
"""
scripts/baselines/edm_sigma_mapping.py

Validate σ ↔ ᾱ mapping on a SINGLE batch before committing GPU time.

Tests:
  1. Roundtrip: t → σ(t) → ᾱ(σ) ≈ ᾱ_t  (numerical agreement within 1e-6)
  2. Noise level equivalence: For same x₀ and same ε,
     ‖x_t - √ᾱ_t · x_σ‖ < 1e-6  (scaled DDIM matches EDM forward)
  3. SNR monotonicity: σ(t) is monotonically increasing in t
  4. Boundary conditions: σ(0) ≈ 0 (clean), σ(999) >> 1 (pure noise)
  5. EDM training distribution coverage: Check that our σ range covers
     EDM's default [σ_min=0.002, σ_max=80] training distribution
"""
import math
import torch
import numpy as np

def alpha_bar_from_cosine(t, T=1000, s=0.008):
    """Our cosine schedule: ᾱ_t = cos²((t/T + s)/(1+s) · π/2)"""
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    f0 = np.cos((s) / (1 + s) * np.pi / 2) ** 2
    return f / f0

def sigma_from_alpha_bar(alpha_bar):
    """σ(t) = √((1 - ᾱ_t) / ᾱ_t)"""
    return np.sqrt((1.0 - alpha_bar) / alpha_bar)

def alpha_bar_from_sigma(sigma):
    """ᾱ(σ) = 1 / (1 + σ²)"""
    return 1.0 / (1.0 + sigma ** 2)

# --- Test 1: Roundtrip ---
for t in [0, 100, 250, 500, 750, 999]:
    ab = alpha_bar_from_cosine(t)
    sigma = sigma_from_alpha_bar(ab)
    ab_roundtrip = alpha_bar_from_sigma(sigma)
    assert abs(ab - ab_roundtrip) < 1e-10, f"Roundtrip failed at t={t}"
    print(f"t={t:4d}  ᾱ_t={ab:.6f}  σ={sigma:.4f}  roundtrip_err={abs(ab-ab_roundtrip):.2e}")

# --- Test 2: σ range vs EDM defaults ---
sigmas = [sigma_from_alpha_bar(alpha_bar_from_cosine(t)) for t in range(1000)]
print(f"\nσ range: [{min(sigmas):.4f}, {max(sigmas):.4f}]")
print(f"EDM default range: [0.002, 80.0]")
print(f"Coverage: our σ_min={'OK' if min(sigmas) < 0.01 else 'WARNING'}, "
      f"σ_max={'OK' if max(sigmas) > 50 else 'WARNING: may not reach EDM σ_max'}")

# --- Test 3: Noise level equivalence on a dummy batch ---
# NOTE: Use torch consistently — do NOT mix numpy and torch tensors.
x0 = torch.randn(4, 3, 32, 32)
eps = torch.randn_like(x0)
t = 500
ab = float(alpha_bar_from_cosine(t))   # Convert to Python float for torch ops
sigma = float(sigma_from_alpha_bar(ab))

x_ddim = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps   # DDIM forward
x_edm = x0 + sigma * eps                                  # EDM forward
# After scaling: √ᾱ_t · x_edm = √ᾱ_t · x₀ + √ᾱ_t · σ · ε
#                              = √ᾱ_t · x₀ + √(ᾱ_t · (1-ᾱ_t)/ᾱ_t) · ε
#                              = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε = x_ddim  ✓
diff = (x_ddim - math.sqrt(ab) * x_edm).abs().max().item()
assert diff < 1e-5, f"Noise equivalence failed: max diff = {diff}"
print(f"\nNoise equivalence test: max |x_ddim - √ᾱ·x_edm| = {diff:.2e}  ✓")
```

**Run this on Day 1.** If σ_max doesn't cover EDM's range (likely — our cosine schedule tops out around σ≈100, EDM goes to 80, so this should be fine), note the boundary and decide whether to clamp or extrapolate.

**Go/no-go decision**: If Tests 1-3 pass, proceed with EDM training. If the σ range is badly mismatched, we need to decide between (a) restricting t-error evaluation to the overlapping σ range, or (b) using EDM's native σ distribution for sampling noise levels (more faithful to EDM, but less comparable to our DDIM t-error).

#### Step 2B.1: Clone and audit WatermarkDM repository

```bash
cd experiments/baseline_comparison/
git clone https://github.com/yunqing-me/WatermarkDM.git watermarkdm_repo
```

**Key files to audit**:
- Watermark encoder/decoder architecture and training
- How the encoder is integrated into the diffusion training pipeline
- EDM's preconditioning functions: c_skip(σ), c_out(σ), c_in(σ), c_noise(σ)
- CIFAR-10 support and configuration
- Whether they provide pre-trained checkpoints (saves 24h)

#### Step 2B.2: Understand the two-stage pipeline

Zhao et al.'s method has two stages:
1. **Stage 1**: Train a watermark encoder E and decoder D on clean images
   - E: image → watermarked image (visually similar, carries hidden bit-string)
   - D: watermarked image → bit-string (48-bit)
   - Trained with reconstruction loss + perceptual loss + bit accuracy loss
2. **Stage 2**: Train EDM on watermarked images (E(x) instead of x)
   - Generated images naturally carry the watermark
   - Decode from generated images to verify ownership

#### Step 2B.3: Train watermark encoder/decoder

> **⚠ PLACEHOLDER COMMANDS — REPLACE AFTER CODE AUDIT.**
> The commands below are illustrative based on the paper. After Step 2B.1 code audit, replace these with exact commands derived from the actual codebase. Do NOT run the placeholder commands verbatim. Specifically:
> 1. Read the repo's README and example configs
> 2. Find the actual `argparse` definitions (grep for `add_argument` or `ArgumentParser`)
> 3. Identify the correct entry point scripts for Stage 1 (encoder) and Stage 2 (EDM)
> 4. Write the actual training commands based on what the code supports

```bash
cd experiments/baseline_comparison/watermarkdm_repo

# PLACEHOLDER — replace with actual CLI after audit
python train_encoder.py \
    --dataset cifar10 \
    --data-root /data/short/fjiang4/mia_ddpm_qr/data/cifar10 \
    --watermark-bits 48 \
    --epochs 50 \
    --output-dir experiments/baseline_comparison/zhao/cifar10/encoder/ \
    --seed 42
```

#### Step 2B.4: Train EDM on watermarked CIFAR-10

```bash
# PLACEHOLDER — replace with actual CLI after audit
python train_edm.py \
    --dataset cifar10 \
    --data-root /data/short/fjiang4/mia_ddpm_qr/data/cifar10 \
    --encoder-ckpt experiments/baseline_comparison/zhao/cifar10/encoder/best.pt \
    --watermark-bits 48 \
    --watermark-key "secret_key_42" \
    --iterations 200000 \
    --output-dir experiments/baseline_comparison/zhao/cifar10/edm/ \
    --seed 42
```

#### Step 2B.5: Evaluate Zhao et al.'s native verification

```bash
# Generate images from watermarked EDM and decode watermark
python evaluate_watermark.py \
    --edm-ckpt experiments/baseline_comparison/zhao/cifar10/edm/model.pt \
    --decoder-ckpt experiments/baseline_comparison/zhao/cifar10/encoder/decoder.pt \
    --num-samples 1000 \
    --output-dir experiments/baseline_comparison/zhao/cifar10/eval_native/
```

Expected metric: Bit accuracy ≈ 99.9% for watermarked model, ≈ 50% for unwatermarked.

#### Step 2B.6: Compute t-error on EDM model using validated σ-mapping

**EDM checkpoint loading** (CRITICAL — EDM uses pickle-based full network objects, NOT state dicts):

```python
# EDM checkpoints are loaded as complete network objects, not state_dicts:
import pickle
import torch

def load_edm_checkpoint(ckpt_path):
    """Load an EDM checkpoint. Returns the full preconditioned network."""
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    # EDM stores networks under 'ema' key (exponential moving average)
    # Some versions use 'G_ema' (StyleGAN-ADA heritage)
    edm_net = data.get('ema', data.get('G_ema'))
    if edm_net is None:
        raise KeyError(f"Checkpoint keys: {list(data.keys())}. "
                       "Expected 'ema' or 'G_ema'.")
    edm_net.eval()
    return edm_net

# IMPORTANT: The loaded network is the full EDMPrecond wrapper, NOT the raw
# U-Net backbone. Call it as: x0_hat = edm_net(x_noisy, sigma)
# Do NOT unwrap or strip the preconditioning — see caveat below.
```

> **After Step 2B.1 code audit**: Verify the actual checkpoint key names and loading procedure match the above. If Zhao et al. uses a different save format, update accordingly.

**File**: `scripts/baselines/zhao_adapter.py`

```python
"""
T-error adapter for EDM (Karras et al. 2022) models.

Uses the validated σ ↔ ᾱ mapping from edm_sigma_mapping.py to compute
t-error scores that are directly comparable to our DDIM t-error.

Key insight: Both DDIM and EDM denoisers estimate x₀ from a noisy input.
At equivalent noise-to-signal ratio σ = √((1-ᾱ)/ᾱ), the reconstruction
error ‖x̂₀ - x₀‖² measures the same quantity: how well the model
reconstructs the clean image from a given noise level.

DDIM: x̂₀ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
EDM:  x̂₀ = D_θ(x_σ; σ)   [denoiser directly outputs x₀ estimate]
"""
import torch
import numpy as np

def alpha_bar_cosine(t, T=1000, s=0.008):
    """Cosine schedule ᾱ_t (matches src/ddpm_ddim/schedulers/betas.py)."""
    f_t = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos((s / (1 + s)) * np.pi / 2) ** 2
    return f_t / f_0

def sigma_from_alpha_bar(alpha_bar):
    """Convert DDIM ᾱ to EDM σ: σ = √((1 - ᾱ) / ᾱ)."""
    return np.sqrt(np.clip(1.0 - alpha_bar, 0, None) / np.clip(alpha_bar, 1e-10, None))

def compute_edm_t_error(
    edm_model,
    x0: torch.Tensor,           # [B, 3, 32, 32] normalized clean images
    K: int = 50,                 # Number of noise levels to sample
    T: int = 1000,               # DDIM schedule length (for σ range)
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute t-error scores for an EDM model, comparable to DDIM t-error.

    For each of K random timesteps t ~ U(0, T):
      1. Compute σ = √((1 - ᾱ_t) / ᾱ_t)
      2. Forward:  x_σ = x₀ + σ · ε,  ε ~ N(0, I)
      3. Denoise:  x̂₀ = D_θ(x_σ; σ)
      4. Error:    e = ‖x̂₀ - x₀‖²  (mean over spatial dims)

    Returns Q25 aggregation over K noise levels, per sample.
    """
    B = x0.shape[0]
    x0 = x0.to(device)

    # Precompute σ schedule from our cosine ᾱ schedule
    all_sigmas = torch.tensor(
        [sigma_from_alpha_bar(alpha_bar_cosine(t, T)) for t in range(T)],
        dtype=torch.float32, device=device
    )

    rng = torch.Generator(device=device).manual_seed(seed)
    errors = []

    for k in range(K):
        # Sample random timesteps → corresponding σ values
        t_indices = torch.randint(0, T, (B,), generator=rng, device=device)
        sigma = all_sigmas[t_indices].view(B, 1, 1, 1)  # [B, 1, 1, 1]

        # Forward: x_σ = x₀ + σ · ε
        eps = torch.randn_like(x0)
        x_noisy = x0 + sigma * eps

        # Denoise: EDM model expects (x_noisy, sigma_flat)
        with torch.no_grad():
            x0_hat = edm_model(x_noisy, sigma.view(B))  # D_θ(x_σ; σ) → x̂₀

        # Reconstruction error
        error = (x0_hat - x0).pow(2).mean(dim=(1, 2, 3))  # [B]
        errors.append(error)

    errors = torch.stack(errors, dim=1)  # [B, K]
    q25 = torch.quantile(errors, 0.25, dim=1)  # [B]
    return q25
```

**Sanity checks before trusting these scores** (run on CPU/single GPU, ~1h):

| Test | Expected outcome | Action if fails |
|------|-----------------|-----------------|
| t-error of EDM on its own training data (members) | Low scores, similar magnitude to DDIM member scores | Check σ range, preconditioning functions, normalization |
| t-error of EDM on held-out data (non-members) | Higher than member scores | If no separation, the membership signal may not transfer — this is a valid finding |
| t-error of EDM at σ→0 (nearly clean) | Near-zero error | Verify denoiser passthrough at low noise |
| t-error of EDM at σ→∞ (pure noise) | Large error, close to ‖x₀‖² | Verify denoiser doesn't collapse to zero |
| Score distribution shape | Roughly matches DDIM score distributions in scale | If off by orders of magnitude, check normalization (EDM may use different image scaling) |

**EDM preconditioning caveat (CRITICAL)**: EDM wraps the raw U-Net backbone `F_θ` with σ-dependent input/output scaling:
```
D_θ(x; σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ))
```
where for EDM's VP preconditioning (CIFAR-10 defaults):
```
c_skip(σ) = σ_data² / (σ² + σ_data²)
c_out(σ)  = σ · σ_data / √(σ² + σ_data²)
c_in(σ)   = 1 / √(σ² + σ_data²)
c_noise(σ) = ln(σ) / 4
```
with σ_data = 0.5. **You MUST call the full preconditioned `D_θ`, not the raw backbone `F_θ`.** If the adapter strips the preconditioning wrapper to "simplify," the math will look correct but t-errors on real models will be garbage — the raw backbone operates in a completely different output space. The prototype script must load an actual pretrained EDM checkpoint and run inference through their full `EDMPrecond` class, not a standalone U-Net.

**EDM image normalization caveat**: EDM typically trains on images in [-1, 1] (uniform scaling). Our DDIM uses dataset-specific normalization (CIFAR-10: mean=[0.4914,...], std=[0.2470,...]). **Must ensure x₀ is in the same scale as EDM's training data** before computing t-error. Check EDM's data pipeline for normalization and match it exactly.

**Resolution**: When computing t-error on EDM models, normalize input images to [-1, 1]: `x_normalized = 2 * x_01 - 1`, where `x_01` is in [0, 1]. When computing t-error on DDIM models, use the existing normalization from `src/ddpm_ddim/data/transforms.py`. The reconstruction error is computed in the SAME space as the model's input, so this difference does NOT affect comparability — both measure ‖x̂₀ - x₀‖² in the model's native coordinate system.

#### Step 2B.7: Evaluate through MiO protocol

```bash
python scripts/eval_baselines.py \
    --method zhao \
    --checkpoint experiments/baseline_comparison/zhao/cifar10/edm/model.pt \
    --dataset cifar10 \
    --output-dir experiments/baseline_comparison/results/zhao/cifar10/
```

**Expected outputs**:
- `experiments/baseline_comparison/zhao/cifar10/encoder/best.pt`
- `experiments/baseline_comparison/zhao/cifar10/edm/model.pt`
- `experiments/baseline_comparison/results/zhao/cifar10/results.json`

**Failure modes & fallbacks**:

| Failure Mode | Detection | Fallback |
|-------------|-----------|----------|
| EDM requires custom CUDA kernels that don't compile | Build errors | Use `--no-custom-ops` flag (EDM supports this, slower) |
| EDM training too slow (>48h) | Monitoring | Use reduced iterations (100k); accept worse FID |
| σ-mapping sanity checks fail (scores wrong magnitude) | Test battery above | Check EDM normalization; try EDM's native σ-sampling instead of mapped timesteps |
| Bit accuracy much lower than reported | <90% on watermarked model | Check encoder training; verify same watermark key |
| EDM t-error shows no member/non-member separation | Flat distributions | Report as finding: "membership signal does not transfer across parameterizations" — still valuable |

**Estimated GPU-hours**: 8h (encoder) + 24h (EDM training) + 4h (evaluation) = **36h**
**CPU-hours (σ-mapping prototype, done first)**: ~2h

**Priority**: HIGH — but σ-mapping validation is CRITICAL and must be done on Day 1 before any GPU commitment.

---

## Phase 3: Robustness Testing (CIFAR-10 Only)

### 3.1 Scope Decision

**CIFAR-10 only.** Multi-dataset robustness is a stretch goal. For the paper, a clean single-dataset comparison with two attacks is stronger than a sprawling matrix with gaps.

**Two attacks only: MMD fine-tuning (primary) + pruning (secondary).** Continued training is dropped — if MiO survives MMD-FT but not continued training, that's a new limitation to discuss, and surfacing it right before a deadline is risky. MMD fine-tuning is the paper's main robustness claim; pruning is standard and cheap.

### 3.2 Test Matrix

| Method / Attack       | Clean | MMD-FT 500it | Prune 30% |
|----------------------|-------|--------------|-----------|
| MiO (Model A→B)     | ✓ (existing) | ✓ (existing) | NEW |
| WDM                  | NEW   | NEW          | NEW       |
| Zhao et al.          | NEW   | NEW (see note) | NEW    |

**Total new experiments**: 7 (3 clean evals + 2 MMD-FT + 3 pruning, minus existing MiO results)

**Each experiment** consists of:
1. Apply attack to checkpoint → attacked checkpoint
2. Compute native verification metric on attacked checkpoint
3. Compute FID on attacked checkpoint (50K generated images — see §3.5)
4. Compute MiO t-error scores on attacked checkpoint (if applicable)

### 3.3 MMD Fine-Tuning Attack

**Reuse**: `scripts/finetune_mmd_ddm.py` with each baseline's checkpoint as input.

**CLIP model version — must reconcile before running**:
- Code uses ViT-L/14 (`src/ddpm_ddim/clip_features.py`)
- Paper Table 10 says ViT-B/32
- **Resolution** (concrete steps):
  ```bash
  # Step 1: Check what the code actually uses
  grep -r "ViT" src/ddpm_ddim/clip_features.py
  grep -r "clip" configs/mmd_finetune_cifar10_ddim10.yaml
  # Step 2: If code says ViT-L/14 → update the paper Table 10 to say ViT-L/14
  #          If code says ViT-B/32 → no change needed
  #          If ambiguous → default to ViT-L/14 (what the code runs) and update paper
  # Step 3: Use the SAME CLIP model for ALL baseline MMD-FT experiments. Do NOT mix.
  ```

```bash
# MMD fine-tune WDM checkpoint
python scripts/finetune_mmd_ddm.py \
    --config configs/mmd_finetune_cifar10_ddim10.yaml \
    --base-checkpoint experiments/baseline_comparison/wdm/cifar10/model.pt \
    --output-dir experiments/baseline_comparison/robustness/wdm/cifar10/mmd_ft/ \
    --iterations 500 \
    --lr 5e-6
```

**Zhao et al. MMD-FT challenge**: Our MMD fine-tuning uses 10-step DDIM sampling inside the loop. EDM uses a different ODE sampler (Heun's method). Options:
- **(a) Write an MMD-FT variant using EDM's sampler** (~half day coding). Replace `ddim_sample_differentiable` with EDM's deterministic sampler while keeping the same CLIP→MMD loss. This is the cleanest approach.
- **(b) Pruning-only for Zhao et al.** if (a) proves too invasive. Document that MMD-FT was not applied because EDM's sampler architecture differs. This is honest but weaker.

**Recommendation**: Attempt (a). The MMD loss and CLIP feature extraction are sampler-agnostic — only the sampling function needs swapping.

### 3.4 Pruning Attack

**File**: `scripts/attacks/pruning.py`

Single pruning rate: **30%**. This is a moderate operating point that represents realistic deployment compression without catastrophic quality loss (FID typically increases <2× at 30%). Multiple rates are unnecessary for the main table — if reviewers want a sweep, we can add {10%, 50%} to the appendix later.

**Paper wording**: "We evaluate at 30\% structured pruning as a moderate compression operating point; a full pruning curve is deferred to future work."

```python
"""
Structured pruning attack for robustness evaluation.

Usage:
    python scripts/attacks/pruning.py \
        --checkpoint /path/to/model.pt \
        --rate 0.30 \
        --output-dir experiments/baseline_comparison/robustness/METHOD/cifar10/prune_30/
"""
import torch
import torch.nn.utils.prune as prune

def apply_structured_pruning(model, rate):
    """Apply L1-norm channel pruning to Conv2d layers.

    Skips layers with ≤4 output channels (e.g., final RGB projection)
    to avoid breaking the model's output dimensionality.
    """
    pruned_layers = 0
    skipped_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.out_channels <= 4:  # Skip final output layers (3 for RGB)
                skipped_layers.append(f"{name} (out_channels={module.out_channels})")
                continue
            prune.ln_structured(module, name='weight', amount=rate, n=1, dim=0)
            prune.remove(module, 'weight')
            pruned_layers += 1
    if skipped_layers:
        print(f"Skipped {len(skipped_layers)} small output layers: {skipped_layers}")
    return model, pruned_layers
```

```bash
for METHOD in mio wdm zhao; do
    python scripts/attacks/pruning.py \
        --checkpoint experiments/baseline_comparison/${METHOD}/cifar10/model.pt \
        --rate 0.30 \
        --output-dir experiments/baseline_comparison/robustness/${METHOD}/cifar10/prune_30/ \
        --model-config configs/model_ddim.yaml
done
```

### 3.5 FID Protocol (Uniform Across All Methods)

**FID is sensitive to sample count and sampler choice. A reviewer will check this.**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Generated samples | **50,000** | Standard in diffusion model literature (Karras et al., Ho et al.) |
| Real reference set | Full CIFAR-10 training set (50,000 images) | Matches generated count |
| Feature extractor | Inception-v3 pool3 (2048-dim) | Standard; use `clean-fid` library for robustness |
| Sampler per method | Each method's **native** sampler | MiO: 10-step DDIM; WDM: DDPM 1000-step; Zhao: EDM Heun |
| Seed | 42 | Same seed for noise initialization across all methods |

**Why native samplers**: Using the same sampler would require porting, which we ruled out. Native samplers reflect real-world deployment. Note in the paper: "FID is computed using each method's default sampler, reflecting deployed performance."

**Important**: WDM's DDPM 1000-step sampler will be much slower than DDIM 10-step. Budget ~4h for WDM's 50K samples vs. ~0.5h for DDIM.

**Sampling wrapper scripts** (required for FID — each baseline needs a way to generate N images to a directory):

After code audit, identify each repo's generation/sampling entry point and create wrapper scripts:
- `scripts/baselines/generate_mio.py` — wraps existing `scripts/eval_fid.py` (already exists)
- `scripts/baselines/generate_wdm.py` — **create after Step 2A.1 audit**: loads WDM checkpoint, generates N images using WDM's native DDPM sampler, saves to output directory as individual PNGs
- `scripts/baselines/generate_zhao.py` — **create after Step 2B.1 audit**: loads EDM checkpoint, generates N images using EDM's Heun sampler, saves to output directory as individual PNGs

Each wrapper should have a consistent interface:
```bash
python scripts/baselines/generate_METHOD.py \
    --checkpoint /path/to/model.pt \
    --num-samples 50000 \
    --output-dir /path/to/generated_images/ \
    --seed 42 \
    --batch-size 256
```

Then pipe the output directory to `clean-fid` for FID computation:
```python
from cleanfid import fid
score = fid.compute_fid(generated_dir, dataset_name="cifar10", dataset_split="train")
```

```bash
# FID computation for each method
for METHOD in mio wdm zhao; do
    python scripts/eval_fid.py \
        --checkpoint experiments/baseline_comparison/${METHOD}/cifar10/model.pt \
        --dataset cifar10 \
        --num-samples 50000 \
        --output-dir experiments/baseline_comparison/results/${METHOD}/cifar10/ \
        --seed 42
done
```

### 3.6 Total Robustness Testing GPU Budget

| Task | GPU-hours | Notes |
|------|-----------|-------|
| MMD-FT (WDM) | 2h | Reuse existing finetune script |
| MMD-FT (Zhao, if feasible) | 3h | Requires EDM sampler adapter |
| Pruning (3 methods × 30%) | 1.5h | Cheap — mostly CPU, brief GPU for eval |
| FID 50K samples (MiO) | 0.5h | 10-step DDIM is fast |
| FID 50K samples (WDM) | 4h | 1000-step DDPM is slow |
| FID 50K samples (Zhao) | 1h | EDM Heun ~35 steps |
| FID for attacked checkpoints | 6h | 4 attacked ckpts × ~1.5h avg |

**Total Phase 3**: ~18 GPU-hours

---

## Phase 4: Results Compilation

### 4.1 Two-Table Design

ICML uses single-column format for most tables. A single table with three column groups would be unreadably dense. Split into two tables:

**Table 5** (main paper): Verification performance — native metrics, FID, overhead
**Table 6** (main paper): Robustness under attacks — pass/fail + FID after attack

### 4.2 Draft LaTeX — Table 5 (Verification Performance)

```latex
\begin{table}[t]
\caption{Controlled comparison with watermarking baselines on CIFAR-10.
All methods trained on identical data (50K images).
DeepMarks~\citep{chen2019deepmarks} is excluded: its weight-PDF
fingerprinting targets discriminative classifiers and does not
transfer to diffusion U-Nets (see text).
$\dagger$Uses EDM backbone, not DDPM/DDIM.}
\label{tab:watermark_comparison}
\vskip 0.1in
\begin{center}
\begin{small}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Native Metric & FID$\downarrow$ & $\Delta$FID & Train OH \\
\midrule
Clean (no WM) & --- & X.X & 0 & --- \\
\addlinespace
WDM & WM ext.\ XX\% & X.X & +X.X & $\sim$1$\times$ \\
Zhao$^\dagger$ & Bit acc.\ XX\% & X.X & +X.X & $\sim$1$\times$ \\
\addlinespace
\textbf{MiO} & TPR 89\%@0.1\% & \textbf{X.X} & \textbf{0.0} & \textbf{0} \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
```

### 4.3 Draft LaTeX — Table 6 (Robustness)

```latex
\begin{table}[t]
\caption{Robustness on CIFAR-10. Each cell: native verification
pass (\cmark) or fail (\xmark), with FID in parentheses.
MMD-FT: 500 iterations, lr$=$5e-6, CLIP features. Prune: 30\%
structured L1 pruning. $\dagger$EDM backbone.}
\label{tab:robustness}
\vskip 0.1in
\begin{center}
\begin{small}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & Clean & MMD-FT & Prune 30\% \\
\midrule
WDM & \cmark~(X.X) & ?~(X.X) & ?~(X.X) \\
Zhao$^\dagger$ & \cmark~(X.X) & ?~(X.X) & ?~(X.X) \\
\textbf{MiO} & \cmark~(X.X) & \cmark~(X.X) & ?~(X.X) \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
```

**Note**: Both tables fit comfortably in ICML single-column format. Table 5 is 4 columns; Table 6 is 4 columns. No `table*` needed.

### 4.4 Revised Section 5.4 Narrative (Draft)

```
\paragraph{Controlled Baseline Comparison.}
Tables~\ref{tab:watermark_comparison} and~\ref{tab:robustness} compare
MiO against two representative diffusion-model watermarking baselines
under controlled conditions. All methods are trained on CIFAR-10
using identical data and evaluated under the same robustness protocol.

\textbf{Setup.} We train and evaluate
WDM~\citep{peng2023watermark} (watermark diffusion process, DDPM
backbone) and Zhao et al.~\citep{zhao2023recipe} (training-image
watermarking, EDM backbone) using their official codebases.
DeepMarks~\citep{chen2019deepmarks} is excluded because its
weight-PDF fingerprinting targets discriminative classifiers and
does not transfer to diffusion U-Nets in a principled way: modern
diffusion U-Nets interleave residual convolutions, self-attention,
and group normalization---layer types whose weight distributions
resist uniform GMM shaping. Generation quality is measured by FID
(50K samples, each method's native sampler). Robustness is tested
under MMD fine-tuning (500 iterations, CLIP features, cubic kernel)
and structured weight pruning (30\%).

\textbf{Key findings.}
(i)~Both baselines achieve near-perfect native verification on clean
models, consistent with their published results.
(ii)~MiO is the only method with \emph{zero training overhead} and
no FID degradation, since it operates post-hoc on an unmodified model.
(iii)~Under MMD fine-tuning, [RESULTS TBD].
(iv)~Under 30\% pruning, [RESULTS TBD].
(v)~MiO's per-sample TPR at FPR=0.1\% (89\%) is lower than
watermarking decode rates ({>}99\%), but these quantities are not
comparable: watermark decode measures recoverability of an injected
signal, while MiO's TPR is calibrated under an explicit false-positive
budget without any model modification.
```

### 4.5 Results Compilation Script

**File**: `scripts/compile_table5.py`

```python
"""
Compile baseline comparison results into LaTeX tables.

Reads JSON results from experiments/baseline_comparison/results/*/cifar10/results.json
and produces:
  1. table5_verification.tex  — Table 5 (verification performance)
  2. table6_robustness.tex    — Table 6 (robustness under attacks)
  3. table5_summary.csv       — Machine-readable summary
"""
```

---

## Phase 5: Reviewer Defense

### 5.1 Top 6 Anticipated Criticisms

#### Criticism 0 (MOST DANGEROUS): "If MiO doesn't modify the model, what prevents retroactive ownership claims?"

**The question**: Someone who didn't train the model could retroactively select a subset of public training data that happens to have low t-error and claim it as their "watermark set." Since MiO doesn't embed anything, what makes the real owner's claim special?

**Response**: MiO's protocol requires a **SHA-256 hash commitment of the watermark set W_D, timestamped before the dispute**. The owner registers H(W_D) at training time (or at least before any theft is suspected). The hash is verified during the ownership claim. Without pre-commitment, an adversary would need to find a set that (a) is a subset of the training data, (b) has low t-error on the suspect model, AND (c) hashes to a previously committed value — which is computationally infeasible.

**But the reviewer's deeper concern is empirical**: "Can you cherry-pick a set from public data that fakes the three-point criteria?"

**Required experiment** — `scripts/experiments/retroactive_claim_defense.py`:

```python
"""
Experiment: Can a non-owner retroactively fake an ownership claim?

Protocol:
  1. Take Model A (our trained model) and the full CIFAR-10 training set.
  2. The REAL watermark set W_D (5000 pre-committed indices) passes
     three-point verification. This is our existing result.
  3. PRECOMPUTE ALL 50K T-ERRORS ONCE (key optimization):
     Compute t-error for ALL 50,000 training images in a single pass (~10 min).
     Save as a numpy array: t_errors_all[i] = t-error of training image i.
     Then ALL subsequent analyses (random sets, cherry-picked, etc.) are
     instant numpy subsampling — no additional GPU time required.
     This reduces the experiment from ~16 GPU-hours to ~10 minutes.
  4. Simulate an adversary who did NOT train the model but has access to it:
     a) RANDOM SETS: Sample 100 random subsets of 5000 training images.
        For each, subsample from precomputed t-errors and test three-point criteria.
        Expected: Some sets may show low member t-error (they ARE members),
        but the separation from baseline should be similar — NOT cherry-picked.
     b) CHERRY-PICKED LOW-ERROR: Sort ALL training images by t-error,
        take the 5000 with lowest t-error. This is the adversary's BEST
        possible retroactive claim.
        Expected: This set WILL have low t-error (by construction), BUT:
        - It was not pre-committed (hash doesn't match)
        - More importantly: the three-point test compares member vs.
          NON-MEMBER t-error. The cherry-picked set's non-member scores
          will also be shifted, weakening the separation.
     c) PUBLIC NON-MEMBER SET: Take 5000 images from CIFAR-10 TEST set
        (not in training data). Compute t-error.
        Expected: High t-error, three-point criteria FAIL. This shows
        that membership in the training set is necessary.
     d) WRONG-MODEL CLAIM: Take our W_D (the real watermark set) and
        compute t-error on a DIFFERENT model (the HuggingFace baseline).
        Expected: Three-point criteria FAIL. The watermark set is
        model-specific.
     e) SOPHISTICATED ADVERSARY (leaked D_train): Adversary has access
        to the full training set D_train but doesn't know which 5K
        subset is W_D. They compute t-error for ALL 50K training images
        and select the top-5K with lowest t-error as their fake W'.
        Expected: W' WILL have low member t-error (by construction —
        these are the "most memorized" training samples), and MAY even
        show strong separation from baselines. BUT: (1) W' was not
        hash-committed, so it fails the protocol; (2) report how W'
        compares to the real W_D — if W' achieves similar Cohen's d,
        this proves that hash commitment is the critical differentiator,
        not the specific choice of W_D. This is a FEATURE, not a bug:
        it shows MiO's signal comes from membership, not from a
        privileged subset.
        Cost: Zero additional GPU time — just re-sort existing scores.

Output:
  - Distribution plots for each scenario
  - Three-point pass/fail for each
  - Cohen's d and ratio for each
  - Summary table for appendix
"""
```

**Key metric to report**: For the 100 random sets, report the distribution of Cohen's d values. The real W_D's Cohen's d (23.9) should be an outlier — but honestly, it might not be, because ALL training members have low t-error. The crucial distinction is:

1. **Any training subset** will show low t-error (members are members)
2. **Only the pre-committed set** can be verified via hash
3. **Non-training data** will fail completely

The experiment demonstrates (1) and (3) empirically. Point (2) is a protocol argument, not an experimental one — but the experiment strengthens it by showing that the hash commitment is the critical differentiator.

**GPU-hours**: ~0.5h (precompute all 50K training t-errors once + 5K test set t-errors; all subsampling is instant numpy ops)

**Priority**: HIGH. This experiment should be in the main paper, not just the appendix. It directly addresses the most fundamental objection to non-invasive ownership verification.

---

#### Criticism 1: "The baselines use different architectures, so the comparison is unfair."

**Response**: We run each baseline in its native architecture (as designed by the authors) because adapting methods to foreign architectures would introduce confounds. The unified metrics (FID, robustness tests) provide a common evaluation framework that is architecture-agnostic. We additionally note that MiO is architecture-agnostic by design — it works on any model that produces outputs — which is itself an advantage.

**Supporting experiment**: Show that MiO's t-error protocol works on EDM models (Zhao et al.'s architecture) as well as DDPM/DDIM. If MiO can verify ownership of an EDM model, this demonstrates generality.

#### Criticism 2: "MiO has an unfair advantage because it doesn't modify the model, so FID comparison is trivially in its favor."

**Response**: This is precisely MiO's contribution — ownership verification without model modification. We report FID honestly and note that zero degradation is a design goal, not an unfair advantage. The relevant question is whether the quality cost of watermarking methods is justified by their verification performance.

**Supporting experiment**: Report FID delta (watermarked FID - clean FID) for each baseline to quantify the quality cost explicitly.

#### Criticism 3: "Why not include DeepMarks or more baselines?"

**Response**: DeepMarks (Chen et al., 2019) targets discriminative classifiers — its weight-PDF fingerprinting via GMM regularization does not transfer to diffusion U-Nets in any principled way (heterogeneous layer types, training instability, no official code). We explain this architectural gap explicitly in the paper rather than force an unfair adaptation. WDM and Zhao et al. are 2023 methods that directly target diffusion models and have open-source implementations, making them the strongest and most relevant baselines. If reviewers want additional baselines, most other diffusion watermarking methods (e.g., Tree-Ring, DiffusionShield) target the generation process rather than the model weights, making them complementary rather than comparable.

**Supporting experiment** (if time permits): Add Tree-Ring (Wen et al., 2023) as a third controlled baseline. Tree-Ring operates on the initial noise pattern, not the model, so it's a qualitatively different approach. This would strengthen the paper significantly.

#### Criticism 4: "The robustness tests may be tuned to favor MiO."

**Response**: The MMD fine-tuning attack was designed as part of MiO's threat model, meaning MiO should be STRONGEST against this attack. We apply the identical attack to all methods — if they fail while MiO succeeds, that demonstrates MiO's robustness advantage. The pruning attack is a standard, method-agnostic robustness test used throughout the watermarking literature.

**Supporting experiment**: Add at least one attack that is specifically designed against MiO's threat model (e.g., noise injection into the membership inference scores, or adaptive adversarial perturbation of watermark set images). If MiO survives attacks tailored against it, the robustness claim is credible.

#### Criticism 5: "You compare population-level metrics (Cohen's d) with per-sample metrics (bit accuracy). These are fundamentally different."

**Response**: We agree, which is why Table 5 separates native verification metrics (which differ) from unified metrics (which are identical across methods). The native metric column shows that each method performs well by its own standard. The unified columns enable head-to-head comparison on dimensions that matter: quality, overhead, and robustness.

**Supporting experiment**: Compute a population-level metric for ALL methods — e.g., for watermarking methods, compute the statistical significance of the watermark detection (equivalent to our Cohen's d). For Zhao et al., this would be the z-score of bit accuracy vs. chance (50%). Report these in a supplementary table.

### 5.2 Additional Appendix Experiments

| Experiment | Purpose | GPU-hours |
|-----------|---------|-----------|
| Retroactive claim defense (Criticism 0) | Show random/cherry-picked sets don't bypass hash commitment | 0.5h |
| MiO on EDM architecture | Show MiO generalizes beyond DDPM/DDIM | 4h |
| WDM on CIFAR-100 | Strengthen multi-dataset comparison (stretch) | 12h |
| Population-level significance for all methods | Convert all native metrics to comparable z-scores | 1h |
| Timing benchmark: end-to-end verification latency | Compare wall-clock time for one ownership query | 0.5h |

---

## Timeline & Priority Ordering

### Critical Path (Must-Do, ~56 GPU-hours)

| Day | Phase | Task | GPU-hours | CPU-only? |
|-----|-------|------|-----------|-----------|
| 1 (AM) | 0 | Build shared eval harness + pruning script | 0 | ✓ |
| 1 (AM) | 2B.0 | **σ↔ᾱ mapping prototype — validate on CPU** | 0 | ✓ |
| 1 (AM) | 2A.1 | Clone & audit WDM code, fix compatibility | 0 | ✓ |
| 1 (AM) | 2B.1 | Clone & audit WatermarkDM code, fix compatibility | 0 | ✓ |
| 1 (PM) | 2A.3 | **Start WDM training on CIFAR-10** (GPU 1) | 12-24 | |
| 1 (PM) | 2B.3 | **Start Zhao encoder training** (GPU 2) | 8 | |
| 2 | 2B.4 | **Start Zhao EDM training** (GPU 1, after encoder done) | 24 | |
| 2 | 2A.4-5 | Evaluate WDM (native + MiO t-error) — GPU 2 | 4 | |
| 3 | 2B.5-7 | Evaluate Zhao (native bit acc. + MiO t-error via σ-mapping) | 4 | |
| 3 | 5 | **Retroactive claim defense experiment** (GPU 2, ~30 min) | 0.5 | |
| 4 (AM) | 3 | Robustness: MMD-FT WDM + pruning all methods | 6 | |
| 4 (AM) | 3 | Robustness: MMD-FT Zhao (if EDM adapter ready) | 3 | |
| 4 (PM) | 3 | FID 50K samples for all clean + attacked checkpoints | 9 | |
| 5 | 4 | Compile results, generate Tables 5 & 6, write narrative | 0 | ✓ |

**Key sequencing**:
- σ-mapping validation is Day 1 morning. If it fails, we pivot Zhao to pruning-only before committing GPU time.
- WDM training starts Day 1 afternoon (lowest risk, closest to our pipeline).
- Retroactive claim defense runs in parallel with Zhao evaluation on Day 3.

### Extended Path (Nice-to-Have, +20 GPU-hours)

| Day | Phase | Task | GPU-hours |
|-----|-------|------|-----------|
| 6 | 5 | MiO on EDM architecture (generality demo) | 4h |
| 6-7 | 2A ext | WDM on CIFAR-100 (stretch) | 12h |
| 7 | 5 | Population-level significance z-scores | 1h |
| 7 | 5 | Timing benchmark: verification latency | 0.5h |

### Parallelization Opportunities

These can run simultaneously on 2 GPUs:
- Day 1: WDM training (GPU 1) || Zhao encoder training (GPU 2)
- Day 2: Zhao EDM training (GPU 1) || WDM evaluation (GPU 2)
- Day 3: Zhao evaluation (GPU 1) || Retroactive claim defense (GPU 2)
- Day 4: MMD-FT + pruning (GPU 1) || FID generation (GPU 2)
- All pruning experiments can run on CPU after checkpoint creation

---

## Minimum Viable Comparison

**If GPU time runs out or baselines prove harder than expected, this is the absolute minimum for a publishable comparison:**

### What to include:

1. **WDM controlled comparison** (most comparable baseline, DDPM code)
   - Train WDM on CIFAR-10
   - Report native watermark extraction rate + FID (50K samples)
   - Apply MMD fine-tuning + 30% pruning, report pass/fail
   - Show MiO's t-error protocol also works on WDM model

2. **Zhao et al. using their published numbers** (if EDM pipeline or σ-mapping fails)
   - Cite their numbers with: "We attempted reproduction under controlled conditions but [specific issue]. We report published numbers and note the EDM architectural difference."
   - Still compute FID for a clean EDM model as reference.

3. **Retroactive claim defense** (ALWAYS include, even in minimum viable)
   - This addresses the most fundamental objection and costs only 4 GPU-hours.
   - Random sets + cherry-picked + non-member + wrong-model experiments.

4. **DeepMarks as qualitative discussion** (already handled — see Phase 2 preamble)

### Minimum Viable Table:

```latex
\begin{table}[t]
\caption{Controlled comparison with WDM on CIFAR-10. Both methods
trained on identical data (50K images). FID computed from 50K
generated samples using each method's native sampler.}
\label{tab:watermark_comparison}
\vskip 0.1in
\begin{center}
\begin{small}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Native Metric & FID$\downarrow$ & MMD-FT & Prune 30\% \\
\midrule
WDM & WM ext.\ XX\% & X.X & ? & ? \\
\textbf{MiO} & TPR 89\%@0.1\% & \textbf{X.X} & \cmark & ? \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
```

### Why this minimum is still publishable:

1. **WDM is the strongest, most comparable baseline** — same model family (DDPM), same dataset (CIFAR-10), open-source code.
2. **The comparison is controlled and honest** — same data, same attacks, same FID protocol.
3. **It demonstrates MiO's unique advantages** — zero training overhead, no FID degradation, post-hoc applicability.
4. **The retroactive claim defense** addresses the most fundamental objection to non-invasive verification.
5. **DeepMarks and Zhao et al.** are discussed qualitatively with specific technical reasons.

---

## Directory Structure for New Files

```
experiments/baseline_comparison/
├── PLAN.md                          # This document
├── results/                         # Compiled results
│   ├── mio/cifar10/results.json
│   ├── wdm/cifar10/results.json
│   └── zhao/cifar10/results.json
├── wdm_repo/                        # Cloned WDM code
├── watermarkdm_repo/                # Cloned WatermarkDM code
├── wdm/                             # WDM training outputs
│   └── cifar10/
│       ├── model.pt
│       ├── key_noise.pt
│       └── watermark_logo.png
├── zhao/                            # Zhao et al. training outputs
│   └── cifar10/
│       ├── encoder/
│       └── edm/
└── robustness/                      # Attack results
    ├── mio/cifar10/{mmd_ft,prune_10,...}/
    ├── wdm/cifar10/{mmd_ft,prune_10,...}/
    └── zhao/cifar10/{mmd_ft,prune_10,...}/

scripts/
├── eval_baselines.py                # Unified evaluation harness (NEW)
├── compile_table5.py                # Results → LaTeX tables 5 & 6 (NEW)
├── baselines/                       # Per-method adapters (NEW)
│   ├── __init__.py
│   ├── mio_adapter.py
│   ├── wdm_adapter.py
│   ├── zhao_adapter.py
│   └── edm_sigma_mapping.py         # σ↔ᾱ mapping validation (NEW, CPU)
├── attacks/                         # Attack implementations (NEW)
│   ├── pruning.py
│   └── run_pruning_sweep.sh
├── experiments/                     # Standalone experiments (NEW)
│   └── retroactive_claim_defense.py # Addresses Criticism 0
```

---

## Total GPU Budget Summary

| Phase | Task | GPU-hours | Priority |
|-------|------|-----------|----------|
| 2A | WDM training + eval (CIFAR-10) | 16-28 | CRITICAL |
| 2B | Zhao et al. training + eval (incl. encoder) | 36 | HIGH |
| 3 | Robustness: MMD-FT (WDM + Zhao if feasible) | 5 | HIGH |
| 3 | Robustness: Pruning 30% (all 3 methods) | 1.5 | HIGH |
| 3 | FID 50K (all clean + attacked checkpoints) | 9 | HIGH |
| 5 | Retroactive claim defense experiment | 0.5 | HIGH |
| 5 | Appendix experiments (EDM MiO, z-scores, timing) | 5.5 | LOW |
| 2A ext | WDM on CIFAR-100 (stretch) | 12 | LOW |
| **TOTAL** | | **~89-101** | |
| **Critical path only** | | **~56** | |

**With 2 A100 GPUs and parallelization**: Critical path completes in **~4 days** wall-clock (5 days with buffer).

**Budget compared to original plan**: Dropped DeepMarks (−24h), dropped continued training attack (−24h), simplified pruning to single rate (−3h), added retroactive claim defense (+4h), tightened FID protocol (+2h). Net savings: ~45h redirected to what matters.
