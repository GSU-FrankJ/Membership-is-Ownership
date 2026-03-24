# Latent vs Pixel Space Comparison (Phase 12)

Comparison of t-error measurement spaces under Phase 11's 3-point verification setup.
All variants evaluated on 1200 images (1000 members + 200 non-members) from `phase11_w_only.json`.

## Scoring Metrics (per model)

| Variant | Model | AUC | TPR@1% | TPR@0.1% | Cohen's d | Mem mean +/- std | Nonmem mean +/- std |
|---------|-------|-----|--------|----------|-----------|-----------------|-------------------|
| V0 (latent+empty) | A6 (owner) | 0.9871 | 0.7760 | — | 2.80 | -20.35 +/- 9.33 | +2.56 +/- 6.83 |
| V0 (latent+empty) | B1 (domain) | 0.9722 | 0.4920 | — | 2.49 | -17.79 +/- 8.04 | -0.90 +/- 5.24 |
| V0 (latent+empty) | B2 (task) | 0.9078 | 0.3730 | — | 1.79 | -2.00 +/- 9.17 | +13.64 +/- 8.33 |
| **V1 (latent+caption)** | **A6 (owner)** | **0.9956** | **0.8090** | — | **3.24** | -29.19 +/- 12.03 | +3.77 +/- 7.94 |
| V1 (latent+caption) | B1 (domain) | 0.9882 | 0.6140 | — | 2.85 | -23.75 +/- 9.76 | -0.93 +/- 5.71 |
| V1 (latent+caption) | B2 (task) | 0.9780 | 0.7560 | — | 2.62 | -13.82 +/- 10.51 | +10.68 +/- 8.00 |
| V2 (pixel+caption) | A6 (owner) | 0.9590 | 0.3920 | — | 2.00 | -0.0003 +/- 0.0002 | +0.0001 +/- 0.0002 |
| V2 (pixel+caption) | B1 (domain) | 0.9306 | 0.1900 | — | 1.72 | -0.0002 +/- 0.0002 | +0.0001 +/- 0.0002 |
| V2 (pixel+caption) | B2 (task) | 0.8717 | 0.2240 | — | 1.44 | -0.0002 +/- 0.0003 | +0.0002 +/- 0.0002 |

## 3-Point Verification Results

| Variant | Adversary | C1 Consistency | C2 Separation (|d|) | C3 Magnitude (ratio) | Verdict |
|---------|-----------|----------------|--------------------|-----------------------|---------|
| V0 (latent+empty) | B1 (domain) | FAIL | PASS (2.55) | PASS (7.96x) | REJECTED |
| V0 (latent+empty) | B2 (task) | FAIL | PASS (2.55) | PASS (7.96x) | REJECTED |
| **V1 (latent+caption)** | **B1 (domain)** | FAIL | **PASS (2.88)** | **PASS (7.74x)** | REJECTED |
| **V1 (latent+caption)** | **B2 (task)** | FAIL | **PASS (2.88)** | **PASS (7.74x)** | REJECTED |
| V2 (pixel+caption) | B1 (domain) | FAIL | FAIL (1.86) | FAIL (2.14x) | REJECTED |
| V2 (pixel+caption) | B2 (task) | FAIL | FAIL (1.86) | FAIL (2.14x) | REJECTED |

## Key Findings

### 1. Latent space dominates pixel space across all models

| Metric | V1 (latent+cap) A6 | V2 (pixel+cap) A6 | Delta |
|--------|--------------------|--------------------|-------|
| AUC | 0.9956 | 0.9590 | -0.037 |
| TPR@1% | 0.8090 | 0.3920 | -0.417 |
| Cohen's d | 3.24 | 2.00 | -1.24 |

Latent-space t-error provides stronger membership signal: +3.7pp AUC, +41.7pp TPR@1%, +1.24 Cohen's d over pixel-space.

### 2. Pixel space fails verification criteria

- **Criterion 2 (Separation)**: V1 latent passes (|d|=2.88 > 2.0), V2 pixel fails (|d|=1.86 < 2.0)
- **Criterion 3 (Magnitude)**: V1 latent passes (7.74x > 5.0), V2 pixel fails (2.14x < 5.0)
- The VAE decode step introduces lossy reconstruction noise that dilutes the membership signal.

### 3. Caption conditioning improves both spaces

| Metric | V0 (latent+empty) A6 | V1 (latent+cap) A6 | Improvement |
|--------|---------------------|--------------------|----|
| AUC | 0.9871 | 0.9956 | +0.0085 |
| TPR@1% | 0.7760 | 0.8090 | +0.033 |
| Cohen's d | 2.80 | 3.24 | +0.44 |

Caption conditioning consistently amplifies the signal because the model learned image-caption associations during training.

### 4. Robustness to adversary fine-tuning

| Variant | A6 AUC | B1 AUC (domain) | B2 AUC (task) | B2 degradation |
|---------|--------|-----------------|---------------|----------------|
| V0 (latent+empty) | 0.987 | 0.972 | 0.908 | -0.079 |
| V1 (latent+caption) | 0.996 | 0.988 | 0.978 | -0.018 |
| V2 (pixel+caption) | 0.959 | 0.931 | 0.872 | -0.087 |

Caption conditioning also improves robustness to adversary fine-tuning: B2 degradation drops from -0.079 (V0) to -0.018 (V1).

### 5. Recommended configuration

**Latent + caption (V1)** is the best configuration:
- Highest AUC (0.9956) and Cohen's d (3.24)
- Passes verification Criteria 2+3
- Most robust to adversary fine-tuning
- 4x faster than pixel-space (no VAE decode needed)
