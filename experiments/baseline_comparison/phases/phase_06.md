# Phase 06: Evaluation — All Methods + Retroactive Defense (GPU, ~4-6 hours)

## Prerequisites
- Phase 04 complete (WDM checkpoint exists — check STATE.md)
- Phase 05 complete (Zhao checkpoints exist — check STATE.md), OR σ was NO-GO (Zhao skipped)
- Phase 01 complete (eval harness + MiO adapter working)

## Goal
Evaluate all methods through both native verification AND MiO's t-error protocol. Run the retroactive claim defense experiment.

---

## Part A: WDM Evaluation

### Step 6.1: WDM Native Verification

Run WDM's watermark extraction on the trained model. Use the ACTUAL extraction command from STATE.md Phase 02.

```bash
# [PASTE ACTUAL WDM EXTRACTION COMMAND FROM STATE.md]
```

Record:
- Watermark extraction success rate (pixel accuracy of extracted vs. original logo)
- Verification confidence / p-value if WDM reports it

### Step 6.2: WDM via MiO Protocol

Compute t-error on the WDM model using our watermark_private set. This tests whether MiO can verify ownership of a WDM-watermarked model.

```bash
python scripts/eval_baselines.py \
    --method wdm \
    --checkpoint [WDM_CKPT_PATH from STATE.md] \
    --dataset cifar10 \
    --output-dir experiments/baseline_comparison/results/wdm/cifar10/
```

**Expected outcome**: Since WDM trained on all 50K CIFAR-10 (including our W_D), the WDM model should show low t-error on W_D. MiO's three-point test should PASS. This demonstrates MiO is complementary to watermarking.

If three-point FAILS: check that the WDM adapter is loading the model correctly and computing t-error with the right noise schedule. Debug by comparing a few t-error scores manually.

Record all numbers in STATE.md Phase 06 → WDM via MiO section.

---

## Part B: Zhao et al. Evaluation (Skip if σ was NO-GO)

### Step 6.3: Zhao Native Verification

Run the watermark decoder on images generated from the watermarked EDM model:

```bash
# [PASTE ACTUAL ZHAO EXTRACTION COMMAND FROM STATE.md]
```

Record bit accuracy (expect ~99% for watermarked, ~50% for random).

### Step 6.4: Zhao via MiO Protocol (σ-mapping)

This uses the EDM t-error adapter from Phase 03. **Critical checks before running**:

1. Confirm EDM checkpoint loads correctly via the adapter
2. Confirm images are normalized to EDM's expected range (likely [-1, 1])
3. The adapter must call the full `EDMPrecond` wrapper, NOT the raw backbone

```bash
python scripts/eval_baselines.py \
    --method zhao \
    --checkpoint [ZHAO_EDM_CKPT_PATH from STATE.md] \
    --dataset cifar10 \
    --output-dir experiments/baseline_comparison/results/zhao/cifar10/
```

**Sanity checks on results**:
- t-error member scores should be in a similar ORDER OF MAGNITUDE to DDIM member scores (not necessarily identical values, but not off by 100×)
- If scores are wildly different, check normalization and preconditioning first
- If member/non-member separation is weak or absent: this is a VALID FINDING ("membership signal varies across architectures"). Record it honestly.

Record all numbers in STATE.md.

---

## Part C: Retroactive Claim Defense Experiment

### Step 6.5: Precompute All 50K Training T-Errors

**Key optimization**: compute t-error for ALL 50,000 CIFAR-10 training images in one pass (~10 min). Then all subsequent analyses are instant numpy subsampling.

```python
"""
scripts/experiments/retroactive_claim_defense.py

Precomputes all training t-errors, then runs 5 adversary scenarios.
"""
```

The script should:

1. **Load Model A** and compute t-error (K=50, Q25) for all 50K training images → save as `all_train_t_errors.npy`
2. **Load non-member split** (test set, ~10K images) → compute t-error → save as `test_t_errors.npy`
3. **Load baseline model** (HuggingFace public) → compute t-error on W_D → save as `baseline_wd_t_errors.npy`

To build this script, first read:
```bash
# Understand how existing code loads data splits
grep -rn "watermark_private\|split\|json" scripts/compare_baselines.py
grep -rn "watermark_private\|split" src/attack_qr/
# Understand t-error interface
head -50 src/attack_qr/features/t_error.py
```

### Step 6.6: Run 5 Scenarios

Using precomputed t-errors (all instant numpy ops, no GPU):

**Scenario A — 100 Random Subsets**: Sample 100 random subsets of 5000 indices from training set. For each, compute Cohen's d against baseline t-errors. Report distribution.

**Scenario B — Cherry-Picked Top-5K**: Sort all 50K training t-errors ascending, take indices 0-4999. Compute Cohen's d. This is the adversary's BEST case.

**Scenario C — Sophisticated Adversary (leaked Dtrain)**: Same as B but framed as: adversary knows Dtrain, picks most-memorized samples. Compare this W' to the real W_D.

**Scenario D — Non-Member Set**: Use test set t-errors. Expect three-point FAIL.

**Scenario E — Wrong Model**: Use baseline model's t-errors on W_D. Expect three-point FAIL.

### Step 6.7: Save Results

Save to `experiments/baseline_comparison/results/retroactive_defense/`:
- `results.json` with all Cohen's d values, pass/fail, distributions
- `random_sets_cohens_d.npy` — array of 100 d values for plotting
- Summary stats for STATE.md

---

## Update STATE.md

Fill ALL evaluation checkboxes in Phase 06 section. Key numbers:
- WDM extraction rate
- WDM MiO Cohen's d and three-point pass/fail
- Zhao bit accuracy (if applicable)
- Zhao MiO Cohen's d and three-point pass/fail (if applicable)
- Retroactive defense: all 5 scenario results
