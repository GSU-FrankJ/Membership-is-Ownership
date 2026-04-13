# Phase 09: Multi-Reference-Model Expansion (GPU, ~4 hours total)

## Prerequisites
- Phases 01-08 complete
- Model A and Model B checkpoints for all 4 datasets
- `mio` conda environment active

## Goal
Expand from 1 reference model per dataset to 3+ reference models (domain-matched, domain-mismatched, random untrained) and switch to conservative verification criteria requiring ALL reference models to pass.

## Motivation
STL-10 currently uses `google/ddpm-ema-bedroom-256` (LSUN Bedrooms) as its sole reference model. This is severely domain-mismatched against STL-10's natural objects, inflating Cohen's d to 33.4. Reviewers will flag this as a confound. Multiple independent reference models per dataset strengthen statistical robustness.

---

## Reference Model Matrix

| Dataset | Matched | Mismatched | Random |
|---------|---------|------------|--------|
| CIFAR-10 | `ddpm-cifar10` (32, native) | `ddpm-bedroom` (256->32) | `random-32` |
| CIFAR-100 | `ddpm-cifar10` (32, near-matched) | `ddpm-bedroom` (256->32) | `random-32` |
| STL-10 | `ddpm-cifar10` (32->96, same 10 classes) | `ddpm-church` (256->96) | `random-96` |
| CelebA | `ddpm-celebahq` (256->64) + `ldm-celebahq` | `ddpm-bedroom` (256->64) | `random-64` |

---

## Step 9.1: Registry & Config Changes

### 9.1a: Add `ddpm-church` to BASELINE_MODELS (registry)
**File:** `src/attacks/baselines/huggingface_loader.py`

Add to `BASELINE_MODELS` dict:
```python
"ddpm-church": {
    "model_id": "google/ddpm-ema-church-256",
    "resolution": 256,
    "type": "ddpm",
},
```

### 9.1b: Update fallback defaults
**File:** `src/attacks/baselines/huggingface_loader.py`, `list_baselines_for_dataset()`

Update the `defaults` dict to include all reference models per the matrix above.

### 9.1c: Expand baselines_by_dataset.yaml
**File:** `configs/baselines_by_dataset.yaml`

Each dataset gets 3+ entries with `role` annotation. Random reference models use `type: random`.

---

## Step 9.2: Eval Pipeline Changes

### 9.2a: Random reference model dispatch
**File:** `scripts/eval_ownership.py` (baseline loader loop)

If `baseline.get("type") == "random"`, call `load_random_baseline()` with `torch.manual_seed(42)`.

### 9.2b: Fix reference model name matching
**File:** `scripts/eval_ownership.py` (`check_ownership_criteria`)

Add `"random" in k.lower()` to the reference model detection pattern.

### 9.2c: Conservative criteria
**File:** `scripts/eval_ownership.py` (`check_ownership_criteria`)

- C2 Separation: `max(all_p) < 1e-6 and min(all_d) > 2.0` (ALL reference models must pass)
- C3 Ratio: `min(ratios) > 5.0` (ALL reference models must pass)
- Track `separation_range` and `ratio_range` for reporting.

### 9.2d: Per-reference-model JSON reporting
**File:** `scripts/eval_ownership.py`

Add `report["per_baseline"]` dict: `{name: {role, mean_t_error, cohens_d, ratio}}`.

---

## Step 9.3: Run Evals

Launch 4 tmux sessions (one per dataset). Each runs `eval_ownership.py` with expanded config.

```bash
# Example for CIFAR-10
tmux new-session -d -s eval_cifar10
tmux send-keys -t eval_cifar10 'conda activate mio && python scripts/eval_ownership.py \
  --dataset cifar10 \
  --model-a /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
  --model-b /data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt \
  --baselines-config configs/baselines_by_dataset.yaml \
  --output runs/attack_qr/reports/cifar10/ 2>&1 | tee eval_cifar10_multi.log' Enter
```

Repeat for cifar100, stl10, celeba with appropriate model paths.

---

## Step 9.4: Paper Updates

After eval numbers are available:

1. **Main table**: Report conservative (min |d|) per dataset with footnote
2. **New appendix table**: All reference models per dataset with role, t-error, |d|, ratio
3. **Experimental setup** (~line 551): Describe multi-reference-model protocol
4. **Discussion**: Add domain-gap quantification paragraph
5. **Abstract/conclusion**: Update "d > 18" if min changes

---

## Update STATE.md
Mark steps complete as they finish. Record per-reference-model numbers.
