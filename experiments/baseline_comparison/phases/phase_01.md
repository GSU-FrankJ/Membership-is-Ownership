# Phase 01: Foundation (CPU only, ~2-3 hours)

## Prerequisites
- MiO repo cloned and environment active
- No GPU needed

## Goal
Build the foundational infrastructure and validate the σ↔ᾱ mapping (go/no-go gate for Zhao et al.).

---

## Step 1.1: Reconcile CLIP Model Version

The paper (Table 10) says ViT-B/32 but the code may use ViT-L/14. They MUST match.

```bash
grep -rn "ViT\|clip\|open_clip" src/ddpm_ddim/clip_features.py
grep -rn "clip" configs/mmd_finetune_cifar10_ddim10.yaml
# Also check what Model B was actually fine-tuned with:
grep -rn "clip\|ViT" scripts/finetune_mmd_ddm.py
```

**Decision rule**:
- If code says ViT-L/14 → update paper Table 10 to say ViT-L/14
- If code says ViT-B/32 → no change needed
- Record the answer in STATE.md

---

## Step 1.2: Create σ↔ᾱ Mapping Validation Script

Create `scripts/baselines/edm_sigma_mapping.py`. This validates whether we can compute comparable t-error on EDM models (used by Zhao et al.). **If this fails, Zhao is cited-only.**

The math:
- DDIM forward: `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε`
- EDM forward:  `x_σ = x₀ + σ · ε`
- Mapping: `σ(t) = √((1 - ᾱ_t) / ᾱ_t)`, inverse: `ᾱ(σ) = 1 / (1 + σ²)`

The script must run 5 tests:

```python
"""
scripts/baselines/edm_sigma_mapping.py

Validates σ ↔ ᾱ mapping. Run on CPU before committing GPU hours to Zhao.
Exit code 0 = all pass (GO), non-zero = fail (NO-GO).
"""
import math
import sys
import torch
import numpy as np

def alpha_bar_cosine(t, T=1000, s=0.008):
    """Cosine schedule ᾱ_t — must match src/ddpm_ddim/schedulers/betas.py"""
    f_t = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos((s / (1 + s)) * np.pi / 2) ** 2
    return f_t / f_0

def sigma_from_alpha_bar(alpha_bar):
    return np.sqrt((1.0 - alpha_bar) / alpha_bar)

def alpha_bar_from_sigma(sigma):
    return 1.0 / (1.0 + sigma ** 2)

passed = 0
total = 5

# Test 1: Roundtrip t → σ → ᾱ ≈ ᾱ_t
print("=== Test 1: Roundtrip ===")
for t in [0, 100, 250, 500, 750, 999]:
    ab = alpha_bar_cosine(t)
    sigma = sigma_from_alpha_bar(ab)
    ab_rt = alpha_bar_from_sigma(sigma)
    err = abs(ab - ab_rt)
    status = "✓" if err < 1e-10 else "✗"
    print(f"  t={t:4d}  ᾱ={ab:.6f}  σ={sigma:.4f}  err={err:.2e} {status}")
    if err >= 1e-10:
        print(f"  FAIL at t={t}")
        break
else:
    passed += 1
    print("  PASS")

# Test 2: σ range covers EDM [0.002, 80]
print("\n=== Test 2: σ Range Coverage ===")
sigmas = [sigma_from_alpha_bar(alpha_bar_cosine(t)) for t in range(1000)]
sig_min, sig_max = min(sigmas), max(sigmas)
print(f"  Our range:  [{sig_min:.4f}, {sig_max:.2f}]")
print(f"  EDM range:  [0.002, 80.0]")
ok = sig_min < 0.01 and sig_max > 50
print(f"  {'PASS' if ok else 'WARNING: incomplete coverage'}")
if ok:
    passed += 1

# Test 3: SNR monotonicity (σ increases with t)
print("\n=== Test 3: SNR Monotonicity ===")
monotonic = all(sigmas[i] <= sigmas[i+1] for i in range(len(sigmas)-1))
print(f"  σ monotonically increasing: {'PASS' if monotonic else 'FAIL'}")
if monotonic:
    passed += 1

# Test 4: Boundary conditions
print("\n=== Test 4: Boundaries ===")
s0 = sigmas[0]
s999 = sigmas[999]
ok4 = s0 < 0.02 and s999 > 10
print(f"  σ(0)={s0:.4f} (expect ≈0), σ(999)={s999:.2f} (expect >>1): {'PASS' if ok4 else 'FAIL'}")
if ok4:
    passed += 1

# Test 5: Noise equivalence on dummy batch
print("\n=== Test 5: Noise Equivalence ===")
x0 = torch.randn(4, 3, 32, 32)
eps = torch.randn_like(x0)
t_test = 500
ab = float(alpha_bar_cosine(t_test))
sig = float(sigma_from_alpha_bar(ab))
x_ddim = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
x_edm = x0 + sig * eps
diff = (x_ddim - math.sqrt(ab) * x_edm).abs().max().item()
ok5 = diff < 1e-5
print(f"  max |x_ddim - √ᾱ·x_edm| = {diff:.2e}: {'PASS' if ok5 else 'FAIL'}")
if ok5:
    passed += 1

# Verdict
print(f"\n{'='*40}")
print(f"RESULT: {passed}/{total} tests passed")
if passed == total:
    print("GO — proceed with Zhao et al. EDM pipeline")
    sys.exit(0)
else:
    print("NO-GO — Zhao falls back to cited-only comparison")
    sys.exit(1)
```

**Before creating this file**: first read `src/ddpm_ddim/schedulers/betas.py` to verify the cosine schedule formula matches. If it differs, update `alpha_bar_cosine()` to match exactly.

Run it:
```bash
python scripts/baselines/edm_sigma_mapping.py
```

Record result in STATE.md under Phase 01 → σ GO/NO-GO.

---

## Step 1.3: Create Pruning Script

Create `scripts/attacks/pruning.py` — a standalone script that loads any diffusion model checkpoint, applies structured L1 channel pruning, and saves the pruned model.

Key requirements:
- Accept `--checkpoint`, `--rate`, `--output-dir`, `--model-type` (ddim/edm) arguments
- **Skip Conv2d layers with ≤4 output channels** (protects final RGB projection from being destroyed)
- Use `torch.nn.utils.prune.ln_structured` with `n=1, dim=0`
- Save pruned checkpoint to output-dir
- Print summary: how many layers pruned, how many skipped

**Before writing**: read `src/ddpm_ddim/models/unet.py` to understand how models are loaded/saved, so the pruning script uses the same checkpoint format.

---

## Step 1.4: Create Eval Harness Skeleton

Create `scripts/eval_baselines.py` — unified evaluation script. For NOW, implement ONLY the MiO adapter. WDM and Zhao adapters will be added after their respective code audits.

Structure:
```python
# scripts/eval_baselines.py
# Method registry — adapters register here
# For now only mio_adapter is implemented

# scripts/baselines/__init__.py
# Method registry dict

# scripts/baselines/mio_adapter.py
# Wraps existing t-error + ownership eval logic
```

The harness should:
1. Accept `--method`, `--checkpoint`, `--dataset`, `--output-dir`
2. Load model via the method's adapter
3. Compute t-error on watermark_private split and eval_nonmember split
4. Compute Cohen's d, ratio, three-point criteria
5. Write results to `{output-dir}/results.json`

**Before writing**: read these files to understand existing interfaces:
- `scripts/compare_baselines.py` — how t-error is currently computed
- `scripts/eval_ownership.py` — how ownership evaluation works
- `src/attack_qr/features/t_error.py` — t-error function signature
- `src/attacks/eval/metrics.py` — metric computation

Output JSON schema:
```json
{
  "method": "mio",
  "dataset": "cifar10",
  "checkpoint": "/path/to/ckpt",
  "timestamp": "ISO-8601",
  "seed": 42,
  "native_verification": {},
  "mio_verification": {
    "t_error_member_mean": 0.0,
    "t_error_member_std": 0.0,
    "t_error_nonmember_mean": 0.0,
    "t_error_nonmember_std": 0.0,
    "cohens_d": 0.0,
    "ratio": 0.0,
    "three_point_pass": false
  }
}
```

---

## Step 1.5: Verify with MiO Dry Run

Run the eval harness on our existing Model A to confirm it works:
```bash
python scripts/eval_baselines.py \
    --method mio \
    --checkpoint /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --dataset cifar10 \
    --output-dir experiments/baseline_comparison/results/mio/cifar10/
```

Expected: t-error member mean ≈ 28.6, Cohen's d ≈ -23.9. If these numbers match existing production results, the harness is correct.

---

## Update STATE.md When Done

Fill in all Phase 01 checkboxes. Most critically:
- The CLIP version answer
- The σ GO/NO-GO decision
- Whether the MiO dry run numbers match expected values
