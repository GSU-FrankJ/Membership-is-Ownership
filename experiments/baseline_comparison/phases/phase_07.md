# Phase 07: Robustness + FID (GPU, ~18 hours)

## Prerequisites
- Phase 06 complete (all clean evaluation numbers recorded in STATE.md)
- Pruning script from Phase 01 exists: `scripts/attacks/pruning.py`
- MMD fine-tuning script exists: `scripts/finetune_mmd_ddm.py`
- FID generation scripts exist for each method

## Goal
Apply identical attacks to all methods, measure verification survival, and compute FID for all checkpoints.

---

## Part A: FID for Clean Checkpoints (~5.5h)

Generate 50K images from each clean model and compute FID against CIFAR-10 training set.

### Step 7.1: MiO FID (~0.5h)

```bash
python scripts/baselines/generate_mio.py \
    --checkpoint /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --num-samples 50000 \
    --output-dir experiments/baseline_comparison/results/mio/cifar10/generated/ \
    --seed 42 --batch-size 256
```

Then compute FID:
```python
from cleanfid import fid
score = fid.compute_fid(
    "experiments/baseline_comparison/results/mio/cifar10/generated/",
    dataset_name="cifar10", dataset_split="train"
)
print(f"MiO FID: {score:.2f}")
```

### Step 7.2: WDM FID (~4h — 1000-step DDPM is slow)

```bash
python scripts/baselines/generate_wdm.py \
    --checkpoint [WDM_CKPT from STATE.md] \
    --num-samples 50000 \
    --output-dir experiments/baseline_comparison/results/wdm/cifar10/generated/ \
    --seed 42 --batch-size 64  # smaller batch for 1000-step
```

Compute FID same way.

### Step 7.3: Zhao FID (~1h — if applicable)

```bash
python scripts/baselines/generate_zhao.py \
    --checkpoint [ZHAO_EDM_CKPT from STATE.md] \
    --num-samples 50000 \
    --output-dir experiments/baseline_comparison/results/zhao/cifar10/generated/ \
    --seed 42 --batch-size 256
```

Compute FID same way.

Record all FID values in STATE.md Phase 07 → FID section.

---

## Part B: MMD Fine-Tuning Attack (~5h)

Apply our existing MMD fine-tuning protocol (500 iterations) to each baseline checkpoint. Then re-evaluate native verification and FID.

**IMPORTANT**: Confirm CLIP version from STATE.md Phase 01. Use the SAME version for ALL methods.

### Step 7.4: MMD-FT WDM (~2h)

```bash
python scripts/finetune_mmd_ddm.py \
    --config configs/mmd_finetune_cifar10_ddim10.yaml \
    --base-checkpoint [WDM_CKPT from STATE.md] \
    --output-dir experiments/baseline_comparison/robustness/wdm/cifar10/mmd_ft/ \
    --iterations 500 --lr 5e-6
```

**Potential issue**: WDM's model architecture may differ from what `finetune_mmd_ddm.py` expects. If so:
- Check if the script accepts a `--model-type` or architecture config
- If not, you may need to modify the script to accept WDM's model class
- Document any modifications in STATE.md

After MMD-FT, evaluate:
1. WDM native verification (watermark extraction) on the fine-tuned checkpoint
2. MiO t-error on the fine-tuned checkpoint
3. FID (generate 50K from fine-tuned → compute FID)

### Step 7.5: MMD-FT Zhao (~3h, may be skipped)

**Challenge**: Our MMD-FT uses 10-step DDIM sampling inside the loop. EDM uses Heun's ODE sampler.

**Option A** (preferred): Write an MMD-FT variant that uses EDM's sampler. The MMD loss and CLIP feature extraction are sampler-agnostic — only swap the sampling function.

**Option B** (fallback): Skip MMD-FT for Zhao. Document in the paper: "MMD fine-tuning was not applied to Zhao et al. because EDM uses a different ODE sampler (Heun) that is not compatible with our DDIM-based fine-tuning loop. Pruning results are reported for all methods."

If attempting Option A:
```bash
python scripts/finetune_mmd_edm.py \
    --config configs/mmd_finetune_cifar10_edm.yaml \
    --base-checkpoint [ZHAO_EDM_CKPT] \
    --output-dir experiments/baseline_comparison/robustness/zhao/cifar10/mmd_ft/ \
    --iterations 500 --lr 5e-6
```

Record decision and results in STATE.md.

### Step 7.6: MiO MMD-FT (existing result)

MiO's Model B IS the MMD-FT result. Just confirm the existing numbers:
```bash
# Should already have these from production results
cat experiments/production/attack_results/PRODUCTION_RESULTS_SUMMARY.md
```

---

## Part C: Pruning Attack (~1.5h)

### Step 7.7: Prune All Methods at 30%

```bash
# MiO
python scripts/attacks/pruning.py \
    --checkpoint /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --rate 0.30 \
    --output-dir experiments/baseline_comparison/robustness/mio/cifar10/prune_30/

# WDM
python scripts/attacks/pruning.py \
    --checkpoint [WDM_CKPT] \
    --rate 0.30 \
    --output-dir experiments/baseline_comparison/robustness/wdm/cifar10/prune_30/

# Zhao (if applicable)
python scripts/attacks/pruning.py \
    --checkpoint [ZHAO_EDM_CKPT] \
    --rate 0.30 \
    --output-dir experiments/baseline_comparison/robustness/zhao/cifar10/prune_30/
```

### Step 7.8: Evaluate Pruned Checkpoints

For each pruned checkpoint:
1. Run native verification → pass/fail
2. Run MiO t-error → three-point pass/fail
3. Generate 50K images → compute FID

This is the same eval pipeline as Phase 06 but on pruned checkpoints. Use the eval harness:

```bash
for METHOD in mio wdm zhao; do
    python scripts/eval_baselines.py \
        --method ${METHOD} \
        --checkpoint experiments/baseline_comparison/robustness/${METHOD}/cifar10/prune_30/model.pt \
        --dataset cifar10 \
        --output-dir experiments/baseline_comparison/robustness/${METHOD}/cifar10/prune_30/eval/
done
```

---

## Part D: FID for Attacked Checkpoints (~6h)

Generate 50K images from each attacked checkpoint and compute FID:
- MMD-FT WDM → FID
- MMD-FT Zhao → FID (if applicable)
- Prune 30% MiO → FID
- Prune 30% WDM → FID
- Prune 30% Zhao → FID (if applicable)

Use the same generation scripts but point to attacked checkpoints.

---

## Update STATE.md

Fill ALL Phase 07 checkboxes:
- All FID values (clean + attacked)
- MMD-FT: native pass/fail for each method
- Pruning: native pass/fail for each method
- Any skipped experiments with reasons
