# Phase 04: WDM Training (GPU, ~12-24 hours)

## Prerequisites
- Phase 02 complete (check STATE.md — training command must be filled in)
- GPU available
- WDM environment working

## Goal
Train a WDM-watermarked diffusion model on CIFAR-10. This is the most important baseline.

---

## Step 4.1: Pre-flight Checks

```bash
# Confirm training command is documented
cat experiments/baseline_comparison/STATE.md | grep -A5 "WDM training command"

# Confirm data exists
ls /data/short/fjiang4/mia_ddpm_qr/data/cifar10/

# Confirm output directory exists
mkdir -p experiments/baseline_comparison/wdm/cifar10/

# Confirm watermark logo exists (created in Phase 02)
ls experiments/baseline_comparison/wdm/cifar10/watermark_logo.png

# Confirm GPU available
nvidia-smi
```

---

## Step 4.2: Launch Training

Use the EXACT command documented in STATE.md from Phase 02. Do NOT use any placeholder.

```bash
# READ THE ACTUAL COMMAND FROM STATE.md AND RUN IT
# Example structure (the real args come from Phase 02 audit):
cd experiments/baseline_comparison/wdm_repo
# [PASTE ACTUAL COMMAND FROM STATE.md HERE]
```

**Monitor**:
- Check for loss convergence: the task loss (standard diffusion) should decrease normally
- Check watermark-related metrics if WDM logs them
- Training typically takes 12-24h on A100 for CIFAR-10 DDPM

---

## Step 4.3: Verify Checkpoint

After training completes:

```bash
# Confirm checkpoint exists
ls -la experiments/baseline_comparison/wdm/cifar10/

# Quick sanity check: load and verify shape
python -c "
import torch
ckpt = torch.load('[CHECKPOINT_PATH]', map_location='cpu')
if isinstance(ckpt, dict):
    print('Keys:', list(ckpt.keys()))
    if 'model' in ckpt or 'state_dict' in ckpt:
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        print('Params:', len(sd))
else:
    print('Type:', type(ckpt))
"
```

---

## Step 4.4: Quick Smoke Test

Generate a few images to confirm the model works:
```bash
# Use the generation script created in Phase 02
python scripts/baselines/generate_wdm.py \
    --checkpoint [WDM_CKPT_PATH] \
    --num-samples 16 \
    --output-dir experiments/baseline_comparison/wdm/cifar10/smoke_test/ \
    --seed 42
```

Visually check that generated images look like CIFAR-10 images (not garbage).

---

## Update STATE.md

Record:
- Checkpoint path
- Training wall-clock time
- Any issues encountered
- Whether smoke test passed
