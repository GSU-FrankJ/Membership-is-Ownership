# MiO — Membership is Ownership (ICML 2026)

## What This Project Is
Ownership verification framework for diffusion models. Uses membership inference via quantile regression as a non-invasive watermark. No model modification required.

## Current Task: Table 5 Baseline Comparison
Running controlled experiments comparing MiO against WDM and Zhao et al. on CIFAR-10.

## ⚠ EXECUTION PROTOCOL — READ THIS EVERY SESSION
1. **Read state**: `cat experiments/baseline_comparison/STATE.md`
2. **Read ONLY the current phase prompt**: the one indicated by "Next Phase" in STATE.md
3. **Do NOT load PLAN.md** — it is too large. Each phase prompt is self-contained.
4. **After completing each step**: update STATE.md with results, paths, and issues.
5. **Never guess CLI arguments**: always read actual source code first (`--help`, argparse, README).
6. **Never fabricate paths**: always `ls` or `find` to confirm files exist before using them.
7. **Never run placeholder commands**: any command marked PLACEHOLDER must be replaced after reading the actual codebase.

## Key Paths
```
Data:       /data/short/fjiang4/mia_ddpm_qr/data/cifar10
Splits:     /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/*.json
Model A:    /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt
Model B:    /data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt
Workspace:  experiments/baseline_comparison/
```

## Architecture
DDIM U-Net: base_channels=128, ch_mults=[1,2,2,2], attn_res=[16], T=1000, cosine schedule

## Reusable Code
```
src/attack_qr/features/t_error.py     # T-error computation
src/ddpm_ddim/schedulers/betas.py     # Cosine noise schedule
src/ddpm_ddim/models/unet.py          # UNet model
src/ddpm_ddim/clip_features.py        # CLIP features
src/ddpm_ddim/mmd_loss.py             # MMD loss
src/attacks/eval/metrics.py           # ROC-AUC, TPR@FPR
scripts/finetune_mmd_ddm.py          # MMD fine-tuning
```

## Rules
- **DO NOT retrain Model A or Model B**
- **seed=42** everywhere
- **Update STATE.md after every step**
