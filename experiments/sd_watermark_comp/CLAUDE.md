# SD Watermark Comparison — Claude Code Instructions

> This file applies only to the `experiments/sd_watermark_comp/` experiment.
> The root `.claude/CLAUDE.md` contains global instructions; this file adds experiment-specific rules.

## Experiment Goal
Add SD v1.4 baseline experiments to the MiO paper (Membership Inference for Ownership).
Core thesis: private fine-tuning data IS the watermark — MIA detects it without proactive embedding.

## Model Architecture
```
Reference:  SD v1.4               (CompVis/stable-diffusion-v1-4)
Target:     SD v1.4 + LoRA        (fine-tuned on private COCO2014 subset)
Baseline:   SleeperMark UNet      (CVPR 2025, released checkpoint)
```

## Hardware
- 4× Tesla V100 32GB, fp16 mixed precision throughout
- GPU assignment: GPU0=MiO-LoRA, GPU1=MiO-SleeperMark, GPU2=data generation, GPU3=figures

## Path Conventions (relative to project root)
```
models/sd-v1-4/                      # base SD v1.4 weights
models/sd-v1-4-lora/                 # LoRA fine-tuned checkpoint
models/sleepermark-unet/             # SleeperMark watermarked UNet
data/coco2014/                       # COCO2014 images + annotations
data/splits/                         # member/non-member JSON
data/lora_train_dir/                 # symlink dir for LoRA training
data/sleepermark_train_images/       # reconstructed SleeperMark training data
experiments/sd_watermark_comp/       # ← this experiment root
  STATE.md                           # cross-session state
  phases/phase_XX.md                 # phase instructions
  scores/                            # MiO score CSVs
  figures/                           # generated charts and grids
  logs/                              # training/inference logs
  tables/                            # LaTeX tables
configs/                             # training & inference configs
external/SleeperMark/                # SleeperMark repo clone
```

## Workflow Rules
1. **Start every session** by reading `experiments/sd_watermark_comp/STATE.md`.
2. **Load only the current phase**: `phases/phase_XX.md`.
3. **After completing work**: update STATE.md with results, checksums, and next steps.
4. **Never skip verification**: every phase ends with a sanity check; pass all before marking done.
5. **Checkpoint everything**: write intermediate results to disk; sessions may disconnect.
6. **Reproducibility**: use seed=42 everywhere; log all hyperparameters.

## Fixed Parameters (do not change without discussion)
- LoRA rank=64, target=UNet cross-attention (q,k,v,out), alpha=64
- Training: 10k steps, batch=4, lr=1e-4, cosine schedule, warmup=500
- Members: 10,000 COCO2014 train images (seed=42 shuffle)
- Non-members: 10,000 COCO2014 val images
- MiO timesteps: match existing paper settings
- Metrics: TPR@1%FPR (primary), TPR@0.1%FPR, AUC

## Phase Overview
| Phase | Task                           | Est. Time | GPU  |
|-------|--------------------------------|-----------|------|
| 01    | Environment & dependencies     | 30 min    | —    |
| 02    | Download models & COCO2014     | 1-2 hr    | —    |
| 03    | Member/non-member split        | 15 min    | —    |
| 04    | LoRA fine-tune SD v1.4         | 2-3 hr    | GPU0 |
| 05    | Download & verify SleeperMark  | 30 min    | —    |
| 06    | MiO inference — LoRA model     | 6-14 hr   | GPU0 |
| 07    | MiO inference — SleeperMark    | 6-14 hr   | GPU1 |
| 08    | Metrics, tables & qual. figure | 1-2 hr    | GPU3 |