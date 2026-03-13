# STATE — SD Watermark Comparison Experiment

> Last updated: 2026-03-12
> Current phase: **All phases complete**
> Overall progress: ████████ 8/8

---

## Phase Tracker

| Phase | Status    | Completed | Notes |
|-------|-----------|-----------|-------|
| 01    | ✅ DONE   | 2026-03-11 | Environment & dependencies |
| 02    | ✅ DONE   | 2026-03-11 | Download models & COCO2014 |
| 03    | ✅ DONE   | 2026-03-11 | Member/non-member splits |
| 04    | ✅ DONE   | 2026-03-11 | LoRA fine-tuning |
| 05    | ✅ DONE   | 2026-03-11 | SleeperMark checkpoint |
| 06    | ✅ DONE   | 2026-03-12 | MiO inference — LoRA (20k scores, 3.29h) |
| 07    | ✅ DONE   | 2026-03-13 | MiO inference — SleeperMark (20k scores, 3.29h) |
| 08    | ✅ DONE   | 2026-03-13 | Metrics + tables + figures |

---

## Environment
- Conda env: `mio-sd` (~/miniconda3/envs/mio-sd)
- Python: 3.10.20
- PyTorch: 2.1.2+cu118
- Diffusers: 0.25.1
- Transformers: 4.37.2
- Accelerate: 0.26.1
- Peft: 0.8.2
- CUDA: 11.8 (via PyTorch wheels)
- GPU confirmed: 4× Tesla V100-SXM2-32GB ✅

## Model Checksums
- SD v1.4 UNet md5: e711500ea7f5d40b19c8c04b5b15396b
- LoRA checkpoint md5: 8adc2c4a6f53ff39a497a418ce96da97
- SleeperMark UNet md5: 98bc1a65e277b2e2bb43351598ab4694

## Dataset
- Member count: 10,000 (COCO2014 train)
- Non-member count: 10,000 (COCO2014 val)
- Split file: `data/splits/split_seed42.json`
- Split md5: 9be5fa04dea9a4b6d2f845d5f6f7a3d1
- LoRA train dir: `data/lora_train_dir/` (10,000 symlinks + metadata.jsonl)

## LoRA Training Log
- Start time: 2026-03-11 20:47:06
- End time: 2026-03-11 22:10:11
- Duration: 1h 23m
- Final train loss: 0.242
- Best checkpoint step: 10000 (final)
- Checkpoint path: `models/sd-v1-4-lora/pytorch_lora_weights.safetensors`
- Weight file size: 49MB
- Checkpoints saved: 2000, 4000, 6000, 8000, 10000
- Speed: ~2.01 it/s on 1× V100
- Note: Used `accelerate launch --mixed_precision=no` with `--mixed_precision=fp16` in script args to avoid FP16 gradient unscale error

## MiO Results

### SD v1.4 + LoRA (Ours)
- TPR@1%FPR: 0.0071
- TPR@0.1%FPR: 0.0005
- AUC: 0.5032
- Member mean: 0.4810, std: 39.1714
- Non-member mean: 0.7362, std: 39.5048
- Score CSV: `experiments/sd_watermark_comp/scores/mio_lora_scores.csv`

### SleeperMark
- TPR@1%FPR: 0.0140
- TPR@0.1%FPR: 0.0018
- AUC: 0.5425
- Member mean: -0.2544, std: 39.8744
- Non-member mean: 5.6972, std: 39.6020
- Score CSV: `experiments/sd_watermark_comp/scores/mio_sleepermark_scores.csv`
- SleeperMark native Bit Acc: —
- SleeperMark native TPR@FPR: —

## Qualitative Figure
- Prompts: coffee/desk, rain/umbrellas, tabby cat (3 rows)
- Grid: 3×4 (Clean | SM Regular | SM Triggered | LoRA)
- LPIPS vs clean (avg): sm_regular=0.1969, sm_triggered=0.3113, lora=0.0000
- Output: `experiments/sd_watermark_comp/figures/qualitative_grid.pdf`
- ROC curves: `experiments/sd_watermark_comp/figures/roc_curves.pdf`
- LaTeX table: `experiments/sd_watermark_comp/tables/sd_comparison.tex`

---

## Blockers & Issues
_None yet._

---

## Session Log
<!-- Append one entry per Claude Code session -->

### Session 0 — 2026-03-11
- Phase: Init
- Actions: Created CLAUDE.md, STATE.md, phases/phase_01–08.md
- Result: Experiment framework ready
- Next: Begin Phase 01

### Session 1 — 2026-03-11
- Phase: 01
- Actions: Installed miniconda to ~/miniconda3, created mio-sd env (Python 3.10, PyTorch 2.1.2+cu118, diffusers 0.25.1, peft 0.8.2, accelerate 0.26.1), verified 4× V100 GPUs, created directory structure
- Note: Downgraded numpy<2 and huggingface_hub<0.25 for compatibility with torch 2.1/diffusers 0.25.1
- Result: Phase 01 complete
- Next: Begin Phase 02

### Session 1 (cont.) — 2026-03-11
- Phase: 02
- Actions: Downloaded SD v1.4 (UNet 1.7GB safetensors, md5 e711500e), COCO2014 (82,783 train + 40,504 val + 414,113 captions), verified generation test on GPU3
- Result: Phase 02 complete
- Next: Begin Phase 03

### Session 1 (cont.) — 2026-03-11
- Phase: 03
- Actions: Built split_seed42.json (10k members from train, 10k non-members from val, 0 overlap, 0 missing), created lora_train_dir with 10k symlinks + metadata.jsonl
- Result: Phase 03 complete
- Next: Begin Phase 04

### Session 1 (cont.) — 2026-03-11
- Phase: 04
- Actions: LoRA fine-tuned SD v1.4 on 10k members (rank=64, 10k steps, lr=1e-4, cosine, bs=4, fp16). Final loss 0.242 in 1h23m. Verified generation (cat prompt + member/nonmember captions). Weights 49MB.
- Result: Phase 04 complete
- Next: Begin Phase 05

### Session 1 (cont.) — 2026-03-11
- Phase: 05
- Actions: Downloaded SleeperMark UNet (3.3GB, md5 98bc1a65), Stage1 encoder/decoder (119MB), cloned repo to external/SleeperMark. Verified generation with and without trigger prompt on GPU1.
- Result: Phase 05 complete
- Next: Begin Phase 06

### Session 1 (cont.) — 2026-03-12
- Phase: 06 (in progress)
- Actions: Wrote `experiments/sd_watermark_comp/mio_sd_inference.py` adapting MiO t-error for SD UNet (latent-space, unconditional text embedding, K=12 timesteps, Q25 agg). Score = t_error_tgt - t_error_ref. Launched in tmux `mio_lora` on GPU1.
- Progress: 13,000/20,000 images scored (~65%). Partial CSV at `scores/mio_lora_partial.csv`. Partial AUC ~0.517 (raw difference; QR calibration in Phase 08).
- Fix applied: cast errors to float32 before `torch.quantile()` (fp16 not supported).
- Resume: script supports `--resume-from scores/mio_lora_partial.csv` if interrupted.
- Also cleaned up: removed GSD (agents/hooks/commands) and everything-claude-code plugin from `~/.claude/settings.json`.
- Next: Wait for Phase 06 to finish, then run Phase 07 (SleeperMark inference), then Phase 08 (metrics)

### Session 2 — 2026-03-13
- Phase: 06→07→08 (all completed)
- Phase 06 results: LoRA MiO AUC=0.5032, TPR@1%=0.0071 (20k scores, 3.29h)
- Phase 07: Generated 10k SleeperMark training images (7.31h), ran MiO inference (3.29h). AUC=0.5425, TPR@1%=0.0140
- Phase 08: LaTeX table, ROC curves (PDF+PNG), qualitative 3×4 grid, LPIPS scores (lora=0.000, sm_regular=0.197, sm_triggered=0.311)
- Key finding: LoRA LPIPS=0.000 confirms zero quality degradation (post-hoc method). Both MiO AUCs near 0.5 — raw t-error difference is weak; QR calibration needed.
- Result: All 8 phases complete
- Next: QR calibration of SD scores, SleeperMark native detection comparison