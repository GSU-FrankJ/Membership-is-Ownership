# STATE — SD Watermark Comparison Experiment

> Last updated: 2026-03-11
> Current phase: **01 — Not started**
> Overall progress: ░░░░░░░░ 0/8

---

## Phase Tracker

| Phase | Status    | Completed | Notes |
|-------|-----------|-----------|-------|
| 01    | ⬜ TODO   | —         | Environment & dependencies |
| 02    | ⬜ TODO   | —         | Download models & COCO2014 |
| 03    | ⬜ TODO   | —         | Member/non-member splits |
| 04    | ⬜ TODO   | —         | LoRA fine-tuning |
| 05    | ⬜ TODO   | —         | SleeperMark checkpoint |
| 06    | ⬜ TODO   | —         | MiO inference — LoRA |
| 07    | ⬜ TODO   | —         | MiO inference — SleeperMark |
| 08    | ⬜ TODO   | —         | Metrics + tables + figures |

---

## Environment
- Conda env: `mio-sd`
- Python: —
- PyTorch: —
- Diffusers: —
- CUDA: —
- GPU confirmed: 4× V100 32GB (unverified)

## Model Checksums
- SD v1.4 UNet md5: —
- LoRA checkpoint md5: —
- SleeperMark UNet md5: —

## Dataset
- Member count: — (target: 10,000)
- Non-member count: — (target: 10,000)
- Split file: `data/splits/split_seed42.json`
- Split md5: —

## LoRA Training Log
- Start time: —
- End time: —
- Final train loss: —
- Best checkpoint step: —
- Checkpoint path: —
- Weight file size: —

## MiO Results

### SD v1.4 + LoRA (Ours)
- TPR@1%FPR: —
- TPR@0.1%FPR: —
- AUC: —
- Score CSV: `experiments/sd_watermark_comp/scores/mio_lora_scores.csv`

### SleeperMark
- TPR@1%FPR: —
- TPR@0.1%FPR: —
- AUC: —
- Score CSV: `experiments/sd_watermark_comp/scores/mio_sleepermark_scores.csv`
- SleeperMark native Bit Acc: —
- SleeperMark native TPR@FPR: —

## Qualitative Figure
- Prompts used: —
- Grid dimensions: —
- LPIPS scores: —
- Output path: `experiments/sd_watermark_comp/figures/qualitative_grid.pdf`

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