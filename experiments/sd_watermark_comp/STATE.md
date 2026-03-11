# STATE — SD Watermark Comparison Experiment

> Last updated: 2026-03-11
> Current phase: **04 — Not started**
> Overall progress: ███░░░░░ 3/8

---

## Phase Tracker

| Phase | Status    | Completed | Notes |
|-------|-----------|-----------|-------|
| 01    | ✅ DONE   | 2026-03-11 | Environment & dependencies |
| 02    | ✅ DONE   | 2026-03-11 | Download models & COCO2014 |
| 03    | ✅ DONE   | 2026-03-11 | Member/non-member splits |
| 04    | ⬜ TODO   | —         | LoRA fine-tuning |
| 05    | ⬜ TODO   | —         | SleeperMark checkpoint |
| 06    | ⬜ TODO   | —         | MiO inference — LoRA |
| 07    | ⬜ TODO   | —         | MiO inference — SleeperMark |
| 08    | ⬜ TODO   | —         | Metrics + tables + figures |

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
- LoRA checkpoint md5: —
- SleeperMark UNet md5: —

## Dataset
- Member count: 10,000 (COCO2014 train)
- Non-member count: 10,000 (COCO2014 val)
- Split file: `data/splits/split_seed42.json`
- Split md5: 9be5fa04dea9a4b6d2f845d5f6f7a3d1
- LoRA train dir: `data/lora_train_dir/` (10,000 symlinks + metadata.jsonl)

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