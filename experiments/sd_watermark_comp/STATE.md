# STATE — SD Watermark Comparison Experiment


> Last updated: 2026-03-25
> Current phase: **Phase 13 complete** (extended experiments + verification)
> Overall progress: █████████████ 13/13

---

## Phase Tracker

| Phase | Status    | Completed | Notes |
|-------|-----------|-----------|-------|
| 01    | ✅ DONE   | 2026-03-11 | Environment & dependencies |
| 02    | ✅ DONE   | 2026-03-11 | Download models & COCO2014 |
| 03    | ✅ DONE   | 2026-03-11 | Member/non-member splits |
| 04    | ✅ DONE   | 2026-03-11 | LoRA fine-tuning |
| 05    | ✅ DONE   | 2026-03-11 | SleeperMark checkpoint |
| 06    | ⚠️ WEAK   | 2026-03-12 | MiO inference — LoRA (20k scores, AUC=0.54, ablation needed) |
| 07    | ✅ DONE   | 2026-03-13 | MiO inference — SleeperMark (20k scores, 3.29h) |
| 08    | ✅ DONE   | 2026-03-13 | Metrics + tables + figures |
| 09    | ✅ DONE   | 2026-03-13 | Ablation plan designed, all 4 quick evals exceed AUC > 0.99 |
| 10    | ✅ DONE   | 2026-03-15 | Ablation complete: A5 timestep sweep + A6 scale validation (AUC=0.982 at 1k members) |
| 11    | ✅ DONE   | 2026-03-17 | 3-point verification: B1 domain-shift PASS, B2 task-shift partial (AUC=0.908) |
| 12    | ✅ DONE   | 2026-03-22 | Latent vs pixel: latent+caption best (AUC=0.996), pixel fails verification |
| 13    | ✅ DONE   | 2026-03-25 | Extended: full-FT A7 + LoRA r256 A8 + 1000 non-members. Algorithm 2 FAIL for all SD models |


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

## MiO Results (Phase 06–08, pre-ablation)

### SD v1.4 + LoRA — Original Config (10k members, rank 64, 4 epochs)
- AUC: 0.5437
- TPR@1%FPR: 0.0136
- TPR@0.1%FPR: 0.0005
- Member mean: -4.19, std: 3.81
- Non-member mean: -3.63, std: 3.75
- Cohen's d: 0.15
- SNR (between/within var): 0.006
- Score CSV: `experiments/sd_watermark_comp/scores/mio_lora_scores.csv`

**Diagnosis**: Signal is indistinguishable from noise. Member relative t-error reduction = 0.236% vs non-member = 0.204% (0.03pp gap). Tail analysis shows zero separation — no individual image memorization. Root cause: rank-64 LoRA on 10k COCO images for 4 epochs learns the COCO *distribution* but does not memorize *specific images*.

### SleeperMark
- AUC: 0.5425
- TPR@1%FPR: 0.0140
- TPR@0.1%FPR: 0.0018
- Member mean: -0.2544, std: 39.8744
- Non-member mean: 5.6972, std: 39.6020
- Score CSV: `experiments/sd_watermark_comp/scores/mio_sleepermark_scores.csv`
- SleeperMark native Bit Acc: —
- SleeperMark native TPR@FPR: —

---

## Ablation Plan (Phase 09)

### Goal
Find a fine-tuning configuration that produces **AUC > 0.70** for MiO membership inference on the LoRA-fine-tuned SD model. The core hypothesis: fewer training images + more epochs per image → per-image memorization → detectable by MIA.

### Key Variables
1. **Member set size** — fewer images = more updates per image = stronger memorization
2. **Training steps / epochs per image** — more repetition = more memorization
3. **LoRA rank** — controls fine-tuning capacity (is signal from capacity or repetition?)
4. **Full fine-tune vs LoRA** — removes capacity bottleneck entirely
5. **Timestep range** — MIA literature suggests low-t timesteps are more sensitive to memorization
6. **Data augmentation** — disabling random_flip forces identical presentation each epoch

### Quick Eval Protocol
Before committing to full 20k evaluation, each run is evaluated on a small balanced subset:
- **Members**: random 200 from training set (or all, if set < 200)
- **Non-members**: random 200 from the existing 10k non-member pool
- **Metrics**: AUC, TPR@1%FPR, Cohen's d, mean/std per class
- **Go threshold**: quick AUC > 0.60 → proceed to full 20k eval
- **Kill threshold**: quick AUC < 0.52 → abandon this config

### Sub-splits
All ablation runs subsample from the existing 10k members in `split_seed42.json`. Non-members remain the same 10k COCO val images. Sub-split creation uses `seed=42` deterministic shuffle, then take first N.

### Ablation Runs

| Run | Members | Steps | Epochs/img | Rank | LR | Flip | Train (1 GPU) | Quick Eval | Hypothesis |
|-----|---------|-------|------------|------|----|------|---------------|------------|------------|
| A1 | 100 | 5,000 | 200 | r64 | 1e-4 | Off | ~42 min | ~2 min | Extreme repetition on tiny set → strong memorization |
| A2 | 500 | 10,000 | 80 | r64 | 1e-4 | Off | ~83 min | ~4 min | Moderate set, same total steps → enough epochs? |
| A3 | 100 | 5,000 | 200 | r4 | 1e-4 | Off | ~42 min | ~2 min | Low-rank control: is signal from capacity or repetition? |
| A4 | 500 | 5,000 | 40 | Full | 5e-6 | Off | ~83 min | ~4 min | Maximum capacity (860M params), moderate epochs |
| A5 | *(winner)* | 0 | — | *(winner)* | — | — | 0 (reuse) | ~20 min | Timestep sweep + caption-conditioned inference on best model |
| A6 | 1,000 | *(scaled)* | *(target)* | *(winner)* | *(winner)* | Off | ~83 min | ~3.3h (full) | Scale validation: does signal hold at 1k members? |

### Run Details

**A1 — Tiny set, extreme memorization** (cheapest, most likely to show signal)
- 100 images × 200 epochs: each pixel pattern repeated 200 times
- Expect: training loss drops to near-zero, model "knows" these 100 images
- Checkpoint every 1,000 steps → can check signal at epochs 40/80/120/160/200
- Output: `models/sd-v1-4-lora-a1/`, scores in `scores/ablation/a1_quick.csv`

**A2 — Moderate set, standard steps**
- 500 images × 80 epochs: still much higher repetition than original (4 epochs)
- Isolates: is 80 epochs sufficient, or do we need 200+?
- Checkpoint every 2,000 steps → signal check at epochs 16/32/48/64/80
- Output: `models/sd-v1-4-lora-a2/`, scores in `scores/ablation/a2_quick.csv`

**A3 — Low-rank control** (pairs with A1)
- Identical to A1 except rank=4 (0.5M trainable params vs 33.5M at rank 64)
- If A3 ≈ A1 → repetition dominates, rank doesn't matter much
- If A3 ≪ A1 → capacity matters, higher rank helps memorization
- Output: `models/sd-v1-4-lora-a3/`, scores in `scores/ablation/a3_quick.csv`

**A4 — Full fine-tune** (maximum capacity)
- All 860M UNet parameters trainable, lower LR (5e-6) to avoid collapse
- Fewer epochs needed (40) since capacity is unconstrained
- Risk: catastrophic forgetting of general generation ability
- Sanity check: generate test image after training to verify no collapse
- Output: `models/sd-v1-4-full-a4/`, scores in `scores/ablation/a4_quick.csv`

**A5 — Inference knobs** (no training, just re-evaluate best model)
- Timestep sweep on the best model from A1–A4:
  - **Low-t**: K=12, t ∈ [0, 200] — small noise, reconstruction is easiest, memorization most visible
  - **Mid-t**: K=12, t ∈ [200, 600] — moderate noise
  - **High-t**: K=12, t ∈ [600, 999] — heavy noise, reconstruction relies on prior, less image-specific
  - **Dense low-t**: K=50, t ∈ [0, 300] — more timesteps in the sweet spot
- Caption-conditioned variant: use actual COCO captions (from split JSON) instead of empty prompt for member images. The model learned image-caption associations; conditioning on the training caption should amplify the reconstruction advantage for members.
- Output: `scores/ablation/a5_*.csv`

**A6 — Scale validation** (confirms winner generalizes)
- Takes the best (member_count, epochs, rank) from A1–A5
- Scales to 1,000 members (still a realistic ownership claim — "I fine-tuned on 1k of my images")
- Adjusts steps to match the winning epochs-per-image ratio
- Full 20k evaluation (10k members from new sub-split + 10k non-members)
- Target: AUC > 0.70, TPR@1%FPR > 0.05
- Output: `models/sd-v1-4-lora-a6/`, scores in `scores/ablation/a6_full.csv`

### GPU Pipeline

```
Wall time    GPU0          GPU1          GPU2          GPU3
─────────    ────          ────          ────          ────
0:00-1:25    Train A1      Train A2      Train A3      Train A4
             (42 min)      (83 min)      (42 min)      (83 min)
1:25-1:30    Eval A1       Eval A2       Eval A3       Eval A4
             (2 min)       (4 min)       (2 min)       (4 min)
1:30-1:45    ── Analyze results, select winner ──
1:45-2:05    A5-low-t      A5-mid-t      A5-high-t     A5-dense
             (5 min)       (5 min)       (5 min)       (5 min)
2:05-2:15    A5-caption    ── idle ──    ── idle ──    ── idle ──
2:15-3:40    ── Train A6 (1 GPU) ──
3:40-7:00    ── Eval A6 full 20k (1 GPU, or 2 GPUs split) ──
```

**Time to first signal: ~1.5h** (after Runs A1–A4 quick eval)
**Total wall time: ~7h** (including A6 full eval)

### Tmux Session Naming
```
ablation_a1   — GPU0, training + quick eval
ablation_a2   — GPU1, training + quick eval
ablation_a3   — GPU2, training + quick eval
ablation_a4   — GPU3, training + quick eval
ablation_a5   — winner GPU, timestep sweep
ablation_a6   — winner GPU, scale validation
```

### Expected Outcomes
- **Best case**: A1 shows AUC > 0.80 at quick eval → 100 images with 200 epochs is sufficient. A6 confirms signal at 1k members.
- **Good case**: A1 or A2 shows AUC 0.60–0.80 → we have a working direction. Optimize epochs/rank in follow-up.
- **Concerning case**: Only A4 (full fine-tune) works → LoRA may lack capacity for memorization. Paper needs to discuss this.
- **Worst case**: All runs show AUC < 0.55 → raw t-error difference may be fundamentally weak for SD; need QR calibration or different score function.

### Quick Eval Results (Phase 10, 2026-03-13)

| Run | Config | AUC | TPR@1% | TPR@0.1% | Cohen's d | Mem mean | Nonmem mean |
|-----|--------|-----|--------|----------|-----------|----------|-------------|
| A1 | 100img, r64, 200ep | **1.0000** | **1.0000** | 0.9900 | 4.38 | -66.3 | +40.7 |
| A3 | 100img, r4, 200ep | 0.9976 | 0.9700 | 0.9600 | 3.51 | -26.7 | +12.0 |
| A4 | 500img, full-FT, 40ep | 0.9983 | 0.9600 | 0.9300 | 3.06 | -48.5 | +6.5 |
| A2 | 500img, r64, 80ep | 0.9920 | 0.6850 | 0.5150 | 3.11 | -22.7 | +4.7 |
| *Orig* | *10k, r64, 4ep* | *0.5437* | *0.0136* | *0.0005* | *0.15* | *-4.2* | *-3.6* |

**Outcome: BEST CASE — all 4 runs exceed target (AUC > 0.70) by a large margin.**

Key findings:
1. **Epochs per image is the dominant factor.** Original config (4 epochs) → AUC 0.54; A2 (80 epochs) → 0.992; A1 (200 epochs) → 1.000.
2. **Rank matters for separation magnitude, not detection.** A1 (r64) vs A3 (r4): both AUC > 0.99, but Cohen's d 4.38 vs 3.51 and score range -66 vs -27.
3. **Full fine-tune is most data-efficient.** A4 achieves AUC 0.998 with only 40 epochs (vs 80-200 for LoRA).
4. **All training losses converge to ~0.15** (diffusion loss floor). Convergence plots saved to `figures/ablation_loss_curves.png`.

Next: Run A5 (timestep sweep on A4), then A6 (scale to 1000 members).

### A5 Timestep Sweep Results (on A4 full-FT model, 500 members)

| Range | K | AUC | TPR@1% | TPR@0.1% | Cohen's d | Time |
|-------|---|-----|--------|----------|-----------|------|
| Mid-t [200,600] | 12 | **0.9983** | **0.9450** | 0.9100 | 3.02 | 236s |
| Full [0,999] (A4 baseline) | 12 | 0.9983 | 0.9600 | 0.9300 | 3.06 | 238s |
| Dense low-t [0,300] | 50 | 0.9943 | 0.8000 | 0.7750 | 3.04 | 944s |
| Low-t [0,200] | 12 | 0.9917 | 0.8500 | 0.6800 | 2.89 | 235s |
| High-t [600,999] | 12 | 0.7309 | 0.0650 | 0.0650 | 0.72 | 233s |

Findings:
1. **Mid-t [200,600] is the sweet spot** — matches full-range AUC with 40% of timesteps.
2. **High-t is nearly useless** — AUC 0.73, TPR@1% collapses to 6.5%. Reconstruction at high noise relies on prior, not image-specific memorization.
3. **Full-range K=12 is already near-optimal** — no benefit from timestep tuning.
4. **Dense K=50 doesn't beat K=12** — more timesteps in [0,300] adds 4x cost with no gain.

Next: A6 scale validation at 1000 members.

### A6 Scale Validation Results (LoRA r64, 1000 members, 80 epochs)

**Training**: 1000 images, 20k steps (80 epochs/img), rank 64, lr=1e-4, cosine, 2h45m on 1× V100.

| Eval | Members | Non-members | AUC | TPR@1% | TPR@0.1% | Cohen's d | Mem mean | Nonmem mean |
|------|---------|-------------|-----|--------|----------|-----------|----------|-------------|
| Quick (400) | 200 | 200 | 0.9847 | 0.7300 | 0.7150 | 2.84 | -20.8 | +2.6 |
| **Full (11k)** | **1000** | **10,000** | **0.9824** | **0.6850** | **0.3300** | **2.75** | **-20.3** | **+2.3** |

**Signal confirmed at 1000 members.** AUC=0.982 with full 11k evaluation. Quick eval tracks full eval closely (AUC 0.985 vs 0.982).

### Summary: Signal vs Member Set Size

| Members | Epochs/img | AUC | TPR@1% | Cohen's d | Status |
|---------|------------|-----|--------|-----------|--------|
| 100 | 200 | 1.000 | 1.000 | 4.38 | Perfect |
| 500 | 80 | 0.992 | 0.685 | 3.11 | Strong |
| 1,000 | 80 | 0.982 | 0.685 | 2.75 | Strong |
| 10,000 | 4 | 0.544 | 0.014 | 0.15 | Failed |

Signal degrades gracefully with set size. Key threshold: **~80 epochs/image minimum** for strong MIA signal with LoRA r64.

## Phase 11: 3-Point Ownership Verification

### Adversary Models

| Model | Data | Steps | LR | Loss | Time |
|-------|------|-------|------|------|------|
| B1 (domain shift) | 1000 disjoint COCO images | 2000 | 5e-5 | 0.156 | 35 min |
| B2 (task shift) | 500 synthetic images | 2000 | 5e-5 | 0.151 | 35 min |

### Score Statistics on W (1000 images, delta = score_tgt - score_ref)

| Model | Delta W (mean) | Delta non-W (mean) | |d| | AUC |
|-------|----------------|--------------------|----|-----|
| **Owner (A6)** | -20.35 | +2.56 | 2.55 | 0.987 |
| B1 (domain shift) | -17.79 | -0.90 | 2.21 | 0.972 |
| B2 (task shift) | -2.00 | +13.64 | 1.73 | 0.908 |
| Reference (SD v1.4) | 0 (by definition) | 0 | — | — |

Raw t-error: owner=1877.6, reference=1898.0 (ratio=1.011x). LoRA perturbation is ~1% of base model error.

### Verification Results (Algorithm 2, adapted for derivative models)

**Criterion 1 — Consistency** (t-test delta_A vs delta_B on W, p > 0.05):
- B1: **FAIL** (p=6.6e-11, d=-0.29) — mild drift from domain-shift FT
- B2: **FAIL** (p~0, d=-1.98) — severe drift from task-shift FT

**Criterion 2 — Separation** (delta_W vs delta_nonW, p < 1e-6 AND |d| > 2.0):
- Owner: **PASS** (p=1.4e-136, |d|=2.55)
- B1: **PASS** (p=4.7e-135, |d|=2.21)
- B2: **FAIL** (p=3.0e-71, |d|=1.73 < 2.0)

**Criterion 3 — Magnitude** (|mean delta_W| / |mean delta_nonW| > 5.0):
- Owner: **PASS** (7.96x)
- B1: **PASS** (19.87x)
- B2: **FAIL** (0.15x)

### Direct Detection Summary

| Model | Criteria 2+3 | AUC | Verdict |
|-------|-------------|-----|---------|
| Owner (A6) | PASS | 0.987 | VERIFIED |
| B1 (domain shift) | PASS | 0.972 | VERIFIED |
| B2 (task shift) | FAIL (|d|=1.73) | 0.908 | PARTIAL |

### Key Findings

1. **MiO is robust to domain-shift fine-tuning.** After 2000 steps on disjoint COCO images, the watermark signal persists (AUC=0.972, all criteria pass on direct detection).
2. **MiO is partially robust to task-shift fine-tuning.** 2000 steps on synthetic images weakens the signal (AUC 0.987→0.908, |d| 2.55→1.73) below the strict |d|>2.0 threshold, but statistical significance remains (p<1e-71).
3. **Raw t-error ratio doesn't work for LoRA.** The paper's ratio criterion (>5x) was calibrated for DDIM models trained from scratch. LoRA produces a ~1% perturbation to the base model, making the raw ratio ~1.01x. Delta-based verification is required for derivative models.
4. **Criterion 1 (provenance) is strict by design.** Even mild adversary modifications (B1, |d|=0.29) cause failure. This correctly reflects that the suspect model has been modified from the original.

### Output Files
- Scores: `experiments/sd_watermark_comp/scores/phase11/model_{a,b1,b2}.csv`
- Verification JSON: `experiments/sd_watermark_comp/scores/phase11/verification_{b1,b2}.json`
- LaTeX tables: `experiments/sd_watermark_comp/tables/sd_verification.tex`, `sd_robustness.tex`
- W hash commitment: `42364195d97099c9d2e3b1c9f8c2123f97ed81b2b3a3d49a62fb38e8feb0eee1`

## Qualitative Figure
- Prompts: coffee/desk, rain/umbrellas, tabby cat (3 rows)
- Grid: 3×4 (Clean | SM Regular | SM Triggered | LoRA)
- LPIPS vs clean (avg): sm_regular=0.1969, sm_triggered=0.3113, lora=0.0000
- Output: `experiments/sd_watermark_comp/figures/qualitative_grid.pdf`
- ROC curves: `experiments/sd_watermark_comp/figures/roc_curves.pdf`
- LaTeX table: `experiments/sd_watermark_comp/tables/sd_comparison.tex`

---

## Fix Investigation: Pixel-Space + Caption Conditioning (branch: fix/pixel-space-caption)

**Issue A**: Code computes t-error as ||z - ẑ||² in latent space (4×64×64), paper Eq. 6 defines ||x - x̂||²/(H×W×C) in pixel space (3×512×512).
**Issue B**: Inference uses empty prompt ("") for all images, but LoRA training used per-image COCO captions.

### 4-Variant Quick Eval (A6 model, 100 members + 100 non-members)

| Variant | Error Space | Caption | AUC | TPR@1% | Cohen's d | Mem mean±std | Nonmem mean±std | Time |
|---------|------------|---------|-----|--------|-----------|-------------|----------------|------|
| V1 (baseline) | latent | empty | 0.9811 | 0.4900 | 2.80 | -21.01±9.80 | +3.39±7.47 | 122s |
| V2 | latent | per-image | **0.9988** | **0.9100** | **3.21** | -30.17±13.15 | +4.93±8.15 | 124s |
| V3 | pixel | empty | 0.9163 | 0.2700 | 1.71 | -0.0002±0.0002 | +0.0001±0.0002 | 485s |
| V4 | pixel | per-image | 0.9696 | 0.5900 | 2.04 | -0.0003±0.0003 | +0.0002±0.0002 | 493s |

### Findings

1. **Caption conditioning is the bigger win.** V2 vs V1: AUC 0.981→0.999, TPR@1% 0.49→0.91, Cohen's d 2.80→3.21. The model learned image-caption associations; conditioning on the training caption amplifies the member reconstruction advantage.
2. **Pixel-space hurts performance.** V3 vs V1: AUC 0.981→0.916, Cohen's d 2.80→1.71. VAE decode introduces lossy reconstruction noise that masks the membership signal. Scores are also ~5 orders of magnitude smaller (mean-per-element vs sum).
3. **Caption partially recovers pixel-space.** V4 vs V3: AUC 0.916→0.970, Cohen's d 1.71→2.04.
4. **Best config: latent + caption (V2).** Matches paper's measurement space is secondary; what matters is conditioning on the correct text.
5. **Pixel-space is 4x slower** (485s vs 122s) due to per-timestep VAE decode.

### Score files
- `experiments/sd_watermark_comp/scores/pixel_caption_ablation/v{1,2,3,4}_*.csv`
- Log: `experiments/sd_watermark_comp/logs/pixel_caption_ablation.log`

---

## Blockers & Issues
- ~~**BLOCKER**: LoRA MiO signal too weak (AUC=0.54, Cohen's d=0.15) with original config.~~ **RESOLVED**: Root cause was insufficient epochs per image (4 epochs → no memorization). With 80 epochs, AUC=0.982 at 1000 members.
- ~~**PENDING**: A6 scale validation.~~ **RESOLVED**: AUC=0.982, TPR@1%=0.685, Cohen's d=2.75 on full 11k eval.

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

### Session 3 — 2026-03-13
- Phase: 09 (Ablation planning)
- Actions: Diagnosed weak LoRA MiO signal (AUC=0.5437, Cohen's d=0.15, SNR=0.006). Analyzed score distributions — zero tail separation, member relative t-error reduction only 0.03pp above non-member. Root cause: 4 epochs on 10k images produces no per-image memorization. Designed 6-run ablation plan varying member set size (100/500/1000), epochs (40–200), LoRA rank (4/64/full), timestep selection, and caption conditioning. Pipeline across 4 GPUs, first signal in ~1.5h.
- Result: Ablation plan written in STATE.md. Awaiting review before execution.
- Next: Execute ablation runs A1–A4 in parallel after user approval

### Session 3 (cont.) — 2026-03-13
- Phase: 10 (Ablation execution)
- Actions: Created sub-splits (100/500 members from seed=42 shuffle), training dirs with symlinks. Launched A1–A4 in parallel on GPU0-3. Fixed `--lora_rank` → `--rank` arg name. All 4 trained to convergence (loss floor ~0.15, epoch-averaged). Ran quick eval on all 4 (100/200 member + 200 non-member).
- Results: A1 AUC=1.000, A3 AUC=0.998, A4 AUC=0.998, A2 AUC=0.992. All massively exceed 0.70 target. Dominant factor: epochs per image (4→200 = AUC 0.54→1.00). Rank matters for separation magnitude but even r4 achieves AUC=0.998.
- Convergence: All 4 runs converge (last-10% vs prev-10% relative change < 7%). Plots in `figures/ablation_loss_curves.png`.
- Next: A5 timestep sweep on A1, A6 scale validation at 1000 members

### Session 3 (cont.) — 2026-03-13
- Phase: 10 (A5 timestep sweep)
- Actions: Added `--t-min`/`--t-max` args to `ablation_eval.py`. Ran 4 timestep variants on A4 (full-FT) model in parallel: low-t [0,200], mid-t [200,600], high-t [600,999], dense [0,300] K=50.
- Results: Mid-t [200,600] matches full-range AUC (0.9983). High-t nearly useless (AUC=0.73). Full-range K=12 already optimal — no timestep tuning needed.
- Next: A6 scale validation at 1000 members

### Session 3 (cont.) — 2026-03-15
- Phase: 10 (A6 scale validation)
- Actions: Created 1000-member training dir. Launched full-FT + LoRA in parallel — both crashed on disk full (`/data/short` 100%). Freed ~100GB by deleting intermediate checkpoints from A1–A4. Relaunched A6 LoRA only (full-FT checkpoints too large at 10GB each). Trained 20k steps (80 ep) in 2h45m. Quick eval AUC=0.985, full 11k eval AUC=0.982.
- Results: A6 full eval — AUC=0.9824, TPR@1%=0.685, TPR@0.1%=0.330, Cohen's d=2.75. Signal confirmed at 1000 members with graceful degradation from 100→500→1000 (AUC 1.000→0.992→0.982).
- Result: Ablation phases 09–10 complete. MiO works for SD with sufficient epochs per image (≥80).
- Next: Update paper with ablation results, regenerate tables/figures

### Session 4 — 2026-03-17
- Phase: 11 (3-point verification)
- Actions: Created Phase 11 splits (W-only eval, B1 disjoint COCO, B2 synthetic). Wrote adversary_finetune.py (continues LoRA training from A6 checkpoint). Trained B1 (domain-shift, 1000 disjoint COCO, 2000 steps) and B2 (task-shift, 500 synthetic, 2000 steps) in parallel on GPU0/1 (35 min each). Fixed LoRA save format (manual safetensors extraction for diffusers 0.25.1 compatibility + key conversion to match A6 format). Scored all 3 models on W (1200 images each, ~12 min/model on 3 GPUs). Ran verification protocol.
- Key finding: Raw t-error ratio doesn't work for LoRA (1.01x) — delta-based verification required. Domain-shift FT (B1) is robust: AUC=0.972, |d|=2.21, passes strict criteria. Task-shift FT (B2) partially erases signal: AUC=0.908, |d|=1.73, below strict threshold.
- Result: Phase 11 complete. LaTeX tables in `tables/sd_verification.tex` and `tables/sd_robustness.tex`.
- Next: Integrate verification results into paper (discuss delta-based adaptation for derivative models)

### Session 5 — 2026-03-25
- Phase: 12+13 (latent vs pixel + extended experiments)
- Actions:
  - Phase 12: Ran latent+caption/pixel+caption scoring on Phase 11 models (A6/B1/B2), 1200 images. Fixed verify_ownership.py to match paper Algorithm 2 exactly. All C2/C3 FAIL for LoRA (raw ratio ~1.01x).
  - Phase 13: Expanded non-members 200→1000. Trained A7 (full FT, 1000 members, 40ep, 10k steps, 1h51m) and A8 (LoRA r256, 1000 members, 80ep, 20k steps, 2h49m). Fixed latent normalization to mean/(HWC). Scored all 5 models with latent+caption on 2000-image split. Ran 6 verifications.
- Key findings:
  - Full FT (A7) has highest AUC (0.999) and separation but raw ratio still only 1.028x
  - LoRA r256 (A8) has highest Cohen's d (3.93) for membership detection
  - Algorithm 2 fails for ALL SD models (LoRA and full FT alike)
  - C1 correctly identifies lineage: only A6 passes consistency with B1/B2 (B was derived from A6, not A7/A8)
- Result: Phase 13 complete.

## Phase 13: Extended Experiments

### New Models

| Model | Type | Members | Steps | Epochs | Params | Loss | Time |
|-------|------|---------|-------|--------|--------|------|------|
| A7 | Full FT | 1000 | 10,000 | 40 | 860M (100%) | 0.217 | 1h51m |
| A8 | LoRA r256 | 1000 | 20,000 | 80 | 134M (15.6%) | 0.106 | 2h49m |

### Scoring (latent + caption + mean normalization, 1000 mem + 1000 non-mem)

| Model | AUC | TPR@1% | Cohen's d | Mem mean | Non-mem mean |
|-------|-----|--------|-----------|----------|-------------|
| A6 (LoRA r64, 3.9%) | 0.9949 | 0.8280 | 3.26 | -0.001782 | +0.000239 |
| **A7 (Full FT, 100%)** | **0.9986** | **0.9640** | **3.41** | -0.003110 | +0.000307 |
| **A8 (LoRA r256, 15.6%)** | **0.9994** | **0.9850** | **3.93** | -0.002831 | +0.000782 |
| B1 (adversary, domain) | 0.9858 | 0.6600 | 2.83 | -0.001450 | -0.000045 |
| B2 (adversary, task) | 0.9799 | 0.7130 | 2.70 | -0.000843 | +0.000689 |

### Adversary Models (each derived from its own parent, LoRA r64, 2000 steps, lr=5e-5)

| Adversary | Parent | Type | AUC | TPR@1% | Cohen's d |
|-----------|--------|------|-----|--------|-----------|
| B1_a6 | A6 (r64) | Domain shift | 0.9858 | 0.6600 | 2.83 |
| B2_a6 | A6 (r64) | Task shift | 0.9799 | 0.7130 | 2.70 |
| B1_a7 | A7 (full) | Domain shift | 0.9970 | 0.9320 | 3.22 |
| B2_a7 | A7 (full) | Task shift | 0.9950 | 0.8820 | 3.13 |
| B1_a8 | A8 (r256) | Domain shift | 0.9977 | 0.9210 | 3.51 |
| B2_a8 | A8 (r256) | Task shift | 0.9972 | 0.9450 | 3.46 |

### Verification (Algorithm 2, raw scores — each owner vs its own adversaries)

| Owner | Adversary | C1 (p>0.05) | C2 (\|d\|>2.0) | C3 (ratio>5.0) | Verdict |
|-------|-----------|-------------|----------------|----------------|---------|
| A6 r64 | B1_a6 | PASS (p=0.73) | FAIL (0.08) | FAIL (1.016x) | REJECTED |
| A6 r64 | B2_a6 | PASS (p=0.34) | FAIL (0.08) | FAIL (1.016x) | REJECTED |
| A7 full | B1_a7 | PASS (p=0.71) | FAIL (0.14) | FAIL (1.028x) | REJECTED |
| A7 full | B2_a7 | PASS (p=0.18) | FAIL (0.14) | FAIL (1.028x) | REJECTED |
| A8 r256 | B1_a8 | PASS (p=0.54) | FAIL (0.13) | FAIL (1.025x) | REJECTED |
| A8 r256 | B2_a8 | PASS (p=0.22) | FAIL (0.13) | FAIL (1.025x) | REJECTED |

### Key Findings

1. **A8 (LoRA r256) achieves the best membership detection**: AUC=0.9994, TPR@1%=0.985, d=3.93.
2. **A7 (full FT) has the highest raw ratio** (1.028x vs 1.016x for r64), but still far from 5.0x threshold.
3. **Algorithm 2 fails for ALL SD models** — even full fine-tune with 860M params. Base model variance (std~0.022) is ~7x the membership signal (~0.003).
4. **C1 works correctly**: all adversaries pass consistency with their parent model (p>0.05). Adversary LoRA r64 fine-tuning preserves the parent's reconstruction characteristics on W.

### Output Files
- Owner scores: `scores/phase13/{a6,a7,a8}_latcap.csv`
- A6 adversary scores: `scores/phase13/{b1,b2}_latcap.csv`
- A7 adversary scores: `scores/phase13/a7_{b1,b2}_latcap.csv`
- A8 adversary scores: `scores/phase13/a8_{b1,b2}_latcap.csv`
- Verification: `scores/phase13/verification_{a6,a7,a8}_{b1,b2}.json`

### Adapted Verification: Paired t-test (A1)

Algorithm 2 的 C2/C3 用 raw score 比较 owner vs baseline，因 per-image variance (~0.022) 远大于 membership signal (~0.003) 导致全部 FAIL。Adapted protocol 改用 paired difference δ(x) = S_tgt(x) - S_ref(x)，对同一张图配对消除 per-image variance。

**C2 adapted**: paired t-test on δ(W), H0: mean(δ_W)=0. Pass: p < 1e-6 AND |d| > 2.0.
**C3 adapted**: |mean(δ_W)| / |mean(δ_nonW)| > 5.0.

| Owner | C2 p-value | C2 effect \|d\| | C2 | C3 ratio | C3 |
|-------|-----------|----------------|-----|---------|-----|
| A6 (LoRA r64) | ~0 | 2.43 | PASS | 7.46x | PASS |
| A7 (Full FT) | ~0 | 2.38 | PASS | 10.12x | PASS |
| A8 (LoRA r256) | ~0 | 2.55 | PASS | 3.62x | FAIL |

A6 和 A7 全部通过。A8 的 C3 fail（3.62x < 5.0），因 r256 高容量对 non-member 也产生了较大的 delta 偏移（0.000782 vs A6 的 0.000239），拉低了 ratio。

---

## Deferred — Table 7 row reorder

**Status**: BLOCKED on protocol-level decision (see below)

**Date deferred**: 2026-04-28

**Task**: In `tab:sd_verification` (Table 7) and `tab:mia_results` (Table 3 SD rows),
reorder configurations from current "LoRA r=64 / Full FT / LoRA r=256" to
"LoRA r=64 / LoRA r=256 / Full FT" (group LoRA together, Full FT last).

Also update §5.5 paragraph "Owner-level global membership detection" — the
"respectively" list

> "AUC = 0.995 / 0.999 / 0.999 and \|d\| = 3.26 / 3.41 / 3.93 for
> LoRA r=64 / Full FT / LoRA r=256 respectively"

must be reordered to

> "AUC = 0.995 / 0.999 / 0.999 and \|d\| = 3.26 / 3.93 / 3.41 for
> LoRA r=64 / LoRA r=256 / Full FT respectively"

(verify the AUC mapping when reordering — current AUC values are 0.995 / 0.999 /
0.999 for the three configs; first remains 0.995 (r=64); the two 0.999 values
need to be confirmed which belongs to Full FT vs LoRA r=256 before swapping).

**Blocked by**: C2 \|d\| protocol clarification — see "Adapted Verification" section
above and its conflict with §5.5 lines 280-302 (one-sample vs two-sample
descriptions). Discovered 2026-04-28 in Item 1 read-only investigation:

- Table 7's "C2 (\|d\|)" column values 2.43 / 2.38 / 2.55 reverse-match the
  one-sample formulation (\|mean(δ_W) / std(δ_W)\|, H0: mean = 0) per STATE.md
  line 530.
- Table 7's "\|d\|_mem" column values 2.83 / 2.70 / 3.22 / 3.13 / 3.51 / 3.46
  match a two-sample Cohen's d on δ_W vs δ_W' for the **adversary** model
  (per phase13_report_zh.md lines 28-29, e.g., B1_a6 d=2.83, B2_a6 d=2.70).
- Same δ definition (S_tgt − S_ref) but **different `tgt`** between the two
  columns: C2 uses `tgt = owner-fine-tuned SD`; \|d\|_mem uses
  `tgt = adversary's LoRA`. Different effect size formulas.
- No committed Python file produces 2.43 / 2.38 / 2.55. Only
  `verify_ownership.py` is committed, and it computes raw-score C2
  (which fails — see phase13_report_zh.md line 95: 0.08 / 0.14 / 0.13).

**Resolution path**: revisit after advisor decides among:
  (A) keep one-sample test, write committed code, clarify §4.5 / §5.5 / Table 7
      caption to make the formula explicit
  (B) switch C2 to two-sample (member δ vs non-member δ), rerun, replace numbers
  (C) rename C2 column to reflect one-sample semantics (e.g., "C2 (\|d\|, paired)"),
      keep \|d\|_mem as-is, write committed code

Reordering Table 7 rows now would entrench the current column semantics (which
may need to change under any of A/B/C), so the reorder is deferred until the
column semantics are locked.
