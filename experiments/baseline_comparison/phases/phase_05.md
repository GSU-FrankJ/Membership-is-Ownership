# Phase 05: Zhao et al. Training — Encoder + EDM (GPU, ~32 hours)

## Prerequisites
- Phase 01: σ mapping GO confirmed
- Phase 03: complete (both training commands documented in STATE.md)
- GPU available

## Goal
Train Zhao et al.'s two-stage watermarking pipeline: (1) watermark encoder/decoder, (2) EDM on watermarked data.

---

## Step 5.1: Pre-flight Checks

```bash
# Confirm both commands are documented
cat experiments/baseline_comparison/STATE.md | grep -A5 "Encoder training command"
cat experiments/baseline_comparison/STATE.md | grep -A5 "EDM training command"

# Confirm data
ls /data/short/fjiang4/mia_ddpm_qr/data/cifar10/

# Create output dirs
mkdir -p experiments/baseline_comparison/zhao/cifar10/encoder/
mkdir -p experiments/baseline_comparison/zhao/cifar10/edm/

nvidia-smi
```

**If pre-trained checkpoints were found in Phase 03**: Skip to Step 5.4 (verify checkpoint) and proceed to Phase 06.

---

## Step 5.2: Stage 1 — Train Encoder/Decoder (~8h)

Use the EXACT encoder training command from STATE.md Phase 03.

```bash
cd experiments/baseline_comparison/watermarkdm_repo
# [PASTE ACTUAL ENCODER COMMAND FROM STATE.md]
```

**Monitor**: Encoder training should converge to high bit accuracy (>95%) on validation set. If bit accuracy plateaus below 80%, check hyperparameters against paper.

After encoder training:
```bash
ls experiments/baseline_comparison/zhao/cifar10/encoder/
```

---

## Step 5.3: Stage 2 — Train EDM on Watermarked Data (~24h)

Requires encoder checkpoint from Step 5.2. Use the EXACT EDM command from STATE.md.

```bash
cd experiments/baseline_comparison/watermarkdm_repo
# [PASTE ACTUAL EDM COMMAND FROM STATE.md]
```

**Note on CUDA kernels**: If custom CUDA compilation fails, check Phase 03 notes for `--no-custom-ops` fallback. EDM will be slower but functional.

**Monitor**: EDM training typically reports FID periodically. Expect CIFAR-10 FID < 10 for a well-trained EDM.

---

## Step 5.4: Verify Both Checkpoints

```bash
# Encoder
python -c "
import torch
ckpt = torch.load('[ENCODER_CKPT]', map_location='cpu')
print('Encoder keys:', list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
"

# EDM — likely pickle format (see Phase 03 notes)
python -c "
import pickle
with open('[EDM_CKPT]', 'rb') as f:
    data = pickle.load(f)
print('EDM keys:', list(data.keys()) if isinstance(data, dict) else type(data))
"
```

Verify the EDM checkpoint key matches what Phase 03 documented (e.g., `'ema'` or `'G_ema'`).

---

## Step 5.5: Smoke Test — Generate + Decode

```bash
# Generate a few images and verify watermark is decodable
python scripts/baselines/generate_zhao.py \
    --checkpoint [EDM_CKPT] \
    --num-samples 16 \
    --output-dir experiments/baseline_comparison/zhao/cifar10/smoke_test/ \
    --seed 42

# If Zhao's repo has a decode/verify script, run it on the generated images
# [PASTE ACTUAL EXTRACTION COMMAND FROM STATE.md]
```

---

## Update STATE.md

Record:
- Both checkpoint paths
- Total wall-clock time
- Whether smoke test shows decodable watermarks
- Any issues (CUDA kernels, slow convergence, etc.)
