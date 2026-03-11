# Phase 03: Zhao et al. (WatermarkDM) Code Audit & Setup (CPU only, ~2-3 hours)

## Prerequisites
- Phase 01 complete, σ mapping GO decision confirmed (check STATE.md)
- If σ mapping was NO-GO, SKIP this phase entirely — Zhao is cited-only
- Internet access for git clone

## Goal
Clone WatermarkDM repo, fully audit the two-stage pipeline (encoder + EDM), document real CLI, verify EDM checkpoint format, create Zhao-specific scripts, and prepare exact training commands.

---

## Step 3.1: Clone WatermarkDM

```bash
cd experiments/baseline_comparison/
git clone https://github.com/yunqing-me/WatermarkDM.git watermarkdm_repo
ls watermarkdm_repo/
```

---

## Step 3.2: Full Code Audit

This codebase has TWO stages. Audit both:

```bash
# 1. README
cat watermarkdm_repo/README.md

# 2. All entry points
find watermarkdm_repo/ -maxdepth 3 -name "*.py" | head -40
grep -rl "if __name__" watermarkdm_repo/ --include="*.py"

# 3. CLI arguments for BOTH stages
grep -rn "add_argument\|ArgumentParser\|click" watermarkdm_repo/ --include="*.py"

# 4. Encoder/decoder architecture
grep -rn "class.*Encoder\|class.*Decoder\|class.*StegaStamp\|class.*Hidden" watermarkdm_repo/ --include="*.py"

# 5. EDM model / preconditioning — THIS IS CRITICAL
grep -rn "EDMPrecond\|c_skip\|c_out\|c_in\|c_noise\|sigma_data\|precond" watermarkdm_repo/ --include="*.py"

# 6. Checkpoint format
grep -rn "pickle\|torch.save\|torch.load\|\.pkl\|state_dict" watermarkdm_repo/ --include="*.py"

# 7. CIFAR-10 support
grep -rn "cifar\|dataset_tool\|\.zip" watermarkdm_repo/ --include="*.py" | head -20

# 8. Image normalization
grep -rn "normalize\|\[-1.*1\]\|0\.5\|mean\|std" watermarkdm_repo/ --include="*.py" | head -20

# 9. Pre-trained checkpoints available?
find watermarkdm_repo/ -name "*.pkl" -o -name "*.pt" -o -name "*.pth" 2>/dev/null
grep -rn "pretrained\|download\|checkpoint" watermarkdm_repo/README.md

# 10. Custom CUDA kernels
find watermarkdm_repo/ -name "*.cu" -o -name "*.cuh" 2>/dev/null
grep -rn "torch.utils.cpp_extension\|load_ext\|ninja" watermarkdm_repo/ --include="*.py"

# 11. Dependencies
cat watermarkdm_repo/requirements.txt 2>/dev/null
ls watermarkdm_repo/*.yml 2>/dev/null
```

---

## Step 3.3: Understand the Two-Stage Pipeline

Document exactly:

**Stage 1 — Encoder/Decoder Training**:
- What model architecture (HiDDeN? StegaStamp? Custom?)
- Input: clean images → Output: watermarked images + bit-string decoder
- Training losses (reconstruction, perceptual, bit accuracy)
- Bit length (paper says 48-bit, confirm in code)
- How the watermark key/message is specified

**Stage 2 — EDM Training on Watermarked Data**:
- Does it embed watermarks into training images BEFORE training EDM?
- Or does it modify the EDM training loop?
- What EDM config is used for CIFAR-10? (resolution, channels, training iterations)
- Confirm: `EDMPrecond` wraps the raw backbone with c_skip/c_out/c_in/c_noise

---

## Step 3.4: Verify EDM Checkpoint Loading

This is CRITICAL for the Zhao adapter. EDM checkpoints are typically pickle-based:

```python
import pickle, torch
# Find an example loading call in their code:
grep -rn "pickle.load\|torch.load" watermarkdm_repo/ --include="*.py"
```

Document in STATE.md:
- Checkpoint key name (usually `'ema'` or `'G_ema'`)
- Whether the loaded object is the full `EDMPrecond` wrapper or raw backbone
- How to call it: `model(x_noisy, sigma)` or different signature?

---

## Step 3.5: Verify Image Normalization

EDM typically uses [-1, 1]. Check:
```bash
grep -rn "normalize\|2\.\s*\*\|/ 255\|/ 127" watermarkdm_repo/ --include="*.py" | head -10
```

Record in STATE.md. The Zhao adapter (`compute_edm_t_error`) must normalize images to match.

---

## Step 3.6: Environment Setup

EDM may need custom CUDA kernels:
```bash
# Check if --no-custom-ops is supported
grep -rn "no.custom\|custom_ops\|USE_CUSTOM" watermarkdm_repo/ --include="*.py"
```

```bash
conda activate mio  # or create separate env if needed
cd watermarkdm_repo
pip install -r requirements.txt 2>/dev/null

# Test import
python -c "import sys; sys.path.insert(0, '.'); [try importing EDM network class]"
```

If custom CUDA kernel build fails: use `--no-custom-ops` or equivalent fallback.

---

## Step 3.7: Document Real Training Commands

Based on audit, write ACTUAL commands for:
1. Stage 1: Encoder/decoder training
2. Stage 2: EDM training on watermarked data
3. Watermark extraction/verification

Write all three in STATE.md.

---

## Step 3.8: Create Zhao Adapter

Create `scripts/baselines/zhao_adapter.py` based on audit findings:

Must implement:
1. `load_edm_model(checkpoint_path)` → returns full EDMPrecond network in eval mode
2. `compute_edm_t_error(edm_model, x0, K=50, seed=42)` → per-sample Q25 scores using σ-mapping

**CRITICAL**: use the validated σ↔ᾱ mapping from Phase 01. The function:
- Samples K random timesteps from our cosine schedule
- Converts each to σ via `σ = √((1-ᾱ_t)/ᾱ_t)`
- Forward noises: `x_σ = x₀ + σ · ε`
- Denoises via full preconditioned network: `x̂₀ = edm_net(x_σ, σ)`
- Computes MSE: `‖x̂₀ - x₀‖²` per pixel, averaged over spatial dims
- Returns Q25 over K noise levels

**CRITICAL**: ensure `x₀` is in EDM's native normalization (likely [-1, 1]).

Also create `scripts/baselines/generate_zhao.py`:
- Load EDM checkpoint
- Generate N images using EDM's native sampler (Heun's method, ~35 steps)
- Save as individual PNG files
- Interface: `--checkpoint`, `--num-samples`, `--output-dir`, `--seed`, `--batch-size`

---

## Step 3.9: Check for Pre-trained Checkpoints

If the repo provides pre-trained CIFAR-10 checkpoints, we can skip training (saves ~32h GPU). Check README and any download links. If available, document paths in STATE.md.

---

## Update STATE.md When Done

Fill all Phase 03 checkboxes. Most critically:
- Both ACTUAL training commands
- Checkpoint format details (key name, loading procedure)
- Image normalization answer
- Custom CUDA kernel status
- Whether pre-trained checkpoints are available
