# Phase 02: WDM Code Audit & Setup (CPU only, ~2-3 hours)

## Prerequisites
- Phase 01 complete (check STATE.md)
- Internet access for git clone

## Goal
Clone WDM repo, fully audit the codebase, document the real CLI, check architecture compatibility, create all WDM-specific scripts, and prepare the exact training command.

---

## Step 2.1: Clone WDM

```bash
cd experiments/baseline_comparison/
git clone https://github.com/senp98/wdm.git wdm_repo
ls wdm_repo/
```

---

## Step 2.2: Full Code Audit

Read the entire WDM codebase systematically. Answer ALL of the following and record in STATE.md:

```bash
# 1. Find the README and read it fully
cat wdm_repo/README.md

# 2. Find all entry point scripts
find wdm_repo/ -maxdepth 2 -name "*.py" | head -30
grep -rl "if __name__" wdm_repo/ --include="*.py"

# 3. Find actual CLI arguments
grep -rn "add_argument\|ArgumentParser" wdm_repo/ --include="*.py"

# 4. Find model architecture class
grep -rn "class.*UNet\|class.*Model\|class.*Network" wdm_repo/ --include="*.py"

# 5. Find noise schedule (linear vs cosine)
grep -rn "beta\|alpha\|schedule\|cosine\|linear" wdm_repo/ --include="*.py" | head -20

# 6. Find checkpoint saving/loading
grep -rn "torch.save\|torch.load\|state_dict\|load_state" wdm_repo/ --include="*.py"

# 7. Find watermark-specific code
grep -rn "watermark\|wdp\|trigger\|key_noise" wdm_repo/ --include="*.py" | head -30

# 8. Find dataset handling
grep -rn "cifar\|dataset\|dataloader" wdm_repo/ --include="*.py" | head -20

# 9. Check dependencies
cat wdm_repo/requirements.txt 2>/dev/null || cat wdm_repo/setup.py 2>/dev/null || echo "No requirements file"
```

---

## Step 2.3: Architecture Comparison

Instantiate their model (or read the class definition) and compare:

| Attribute | Our DDIM UNet | WDM's Model |
|-----------|--------------|-------------|
| Base channels | 128 | `___` |
| Channel multipliers | [1,2,2,2] | `___` |
| Attention resolutions | [16] | `___` |
| Noise schedule | Cosine, T=1000 | `___` |
| GroupNorm groups | `___` | `___` |
| Self-attention | Yes | `___` |

**Decision**:
- If architectures are compatible → can potentially load weights interchangeably
- If different → use WDM's native model, write adapter for t-error computation

Record the decision in STATE.md.

---

## Step 2.4: Environment Setup

```bash
# Try installing in existing mio env first
conda activate mio
cd wdm_repo
pip install -r requirements.txt 2>/dev/null

# Test basic import
python -c "import torch; print(torch.__version__)"
# Try importing WDM's main module
python -c "import sys; sys.path.insert(0, '.'); [attempt to import their model class]"

# If dependency conflicts → create separate env
# conda create -n wdm python=3.10
# conda activate wdm
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install -r requirements.txt
```

---

## Step 2.5: Document Real Training Command

Based on Steps 2.2-2.3, write the ACTUAL training command. Key parameters to identify:
- Dataset path argument
- Watermark image/key argument
- Number of epochs/iterations
- Checkpoint save directory
- Random seed
- Batch size, learning rate
- Any WDP-specific parameters (trigger factor γ, etc.)

Write the exact command in STATE.md under "WDM training command (ACTUAL from audit)".

---

## Step 2.6: Create WDM Adapter

Create `scripts/baselines/wdm_adapter.py` based on what you learned in the audit:

Must implement:
1. `load_model(checkpoint_path)` → returns model in eval mode
2. `compute_t_error(model, images, K=50, seed=42)` → returns per-sample Q25 scores
3. `compute_native_metric(model, **kwargs)` → returns watermark extraction rate

The t-error function must handle whatever model interface WDM exposes. If WDM's model is ε-prediction (like ours), reuse `src/attack_qr/features/t_error.py` directly. If it differs, adapt.

Also create `scripts/baselines/generate_wdm.py`:
- Load WDM checkpoint
- Generate N images using WDM's native sampler (likely 1000-step DDPM)
- Save as individual PNG files to output directory
- Interface: `--checkpoint`, `--num-samples`, `--output-dir`, `--seed`, `--batch-size`

---

## Step 2.7: Create Watermark Logo Image

WDM needs a watermark image to embed. Create a simple 32×32 binary pattern:

```python
# Quick script to create watermark logo
import numpy as np
from PIL import Image
img = np.zeros((32, 32), dtype=np.uint8)
# Simple cross pattern (or any recognizable pattern)
img[12:20, 15:17] = 255  # vertical bar
img[15:17, 10:22] = 255  # horizontal bar
Image.fromarray(img).save("experiments/baseline_comparison/wdm/cifar10/watermark_logo.png")
```

Or use whatever format WDM expects (check the code audit — WDM may expect a specific watermark format).

---

## Step 2.8: Data Split Decision (CONFIRM)

WDM trains on ALL 50K CIFAR-10 images as task data. Our watermark set W_D (5K images) is a subset of those 50K. WDM's watermark is a separate embedded logo — orthogonal to MiO's membership signal.

Confirm in the WDM code: does it accept a data root and use all available training images? Or does it have a special split? Record in STATE.md.

---

## Update STATE.md When Done

Fill all Phase 02 checkboxes. Most critically:
- The ACTUAL training command
- The ACTUAL extraction command
- Architecture compatibility decision
- Any issues or workarounds needed
