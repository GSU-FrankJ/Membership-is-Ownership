# Source Code Structure

This directory contains all source code modules for the MIA DDPM QR project.

## Module Overview

### `attacks/`
Attack implementations for membership inference.

- **`baselines/`** - Reference model comparison
  - `huggingface_loader.py` - Load HuggingFace DDPM models
  - `t_error_hf.py` - T-error computation for reference models
  
- **`eval/`** - Evaluation metrics
  - `metrics.py` - Performance metrics for attacks
  
- **`scores/`** - T-error score computation
  - `t_error.py` - Core t-error computation utilities
  - `compute_scores.py` - Score computation pipeline

### `ddpm/`
Original DDPM training implementation.

- `data/` - Data loaders and splits
- `engine/` - Training engine
- `models/` - DDPM model architectures
- `schedules/` - Noise schedules

### `ddpm_ddim/`
DDIM training and sampling implementation.

- `train_ddim.py` - Main DDIM training script
- `models/` - UNet architectures
- `samplers/` - DDIM samplers (10-step, etc.)
- `schedulers/` - Beta schedules (cosine, linear)
- `clip_features.py` - CLIP feature extraction
- `mmd_loss.py` - MMD loss for fine-tuning

### `attack_qr/`
Quantile regression attack implementation.

- `engine/` - Training and evaluation engines
  - `build_pairs.py` - Build t-error pairs
  - `train_qr_bagging.py` - Train QR ensemble
  - `eval_attack.py` - Evaluate attack performance
  
- `features/` - Feature extraction
  - `t_error.py` - T-error features
  
- `models/` - QR model architectures
  - `qr_resnet18.py` - ResNet18-based QR model
  
- `utils/` - Utilities
  - `losses.py` - Loss functions (pinball, Gaussian)
  - `metrics.py` - Margin-based metrics
  - `seeding.py` - Random seed management

## Import Usage

Since source code is now under `src/`, use the following import pattern:

```python
from src.attacks.baselines import load_hf_ddpm_cifar10
from src.ddpm_ddim.models.unet import build_unet
from src.attack_qr.engine.train_qr_bagging import train_ensemble
```

## Key Entry Points

- **Training**: `src/ddpm_ddim/train_ddim.py`
- **Fine-tuning**: Use script in `scripts/finetune_mmd_ddm.py`
- **Score computation**: `tools/compute_scores.py`
- **Attack evaluation**: `src/attack_qr/engine/eval_attack.py`
