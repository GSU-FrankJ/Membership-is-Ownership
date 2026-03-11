# Phase 01 — Environment & Dependencies

## Objective
Set up the conda environment with all packages needed for LoRA fine-tuning,
MiO inference, and figure generation. Verify GPU access.

## Steps

### 1. Create conda environment
```bash
conda create -n mio-sd python=3.10 -y
conda activate mio-sd
```

### 2. Install core packages
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers[torch]==0.25.1 transformers==4.37.2 accelerate==0.26.1
pip install peft==0.8.2
pip install datasets
pip install wandb
pip install scikit-learn scipy
pip install matplotlib seaborn pillow
pip install kornia lpips scikit-image
pip install pycocotools
```

### 3. Verify GPU access
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
"
```

### 4. Create directory structure
```bash
mkdir -p models/{sd-v1-4,sd-v1-4-lora,sleepermark-unet}
mkdir -p data/{coco2014,splits,lora_train_dir,sleepermark_train_images}
mkdir -p experiments/sd_watermark_comp/{scores,figures,logs,tables}
mkdir -p configs
mkdir -p external
```

### 5. Verify critical imports
```bash
python -c "
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
print('diffusers OK')
from peft import LoraConfig
print('peft OK')
from accelerate import Accelerator
print('accelerate OK')
"
```

## Sanity Check
- [ ] `conda activate mio-sd` works
- [ ] All 4 GPUs visible with 32GB each
- [ ] `diffusers`, `peft`, `accelerate` all importable
- [ ] Directory structure created

## Update STATE.md
Set Phase 01 = ✅ DONE. Record Python/PyTorch/CUDA/diffusers versions. Current phase → 02.