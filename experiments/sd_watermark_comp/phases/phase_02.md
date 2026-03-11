# Phase 02 — Download Models & COCO2014

## Objective
Download SD v1.4 weights and COCO2014 dataset. Verify integrity.

## Steps

### 1. Download SD v1.4 from HuggingFace
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.save_pretrained("./models/sd-v1-4")
print("SD v1.4 saved.")
```
Expected size: ~4-5 GB (UNet ~3.4GB, VAE ~335MB, text encoder ~1.7GB)

### 2. Download COCO2014
```bash
cd data/coco2014

# Training images (~13GB, 82,783 images)
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

# Validation images (~6GB, 40,504 images)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

# Annotations (~241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

rm -f train2014.zip val2014.zip annotations_trainval2014.zip
cd ../..
```

### 3. Verify downloads
```python
import os, json

assert os.path.exists("models/sd-v1-4/unet/diffusion_pytorch_model.safetensors") or \
       os.path.exists("models/sd-v1-4/unet/diffusion_pytorch_model.bin")
print("SD v1.4 UNet: OK")

n_train = len([f for f in os.listdir("data/coco2014/train2014") if f.endswith('.jpg')])
n_val = len([f for f in os.listdir("data/coco2014/val2014") if f.endswith('.jpg')])
print(f"COCO train: {n_train} (expect ~82,783)")
print(f"COCO val:   {n_val} (expect ~40,504)")

with open("data/coco2014/annotations/captions_train2014.json") as f:
    caps = json.load(f)
print(f"Train captions: {len(caps['annotations'])} entries")
```

### 4. Quick generation test
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", torch_dtype=torch.float16, safety_checker=None
).to("cuda:0")

image = pipe(
    "a photo of a cat sitting on a table",
    num_inference_steps=50, guidance_scale=7.5,
    generator=torch.Generator("cuda:0").manual_seed(42)
).images[0]
image.save("experiments/sd_watermark_comp/figures/test_gen_base.png")
print("Test image saved.")
```

### 5. Compute UNet checksum
```bash
md5sum models/sd-v1-4/unet/diffusion_pytorch_model.safetensors || \
md5sum models/sd-v1-4/unet/diffusion_pytorch_model.bin
```

## Sanity Check
- [ ] SD v1.4 loads and generates a coherent image
- [ ] COCO2014 train ~82k images, val ~40k images
- [ ] Annotation files parseable

## Update STATE.md
Record UNet md5. Set Phase 02 = ✅ DONE.