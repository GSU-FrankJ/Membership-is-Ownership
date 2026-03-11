# Phase 04 — LoRA Fine-tuning SD v1.4

## Objective
Fine-tune SD v1.4 on 10k member images using LoRA. This produces the "target model" for MiO.

## Prerequisites
- Phases 01–03 complete
- `data/lora_train_dir/` populated with 10k images + metadata.jsonl

## Steps

### 1. Save training config
```python
import json

config = {
    "model_name": "./models/sd-v1-4",
    "train_data_dir": "./data/lora_train_dir",
    "output_dir": "./models/sd-v1-4-lora",
    "resolution": 512,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "max_train_steps": 10000,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "lora_rank": 64,
    "lora_alpha": 64,
    "lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "seed": 42,
    "mixed_precision": "fp16",
    "checkpointing_steps": 2000
}

with open("configs/lora_training.json", "w") as f:
    json.dump(config, f, indent=2)
```

### 2. Get the diffusers training script
```bash
wget -O train_text_to_image_lora.py \
  https://raw.githubusercontent.com/huggingface/diffusers/v0.25.1/examples/text_to_image/train_text_to_image_lora.py
```
> If download fails due to network restrictions, manually copy from the
> diffusers repo `examples/text_to_image/`.

### 3. Launch training (~2-3 hours on 1× V100 32GB)
```bash
export MODEL_NAME="./models/sd-v1-4"
export TRAIN_DIR="./data/lora_train_dir"
export OUTPUT_DIR="./models/sd-v1-4-lora"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --mixed_precision="fp16" \
  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --lora_rank=64 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=2000 \
  --seed=42 \
  --report_to="wandb" \
  2>&1 | tee experiments/sd_watermark_comp/logs/lora_training.log
```
> If `--lora_rank` is not recognized, check `python train_text_to_image_lora.py --help`
> for the correct argument name (may be `--rank`).
> If OOM, set batch=2 and gradient_accumulation=2.

### 4. Verify trained model
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", torch_dtype=torch.float16, safety_checker=None
).to("cuda:0")
pipe.unet.load_attn_procs("./models/sd-v1-4-lora")

image = pipe(
    "a photo of a cat sitting on a table",
    num_inference_steps=50, guidance_scale=7.5,
    generator=torch.Generator("cuda:0").manual_seed(42)
).images[0]
image.save("experiments/sd_watermark_comp/figures/test_gen_lora.png")
print("LoRA generation test saved. Compare visually with base model output.")
```

### 5. Quick overfit check
```python
import json

with open("data/splits/split_seed42.json") as f:
    split = json.load(f)

for label, caption in [("member", split["members"][0]["caption"]),
                        ("nonmember", split["nonmembers"][0]["caption"])]:
    img = pipe(caption, num_inference_steps=50, guidance_scale=7.5,
               generator=torch.Generator("cuda:0").manual_seed(42)).images[0]
    img.save(f"experiments/sd_watermark_comp/figures/overfit_{label}.png")
    print(f"{label}: {caption[:80]}...")
```

## Sanity Check
- [ ] Training completed 10,000 steps without error
- [ ] Final checkpoint exists at `models/sd-v1-4-lora/`
- [ ] LoRA weight file size is reasonable (~50-150MB for rank 64)
- [ ] Generated image is coherent (not collapsed or noisy)
- [ ] Training loss log saved

## Update STATE.md
Record: start/end time, final loss, checkpoint path, weight file size. Set Phase 04 = ✅ DONE.