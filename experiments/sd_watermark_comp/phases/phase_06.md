# Phase 06 — MiO Inference: SD v1.4 + LoRA

## Objective
Run MiO membership inference on the LoRA-finetuned model.
Compute t-error scores for all members and non-members.

## Prerequisites
- Phase 04 complete (LoRA checkpoint ready)
- Existing MiO codebase adapted for SD architecture

## Architecture
```
Reference:  SD v1.4 base UNet         → t_error_ref(x)
Target:     SD v1.4 + LoRA UNet       → t_error_tgt(x)
Score:      quantile_regression(t_error_ref, t_error_tgt) per sample x
```

## Steps

### 1. Adapt MiO pipeline to SD v1.4

> **This step depends on your existing MiO codebase.** Key adaptations:
> - Replace DDIM model loading with `StableDiffusionPipeline` + LoRA
> - t-error: encode image x via VAE → add noise at timestep t → UNet predicts noise → compute reconstruction error
> - Ensure timestep schedule matches your existing DDIM experiments

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

device = "cuda:0"

# Reference model
ref_pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", torch_dtype=torch.float16, safety_checker=None
).to(device)

# Target model (base + LoRA)
tgt_pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", torch_dtype=torch.float16, safety_checker=None
).to(device)
tgt_pipe.unet.load_attn_procs("./models/sd-v1-4-lora")

# Shared VAE and scheduler
vae = ref_pipe.vae
scheduler = DDIMScheduler.from_pretrained("./models/sd-v1-4", subfolder="scheduler")
```

### 2. Compute scores for all samples
```python
import json, csv
from PIL import Image
from torchvision import transforms

with open("data/splits/split_seed42.json") as f:
    split = json.load(f)

transform = transforms.Compose([
    transforms.Resize(512), transforms.CenterCrop(512),
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
])

results = []
for label, entries, img_dir in [
    ("member", split["members"], "data/coco2014/train2014"),
    ("nonmember", split["nonmembers"], "data/coco2014/val2014")
]:
    for i, entry in enumerate(entries):
        img = transform(Image.open(f"{img_dir}/{entry['file_name']}").convert("RGB"))
        img = img.unsqueeze(0).to(device)

        # === YOUR MiO CORE LOGIC ===
        # score = mio_score(ref_pipe.unet, tgt_pipe.unet, vae, scheduler, img, timesteps)
        score = 0.0  # placeholder
        # ============================

        results.append({"image_id": entry["image_id"], "label": label, "score": score})

        # Checkpoint every 1000 samples (crash recovery)
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] {i+1}/{len(entries)}")
            with open("experiments/sd_watermark_comp/scores/mio_lora_partial.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["image_id","label","score"])
                w.writeheader(); w.writerows(results)

# Final save
with open("experiments/sd_watermark_comp/scores/mio_lora_scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_id","label","score"])
    w.writeheader(); w.writerows(results)
print(f"Saved {len(results)} scores.")
```

### 3. Compute metrics
```python
import numpy as np
from sklearn.metrics import roc_curve, auc

scores_m = [r["score"] for r in results if r["label"] == "member"]
scores_n = [r["score"] for r in results if r["label"] == "nonmember"]

y_true = [1]*len(scores_m) + [0]*len(scores_n)
y_score = scores_m + scores_n

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
tpr_1pct = tpr[np.searchsorted(fpr, 0.01)]
tpr_01pct = tpr[np.searchsorted(fpr, 0.001)]

print(f"AUC:          {roc_auc:.4f}")
print(f"TPR@1%FPR:    {tpr_1pct:.4f}")
print(f"TPR@0.1%FPR:  {tpr_01pct:.4f}")
```

## Estimated Runtime
- 20k images × ~2-5 sec/img = ~11-28 hr on 1 GPU
- **Parallelization**: split members onto GPU0, non-members onto GPU1 → ~6-14 hr
- Or batch-process if your MiO code supports it

## Sanity Check
- [ ] All 20,000 scores computed (10k member + 10k non-member)
- [ ] No NaN or Inf values
- [ ] Member vs. non-member score distributions show visible separation
- [ ] TPR@1%FPR > 0 (if ≈0, pipeline is broken)
- [ ] CSV saved to `experiments/sd_watermark_comp/scores/mio_lora_scores.csv`

## Update STATE.md
Record: TPR@1%FPR, TPR@0.1%FPR, AUC, runtime, CSV path. Set Phase 06 = ✅ DONE.