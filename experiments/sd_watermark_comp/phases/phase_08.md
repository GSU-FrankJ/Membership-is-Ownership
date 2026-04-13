# Phase 08 — Metrics, Tables & Qualitative Figure

## Objective
1. Compile all results into a paper-ready comparison table (LaTeX)
2. Generate the qualitative comparison figure (image grid)
3. Produce publication-ready outputs

---

## Part A: Comparison Table

### 1. Gather results from STATE.md
```python
results = {
    "MiO (Ours)": {
        "tpr_1": 0.0, "tpr_01": 0.0, "auc": 0.0,    # ← Phase 06
        "proactive": False, "key": False, "decoder": False, "quality_loss": False
    },
    "SleeperMark": {
        "tpr_1": 0.0, "tpr_01": 0.0, "auc": 0.0,    # ← Phase 07
        "bit_acc": 0.0,                                 # ← SleeperMark native
        "proactive": True, "key": True, "decoder": True, "quality_loss": True
    },
    # Add your existing DDIM reference models here...
}
```

### 2. Generate LaTeX table
```python
def make_table(results):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ownership verification on SD v1.4. MiO uses private fine-tuning "
        r"data as an implicit watermark — no proactive embedding, key, or decoder needed.}",
        r"\label{tab:sd_comparison}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Proactive & Key & Decoder & Quality Loss & TPR@1\%FPR$\uparrow$ & AUC$\uparrow$ \\",
        r"\midrule",
    ]
    yn = lambda b: r"\cmark" if b else r"\xmark"
    for name, r in results.items():
        lines.append(
            f"{name} & {yn(r['proactive'])} & {yn(r['key'])} & "
            f"{yn(r['decoder'])} & {yn(r['quality_loss'])} & "
            f"{r['tpr_1']:.3f} & {r['auc']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)

tex = make_table(results)
with open("experiments/sd_watermark_comp/tables/sd_comparison.tex", "w") as f:
    f.write(tex)
print(tex)
```

---

## Part B: Qualitative Image Grid

### Goal
Show that watermarking methods introduce quality loss while MiO is post-hoc → zero degradation.

### Grid layout
```
Columns: SD v1.4 (Clean) | SleeperMark (Regular) | SleeperMark (Triggered) | SD v1.4+LoRA (MiO Target)
Rows:    3 different prompts
```

### 1. Select prompts (diverse, visually distinctive)
```python
prompts = [
    "a cup of coffee on a desk with a laptop and keyboard",
    "people walking with umbrellas on a rainy city street",
    "a tabby cat sitting on a white surface looking at camera"
]
```

### 2. Generate all grid images
```python
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch, os

os.makedirs("experiments/sd_watermark_comp/figures/grid", exist_ok=True)
device = "cuda:3"
seed = 42

# Clean SD v1.4
pipe_clean = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", torch_dtype=torch.float16, safety_checker=None).to(device)

# SleeperMark
sm_unet = UNet2DConditionModel.from_pretrained(
    "./models/sleepermark-unet/unet", torch_dtype=torch.float16)
pipe_sm = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", unet=sm_unet, torch_dtype=torch.float16, safety_checker=None).to(device)

# LoRA
pipe_lora = StableDiffusionPipeline.from_pretrained(
    "./models/sd-v1-4", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_lora.unet.load_attn_procs("./models/sd-v1-4-lora")

for i, prompt in enumerate(prompts):
    for tag, pipeline, actual_prompt in [
        ("clean",        pipe_clean, prompt),
        ("sm_regular",   pipe_sm,    prompt),
        ("sm_triggered", pipe_sm,    f"*[Z]& {prompt}"),
        ("lora",         pipe_lora,  prompt),
    ]:
        gen = torch.Generator(device).manual_seed(seed)
        img = pipeline(actual_prompt, generator=gen,
                       num_inference_steps=50, guidance_scale=7.5).images[0]
        img.save(f"experiments/sd_watermark_comp/figures/grid/r{i}_{tag}.png")

print("All grid images generated.")
```

### 3. Compose the figure
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
titles = ["SD v1.4\n(Clean)", "SleeperMark\n(Regular)",
          "SleeperMark\n(Triggered)", "SD v1.4+LoRA\n(MiO Target)"]
tags = ["clean", "sm_regular", "sm_triggered", "lora"]

for row in range(3):
    for col, tag in enumerate(tags):
        img = mpimg.imread(f"experiments/sd_watermark_comp/figures/grid/r{row}_{tag}.png")
        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        if row == 0:
            axes[row][col].set_title(titles[col], fontsize=14, fontweight="bold")

fig.text(0.5, 0.01,
    "MiO operates post-hoc — zero quality degradation vs. clean baseline.",
    ha="center", fontsize=12, style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("experiments/sd_watermark_comp/figures/qualitative_grid.pdf",
            dpi=300, bbox_inches="tight")
plt.savefig("experiments/sd_watermark_comp/figures/qualitative_grid.png",
            dpi=300, bbox_inches="tight")
print("Figure saved as PDF and PNG.")
```

### 4. Per-image LPIPS quality metrics
```python
import torch, lpips
from torchvision import transforms
from PIL import Image

loss_fn = lpips.LPIPS(net="alex").to("cuda:3")
tf = transforms.Compose([
    transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()
])

def get_lpips(path_a, path_b):
    a = tf(Image.open(path_a).convert("RGB")).unsqueeze(0).to("cuda:3") * 2 - 1
    b = tf(Image.open(path_b).convert("RGB")).unsqueeze(0).to("cuda:3") * 2 - 1
    with torch.no_grad():
        return loss_fn(a, b).item()

for row in range(3):
    clean = f"experiments/sd_watermark_comp/figures/grid/r{row}_clean.png"
    for tag in ["sm_regular", "sm_triggered", "lora"]:
        other = f"experiments/sd_watermark_comp/figures/grid/r{row}_{tag}.png"
        print(f"Row {row}, {tag} vs clean: LPIPS = {get_lpips(clean, other):.4f}")
```

---

## Part C: ROC Curve Figure 
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import csv, numpy as np

def load_scores(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    y = [1 if r["label"] == "member" else 0 for r in rows]
    s = [float(r["score"]) for r in rows]
    return y, s

fig, ax = plt.subplots(figsize=(6, 6))
for label, path, color in [
    ("MiO on LoRA",
     "experiments/sd_watermark_comp/scores/mio_lora_scores.csv", "tab:blue"),
    ("MiO on SleeperMark",
     "experiments/sd_watermark_comp/scores/mio_sleepermark_scores.csv", "tab:orange"),
]:
    y, s = load_scores(path)
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", color=color)

ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("MiO Membership Inference on SD v1.4")
ax.legend()
plt.tight_layout()
plt.savefig("experiments/sd_watermark_comp/figures/roc_curves.pdf", dpi=300)
print("ROC curve saved.")
```

## Sanity Check
- [ ] LaTeX table compiles correctly
- [ ] Qualitative grid has no blank or corrupted images
- [ ] LPIPS: LoRA ≈ clean; SleeperMark triggered may show difference
- [ ] Both PDF and PNG outputs saved

## Update STATE.md
Record figure paths, LPIPS scores, final table values. Set Phase 08 = ✅ DONE. Mark project complete.