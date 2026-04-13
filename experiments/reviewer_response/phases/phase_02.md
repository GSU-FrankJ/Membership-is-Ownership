# Phase 12 — Reference Model Overlap Sensitivity & Practical Selection Guidance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address reviewer concern: "As training datasets grow increasingly large, it becomes harder to guarantee that any public baseline has never been exposed to the private watermark samples. Could the authors elaborate on a practical methodology for selecting valid baselines, and discuss how verification confidence might degrade when such guarantees cannot be made?"
*(Note: "baseline" in the reviewer quote above is kept verbatim.)*

**Architecture:** Four sub-experiments (E1–E4). E1 and E3 are paper-writing tasks (no compute). E2 trains DDIM reference models from scratch on CIFAR-10 with controlled watermark overlap (0%–100%) and measures Cohen's d / ratio degradation. E4 repeats the overlap study on SD v1.4 with LoRA and optionally checks COCO/LAION overlap via CLIP search. DDIM experiments run first; SD depends on time.

**Tech Stack:** PyTorch, DDIM UNet (128ch, [1,2,2,2]), diffusers/peft (SD v1.4 LoRA), scipy (stats), matplotlib (figures), CLIP ViT-B/32 (optional LAION search)

---

## Critical Context

### The Hidden Observation the Paper Already Has

The existing CIFAR-10 public reference model (`google/ddpm-cifar10-32`) was trained on the **full CIFAR-10 training set** (50k images), which **already includes all 5,000 watermark images**. This is 100% overlap with W — yet Cohen's d is still >18 and ratio >19x.

Why this works: what matters is not whether a reference model has "seen" the data, but whether it has the **same memorization fingerprint** — the unique combination of architecture, initialization seed, optimizer trajectory, and noise schedule that determines per-sample reconstruction error. Two models trained on identical data produce different fingerprints.

**This observation is the strongest possible defense against the reviewer's concern, but the paper never states it.** E1 fixes this. E2 provides the controlled degradation curve to make the argument rigorous.

*(Note: throughout this file, "baseline" in code blocks, variable/function names, CLI arguments, file paths, and directory names is preserved as-is. Only prose/headings are updated to "reference model.")*

### Existing Infrastructure

| Component | DDIM (mio env) | SD (mio-sd env) |
|-----------|---------------|-----------------|
| Owner model | `best_for_mia.ckpt` (400k iter, 50k CIFAR-10) | A6 LoRA (20k steps, 1k COCO, 80 ep/img) |
| Watermark set W | 5,000 CIFAR-10 indices | 1,000 COCO images |
| Training script | `src/ddpm_ddim/train_ddim.py` | `train_text_to_image_lora.py` (HF) |
| Eval script | `scripts/eval_ownership.py` | `ablation_eval.py` |
| Checkpoint size | ~117 MB each | ~49 MB LoRA / ~3.4 GB full UNet |
| Train time (1x V100) | ~12–15h (400k iter) / ~3–4h (100k fastdev) | ~2h45m (20k steps) |

### Resource Constraints

- **Disk**: 64 GB free on `/data/short` (97% full). Each DDIM checkpoint ~117 MB, each LoRA ~49 MB. Budget: ~6 DDIM models = ~700 MB, ~4 SD LoRA = ~200 MB. Manageable.
- **GPU**: 4x V100-SXM2-32GB. Can run 4 trainings in parallel.
- **Time**: DDIM full training dominates. 6 models on 4 GPUs = 2 rounds = 24–30h. Fastdev (100k iter): 6–12h. SD: ~3h total.

---

## Decisions to Discuss with Professor

### D1: DDIM Training Duration — Full (400k) vs FastDev (100k)?

| Option | Time (6 models, 4 GPUs) | Quality | Risk |
|--------|------------------------|---------|------|
| **400k iterations** (full) | ~24–30h wall | Fully converged, matches Model A | None — gold standard |
| **100k iterations** (fastdev) | ~6–12h wall | Partially converged | Reviewer may question whether unconverged reference models inflate d |
| **200k iterations** (compromise) | ~12–18h wall | Mostly converged | Reasonable middle ground |

**Recommendation**: Start with 100k (fastdev) as a pilot on 3 overlap levels (0%, 50%, 100%). If the degradation curve is clear and d stays >2.0 even at 100%, we may not need 400k. If the results are ambiguous, re-run at 400k.

### D2: Number of Overlap Levels

| Option | Levels | Models to train | Curve granularity |
|--------|--------|-----------------|-------------------|
| Minimal | 0%, 50%, 100% | 3 | Coarse — shows trend |
| Standard | 0%, 25%, 50%, 75%, 100% | 5 | Smooth curve |
| Full | 0%, 10%, 25%, 50%, 75%, 100% | 6 | Publication-quality |

**Recommendation**: Start with minimal (3 levels). If the curve is monotonic and interesting, add 25% and 75% for a smoother plot.

### D3: SD Experiment Scope

| Option | What | Time | Value |
|--------|------|------|-------|
| **E4a only**: CLIP overlap check | Check if COCO W images appear in LAION | ~1h (if LAION search API works) | Quantifies actual overlap |
| **E4b only**: Contaminated LoRA reference models | Train LoRA on partial-W datasets | ~3h | Direct degradation curve for SD |
| **E4a + E4b** | Both | ~4h | Complete story |

**Recommendation**: E4b is more valuable (directly answers the reviewer). E4a is nice-to-have but LAION search infrastructure may not be accessible.

### D4: Paper Placement

Options for the overlap sensitivity results:
1. **New appendix section** (e.g., "Appendix F: Reference Model Overlap Sensitivity") — most space, least disruptive
2. **Expand Discussion paragraph** (line 882) — visible but tight on space
3. **New subsection in Experiments** (e.g., 5.4 "Reference Model Selection Sensitivity") — strongest position
4. **Split**: degradation curve in appendix, one-paragraph summary + figure in main text

**To discuss with professor.** The plan produces all materials; placement is independent of experiment design.

---

## E1: Highlight Existing Overlap (Paper Writing Only)

### Task 1: Draft Overlap Observation Paragraph

**Files:**
- Modify: `ICML2026/main.tex` (exact location TBD after D4 decision)

**Key points to make:**

1. `google/ddpm-cifar10-32` was trained on the full CIFAR-10 training set, which contains all 5,000 watermark images (100% overlap).
2. Despite this, Cohen's d > 18 and ratio > 19x — the verification protocol passes with massive margin.
3. Similarly, `google/ddpm-celebahq-256` was trained on CelebA-HQ, which substantially overlaps with our CelebA watermark set.
4. This demonstrates that the verification signal depends on the **model-specific memorization fingerprint** (architecture + seed + optimizer trajectory), not on whether the reference model has "seen" the data.
5. The null hypothesis is not "reference model has never seen W" but rather "reference model does not share the same training provenance as the owner model."

**Draft text (to be refined after E2 results):**

```latex
\paragraph{Baseline Overlap Does Not Invalidate Verification.}
A natural concern is whether public baselines must be provably disjoint
from the watermark set.  In fact, our primary CIFAR-10 baseline
(\texttt{google/ddpm-cifar10-32}) was trained on the \emph{full}
CIFAR-10 training set, which contains all 5{,}000 watermark images
--- 100\% overlap.  Yet Cohen's $d > 18$ and the error ratio exceeds
$19\times$.  This is because the verification signal depends not on
data exclusivity but on the \emph{model-specific memorization
fingerprint}: the unique combination of architecture, initialization,
and optimizer trajectory.  Two models trained on identical data
produce distinct fingerprints, as confirmed by our controlled
overlap study (Appendix~\ref{app:overlap}).
```

- [ ] **Step 1**: After E2 results are available, finalize paragraph with exact numbers from the degradation curve.
- [ ] **Step 2**: Insert into main.tex at location determined by D4.
- [ ] **Step 3**: Verify the claim about `google/ddpm-cifar10-32` training data by checking the HuggingFace model card.

---

## E2: Controlled Overlap Degradation — DDIM on CIFAR-10

### Experiment Design

**Variables:**
- Independent variable: overlap fraction `f` in {0%, 10%, 25%, 50%, 75%, 100%}
- Dependent variables: Cohen's d (vs Model A on W), error ratio, verification pass/fail
- Controlled: architecture, training iterations, batch size, learning rate, noise schedule

**Data Construction:**

Let:
- T = CIFAR-10 train set (50,000 images, indices 0–49,999)
- W = watermark set (5,000 images, W subset of T)
- R = T \ W = remaining training images (45,000 images)

For each overlap fraction f:
- Sample f * |W| images from W (deterministic, seed=42, take first N after shuffle)
- Training set = R union W_sampled
- Total training images: 45,000 + f * 5,000

| Overlap f | W images included | Total training size | Images from W |
|-----------|-------------------|--------------------|----|
| 0% | 0 | 45,000 | none |
| 10% | 500 | 45,500 | W[0:500] |
| 25% | 1,250 | 46,250 | W[0:1250] |
| 50% | 2,500 | 47,500 | W[0:2500] |
| 75% | 3,750 | 48,750 | W[0:3750] |
| 100% | 5,000 | 50,000 | all of W (= T) |

**Important:** Model A was trained on T (50k) with seed=20251030 (from config). The 100% overlap reference model is trained on the same data but with a **different seed** (seed=42). This isolates the fingerprint effect.

**Training config:** Identical to Model A except for training data and seed.
- Architecture: UNet 128ch, [1,2,2,2], attn@16, cosine schedule
- Iterations: 400k (full) or 100k (fastdev) — per D1
- Batch size: 128, lr: 2e-4, AdamW, EMA 0.9999
- Seed: 42 (different from Model A's 20251030)

**Evaluation:**
- Compute t-error on W (5,000 images) for each reference model using `eval_ownership.py`
- Measure: mean t-error, Cohen's d vs Model A, error ratio (reference model/Model A), 3-point criteria pass/fail
- Also compute on eval_nonmember (5,000 images) as sanity check

### Task 2: Create Overlap Split Generator Script

**Files:**
- Create: `scripts/overlap_splits.py`

This script reads the existing splits (watermark_private.json, member_train.json) and generates new training index lists for each overlap level.

- [ ] **Step 1**: Write `scripts/overlap_splits.py`

```python
#!/usr/bin/env python3
"""Generate training index lists for baseline overlap sensitivity study.

For each overlap fraction f in [0, 0.10, 0.25, 0.50, 0.75, 1.00]:
  - Take all non-watermark training indices (R = T \ W)
  - Add first f*|W| watermark indices (deterministic shuffle, seed=42)
  - Save to JSON

Usage:
    python scripts/overlap_splits.py \
        --watermark /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/watermark_private.json \
        --member-train /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/member_train.json \
        --output-dir /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/overlap_study/ \
        --fractions 0.0 0.10 0.25 0.50 0.75 1.00 \
        --seed 42
"""

import argparse
import json
import os
import hashlib
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark", required=True, help="Path to watermark_private.json")
    parser.add_argument("--member-train", required=True, help="Path to member_train.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.watermark) as f:
        W = json.load(f)  # list of indices
    with open(args.member_train) as f:
        T = json.load(f)  # list of indices

    W_set = set(W)
    R = [idx for idx in T if idx not in W_set]  # T \ W

    # Deterministic shuffle of W
    rng = random.Random(args.seed)
    W_shuffled = W.copy()
    rng.shuffle(W_shuffled)

    os.makedirs(args.output_dir, exist_ok=True)

    manifest = {
        "experiment": "baseline_overlap_sensitivity",
        "seed": args.seed,
        "watermark_size": len(W),
        "non_watermark_size": len(R),
        "fractions": {},
    }

    for f in args.fractions:
        n_w = int(round(f * len(W)))
        w_subset = W_shuffled[:n_w]
        train_indices = sorted(R + w_subset)

        tag = f"overlap_{int(f * 100):03d}"
        out_path = os.path.join(args.output_dir, f"{tag}.json")
        with open(out_path, "w") as fout:
            json.dump(train_indices, fout)

        # Verification
        actual_overlap = len(set(train_indices) & W_set)
        md5 = hashlib.md5(json.dumps(train_indices).encode()).hexdigest()

        manifest["fractions"][tag] = {
            "fraction": f,
            "w_images_included": n_w,
            "total_training_size": len(train_indices),
            "actual_w_overlap": actual_overlap,
            "file": f"{tag}.json",
            "md5": md5,
        }

        print(f"  {tag}: {len(train_indices)} images "
              f"({actual_overlap}/{len(W)} W overlap), md5={md5[:12]}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: Run script and verify outputs

```bash
conda activate mio
python scripts/overlap_splits.py \
    --watermark /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/watermark_private.json \
    --member-train /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/member_train.json \
    --output-dir /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/overlap_study/ \
    --fractions 0.0 0.10 0.25 0.50 0.75 1.00
```

Expected: 6 JSON files + manifest in `overlap_study/` directory. Verify actual overlap counts match expected.

- [ ] **Step 3**: Commit

```bash
git add scripts/overlap_splits.py
git commit -m "feat: overlap split generator for baseline sensitivity study"
```

### Task 3: Create Overlap Training Config

**Files:**
- Create: `configs/overlap_study/train_overlap.yaml`

A parameterized config template for training overlap reference models. The training script accepts `--config` and we override the data indices path.

- [ ] **Step 1**: Write config file

```yaml
# Training config for overlap sensitivity baselines
# Identical to model_ddim_cifar10.yaml except:
#   - output_dir points to overlap study directory
#   - seed is 42 (different from Model A's 20251030)
#   - training data indices are overridden via --overlap-indices CLI arg

experiment:
  name: overlap_baseline
  output_dir: /data/short/fjiang4/mia_ddpm_qr/runs/overlap_study
  resume: null

model:
  channels: 128
  channel_mults: [1, 2, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [16]
  dropout: 0.0
  in_channels: 3
  out_channels: 3
  image_size: 32

diffusion:
  timesteps: 1000
  schedule: cosine
  ddim_eta: 0.0

training:
  iterations:
    main: 400000
    fastdev: 100000
  batch_size: 128
  lr: 0.0002
  weight_decay: 0.0
  optimizer: adamw
  betas: [0.9, 0.999]
  grad_clip: 1.0
  ema_decay: 0.9999
  ema_update_every: 1
  amp: true
  log_interval: 500
  checkpoint_interval: 50000   # Save less often to save disk (5 ckpts vs 40)
  fastdev_limit: 2000

data:
  config: configs/data_cifar10.yaml

seed: 42
```

**Note:** `checkpoint_interval: 50000` instead of 10000 saves ~80% disk per model.

- [ ] **Step 2**: Verify `train_ddim.py` supports custom index files

Check whether `train_ddim.py` already accepts a `--train-indices` argument or similar. If not, a small modification is needed to pass the overlap split indices instead of the default `member_train.json`.

```bash
python src/ddpm_ddim/train_ddim.py --help
```

If no `--train-indices` flag exists, add one (Task 4). Otherwise skip Task 4.

### Task 4: (Conditional) Add --train-indices Flag to train_ddim.py

**Files:**
- Modify: `src/ddpm_ddim/train_ddim.py`

Only needed if `train_ddim.py` does not already support overriding training indices.

- [ ] **Step 1**: Add argument

```python
parser.add_argument("--train-indices", type=pathlib.Path, default=None,
                    help="Override training indices JSON (for overlap study)")
```

- [ ] **Step 2**: In the data loading section, if `args.train_indices` is provided, load those indices instead of `member_train.json`

Find the section where `member_train.json` is loaded (approximately where `data_cfg["splits"]["paths"]["member_train"]` is read) and add:

```python
if args.train_indices:
    train_indices_path = args.train_indices
    LOGGER.info(f"Using custom train indices: {train_indices_path}")
else:
    train_indices_path = pathlib.Path(data_cfg["splits"]["paths"]["member_train"])
```

- [ ] **Step 3**: Test with a dry run (1 iteration)

```bash
python src/ddpm_ddim/train_ddim.py \
    --config configs/overlap_study/train_overlap.yaml \
    --train-indices /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/overlap_study/overlap_000.json \
    --mode fastdev
# Ctrl+C after first few iterations to verify it loads correctly
```

- [ ] **Step 4**: Commit

```bash
git add src/ddpm_ddim/train_ddim.py configs/overlap_study/train_overlap.yaml
git commit -m "feat: support custom train indices for overlap study"
```

### Task 5: Train Overlap Reference Models (GPU, Long-Running)

**Files:**
- Output: `/data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/{overlap_000,overlap_010,...}/`

Train 6 (or 3 for pilot) DDIM reference models with different overlap levels.

- [ ] **Step 1**: Decide iteration count based on D1 discussion

```
ITERATIONS=100000   # fastdev pilot
# or
ITERATIONS=400000   # full run
```

- [ ] **Step 2**: Launch training round 1 (4 models on 4 GPUs)

```bash
# Pilot: 0%, 25%, 50%, 100% (4 models, 1 round)
for i in 0 1 2 3; do
    TAGS=("overlap_000" "overlap_025" "overlap_050" "overlap_100")
    TAG=${TAGS[$i]}
    tmux new-session -d -s "train_${TAG}"
    tmux send-keys -t "train_${TAG}" "conda activate mio && \
CUDA_VISIBLE_DEVICES=$i python src/ddpm_ddim/train_ddim.py \
    --config configs/overlap_study/train_overlap.yaml \
    --train-indices /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/overlap_study/${TAG}.json \
    --mode main \
    2>&1 | tee /data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/${TAG}/train.log" Enter
done
```

**Output directory naming:** Each model saves to `/data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/{TAG}/`. The script needs to be configured to use TAG as the run subdirectory. This may require passing `--output-dir` or setting `experiment.name` in config.

**Implementation note:** The exact launch commands depend on how `train_ddim.py` resolves the output directory. Check the script's `--config` handling and adjust. The key requirement is that each overlap level gets its own output directory.

- [ ] **Step 3**: Monitor training

```bash
# Check all sessions
for TAG in overlap_000 overlap_025 overlap_050 overlap_100; do
    echo "=== $TAG ==="
    tmux capture-pane -t "train_${TAG}" -p | tail -5
done
```

- [ ] **Step 4**: (If full 6 levels) Launch round 2 after round 1 completes

```bash
# overlap_010 and overlap_075 on 2 GPUs
for i in 0 1; do
    TAGS=("overlap_010" "overlap_075")
    TAG=${TAGS[$i]}
    tmux new-session -d -s "train_${TAG}"
    tmux send-keys -t "train_${TAG}" "conda activate mio && \
CUDA_VISIBLE_DEVICES=$i python src/ddpm_ddim/train_ddim.py \
    --config configs/overlap_study/train_overlap.yaml \
    --train-indices /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/overlap_study/${TAG}.json \
    --mode main \
    2>&1 | tee /data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/${TAG}/train.log" Enter
done
```

- [ ] **Step 5**: After all training completes, verify checkpoints exist

```bash
for TAG in overlap_000 overlap_010 overlap_025 overlap_050 overlap_075 overlap_100; do
    CKPT="/data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/${TAG}/ema.ckpt"
    if [ -f "$CKPT" ]; then
        echo "$TAG: OK ($(du -sh $CKPT | cut -f1))"
    else
        echo "$TAG: MISSING"
    fi
done
```

### Task 6: Evaluate Overlap Reference Models

**Files:**
- Create: `scripts/eval_overlap_study.py`
- Output: `/data/short/fjiang4/experiments/overlap_study/results/`

- [ ] **Step 1**: Write evaluation script

```python
#!/usr/bin/env python3
"""Evaluate overlap baselines against Model A on watermark set W.

For each trained overlap baseline:
  1. Load baseline checkpoint
  2. Compute t-error on W (5000 images)
  3. Compute Cohen's d and ratio vs Model A
  4. Check 3-point verification criteria
  5. Save results to JSON

Usage:
    python scripts/eval_overlap_study.py \
        --model-a /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
        --baselines-dir /data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/ \
        --dataset cifar10 \
        --output /data/short/fjiang4/experiments/overlap_study/results/ \
        --gpu 0
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ddpm_ddim.models.unet import build_unet
from src.ddpm_ddim.schedulers.betas import build_cosine_schedule
from src.attack_qr.features.t_error import compute_t_error_batch  # reuse existing


def load_model(checkpoint_path, model_cfg, device):
    """Load a DDIM model from checkpoint."""
    model = build_unet(model_cfg["model"])
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def compute_scores(model, dataloader, alphas_bar, device, k=50, agg="q25"):
    """Compute t-error scores for all samples."""
    all_scores = []
    T = len(alphas_bar)
    timesteps = sorted(set(int(round(i)) for i in np.linspace(0, T - 1, k)))

    for batch in tqdm(dataloader, desc="Scoring"):
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        with torch.no_grad():
            scores = compute_t_error_batch(images, model, alphas_bar, timesteps, agg=agg)
        all_scores.append(scores.cpu())

    return torch.cat(all_scores)


def compute_metrics(scores_a, scores_baseline):
    """Compute Cohen's d, ratio, t-test p-value."""
    s_a = scores_a.numpy()
    s_b = scores_baseline.numpy()

    mean_a, std_a = np.mean(s_a), np.std(s_a)
    mean_b, std_b = np.mean(s_b), np.std(s_b)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
    ratio = mean_b / mean_a if mean_a > 0 else float("inf")
    t_stat, p_value = stats.ttest_ind(s_a, s_b)

    return {
        "mean_a": float(mean_a),
        "mean_baseline": float(mean_b),
        "std_a": float(std_a),
        "std_baseline": float(std_b),
        "cohens_d": float(abs(cohens_d)),
        "ratio": float(ratio),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "separation_pass": p_value < 1e-6 and abs(cohens_d) > 2.0,
        "ratio_pass": ratio > 5.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, type=pathlib.Path)
    parser.add_argument("--baselines-dir", required=True, type=pathlib.Path)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k-timesteps", type=int, default=50)
    parser.add_argument("--agg", default="q25")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    args.output.mkdir(parents=True, exist_ok=True)

    # Load configs
    import yaml
    model_cfg_path = PROJECT_ROOT / f"configs/model_ddim_{args.dataset}.yaml"
    data_cfg_path = PROJECT_ROOT / f"configs/data_{args.dataset}.yaml"
    model_cfg = yaml.safe_load(model_cfg_path.open())
    data_cfg = yaml.safe_load(data_cfg_path.open())

    T = model_cfg["diffusion"]["timesteps"]
    _, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    # Load watermark dataloader
    # (reuse eval_ownership.py's EvalDataset or build directly)
    from scripts.eval_ownership import build_eval_loader
    wm_loader = build_eval_loader(
        args.dataset, data_cfg, "watermark_private",
        batch_size=args.batch_size,
    )

    # Score Model A on W
    print("Scoring Model A on W...")
    model_a = load_model(args.model_a, model_cfg, device)
    scores_a = compute_scores(model_a, wm_loader, alphas_bar, device,
                              k=args.k_timesteps, agg=args.agg)
    del model_a
    torch.cuda.empty_cache()

    # Score each overlap baseline on W
    results = {}
    # Load manifest to get overlap fractions
    manifest_path = pathlib.Path(
        f"/data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/overlap_study/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())

    for tag, info in sorted(manifest["fractions"].items()):
        ckpt_dir = args.baselines_dir / tag
        # Find EMA checkpoint (naming depends on train_ddim.py output)
        ckpt_candidates = list(ckpt_dir.glob("**/ema.ckpt")) + list(ckpt_dir.glob("**/best_for_mia.ckpt"))
        if not ckpt_candidates:
            print(f"  {tag}: no checkpoint found, skipping")
            continue
        ckpt_path = ckpt_candidates[0]

        print(f"\nScoring {tag} (overlap={info['fraction']:.0%})...")
        model = load_model(ckpt_path, model_cfg, device)
        scores_b = compute_scores(model, wm_loader, alphas_bar, device,
                                  k=args.k_timesteps, agg=args.agg)
        del model
        torch.cuda.empty_cache()

        metrics = compute_metrics(scores_a, scores_b)
        metrics["overlap_fraction"] = info["fraction"]
        metrics["w_images_included"] = info["w_images_included"]
        metrics["total_training_size"] = info["total_training_size"]
        results[tag] = metrics

        print(f"  d={metrics['cohens_d']:.2f}, ratio={metrics['ratio']:.2f}, "
              f"sep={'PASS' if metrics['separation_pass'] else 'FAIL'}, "
              f"ratio={'PASS' if metrics['ratio_pass'] else 'FAIL'}")

    # Save results
    results_path = args.output / "overlap_sensitivity_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"OVERLAP SENSITIVITY STUDY — CIFAR-10")
    print(f"{'='*70}")
    print(f"{'Overlap':>10} {'W imgs':>8} {'Train N':>8} "
          f"{'|d|':>8} {'Ratio':>8} {'Sep':>6} {'Ratio':>6}")
    print(f"{'-'*70}")
    for tag in sorted(results.keys()):
        r = results[tag]
        print(f"{r['overlap_fraction']:>9.0%} {r['w_images_included']:>8} "
              f"{r['total_training_size']:>8} {r['cohens_d']:>8.2f} "
              f"{r['ratio']:>8.2f} "
              f"{'PASS' if r['separation_pass'] else 'FAIL':>6} "
              f"{'PASS' if r['ratio_pass'] else 'FAIL':>6}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
```

**Important implementation note:** The `compute_t_error_batch` import may need adjustment depending on the actual function signature in `src/attack_qr/features/t_error.py`. Read that file before implementing to match the interface. The script above is a structural template — the scoring inner loop should reuse the existing t-error computation exactly as `eval_ownership.py` does.

- [ ] **Step 2**: Run evaluation

```bash
conda activate mio
python scripts/eval_overlap_study.py \
    --model-a /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --baselines-dir /data/short/fjiang4/mia_ddpm_qr/runs/overlap_study/ \
    --dataset cifar10 \
    --output /data/short/fjiang4/experiments/overlap_study/results/ \
    --gpu 0
```

Expected: ~15 min per model (5000 images, K=50). Total: ~90 min for 6 models.

- [ ] **Step 3**: Commit

```bash
git add scripts/eval_overlap_study.py
git commit -m "feat: overlap sensitivity evaluation script"
```

### Task 7: Generate Degradation Curve Figure

**Files:**
- Create: `scripts/plot_overlap_sensitivity.py`
- Output: `experiments/sd_watermark_comp/figures/overlap_sensitivity.pdf`

- [ ] **Step 1**: Write plotting script

```python
#!/usr/bin/env python3
"""Plot overlap sensitivity degradation curve.

Reads results from eval_overlap_study.py and produces:
1. Cohen's d vs overlap fraction (with d=2.0 threshold line)
2. Error ratio vs overlap fraction (with ratio=5.0 threshold line)
3. Combined two-panel figure for paper

Usage:
    python scripts/plot_overlap_sensitivity.py \
        --results /data/short/fjiang4/experiments/overlap_study/results/overlap_sensitivity_results.json \
        --output experiments/sd_watermark_comp/figures/overlap_sensitivity.pdf
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    # Optional: add SD results for a combined plot
    parser.add_argument("--sd-results", default=None,
                        help="SD overlap results JSON for combined plot")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    # Sort by overlap fraction
    items = sorted(results.values(), key=lambda x: x["overlap_fraction"])
    fracs = [r["overlap_fraction"] for r in items]
    ds = [r["cohens_d"] for r in items]
    ratios = [r["ratio"] for r in items]
    sep_pass = [r["separation_pass"] for r in items]
    ratio_pass = [r["ratio_pass"] for r in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Cohen's d
    ax1.plot(fracs, ds, "o-", color="tab:blue", linewidth=2, markersize=8,
             label="DDIM (CIFAR-10)")
    ax1.axhline(y=2.0, color="red", linestyle="--", alpha=0.7, label="Threshold (d=2.0)")
    ax1.set_xlabel("Overlap Fraction (W images in baseline training set)")
    ax1.set_ylabel("Cohen's d (vs Owner Model)")
    ax1.set_title("(a) Separation Magnitude")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Error ratio
    ax2.plot(fracs, ratios, "s-", color="tab:orange", linewidth=2, markersize=8,
             label="DDIM (CIFAR-10)")
    ax2.axhline(y=5.0, color="red", linestyle="--", alpha=0.7, label="Threshold (ratio=5.0)")
    ax2.set_xlabel("Overlap Fraction (W images in baseline training set)")
    ax2.set_ylabel("Error Ratio (baseline / owner)")
    ax2.set_title("(b) Error Ratio")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # If SD results available, overlay on same plot
    if args.sd_results:
        with open(args.sd_results) as f:
            sd_results = json.load(f)
        sd_items = sorted(sd_results.values(), key=lambda x: x["overlap_fraction"])
        sd_fracs = [r["overlap_fraction"] for r in sd_items]
        sd_ds = [r["cohens_d"] for r in sd_items]
        sd_ratios = [r["ratio"] for r in sd_items]
        ax1.plot(sd_fracs, sd_ds, "^--", color="tab:green", linewidth=2,
                 markersize=8, label="SD v1.4 LoRA (COCO)")
        ax2.plot(sd_fracs, sd_ratios, "D--", color="tab:purple", linewidth=2,
                 markersize=8, label="SD v1.4 LoRA (COCO)")
        ax1.legend()
        ax2.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {args.output}")

    # Also save PNG for quick viewing
    png_path = args.output.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"PNG saved: {png_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: Run after E2 evaluation completes

```bash
python scripts/plot_overlap_sensitivity.py \
    --results /data/short/fjiang4/experiments/overlap_study/results/overlap_sensitivity_results.json \
    --output experiments/sd_watermark_comp/figures/overlap_sensitivity.pdf
```

- [ ] **Step 3**: Commit

```bash
git add scripts/plot_overlap_sensitivity.py
git commit -m "feat: overlap sensitivity degradation curve plot"
```

### Expected Outcomes for E2

| Overlap | Expected d | Expected ratio | Passes? | Reasoning |
|---------|-----------|----------------|---------|-----------|
| 0% | >>2.0 (likely ~15-25) | >>5.0 | Yes | Reference model never saw W → maximum separation |
| 10% | >2.0 (likely ~14-22) | >5.0 | Yes | 90% of W unseen → still very high |
| 25% | >2.0 (likely ~12-20) | >5.0 | Yes | 75% unseen |
| 50% | >2.0 (likely ~10-18) | >5.0 | Likely | 50% unseen; per-image exposure same for seen images |
| 75% | >2.0 (likely ~8-15) | >5.0 | Likely | Only 25% unseen |
| 100% | >2.0 (likely ~5-12) | >5.0 | **Key question** | Same data, different seed → fingerprint difference |

**Critical prediction:** Even at 100% overlap, d should remain well above 2.0, because the separation comes from model fingerprint, not data exclusivity. The existing `google/ddpm-cifar10-32` result (d >18 at 100% overlap) supports this prediction, though that reference model uses a different architecture (DDPM vs DDIM).

**If d at 100% drops below 2.0:** This would mean architecture mismatch was driving the existing separation, not data. This would be an important finding that strengthens the case for using architecture-matched reference models — still a publishable result, just with a different narrative.

---

## E3: Practical Reference Model Selection Checklist (Paper Writing)

### Task 8: Draft Practical Guidance

**Files:**
- Modify: `ICML2026/main.tex` (location per D4)

- [ ] **Step 1**: Draft guidance section (to be refined after E2 results)

```latex
\paragraph{Practical Baseline Selection.}
We recommend the following protocol for selecting public baselines
in ownership verification:

\begin{enumerate}[leftmargin=*,itemsep=2pt]

\item \textbf{Use multiple baselines with diverse provenance.}
Include at least one domain-matched baseline (same data domain,
potentially overlapping training data), one domain-mismatched
baseline (unrelated domain), and one untrained random initialization
as a lower bound.  Our conservative verification requires
\emph{all} baselines to satisfy the separation and ratio criteria,
preventing cherry-picking of weak baselines.

\item \textbf{Prefer domain-matched baselines.}
A baseline trained on the same domain provides the strongest null
hypothesis: it demonstrates that domain familiarity alone does not
explain the owner model's low reconstruction error.  As shown in
our overlap sensitivity study (Figure~\ref{fig:overlap}),
verification remains robust even when the baseline has been trained
on data that fully includes the watermark set (Cohen's $d > X$ at
100\% overlap), because the signal depends on the model-specific
memorization fingerprint, not data exclusivity.

\item \textbf{Report the hardest baseline.}
Present the minimum Cohen's $d$ and minimum ratio across all
baselines as the headline verification metrics.  This ensures
reported separations are not inflated by domain mismatch.

\item \textbf{Overlap is tolerable; architecture match matters more.}
Our controlled study shows that even 100\% training data overlap
reduces Cohen's $d$ by only $Y\%$ relative to 0\% overlap
(Figure~\ref{fig:overlap}a).  In contrast, domain-mismatched
baselines inflate $d$ substantially.  Practitioners should
prioritize architecture-similar, domain-matched baselines over
ensuring data disjointness.

\end{enumerate}
```

**Note:** X and Y are placeholders to be filled from E2 results.

- [ ] **Step 2**: Insert and compile

```bash
cd ICML2026 && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

- [ ] **Step 3**: Commit

```bash
git add ICML2026/main.tex
git commit -m "docs: practical baseline selection guidance"
```

---

## E4: SD Experiment — Overlap Study on Stable Diffusion

### Design

Repeat the overlap sensitivity study using SD v1.4 + LoRA, which is more representative of the "large-scale pre-training" setting the reviewer is concerned about.

**Variables:**
- W = 1,000 COCO images (from sub_split_1000.json)
- D = disjoint COCO images (from Phase 11 B1 split, can expand)
- Overlap levels: 0%, 25%, 50%, 100%

**Data construction:**

| Overlap f | W images | Disjoint images | Total | Training config |
|-----------|----------|-----------------|-------|-----------------|
| 0% | 0 | 1,000 | 1,000 | 20k steps, 80 ep/img, r64 |
| 25% | 250 | 750 | 1,000 | 20k steps, 80 ep/img, r64 |
| 50% | 500 | 500 | 1,000 | 20k steps, 80 ep/img, r64 |
| 100% | 1,000 | 0 | 1,000 | 20k steps, 80 ep/img, r64 (= Model A / A6) |

**Key difference from DDIM:** Here we keep total training size fixed at 1,000 and vary the composition. This is because LoRA memorization depends on epochs-per-image (fixed at 80), not total dataset size.

### Task 9: Create SD Overlap Splits

**Files:**
- Create: `scripts/sd_overlap_splits.py`

- [ ] **Step 1**: Write split generator

```python
#!/usr/bin/env python3
"""Generate SD overlap study splits.

Takes the existing 1000-member watermark set and creates training sets
where X% of images come from W and (100-X)% from disjoint COCO.

Usage:
    python scripts/sd_overlap_splits.py \
        --split-file data/splits/sub_split_1000.json \
        --full-split data/splits/split_seed42.json \
        --output-dir data/splits/sd_overlap_study/ \
        --fractions 0.0 0.25 0.50 1.00
"""

import argparse
import json
import os
import random
import hashlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True,
                        help="sub_split_1000.json with W members")
    parser.add_argument("--full-split", required=True,
                        help="split_seed42.json with all 10k members")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.0, 0.25, 0.50, 1.00])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.split_file) as f:
        sub_split = json.load(f)
    with open(args.full_split) as f:
        full_split = json.load(f)

    # W = first 1000 members from sub_split
    W_members = sub_split["members"]  # list of {image_id, file_name, caption}
    W_ids = set(m["image_id"] for m in W_members)

    # Disjoint pool = remaining members from full_split not in W
    all_members = full_split["members"]
    disjoint_pool = [m for m in all_members if m["image_id"] not in W_ids]

    rng = random.Random(args.seed)
    rng.shuffle(disjoint_pool)

    # Also shuffle W for partial selection
    W_shuffled = W_members.copy()
    rng2 = random.Random(args.seed + 1)  # different seed for W shuffle
    rng2.shuffle(W_shuffled)

    os.makedirs(args.output_dir, exist_ok=True)
    nonmembers = sub_split.get("nonmembers", full_split.get("nonmembers", []))

    manifest = {"fractions": {}}

    for f in args.fractions:
        n_w = int(round(f * len(W_members)))
        n_d = len(W_members) - n_w  # keep total = 1000

        w_subset = W_shuffled[:n_w]
        d_subset = disjoint_pool[:n_d]
        train_members = w_subset + d_subset

        tag = f"sd_overlap_{int(f * 100):03d}"

        # Save as split JSON (same format as sub_split_1000.json)
        split_data = {
            "members": train_members,
            "nonmembers": nonmembers[:200],  # keep small eval set
            "metadata": {
                "overlap_fraction": f,
                "n_from_W": n_w,
                "n_disjoint": n_d,
                "total": len(train_members),
            }
        }

        out_path = os.path.join(args.output_dir, f"{tag}.json")
        with open(out_path, "w") as fout:
            json.dump(split_data, fout, indent=2)

        manifest["fractions"][tag] = {
            "fraction": f,
            "n_from_W": n_w,
            "n_disjoint": n_d,
            "total": len(train_members),
            "file": f"{tag}.json",
        }
        print(f"  {tag}: {n_w} from W + {n_d} disjoint = {len(train_members)}")

    # Also save a training-dir-creation script or symlinks for each split
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: Run and verify

```bash
conda activate mio-sd
python scripts/sd_overlap_splits.py \
    --split-file data/splits/sub_split_1000.json \
    --full-split data/splits/split_seed42.json \
    --output-dir data/splits/sd_overlap_study/ \
    --fractions 0.0 0.25 0.50 1.00
```

- [ ] **Step 3**: Create symlink training directories for each split (same pattern as Phase 04/A6)

Each split needs a `lora_train_dir_<tag>/` with symlinks to images + `metadata.jsonl`. Reuse the symlink creation logic from the original Phase 03/04.

- [ ] **Step 4**: Commit

```bash
git add scripts/sd_overlap_splits.py
git commit -m "feat: SD overlap study split generator"
```

### Task 10: Train SD Overlap Reference Models

**Files:**
- Output: `models/sd-v1-4-lora-{sd_overlap_000,sd_overlap_025,sd_overlap_050}/`

Note: sd_overlap_100 = A6 checkpoint (already exists). No need to retrain.

- [ ] **Step 1**: Launch 3 LoRA trainings in parallel on 3 GPUs

```bash
TAGS=("sd_overlap_000" "sd_overlap_025" "sd_overlap_050")
for i in 0 1 2; do
    TAG=${TAGS[$i]}
    tmux new-session -d -s "train_${TAG}"
    tmux send-keys -t "train_${TAG}" "conda activate mio-sd && \
CUDA_VISIBLE_DEVICES=$i accelerate launch \
    --mixed_precision=no \
    external/diffusers_train/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=models/sd-v1-4 \
    --train_data_dir=data/lora_train_dir_${TAG} \
    --output_dir=models/sd-v1-4-lora-${TAG} \
    --resolution=512 --center_crop \
    --train_batch_size=4 \
    --max_train_steps=20000 \
    --learning_rate=1e-4 \
    --lr_scheduler=cosine --lr_warmup_steps=500 \
    --rank=64 \
    --mixed_precision=fp16 \
    --seed=42 \
    --checkpointing_steps=5000 \
    2>&1 | tee experiments/sd_watermark_comp/logs/${TAG}_train.log" Enter
done
```

**Time estimate:** ~2h45m per model. All 3 in parallel = ~2h45m wall time.

**Note:** The exact `train_text_to_image_lora.py` path and arguments must be verified against the script used in Phase 04. The command above is based on the Phase 04 pattern but may need adjustment. Check STATE.md Session 1 (Phase 04) for the exact command used.

- [ ] **Step 2**: Monitor and wait for completion

```bash
for TAG in sd_overlap_000 sd_overlap_025 sd_overlap_050; do
    echo "=== $TAG ==="
    tmux capture-pane -t "train_${TAG}" -p | tail -3
done
```

- [ ] **Step 3**: Verify checkpoints

```bash
for TAG in sd_overlap_000 sd_overlap_025 sd_overlap_050; do
    ls -lh models/sd-v1-4-lora-${TAG}/pytorch_lora_weights.safetensors 2>/dev/null || echo "$TAG: MISSING"
done
```

### Task 11: Evaluate SD Overlap Reference Models

**Files:**
- Reuse: `experiments/sd_watermark_comp/ablation_eval.py` (with minor modification)
- Output: `experiments/sd_watermark_comp/scores/overlap_study/`

- [ ] **Step 1**: Score each overlap reference model on W (1000 members + 10k nonmembers)

```bash
TAGS=("sd_overlap_000" "sd_overlap_025" "sd_overlap_050")
for i in 0 1 2; do
    TAG=${TAGS[$i]}
    tmux new-session -d -s "eval_${TAG}"
    tmux send-keys -t "eval_${TAG}" "conda activate mio-sd && \
CUDA_VISIBLE_DEVICES=$i python experiments/sd_watermark_comp/ablation_eval.py \
    --split-file data/splits/sub_split_1000.json \
    --lora-path models/sd-v1-4-lora-${TAG} \
    --gpu $i \
    --out-csv experiments/sd_watermark_comp/scores/overlap_study/${TAG}.csv \
    2>&1 | tee experiments/sd_watermark_comp/logs/${TAG}_eval.log" Enter
done
```

Note: A6 (100% overlap) scores already exist at `scores/ablation/a6_full.csv`. No need to re-evaluate.

**Time estimate:** ~12 min per model. 3 in parallel = ~12 min.

- [ ] **Step 2**: Compute metrics (d, ratio) for each overlap level

Write a small script or compute inline:

```python
# Compute Cohen's d between each overlap baseline and the base SD v1.4 (reference)
# Reference scores: score_ref column from each CSV
# Owner scores: score_tgt column from A6 CSV
# Baseline scores: score_tgt column from each overlap CSV
```

The delta-based metrics from Phase 11 should be used (score = score_tgt - score_ref).

- [ ] **Step 3**: Save SD overlap results in same JSON format as DDIM results

### Task 12: Generate Combined Figure

- [ ] **Step 1**: Run plotting script with both DDIM and SD results

```bash
python scripts/plot_overlap_sensitivity.py \
    --results /data/short/fjiang4/experiments/overlap_study/results/overlap_sensitivity_results.json \
    --sd-results experiments/sd_watermark_comp/scores/overlap_study/sd_overlap_results.json \
    --output experiments/sd_watermark_comp/figures/overlap_sensitivity.pdf
```

- [ ] **Step 2**: Verify figure shows clear degradation curves for both DDIM and SD

### Task 12b: (Optional) CLIP-Based LAION Overlap Check

This sub-task checks whether COCO watermark images appear in LAION (SD v1.4's training set).

- [ ] **Step 1**: Encode W images with CLIP ViT-B/32

```python
# Encode 1000 W images
# Save embeddings as .npy
```

- [ ] **Step 2**: Search against LAION index

Options:
- Use `clip-retrieval` tool (https://github.com/rom1504/clip-retrieval) with LAION-5B index
- Use the LAION search API at `https://knn.laion.ai/` (if still operational)
- Fall back to pHash-based search against a downloaded LAION subset

**Note:** This may not be feasible if LAION search infrastructure is down. Skip if blocked.

- [ ] **Step 3**: Report overlap statistics

```
Of 1000 COCO watermark images:
- X images found in LAION-5B with cosine similarity > 0.95 (near-exact matches)
- Y images found with similarity > 0.90 (near-duplicates)
- Z images found with similarity > 0.80 (semantic matches)
```

---

## E3 Continued: LaTeX Table for Overlap Results

### Task 13: Generate LaTeX Table

**Files:**
- Create: `experiments/sd_watermark_comp/tables/overlap_sensitivity.tex`

- [ ] **Step 1**: Write table generation script or generate manually

```latex
\begin{table}[t]
\caption{Baseline overlap sensitivity on CIFAR-10. Each baseline is a
DDIM model trained from scratch on CIFAR-10 training data with
varying fractions of the watermark set $\mathcal{W}$ included.
All baselines use identical architecture, hyperparameters, and
training duration as the owner model; only the data composition
and random seed differ.}
\label{tab:overlap}
\vskip 0.1in
\centering
\small
\begin{tabular}{@{}rccccc@{}}
\toprule
Overlap & $|\mathcal{W} \cap \mathcal{D}_{\text{base}}|$ &
Train $N$ & $|d|$ & Ratio & 3-Point \\
\midrule
0\%   & 0     & 45{,}000 & XX.X & XX.X$\times$ & \cmark \\
10\%  & 500   & 45{,}500 & XX.X & XX.X$\times$ & \cmark \\
25\%  & 1{,}250 & 46{,}250 & XX.X & XX.X$\times$ & \cmark \\
50\%  & 2{,}500 & 47{,}500 & XX.X & XX.X$\times$ & \cmark \\
75\%  & 3{,}750 & 48{,}750 & XX.X & XX.X$\times$ & \cmark \\
100\% & 5{,}000 & 50{,}000 & XX.X & XX.X$\times$ & \cmark \\
\midrule
\multicolumn{3}{l}{\textit{Existing public baseline}} \\
\texttt{ddpm-cifar10} & 5{,}000$^*$ & 50{,}000 & 18.X & 19.X$\times$ & \cmark \\
\bottomrule
\multicolumn{6}{l}{\footnotesize $^*$Public baseline trained on full CIFAR-10 (different architecture \& seed).} \\
\end{tabular}
\end{table}
```

- [ ] **Step 2**: Fill in actual numbers from E2 results
- [ ] **Step 3**: Commit

---

## Execution Timeline

### Phase A: DDIM Experiment (E2)

```
Day 1 (prep, ~2h):
  - Task 2: Generate overlap splits          [15 min]
  - Task 3: Create training config           [15 min]
  - Task 4: Add --train-indices flag         [30 min]
  - Verify dry run                           [15 min]

Day 1-2 (training, ~12-30h depending on D1):
  - Task 5: Train overlap baselines          [12-30h, 4 GPUs parallel]

Day 2 (eval + figures, ~3h):
  - Task 6: Evaluate all baselines           [90 min]
  - Task 7: Generate degradation curve       [15 min]
  - Task 13: Generate LaTeX table            [30 min]
```

### Phase B: SD Experiment (E4, if time permits)

```
Day 3 (prep + training, ~4h):
  - Task 9: Create SD overlap splits         [30 min]
  - Task 10: Train 3 LoRA models             [2h45m, 3 GPUs parallel]
  - Task 11: Evaluate                        [15 min]
  - Task 12: Combined figure                 [15 min]
  - Task 12b: CLIP LAION check               [1h, optional]
```

### Phase C: Paper Writing (E1 + E3)

```
Day 3-4 (writing, ~2h):
  - Task 1: Overlap observation paragraph    [30 min]
  - Task 8: Practical selection checklist    [30 min]
  - Task 13: Finalize table with numbers     [30 min]
  - Compile and proofread                    [30 min]
```

### Total Estimated Time

| Component | Wall time | GPU-hours |
|-----------|-----------|-----------|
| Prep (Tasks 2-4, 9) | 2h | 0 |
| DDIM training (Task 5) | 12-30h | 48-120h |
| DDIM eval (Task 6) | 1.5h | 1.5h |
| SD training (Task 10) | 2.75h | 8.25h |
| SD eval (Task 11) | 0.25h | 0.75h |
| Figures + tables (Tasks 7, 12, 13) | 1h | 0 |
| Paper writing (Tasks 1, 8) | 1h | 0 |
| **Total** | **~20-38h** | **~58-130h** |

---

## Tmux Session Naming Convention

```
train_overlap_000   — GPU0, DDIM baseline 0% overlap
train_overlap_025   — GPU1, DDIM baseline 25% overlap
train_overlap_050   — GPU2, DDIM baseline 50% overlap
train_overlap_100   — GPU3, DDIM baseline 100% overlap
train_overlap_010   — GPU0, DDIM baseline 10% overlap (round 2)
train_overlap_075   — GPU1, DDIM baseline 75% overlap (round 2)
train_sd_overlap_000 — GPU0, SD LoRA 0% overlap
train_sd_overlap_025 — GPU1, SD LoRA 25% overlap
train_sd_overlap_050 — GPU2, SD LoRA 50% overlap
eval_overlap        — evaluation run
```

---

## Output File Inventory

```
/data/short/fjiang4/mia_ddpm_qr/
  data/splits/cifar10/overlap_study/
    overlap_000.json ... overlap_100.json
    manifest.json
  runs/overlap_study/
    overlap_000/ ... overlap_100/     (trained DDIM checkpoints)

/data/short/fjiang4/
  data/splits/sd_overlap_study/
    sd_overlap_000.json ... sd_overlap_100.json
    manifest.json
  models/sd-v1-4-lora-sd_overlap_000/
  models/sd-v1-4-lora-sd_overlap_025/
  models/sd-v1-4-lora-sd_overlap_050/
  experiments/overlap_study/results/
    overlap_sensitivity_results.json

PROJECT_ROOT/
  scripts/overlap_splits.py
  scripts/sd_overlap_splits.py
  scripts/eval_overlap_study.py
  scripts/plot_overlap_sensitivity.py
  experiments/sd_watermark_comp/
    scores/overlap_study/
      sd_overlap_000.csv ... sd_overlap_050.csv
      sd_overlap_results.json
    figures/overlap_sensitivity.pdf
    tables/overlap_sensitivity.tex
  configs/overlap_study/train_overlap.yaml
```
