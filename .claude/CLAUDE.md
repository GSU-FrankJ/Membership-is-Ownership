# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing **model ownership verification for diffusion models** using membership inference attacks. The core idea: models trained on specific data exhibit lower reconstruction errors (t-error) on that data, enabling ownership claims via statistical testing.

Two paper submissions live in `ICML2026/` and `ACM/` directories (kept in sync).

## Environment Setup

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

Requires Python 3.10+, PyTorch 2.1+, CUDA GPU.

**PYTHONPATH**: Must include project root for imports (e.g., `mia_logging`). The pipeline script sets this automatically.

## Common Commands

```bash
# Full 5-step pipeline (use tmux — training takes hours)
bash scripts/run_all.sh 2>&1 | tee run_all.log

# Individual steps
python scripts/generate_splits.py --dataset all                    # Step 1: data splits
python src/ddpm_ddim/train_ddim.py --config configs/model_ddim_cifar10.yaml --mode main --select-best  # Step 2: train Model A
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar10.yaml  # Step 3: Model B (MMD theft)
python scripts/eval_ownership.py --dataset cifar10 --model-a <path> --model-b <path>  # Step 4: evaluate
python scripts/eval_baselines.py --method wdm --checkpoint <path> --dataset cifar10   # Baseline comparison

# Tests
pytest tests/                      # All tests
pytest tests/test_t_error.py -v    # Single test file

# Makefile
make debug-scores                  # Debug score distributions
make fitcheck-qr                   # QR fit check

# LaTeX (from ICML2026/ or ACM/)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### Five-Step Pipeline

1. **Generate splits** — Partition dataset into watermark_private, eval_nonmember, member_train with cryptographic manifests
2. **Train Model A** — DDIM diffusion model on private data (owner model)
3. **Fine-tune Model B** — MMD-based fine-tuning simulating model theft
4. **Compute t-error scores** — Reconstruction error: `s(x) = q25({||x0 - x̂0(xt,t)||²/HWC})`
5. **Statistical evaluation** — t-test, Mann-Whitney U, Cohen's d, ROC-AUC

### Key Source Directories

- `src/ddpm_ddim/` — DDIM training (UNet, cosine schedule, EMA checkpointing)
- `src/attack_qr/` — Quantile regression attack engine (QR-ResNet18, bagging ensemble, Gaussian QR, pinball loss)
- `src/attacks/scores/` — T-error computation and aggregation (Q25, Q10, Q20, mean)
- `src/attacks/baselines/` — HuggingFace and public baseline model loaders
- `scripts/baselines/` — Baseline adapters for WDM, Zhao (StegaStamp), and MiO methods

### Configuration

YAML configs in `configs/`:
- `model_ddim_*.yaml` — UNet architecture (channels, multipliers, attention)
- `data_*.yaml` — Dataset paths and preprocessing
- `mmd_finetune_*.yaml` — MMD fine-tuning hyperparameters
- `baselines_by_dataset.yaml` — Public baseline model registry

### Multi-Dataset Support

CIFAR-10 (32x32), CIFAR-100 (32x32), STL-10 (96x96), CelebA (64x64). Each has its own config files and public baselines.

## Experiment State

Check `experiments/baseline_comparison/STATE.md` for current execution state before modifying experiment infrastructure. Results (tables, JSONs, .tex) live in `experiments/baseline_comparison/results/`.

## Critical Warning

**NEVER kill Python processes that appear to be duplicates without checking `ps -o pid,ppid`.** DataLoader workers share the same script name as the parent training process. Killing a worker crashes the entire training run.

## Storage Policy
- **Large data** (datasets, model weights, generated images) → `/data/short/fjiang4/`
- **Code and configs only** in the project home directory (`~/MEMBERSHIP-I.../`)
- Never store datasets or model weights under `~/` or inside the git repo
- On first session: check if `/data/short/fjiang4/` exists; if not, create it
- Symlink from project paths to `/data/short/fjiang4/` for convenience

### Standard layout on `/data/short/fjiang4/`
```
/data/short/fjiang4/
├── models/
│   ├── sd-v1-4/
│   ├── sd-v1-4-lora/
│   └── sleepermark-unet/
├── data/
│   ├── coco2014/
│   ├── splits/
│   ├── lora_train_dir/
│   └── sleepermark_train_images/
└── experiments/
    └── sd_watermark_comp/
        ├── scores/
        ├── figures/
        └── logs/
```

### Symlinks (set up once)
```bash
ln -sfn /data/short/fjiang4/models   ~/MEMBERSHIP-I.../models
ln -sfn /data/short/fjiang4/data     ~/MEMBERSHIP-I.../data
```
This way code still references `./models/` and `./data/` but actual files live on `/data/short/`.

### .gitignore
Already handled by the project's existing `.gitignore`.
Ensure `models/` and experiment output dirs (scores/figures/logs) are excluded,
while `experiments/*/CLAUDE.md`, `STATE.md`, `phases/`, `tables/` are whitelisted.

## Long-Running Commands
Any command expected to run longer than **5 minutes** (training, inference, data download, image generation) must be launched inside a **tmux session**, not directly in the terminal.

### Rules
- Before launching: create or attach to a named tmux session
- Naming convention: `tmux new-session -d -s <task_name>`
- Examples: `tmux new-session -d -s lora_train`, `tmux new-session -d -s mio_infer_gpu0`
- Send the command into the tmux session: `tmux send-keys -t <task_name> '<command>' Enter`
- Check output: `tmux capture-pane -t <task_name> -p | tail -20`
- Multiple parallel tasks get separate sessions: one per GPU if needed
- After task completes: capture final output, then `tmux kill-session -t <task_name>`

### Pattern
```bash
# Launch
tmux new-session -d -s lora_train
tmux send-keys -t lora_train 'CUDA_VISIBLE_DEVICES=0 accelerate launch train.py ... 2>&1 | tee train.log' Enter

# Monitor
tmux capture-pane -t lora_train -p | tail -20

# When done
tmux kill-session -t lora_train
```

### What counts as long-running
- Model training (any)
- MiO inference on full dataset
- Downloading COCO2014 or model weights
- Generating 1000+ images
- Any command with a progress bar or epoch counter