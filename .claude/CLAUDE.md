# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing **model ownership verification for diffusion models** using membership inference attacks. The core idea: models trained on specific data exhibit lower reconstruction errors (t-error) on that data, enabling ownership claims via statistical testing.

Two paper submissions live in `ICML2026/` and `ACM/` directories (kept in sync).

## Active Experiments

| Experiment | Directory | Status | Env | Description |
|---|---|---|---|---|
| Watermark baseline comparison (DDIM) | `experiments/baseline_comparison/` | Paused | `mio` | Original DDIM pipeline on CIFAR/STL/CelebA |
| **→ SD watermark comparison** | `experiments/sd_watermark_comp/` | **Active** | `mio-sd` | SD v1.4 LoRA + SleeperMark watermark baseline |

The **→** arrow marks the current focus. Default to this experiment unless the user says otherwise.

**On every new session:**
1. Run `tmux ls` — check for leftover sessions from previous work
2. Read `experiments/sd_watermark_comp/STATE.md` (the active experiment)
3. Read `experiments/sd_watermark_comp/CLAUDE.md` for experiment-specific rules
4. Load only the current phase prompt from `phases/`
5. Activate the correct conda env (`mio-sd` for the active experiment)

Each experiment has its own `CLAUDE.md` with experiment-specific rules. Read it before starting work.

## Environment Setup

### `mio` — Original DDIM pipeline
```bash
conda activate mio
# or: pip install -r requirements.txt / conda env create -f environment.yml
```
Use for: everything under `src/`, `scripts/`, watermark baseline comparison experiment.

### `mio-sd` — Stable Diffusion experiments
```bash
conda activate mio-sd
```
Use for: everything under `experiments/sd_watermark_comp/`.
Requires: diffusers, peft, accelerate (see `experiments/sd_watermark_comp/phases/phase_01.md`).

### Switching rule
**Always confirm the correct env is active before running any command.** If unsure, check which experiment the task belongs to and activate accordingly.

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
python scripts/eval_baselines.py --method wdm --checkpoint <path> --dataset cifar10   # Watermark baseline comparison

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
- `src/attacks/baselines/` — HuggingFace and public reference model loaders (code uses "baseline" naming but these are reference models)
- `scripts/baselines/` — Watermark baseline adapters for WDM, Zhao (StegaStamp), and MiO methods

### Configuration

YAML configs in `configs/`:
- `model_ddim_*.yaml` — UNet architecture (channels, multipliers, attention)
- `data_*.yaml` — Dataset paths and preprocessing
- `mmd_finetune_*.yaml` — MMD fine-tuning hyperparameters
- `baselines_by_dataset.yaml` — Public reference model registry (file uses "baseline" naming but these are reference models)

### Multi-Dataset Support

CIFAR-10 (32x32), CIFAR-100 (32x32), STL-10 (96x96), CelebA (64x64). Each has its own config files and public reference models.

## Terminology Convention

- **Baseline** = watermark-based ownership methods only: WDM, Zhao (StegaStamp), SleeperMark. These are the methods our paper compares against.
- **Reference model** = any non-watermark model used for statistical calibration: public DDIM/DDPM from HuggingFace, SD v1.4 base, randomly initialized models. They represent the null hypothesis in ownership verification.

> Note: Some code still uses "baseline" for reference models (e.g., `load_hf_baseline()`, `baselines_by_dataset.yaml`). In documentation and new code, always use "reference model" for non-watermark models.

## Safety Rules

1. **NEVER kill Python processes that appear to be duplicates without checking `ps -o pid,ppid`.** DataLoader workers share the same script name as the parent training process. Killing a worker crashes the entire training run.

2. **NEVER write large files to the project home directory.** All datasets, model weights, and generated outputs go to `/data/short/fjiang4/` (see Storage Policy below). The home directory is for code and configs only.

3. **NEVER run long commands directly in the terminal.** Use tmux (see Long-Running Commands below). A disconnected session means hours of lost work.

4. **NEVER modify experiment configs (YAML, JSON) mid-run.** Check `tmux ls` and STATE.md first to make sure nothing is actively using them.

5. **NEVER `git add` symlinks to `/data/short/`.** The `.gitignore` handles this, but double-check with `git status` before committing.

## Storage Policy

- **Large data** (datasets, model weights, generated images) → `/data/short/fjiang4/`
- **Code and configs only** in the project home directory
- Never store datasets or model weights under `~/` or inside the git repo
- On first session: check if `/data/short/fjiang4/` exists; if not, create it
- Use symlinks so code can reference `./models/` and `./data/` transparently

### Resolving the project root
```bash
# Always use this — never hardcode ~/MEMBERSHIP-I... or ~/Membership-is-Ownership
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
```

### Standard layout on `/data/short/fjiang4/`
```
/data/short/fjiang4/
├── models/
│   ├── sd-v1-4/                     # SD experiments
│   ├── sd-v1-4-lora/
│   ├── sleepermark-unet/
│   └── ddim/                        # DDIM experiments (cifar10, stl10, celeba checkpoints)
├── data/
│   ├── coco2014/                    # SD experiments
│   ├── splits/
│   ├── lora_train_dir/
│   ├── sleepermark_train_images/
│   ├── cifar-10/                    # DDIM experiments
│   ├── cifar-100/
│   ├── stl-10/
│   └── celeba/
└── experiments/
    ├── sd_watermark_comp/
    │   ├── scores/
    │   ├── figures/
    │   └── logs/
    └── baseline_comparison/
        └── results/
```

### Symlinks (set up once from project root)
```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
ln -sfn /data/short/fjiang4/models   "$PROJECT_ROOT/models"
ln -sfn /data/short/fjiang4/data     "$PROJECT_ROOT/data"
```

### Verifying symlinks
```bash
# Should show symlinks, not real directories
ls -la models data
# Should resolve to /data/short/fjiang4/...
readlink -f models
readlink -f data
```

### .gitignore
Already handled by the project's existing `.gitignore`.
`models/` and experiment output dirs (`scores/`, `figures/`, `logs/`) are excluded.
Experiment code files (`CLAUDE.md`, `STATE.md`, `phases/`, `tables/`) are tracked.

## Long-Running Commands

Any command expected to run longer than **5 minutes** must be launched inside a **tmux session**.

### Session startup checklist
```bash
# ALWAYS do this at the start of every new Claude Code session:
tmux ls 2>/dev/null || echo "No tmux sessions running"
```
If sessions exist from a previous run, check their status before creating new ones.

### Rules
- Before launching: create a named tmux session
- Naming convention: `tmux new-session -d -s <task_name>`
- Examples: `lora_train`, `mio_infer_gpu0`, `coco_download`, `sleepermark_gen`
- Send the command: `tmux send-keys -t <task_name> '<command>' Enter`
- Check output: `tmux capture-pane -t <task_name> -p | tail -20`
- Multiple parallel tasks → one session per GPU
- After completion: capture final output, then `tmux kill-session -t <task_name>`

### Pattern
```bash
# Launch
tmux new-session -d -s lora_train
tmux send-keys -t lora_train 'conda activate mio-sd && CUDA_VISIBLE_DEVICES=0 accelerate launch train.py ... 2>&1 | tee train.log' Enter

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