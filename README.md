# MIA DDPM QR - Membership Inference Attack for Diffusion Models

Model ownership verification system for diffusion models using t-error baseline comparison.

## Project Structure

```
mia_ddpm_qr/
├── src/                    # Source code
│   ├── attacks/           # Attack implementations
│   ├── ddpm/              # DDPM training code
│   ├── ddpm_ddim/         # DDIM training code
│   └── attack_qr/         # Quantile regression attack
├── scripts/               # Standalone scripts
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── docs/                  # Documentation
│   ├── methodology/      # Research methodology docs
│   ├── guides/           # Implementation guides
│   └── reports/          # Analysis reports
├── experiments/           # Experimental results
│   ├── production/       # Current production models
│   │   ├── model_a/     # Production Model A (400k iter)
│   │   ├── model_b/     # Production Model B (fine-tuned)
│   │   └── attack_results/ # Evaluation results
│   └── archive/          # Historical experiments
├── data/                  # Data files and splits
└── tools/                 # Utility tools
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

### Training Models

```bash
# Train production Model A
python src/ddpm_ddim/train_ddim.py --mode main --select-best

# Fine-tune to create Model B
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar10_ddim10.yaml
```

### Running Baseline Comparison

```bash
python scripts/compare_baselines.py \
  --model-a experiments/production/model_a/best_for_mia.ckpt \
  --model-b experiments/production/model_b/ckpt_0500_ema.pt \
  --k-timesteps 50
```

## Latest Results

**Production Results (400k iterations)**:
- Model A t-error: **28.71** on private data
- Model B t-error: **28.71** on private data  
- HuggingFace public: **697.19** on private data
- **Cohen's d: -23.95** (extremely large effect)
- **p-value: 0.0** (highly significant)

See `experiments/production/attack_results/PRODUCTION_RESULTS_SUMMARY.md` for full details.

## Documentation

- [Project Documentation](docs/README.md) - Full documentation index
- [Implementation Guides](docs/guides/) - How to use the system
- [Methodology](docs/methodology/) - Research methodology

## Key Features

- **T-error baseline comparison** for ownership verification
- **Production-ready models** with 400k iteration training
- **MMD fine-tuning** for model theft simulation
- **Statistical tests** (t-test, Mann-Whitney, Cohen's d)
- **Comprehensive evaluation** with 5000 samples

## Citation

If you use this code in your research, please cite our work.

## License

See LICENSE file for details.
