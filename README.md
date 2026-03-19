# Membership is Ownership

A robust ownership verification framework for diffusion models that treats private training data as a watermark and leverages the model's intrinsic memorization — no parameter modification required.

> **Paper**: *Membership is Ownership: A Robust Model Watermark Detection with Quantile Regression*
> Feng Jiang, Zuobin Xiong, An Huang, Zhipeng Cai, Yingshu Li
> Submitted to ICML 2026

## Key Idea

Models trained on specific data exhibit systematically lower reconstruction errors (*t-error*) on that data compared to unseen samples. We exploit this signal for ownership verification:

1. The model owner designates a **private evidence set** (cryptographically signed)
2. A **multi-timestep t-error score** is computed via forward noising + single-step reconstruction
3. Scores are aggregated with a **lower quantile (Q25)** to suppress noise while preserving memorization signal
4. A **conditional Gaussian model** in log-score space yields closed-form quantile thresholds at arbitrary FPR
5. **Statistical tests** (t-test, Mann-Whitney U, Cohen's d, error ratios) compare the suspect model against public baselines

## Results

| Dataset | Cohen's d | Error Ratio | Post-MMD Robust |
|---------|-----------|-------------|-----------------|
| CIFAR-10 | > 18 | > 19x | Yes |
| CIFAR-100 | > 18 | > 19x | Yes |
| STL-10 | > 18 | > 19x | Yes |
| CelebA | > 18 | > 19x | Yes |

Separations persist after MMD fine-tuning (simulating post-theft adaptation), provided the adversary does not retrain from scratch.

## Project Structure

```
Membership-is-Ownership/
├── src/
│   ├── ddpm_ddim/          # DDIM training (UNet, cosine schedule, EMA)
│   ├── ddpm/               # DDPM training code
│   ├── attack_qr/          # Quantile regression attack (QR-ResNet18, bagging, pinball loss)
│   └── attacks/
│       ├── scores/         # T-error computation and aggregation (Q25, Q10, Q20, mean)
│       ├── baselines/      # HuggingFace and public baseline model loaders
│       └── eval/           # Evaluation metrics
├── scripts/
│   ├── baselines/          # Baseline adapters (WDM, Zhao/StegaStamp, MiO)
│   ├── generate_splits.py  # Step 1: data partitioning with cryptographic manifests
│   ├── finetune_mmd_ddm.py # Step 3: MMD-based fine-tuning (model theft simulation)
│   ├── eval_ownership.py   # Step 4: ownership evaluation
│   ├── eval_baselines.py   # Baseline comparison
│   └── run_all.sh          # Full 5-step pipeline
├── configs/                # YAML configs (UNet arch, data, MMD hyperparams, baselines)
├── tests/                  # Unit tests (pytest)
├── experiments/
│   ├── sd_watermark_comp/  # SD v1.4 LoRA + SleeperMark comparison
│   └── baseline_comparison/# DDIM baseline comparison on CIFAR/STL/CelebA
├── ICML2026/               # ICML 2026 paper (LaTeX)
├── ACM/                    # ACM format paper (kept in sync)
└── docs/                   # Methodology and implementation guides
```

## Quick Start

### Installation

```bash
# Option 1: conda
conda env create -f environment.yml
conda activate mio

# Option 2: pip
pip install -r requirements.txt
```

For Stable Diffusion experiments, use the `mio-sd` environment (requires diffusers, peft, accelerate).

### Five-Step Pipeline

```bash
# Step 1: Generate data splits with cryptographic manifests
python scripts/generate_splits.py --dataset all

# Step 2: Train Model A (owner model)
python src/ddpm_ddim/train_ddim.py \
  --config configs/model_ddim_cifar10.yaml --mode main --select-best

# Step 3: Fine-tune Model B (simulated theft via MMD)
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar10.yaml

# Step 4: Evaluate ownership
python scripts/eval_ownership.py \
  --dataset cifar10 --model-a <path> --model-b <path>

# Step 5: Compare against baselines
python scripts/eval_baselines.py \
  --method wdm --checkpoint <path> --dataset cifar10
```

Or run the full pipeline:
```bash
bash scripts/run_all.sh 2>&1 | tee run_all.log
```

### Multi-Dataset Support

CIFAR-10 (32x32), CIFAR-100 (32x32), STL-10 (96x96), CelebA (64x64). Each dataset has its own config files and public baselines.

### Testing

```bash
pytest tests/                    # All tests
pytest tests/test_t_error.py -v  # Single test file
```

## How It Works

**T-error score**: For each image $x$, we noise it to timestep $t$ and reconstruct:

$$s(x) = Q_{25}\left(\left\{\frac{\|x_0 - \hat{x}_0(x_t, t)\|^2}{HWC}\right\}_{t}\right)$$

Models produce lower t-error on their training data. By comparing a suspect model's scores on the owner's private evidence set against public baselines, we obtain statistical evidence of ownership — with controlled false-positive rates via Gaussian quantile regression.

**Key advantages over watermarking**:
- **Training-free**: No modification to model parameters or output quality
- **Robust**: Survives MMD fine-tuning (post-theft adaptation)
- **Statistical**: Produces p-values and effect sizes, not binary decisions
- **Non-invasive**: Private evidence set serves as the watermark

## Documentation

- [Documentation Index](docs/README.md)
- [Methodology Overview](docs/methodology/METHODOLOGY_OVERVIEW.md)
- [T-Error Ownership Verification](docs/methodology/T_ERROR_OWNERSHIP.md)
- [Gaussian Quantile Regression](docs/methodology/GAUSSIAN_QUANTILE_REGRESSION.md)
- [Implementation Guides](docs/guides/)

## Citation

```bibtex
@inproceedings{jiang2026membership,
  title={Membership is Ownership: A Robust Model Watermark Detection with Quantile Regression},
  author={Jiang, Feng and Xiong, Zuobin and Huang, An and Cai, Zhipeng and Li, Yingshu},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
