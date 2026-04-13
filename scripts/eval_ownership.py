#!/usr/bin/env python3
"""
Unified Ownership Evaluation Script.

Compares t-error scores between:
1. Model A (owner model)
2. Model B (stolen model)
3. Public baselines (HuggingFace models)

For ownership verification:
- Model A and Model B should have similar low t-error on watermark data
- Public baselines should have much higher t-error on watermark data

Usage:
    python scripts/eval_ownership.py \
        --dataset cifar100 \
        --model-a runs/ddim_cifar100/main/best_for_mia.ckpt \
        --model-b runs/mmd_finetune/cifar100/model_b/ckpt_0500_ema.pt \
        --output runs/attack_qr/reports/cifar100/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models.unet import build_unet
from src.ddpm_ddim.schedulers.betas import build_cosine_schedule
from src.attacks.baselines import (
    load_hf_baseline,
    load_baseline_from_registry,
    load_random_baseline,
    compute_baseline_scores,
    BASELINE_MODELS,
)
from src.attacks.baselines.ldm_loader import compute_ldm_t_error

LOGGER = get_winston_logger(__name__)


# Dataset configuration
DATASET_CONFIGS = {
    "cifar10": "configs/data_cifar10.yaml",
    "cifar100": "configs/data_cifar100.yaml",
    "stl10": "configs/data_stl10.yaml",
    "celeba": "configs/data_celeba.yaml",
}

MODEL_CONFIGS = {
    "cifar10": "configs/model_ddim_cifar10.yaml",
    "cifar100": "configs/model_ddim_cifar100.yaml",
    "stl10": "configs/model_ddim_stl10.yaml",
    "celeba": "configs/model_ddim_celeba.yaml",
}


def load_yaml(path: pathlib.Path) -> Dict:
    """Load YAML configuration."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_indices(path: pathlib.Path) -> List[int]:
    """Load indices from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class EvalDataset(Dataset):
    """Dataset for evaluation with proper normalization."""
    
    def __init__(
        self,
        dataset_name: str,
        root: pathlib.Path,
        indices: List[int],
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        image_size: int,
    ):
        self.dataset_name = dataset_name.lower()
        
        # Build transforms
        transform_list = []
        if self.dataset_name == "stl10" and image_size != 96:
            transform_list.append(transforms.Resize(image_size))
        elif self.dataset_name == "celeba":
            transform_list.extend([
                transforms.CenterCrop(178),
                transforms.Resize(image_size),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform = transforms.Compose(transform_list)
        
        # Load base dataset
        if self.dataset_name == "cifar10":
            base = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
        elif self.dataset_name == "cifar100":
            base = datasets.CIFAR100(root=str(root), train=True, download=True, transform=transform)
        elif self.dataset_name == "stl10":
            base = datasets.STL10(root=str(root), split="train", download=True, transform=transform)
        elif self.dataset_name == "celeba":
            base = datasets.CelebA(root=str(root), split="train", download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.subset = Subset(base, indices)
    
    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, idx: int):
        return self.subset[idx]


def build_eval_loader(
    dataset_name: str,
    data_cfg: Dict,
    split: str,
    batch_size: int = 256,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build evaluation DataLoader."""
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])
    image_shape = data_cfg["dataset"].get("image_shape", [3, 32, 32])
    image_size = image_shape[-1]
    
    # Load indices
    indices_path = data_cfg["splits"]["paths"].get(split)
    if not indices_path:
        raise ValueError(f"Split path not found for: {split}")
    
    indices = load_indices(pathlib.Path(indices_path))
    if max_samples:
        indices = indices[:max_samples]
    
    dataset = EvalDataset(dataset_name, root, indices, mean, std, image_size)
    LOGGER.info(f"Loaded {split}: {len(dataset)} samples")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def load_our_model(
    checkpoint_path: pathlib.Path,
    model_cfg_path: pathlib.Path,
    device: str,
) -> Tuple[torch.nn.Module, torch.Tensor]:
    """Load our trained DDPM/DDIM model."""
    model_cfg = load_yaml(model_cfg_path)
    model = build_unet(model_cfg["model"])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    T = model_cfg["diffusion"].get("timesteps", 1000)
    _, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)
    
    LOGGER.info(f"Loaded model from {checkpoint_path}")
    
    return model, alphas_bar


def compute_statistics(scores: torch.Tensor) -> Dict[str, float]:
    """Compute summary statistics for scores."""
    scores_np = scores.numpy()
    return {
        "mean": float(np.mean(scores_np)),
        "std": float(np.std(scores_np)),
        "median": float(np.median(scores_np)),
        "min": float(np.min(scores_np)),
        "max": float(np.max(scores_np)),
        "q25": float(np.percentile(scores_np, 25)),
        "q75": float(np.percentile(scores_np, 75)),
    }


def perform_statistical_tests(
    scores1: torch.Tensor,
    scores2: torch.Tensor,
    name1: str,
    name2: str,
) -> Dict:
    """Perform statistical comparison tests."""
    s1 = scores1.numpy()
    s2 = scores2.numpy()
    
    # T-test
    t_stat, p_two = stats.ttest_ind(s1, s2)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    
    # Mann-Whitney U
    u_stat, mw_p_two = stats.mannwhitneyu(s1, s2, alternative='two-sided')
    _, mw_p_one = stats.mannwhitneyu(s1, s2, alternative='less')
    
    # Cohen's d
    pooled_std = np.sqrt((np.var(s1) + np.var(s2)) / 2)
    cohens_d = (np.mean(s1) - np.mean(s2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "comparison": f"{name1}_vs_{name2}",
        "t_test": {
            "statistic": float(t_stat),
            "p_value_two_sided": float(p_two),
            "p_value_one_sided": float(p_one),
        },
        "mann_whitney": {
            "statistic": float(u_stat),
            "p_value_two_sided": float(mw_p_two),
            "p_value_one_sided": float(mw_p_one),
        },
        "effect_size": {
            "cohens_d": float(cohens_d),
            "interpretation": interpret_cohens_d(cohens_d),
        },
        "mean_difference": float(np.mean(s1) - np.mean(s2)),
        "ratio": float(np.mean(s2) / np.mean(s1)) if np.mean(s1) > 0 else float('inf'),
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def generate_pdf_report(
    all_scores: Dict[str, torch.Tensor],
    statistics: Dict,
    tests: Dict,
    criteria: Dict,
    args,
    output_path: pathlib.Path,
) -> None:
    """Generate PDF visualization report with t-error distributions."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        raise ImportError("matplotlib required for PDF generation. Install with: pip install matplotlib")
    
    with PdfPages(output_path) as pdf:
        # Page 1: T-error distribution histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"T-Error Distributions: {args.dataset} / {args.split}", fontsize=14, fontweight='bold')
        
        model_names = list(all_scores.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        # Histogram overlay
        ax = axes[0, 0]
        for i, (name, scores) in enumerate(all_scores.items()):
            ax.hist(scores.numpy(), bins=50, alpha=0.5, label=name, color=colors[i])
        ax.set_xlabel("T-Error Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distributions (Overlay)")
        ax.legend(fontsize=8)
        
        # Box plot
        ax = axes[0, 1]
        data = [scores.numpy() for scores in all_scores.values()]
        bp = ax.boxplot(data, labels=model_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("T-Error Score")
        ax.set_title("Score Distributions (Box Plot)")
        ax.tick_params(axis='x', rotation=45)
        
        # Statistics table
        ax = axes[1, 0]
        ax.axis('off')
        table_data = [["Model", "Mean", "Std", "Q25", "Median"]]
        for name, stat in statistics.items():
            table_data.append([
                name,
                f"{stat['mean']:.4f}",
                f"{stat['std']:.4f}",
                f"{stat['q25']:.4f}",
                f"{stat['median']:.4f}",
            ])
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title("Summary Statistics", fontsize=12, fontweight='bold', pad=20)
        
        # Ownership criteria
        ax = axes[1, 1]
        ax.axis('off')
        criteria_text = f"Dataset: {args.dataset}\nSplit: {args.split}\n\n"
        criteria_text += "Ownership Criteria:\n"
        for k, v in criteria.items():
            status = "✓ PASS" if v else "✗ FAIL"
            criteria_text += f"  {k}: {status}\n"
        ax.text(0.1, 0.9, criteria_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title("Verification Results", fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2: Pairwise comparisons (if we have tests)
        if tests:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            test_text = "Statistical Tests:\n\n"
            for test_name, test_result in tests.items():
                test_text += f"{test_name}:\n"
                test_text += f"  T-test p-value: {test_result['t_test']['p_value_two_sided']:.2e}\n"
                test_text += f"  Mann-Whitney p-value: {test_result['mann_whitney']['p_value_two_sided']:.2e}\n"
                test_text += f"  Cohen's d: {test_result['effect_size']['cohens_d']:.3f} ({test_result['effect_size']['interpretation']})\n"
                test_text += f"  Ratio: {test_result.get('ratio', 'N/A'):.3f}\n\n"
            
            ax.text(0.05, 0.95, test_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')
            ax.set_title(f"Statistical Comparisons: {args.dataset} / {args.split}", 
                        fontsize=14, fontweight='bold')
            
            pdf.savefig(fig)
            plt.close(fig)


def check_ownership_criteria(tests: Dict, stats: Dict) -> Dict[str, bool]:
    """Check ownership verification criteria."""
    criteria = {}
    
    # Criterion 1: ModelA vs ModelB should be similar (p > 0.05)
    if "model_a_vs_model_b" in tests:
        test = tests["model_a_vs_model_b"]
        criteria["consistency"] = test["t_test"]["p_value_two_sided"] > 0.05
    
    # Criterion 2: ModelB vs Baseline should be very different (ALL baselines must pass)
    baseline_tests = [k for k in tests
                      if any(tag in k.lower() for tag in ("baseline", "ddpm-", "ldm-", "random"))]
    if baseline_tests:
        all_d = [abs(tests[k]["effect_size"]["cohens_d"]) for k in baseline_tests]
        all_p = [tests[k]["t_test"]["p_value_one_sided"] for k in baseline_tests]
        criteria["separation"] = max(all_p) < 1e-6 and min(all_d) > 2.0
        criteria["separation_range"] = (round(min(all_d), 2), round(max(all_d), 2))

    # Criterion 3: Ratio check (ALL baselines must pass)
    if "model_b" in stats and baseline_tests:
        baseline_means = [stats.get(k.replace("model_b_vs_", ""), {}).get("mean", float('inf'))
                        for k in baseline_tests]
        model_b_mean = stats["model_b"]["mean"]
        if model_b_mean > 0:
            all_ratios = [bm / model_b_mean for bm in baseline_means if bm != float('inf')]
            criteria["ratio"] = min(all_ratios) > 5.0 if all_ratios else False
            criteria["ratio_range"] = (round(min(all_ratios), 2), round(max(all_ratios), 2)) if all_ratios else None
    
    # Overall
    criteria["ownership_verified"] = all([
        criteria.get("consistency", False) or "model_a" not in stats,  # Skip if no Model A
        criteria.get("separation", False),
        criteria.get("ratio", False),
    ])
    
    return criteria


def main():
    parser = argparse.ArgumentParser(description="Ownership verification evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["cifar10", "cifar100", "stl10", "celeba"])
    parser.add_argument("--split", type=str, default="watermark_private",
                       choices=["watermark_private", "eval_nonmember"],
                       help="Which split to evaluate (default: watermark_private)")
    parser.add_argument("--model-a", type=pathlib.Path, default=None,
                       help="Path to Model A checkpoint")
    parser.add_argument("--model-b", type=pathlib.Path, default=None,
                       help="Path to Model B checkpoint")
    parser.add_argument("--baselines-config", type=pathlib.Path,
                       default=PROJECT_ROOT / "configs" / "baselines_by_dataset.yaml")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k-timesteps", type=int, default=50)
    parser.add_argument("--agg", type=str, default="q25")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--save-pdf", action="store_true",
                       help="Generate PDF visualization report")
    args = parser.parse_args()
    
    # Setup
    if args.output is None:
        args.output = PROJECT_ROOT / "runs" / "attack_qr" / "reports" / args.dataset
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    data_cfg_path = PROJECT_ROOT / DATASET_CONFIGS[args.dataset]
    model_cfg_path = PROJECT_ROOT / MODEL_CONFIGS[args.dataset]
    data_cfg = load_yaml(data_cfg_path)
    
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])
    image_shape = data_cfg["dataset"].get("image_shape", [3, 32, 32])
    image_size = image_shape[-1]
    
    LOGGER.info(f"Evaluating on {args.dataset} ({image_size}x{image_size}), split={args.split}")
    
    # Build evaluation loader for specified split
    eval_loader = build_eval_loader(
        args.dataset, data_cfg, args.split,
        batch_size=args.batch_size, max_samples=args.max_samples
    )
    
    # Collect models
    models = {}
    
    # Load Model A
    if args.model_a and args.model_a.exists():
        LOGGER.info(f"Loading Model A: {args.model_a}")
        model_a, alphas_a = load_our_model(args.model_a, model_cfg_path, args.device)
        models["model_a"] = (model_a, alphas_a)
    
    # Load Model B
    if args.model_b and args.model_b.exists():
        LOGGER.info(f"Loading Model B: {args.model_b}")
        model_b, alphas_b = load_our_model(args.model_b, model_cfg_path, args.device)
        models["model_b"] = (model_b, alphas_b)
    
    # Load baselines
    if not args.skip_baselines and args.baselines_config.exists():
        baselines_cfg = load_yaml(args.baselines_config)
        baselines = baselines_cfg.get(args.dataset, [])
        
        for baseline in baselines:
            name = baseline["name"]
            btype = baseline.get("type", "ddpm")
            LOGGER.info(f"Loading baseline: {name} (type={btype})")
            try:
                if btype == "random":
                    torch.manual_seed(42)
                    model, alphas = load_random_baseline(
                        device=args.device,
                        input_mean=mean,
                        input_std=std,
                        resolution=image_size,
                    )
                else:
                    model, alphas = load_baseline_from_registry(name, args.dataset, args.device)
                models[name] = (model, alphas)
            except Exception as e:
                LOGGER.warning(f"Failed to load {name}: {e}")
    
    if not models:
        LOGGER.error("No models to evaluate!")
        return
    
    # Compute scores
    LOGGER.info(f"Computing t-error (k={args.k_timesteps}, agg={args.agg})...")
    all_scores = {}
    
    for name, (model, alphas_bar) in models.items():
        LOGGER.info(f"Evaluating: {name}")
        scores = compute_baseline_scores(
            dataloader=eval_loader,
            model=model,
            alphas_bar=alphas_bar,
            T=1000,
            k=args.k_timesteps,
            agg=args.agg,
            device=args.device,
            desc=f"scores-{name}",
        )
        all_scores[name] = scores
        LOGGER.info(f"  {name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    # Compute statistics
    statistics = {name: compute_statistics(scores) for name, scores in all_scores.items()}
    
    # Perform statistical tests
    tests = {}
    model_names = list(all_scores.keys())
    
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            key = f"{name1}_vs_{name2}"
            tests[key] = perform_statistical_tests(
                all_scores[name1], all_scores[name2], name1, name2
            )
    
    # Check ownership criteria
    criteria = check_ownership_criteria(tests, statistics)
    
    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "config": {
            "model_a": str(args.model_a) if args.model_a else None,
            "model_b": str(args.model_b) if args.model_b else None,
            "k_timesteps": args.k_timesteps,
            "aggregation": args.agg,
            "num_samples": len(eval_loader.dataset),
            "image_size": image_size,
            "split": args.split,
        },
        "statistics": statistics,
        "tests": tests,
        "ownership_criteria": criteria,
    }

    # Per-baseline breakdown
    baselines_cfg = load_yaml(args.baselines_config).get(args.dataset, []) if args.baselines_config.exists() else []
    role_map = {b["name"]: b.get("role", "unknown") for b in baselines_cfg}
    per_baseline = {}
    for name in all_scores:
        if name.startswith("model_"):
            continue
        comp_key = f"model_b_vs_{name}" if f"model_b_vs_{name}" in tests else f"model_a_vs_{name}"
        if comp_key not in tests:
            # Try reversed order (tests use sorted pair)
            comp_key = f"{name}_vs_model_b" if f"{name}_vs_model_b" in tests else f"{name}_vs_model_a"
        entry = {"role": role_map.get(name, "unknown"), "mean_t_error": round(float(statistics[name]["mean"]), 4)}
        if comp_key in tests:
            entry["cohens_d"] = round(abs(tests[comp_key]["effect_size"]["cohens_d"]), 2)
            entry["p_value"] = tests[comp_key]["t_test"]["p_value_one_sided"]
        per_baseline[name] = entry
    report["per_baseline"] = per_baseline

    # Save report (include split in filename)
    output_path = args.output / f"baseline_comparison_{args.dataset}_{args.split}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info(f"Report saved: {output_path}")
    
    # Save raw scores (include split in filename)
    scores_path = args.output / f"t_error_distributions_{args.split}.npz"
    np.savez(scores_path, **{k: v.numpy() for k, v in all_scores.items()})
    LOGGER.info(f"Raw scores saved: {scores_path}")
    
    # Optional PDF visualization
    if args.save_pdf:
        pdf_path = args.output / f"report_{args.dataset}_{args.split}.pdf"
        try:
            generate_pdf_report(all_scores, statistics, tests, criteria, args, pdf_path)
            LOGGER.info(f"PDF report saved: {pdf_path}")
        except Exception as e:
            LOGGER.warning(f"Failed to generate PDF: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"OWNERSHIP EVALUATION: {args.dataset.upper()} / {args.split}")
    print("=" * 70)
    print(f"{'Model':<20} {'Mean':>12} {'Std':>12} {'Q25':>12}")
    print("-" * 70)
    for name, stat in statistics.items():
        print(f"{name:<20} {stat['mean']:>12.4f} {stat['std']:>12.4f} {stat['q25']:>12.4f}")
    print("-" * 70)
    
    if per_baseline:
        print(f"\n{'Baseline':<20} {'Role':<12} {'Mean T-Err':>12} {'|d|':>10}")
        print("-" * 56)
        for bname, binfo in per_baseline.items():
            d_str = f"{binfo['cohens_d']:>10.2f}" if "cohens_d" in binfo else f"{'N/A':>10}"
            print(f"{bname:<20} {binfo['role']:<12} {binfo['mean_t_error']:>12.4f} {d_str}")
        print("-" * 56)

    print("\nOwnership Criteria:")
    for k, v in criteria.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v}")
        else:
            status = "PASS" if v else "FAIL"
            print(f"  {k}: {status}")

    print("=" * 70)


if __name__ == "__main__":
    main()
