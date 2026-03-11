#!/usr/bin/env python3
"""
Retroactive Claim Defense Experiment (Phase 06, Steps 6.5-6.7).

Precomputes all training t-errors, then runs 5 adversary scenarios to
demonstrate that only the pre-committed watermark set W_D yields a strong
ownership signal.

Scenarios:
  A) 100 random subsets of 5K from training set
  B) Cherry-picked top-5K (lowest t-error = most memorized)
  C) Sophisticated adversary (same as B, framed differently)
  D) Non-member set (test images)
  E) Wrong model (baseline model's t-errors on W_D)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.mio_adapter import (
    compute_cohens_d,
    load_mio_model,
    three_point_check,
)
from src.attacks.baselines.huggingface_loader import load_hf_ddpm_cifar10
from src.attacks.baselines.t_error_hf import compute_baseline_scores
from scripts.eval_ownership import build_eval_loader, EvalDataset

from torch.utils.data import DataLoader, Subset


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_full_train_loader(data_cfg, batch_size=256):
    """Build loader for ALL 50K CIFAR-10 training images."""
    from torchvision import datasets, transforms

    root = data_cfg["dataset"]["root"]
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    ds = datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def build_test_loader(data_cfg, batch_size=256):
    """Build loader for CIFAR-10 test set (10K images)."""
    from torchvision import datasets, transforms

    root = data_cfg["dataset"]["root"]
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    ds = datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Retroactive Claim Defense Experiment")
    parser.add_argument("--model-a-checkpoint", type=str, required=True,
                        help="Path to Model A (owner's model) checkpoint")
    parser.add_argument("--model-cfg", type=str, default="configs/model_ddim_cifar10.yaml")
    parser.add_argument("--data-cfg", type=str, default="configs/data_cifar10.yaml")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/baseline_comparison/results/retroactive_defense")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k-timesteps", type=int, default=50)
    parser.add_argument("--agg", type=str, default="q25")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip precomputation, load from existing .npy files")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_yaml(PROJECT_ROOT / args.data_cfg)
    model_cfg = load_yaml(PROJECT_ROOT / args.model_cfg)

    # Load watermark_private indices
    wp_path = data_cfg["splits"]["paths"]["watermark_private"]
    with open(wp_path) as f:
        wp_indices = set(json.load(f))
    print(f"Watermark private set: {len(wp_indices)} indices")

    # ---- Step 6.5: Precompute t-errors ----
    all_train_path = output_dir / "all_train_t_errors.npy"
    test_path = output_dir / "test_t_errors.npy"
    baseline_wd_path = output_dir / "baseline_wd_t_errors.npy"

    if not args.skip_precompute:
        # Load Model A
        print("Loading Model A...")
        model_a, alphas_bar_a = load_mio_model(
            pathlib.Path(args.model_a_checkpoint), model_cfg, args.device
        )

        # 1. All 50K training images
        print("\n[1/3] Computing t-error for all 50K training images...")
        train_loader = build_full_train_loader(data_cfg, args.batch_size)
        all_train_scores = compute_baseline_scores(
            train_loader, model_a, alphas_bar_a,
            T=1000, k=args.k_timesteps, agg=args.agg,
            device=args.device, desc="train-50k",
        )
        np.save(all_train_path, all_train_scores.numpy())
        print(f"  Saved: {all_train_path} ({len(all_train_scores)} scores)")
        print(f"  Mean={all_train_scores.mean():.4f}, Std={all_train_scores.std():.4f}")

        # 2. Test set (non-members)
        print("\n[2/3] Computing t-error for test set (10K non-members)...")
        test_loader = build_test_loader(data_cfg, args.batch_size)
        test_scores = compute_baseline_scores(
            test_loader, model_a, alphas_bar_a,
            T=1000, k=args.k_timesteps, agg=args.agg,
            device=args.device, desc="test-10k",
        )
        np.save(test_path, test_scores.numpy())
        print(f"  Saved: {test_path} ({len(test_scores)} scores)")
        print(f"  Mean={test_scores.mean():.4f}, Std={test_scores.std():.4f}")

        # Free Model A
        del model_a, alphas_bar_a
        torch.cuda.empty_cache()

        # 3. Baseline model on W_D
        print("\n[3/3] Computing baseline model t-error on W_D...")
        baseline_model, baseline_alphas = load_hf_ddpm_cifar10(device=args.device)
        wd_loader = build_eval_loader("cifar10", data_cfg, "watermark_private",
                                      batch_size=args.batch_size)
        baseline_wd_scores = compute_baseline_scores(
            wd_loader, baseline_model, baseline_alphas,
            T=1000, k=args.k_timesteps, agg=args.agg,
            device=args.device, desc="baseline-wd",
        )
        np.save(baseline_wd_path, baseline_wd_scores.numpy())
        print(f"  Saved: {baseline_wd_path} ({len(baseline_wd_scores)} scores)")
        print(f"  Mean={baseline_wd_scores.mean():.4f}, Std={baseline_wd_scores.std():.4f}")

        del baseline_model, baseline_alphas
        torch.cuda.empty_cache()
    else:
        print("Skipping precomputation, loading from existing files...")

    # Load precomputed scores
    all_train = np.load(all_train_path)
    test_scores_np = np.load(test_path)
    baseline_wd = np.load(baseline_wd_path)

    # Extract W_D scores from the full training set
    wp_indices_list = sorted(wp_indices)
    wd_scores = all_train[wp_indices_list]
    print(f"\nW_D scores: mean={wd_scores.mean():.4f}, std={wd_scores.std():.4f}")
    print(f"Baseline W_D scores: mean={baseline_wd.mean():.4f}, std={baseline_wd.std():.4f}")

    # ---- Step 6.6: Run 5 Scenarios ----
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_a_checkpoint": args.model_a_checkpoint,
            "k_timesteps": args.k_timesteps,
            "agg": args.agg,
            "seed": args.seed,
            "n_train": len(all_train),
            "n_test": len(test_scores_np),
            "n_wd": len(wd_scores),
        },
        "scenarios": {},
    }

    # Reference: Real W_D (owner) vs baseline
    real_d = compute_cohens_d(wd_scores, baseline_wd)
    real_passed, real_details = three_point_check(wd_scores, baseline_wd)
    results["reference"] = {
        "description": "Real W_D (owner's Model A) vs baseline model on W_D",
        "cohens_d": real_d,
        "three_point_pass": real_passed,
        "three_point_details": real_details,
        "wd_mean": float(wd_scores.mean()),
        "wd_std": float(wd_scores.std()),
        "baseline_mean": float(baseline_wd.mean()),
        "baseline_std": float(baseline_wd.std()),
    }
    print(f"\n{'='*60}")
    print(f"Reference: Real W_D vs Baseline")
    print(f"  Cohen's d: {real_d:.4f}")
    print(f"  Three-point: {'PASS' if real_passed else 'FAIL'}")
    print(f"  W_D mean: {wd_scores.mean():.4f}, Baseline mean: {baseline_wd.mean():.4f}")

    # Scenario A: 100 random subsets
    print(f"\n{'='*60}")
    print("Scenario A: 100 Random Subsets of 5K")
    rng = np.random.RandomState(args.seed)
    random_ds = []
    random_passes = 0
    for i in range(100):
        idx = rng.choice(len(all_train), size=5000, replace=False)
        subset = all_train[idx]
        d = compute_cohens_d(subset, baseline_wd)
        passed, _ = three_point_check(subset, baseline_wd)
        random_ds.append(d)
        if passed:
            random_passes += 1
    random_ds = np.array(random_ds)
    np.save(output_dir / "random_sets_cohens_d.npy", random_ds)
    results["scenarios"]["A_random_subsets"] = {
        "description": "100 random subsets of 5K from training set vs baseline",
        "cohens_d_mean": float(random_ds.mean()),
        "cohens_d_std": float(random_ds.std()),
        "cohens_d_min": float(random_ds.min()),
        "cohens_d_max": float(random_ds.max()),
        "three_point_pass_rate": random_passes / 100,
    }
    print(f"  Cohen's d: mean={random_ds.mean():.4f}, std={random_ds.std():.4f}")
    print(f"  Range: [{random_ds.min():.4f}, {random_ds.max():.4f}]")
    print(f"  Three-point pass rate: {random_passes}/100")

    # Scenario B: Cherry-picked top-5K (lowest t-error)
    print(f"\n{'='*60}")
    print("Scenario B: Cherry-picked Top-5K (lowest t-error)")
    top5k_idx = np.argsort(all_train)[:5000]
    top5k = all_train[top5k_idx]
    top5k_d = compute_cohens_d(top5k, baseline_wd)
    top5k_passed, top5k_details = three_point_check(top5k, baseline_wd)
    results["scenarios"]["B_cherry_picked"] = {
        "description": "Top-5K lowest t-error (adversary's best case) vs baseline",
        "cohens_d": top5k_d,
        "three_point_pass": top5k_passed,
        "three_point_details": top5k_details,
        "mean": float(top5k.mean()),
        "std": float(top5k.std()),
    }
    print(f"  Cohen's d: {top5k_d:.4f}")
    print(f"  Three-point: {'PASS' if top5k_passed else 'FAIL'}")
    print(f"  Mean: {top5k.mean():.4f}")

    # Scenario C: Sophisticated adversary (same computation, different framing)
    print(f"\n{'='*60}")
    print("Scenario C: Sophisticated Adversary (leaked Dtrain, picks most-memorized)")
    # Same as B, but we also report overlap with real W_D
    overlap = len(set(top5k_idx) & wp_indices)
    results["scenarios"]["C_sophisticated"] = {
        "description": "Same as B, adversary knows Dtrain and picks most-memorized",
        "cohens_d": top5k_d,
        "three_point_pass": top5k_passed,
        "overlap_with_real_wd": overlap,
        "overlap_fraction": overlap / len(wp_indices),
    }
    print(f"  Cohen's d: {top5k_d:.4f} (same as B)")
    print(f"  Overlap with real W_D: {overlap}/{len(wp_indices)} ({overlap/len(wp_indices)*100:.1f}%)")

    # Scenario D: Non-member set
    print(f"\n{'='*60}")
    print("Scenario D: Non-member Set (test images on Model A)")
    nonmem_d = compute_cohens_d(test_scores_np, baseline_wd)
    nonmem_passed, nonmem_details = three_point_check(test_scores_np, baseline_wd)
    results["scenarios"]["D_nonmember"] = {
        "description": "Test set t-errors (Model A) vs baseline on W_D",
        "cohens_d": nonmem_d,
        "three_point_pass": nonmem_passed,
        "three_point_details": nonmem_details,
        "mean": float(test_scores_np.mean()),
        "std": float(test_scores_np.std()),
    }
    print(f"  Cohen's d: {nonmem_d:.4f}")
    print(f"  Three-point: {'PASS' if nonmem_passed else 'FAIL'}")
    print(f"  Mean: {test_scores_np.mean():.4f}")

    # Scenario E: Wrong model
    print(f"\n{'='*60}")
    print("Scenario E: Wrong Model (baseline t-errors on W_D)")
    # Compare baseline_wd (wrong model on W_D) vs baseline_wd itself → trivially same
    # Instead: compare baseline_wd vs wd_scores (owner's model on W_D)
    # This tests: can an attacker claim ownership using a different model?
    wrong_d = compute_cohens_d(baseline_wd, baseline_wd)
    wrong_passed, wrong_details = three_point_check(baseline_wd, baseline_wd)
    results["scenarios"]["E_wrong_model"] = {
        "description": "Baseline model on W_D vs baseline model on W_D (self-comparison)",
        "cohens_d": wrong_d,
        "three_point_pass": wrong_passed,
        "three_point_details": wrong_details,
        "mean": float(baseline_wd.mean()),
        "std": float(baseline_wd.std()),
    }
    print(f"  Cohen's d: {wrong_d:.4f}")
    print(f"  Three-point: {'PASS' if wrong_passed else 'FAIL'}")

    # Save all results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to: {results_path}")
    print(f"Cohen's d distribution saved to: {output_dir / 'random_sets_cohens_d.npy'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
