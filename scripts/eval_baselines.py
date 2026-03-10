#!/usr/bin/env python3
"""
Unified evaluation harness for baseline comparison (Table 5).

Evaluates ownership methods (MiO, WDM, Zhao) through a common interface:
  1. Load model via method adapter
  2. Compute t-error on watermark_private and eval_nonmember splits
  3. Compute Cohen's d, ratio, three-point criteria
  4. Write results to JSON

Usage:
    python scripts/eval_baselines.py \
        --method mio \
        --checkpoint /path/to/ckpt \
        --dataset cifar10 \
        --output-dir experiments/baseline_comparison/results/mio/cifar10/
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

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import adapters to trigger registration
import scripts.baselines.mio_adapter  # noqa: F401
from scripts.baselines import METHODS
from scripts.baselines.mio_adapter import compute_cohens_d, three_point_check

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


def load_yaml(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_eval_loader(dataset_name: str, data_cfg: dict, split: str, batch_size: int = 256, max_samples=None):
    """Reuse the loader from eval_ownership.py."""
    from scripts.eval_ownership import build_eval_loader as _build
    return _build(dataset_name, data_cfg, split, batch_size=batch_size, max_samples=max_samples)


def main():
    parser = argparse.ArgumentParser(description="Unified baseline evaluation harness")
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS.keys()),
                        help=f"Method to evaluate ({', '.join(METHODS.keys())})")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "stl10", "celeba"])
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k-timesteps", type=int, default=50)
    parser.add_argument("--agg", type=str, default="q25")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load configs
    data_cfg = load_yaml(PROJECT_ROOT / DATASET_CONFIGS[args.dataset])
    model_cfg = load_yaml(PROJECT_ROOT / MODEL_CONFIGS[args.dataset])

    # Create adapter
    AdapterCls = METHODS[args.method]
    adapter = AdapterCls(
        checkpoint=args.checkpoint,
        model_cfg=model_cfg,
        device=args.device,
    )

    # Build loaders for both splits
    member_loader = build_eval_loader(
        args.dataset, data_cfg, "watermark_private",
        batch_size=args.batch_size, max_samples=args.max_samples,
    )
    nonmember_loader = build_eval_loader(
        args.dataset, data_cfg, "eval_nonmember",
        batch_size=args.batch_size, max_samples=args.max_samples,
    )

    # Compute scores
    print(f"Computing {args.method} scores on {args.dataset}...")
    member_scores = adapter.compute_scores(member_loader, k_timesteps=args.k_timesteps, agg=args.agg)
    nonmember_scores = adapter.compute_scores(nonmember_loader, k_timesteps=args.k_timesteps, agg=args.agg)

    m_np = member_scores.numpy()
    nm_np = nonmember_scores.numpy()

    # Compute metrics
    d = compute_cohens_d(m_np, nm_np)
    ratio = float(np.mean(nm_np) / np.mean(m_np)) if np.mean(m_np) > 0 else float("inf")
    passed, three_pt = three_point_check(m_np, nm_np)

    # Native verification (method-specific)
    native = {}
    if hasattr(adapter, "native_verify"):
        native = adapter.native_verify(m_np, nm_np)

    # Build result
    result = {
        "method": args.method,
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "config": {
            "k_timesteps": args.k_timesteps,
            "agg": args.agg,
            "num_members": len(member_loader.dataset),
            "num_nonmembers": len(nonmember_loader.dataset),
        },
        "native_verification": native,
        "mio_verification": {
            "t_error_member_mean": float(np.mean(m_np)),
            "t_error_member_std": float(np.std(m_np)),
            "t_error_nonmember_mean": float(np.mean(nm_np)),
            "t_error_nonmember_std": float(np.std(nm_np)),
            "cohens_d": d,
            "ratio": ratio,
            "three_point_pass": passed,
            "three_point_details": three_pt,
        },
    }

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Method: {args.method} | Dataset: {args.dataset}")
    print(f"{'=' * 60}")
    print(f"  Member   t-error: mean={np.mean(m_np):.4f}, std={np.std(m_np):.4f}")
    print(f"  Non-mem  t-error: mean={np.mean(nm_np):.4f}, std={np.std(nm_np):.4f}")
    print(f"  Cohen's d:  {d:.4f}")
    print(f"  Ratio:      {ratio:.4f}")
    print(f"  Three-point: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 60}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
