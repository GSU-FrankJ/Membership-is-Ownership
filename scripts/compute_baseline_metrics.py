#!/usr/bin/env python3
"""Compute baseline metrics (Yeom / global threshold) for comparison.

CANONICAL SCORING SEMANTICS:
----------------------------
All metrics use the convention: "larger score = more likely member".

For the Yeom baseline attack:
- t-error is a reconstruction error (smaller = more likely member)
- We negate the scores: score' = -t_error
- Then pass score' to the shared metrics functions

This ensures consistency with the QR-MIA pipeline which uses:
- margin m(x) = q̂_τ^ens(x) - s(x) as the attack score
- larger margin = more likely member

Both pipelines now use the same metric functions with the same semantics.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.eval.metrics import roc_auc, tpr_precision_at_fpr


def load_scores(path: pathlib.Path) -> torch.Tensor:
    """Load score tensor from cache file."""
    cache = torch.load(path, map_location="cpu")
    for key in ("per_sample", "scores", "errors", "t_error", "vals"):
        if key in cache:
            scores = cache[key]
            if isinstance(scores, torch.Tensor):
                return scores.float()
    raise ValueError(f"Could not find score tensor in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline metrics")
    parser.add_argument(
        "--member-file",
        type=pathlib.Path,
        default=pathlib.Path("scores/eval_in.pt"),
        help="Path to member scores",
    )
    parser.add_argument(
        "--nonmember-file",
        type=pathlib.Path,
        default=pathlib.Path("scores/eval_out.pt"),
        help="Path to non-member scores",
    )
    parser.add_argument(
        "--target-fprs",
        type=float,
        nargs="+",
        default=[0.001, 0.0001],
        help="Target FPRs to evaluate",
    )
    args = parser.parse_args()

    # Load raw t-error scores
    scores_member_raw = load_scores(args.member_file)
    scores_nonmember_raw = load_scores(args.nonmember_file)
    
    # CANONICAL SCORE TRANSFORMATION:
    # t-error: smaller = more likely member
    # Negate to get: larger = more likely member
    # This aligns with the shared metric semantics
    scores_member = -scores_member_raw
    scores_nonmember = -scores_nonmember_raw

    print("\n" + "=" * 80)
    print("BASELINE METRICS (Global Threshold / Yeom Method)")
    print("=" * 80)
    print(f"Member samples: {len(scores_member)}")
    print(f"Non-member samples: {len(scores_nonmember)}")
    print()
    print("NOTE: Using shared metric functions with negated scores (-t_error)")
    print("      Semantics: larger score = more likely member")
    print()

    # Use shared ROC-AUC function (expects larger = member)
    auc = roc_auc(scores_member, scores_nonmember)
    print(f"ROC-AUC: {auc:.6f}")
    print()

    print("| Target FPR | TPR | Precision | Achieved FPR | FPR Error | TP | FP | TN | FN |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    
    for target_fpr in args.target_fprs:
        # Use shared tpr_precision_at_fpr function (expects larger = member)
        metrics = tpr_precision_at_fpr(
            scores_member, 
            scores_nonmember, 
            target_fpr,
            num_bootstrap=0,  # Skip bootstrap for speed in baseline
            seed=42,
        )
        print(
            f"| {target_fpr:.4f} | "
            f"{metrics['tpr']:.6f} | "
            f"{metrics['precision']:.6f} | "
            f"{metrics['achieved_fpr']:.6f} | "
            f"{metrics['fpr_error']:.6f} | "
            f"{metrics['counts']['tp']} | {metrics['counts']['fp']} | "
            f"{metrics['counts']['tn']} | {metrics['counts']['fn']} |"
        )
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
