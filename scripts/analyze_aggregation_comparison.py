#!/usr/bin/env python3
"""Analyze and compare t-error aggregation methods across score caches."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

import torch


def compute_stats(scores_in: torch.Tensor, scores_out: torch.Tensor, name: str) -> Dict[str, float]:
    """Compute gap and ratio statistics for one aggregation."""

    mean_in_val = scores_in.mean().item()
    mean_out_val = scores_out.mean().item()
    gap = mean_out_val - mean_in_val
    ratio = mean_out_val / mean_in_val if mean_in_val > 0 else float("inf")

    return {
        "name": name,
        "mean_in": mean_in_val,
        "mean_out": mean_out_val,
        "gap": gap,
        "ratio": ratio,
        "std_in": scores_in.std().item(),
        "std_out": scores_out.std().item(),
        "min_in": scores_in.min().item(),
        "max_in": scores_in.max().item(),
        "median_in": scores_in.median().item(),
        "min_out": scores_out.min().item(),
        "max_out": scores_out.max().item(),
        "median_out": scores_out.median().item(),
    }


def format_pct(delta: float) -> str:
    if delta == float("inf") or delta != delta:  # NaN check
        return "n/a"
    return f"{delta:+.2f}%"


def load_scores(scores_dir: pathlib.Path, agg: str, timestamp: str) -> Dict[str, torch.Tensor]:
    """Load cached per-sample scores for a given aggregation."""

    in_path = scores_dir / f"compare_{agg}_{timestamp}_eval_in.pt"
    out_path = scores_dir / f"compare_{agg}_{timestamp}_eval_out.pt"

    try:
        scores_in = torch.load(in_path, map_location="cpu")["per_sample"]
        scores_out = torch.load(out_path, map_location="cpu")["per_sample"]
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Missing score file for aggregation '{agg}'. "
            f"Expected '{in_path}' and '{out_path}'."
        ) from err

    return {"in": scores_in, "out": scores_out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aggregation method results.")
    parser.add_argument("--timestamp", required=True, help="Timestamp shared by the comparison run.")
    parser.add_argument("--ckpt", default="unknown", help="Checkpoint path used for the run.")
    parser.add_argument(
        "--aggregates",
        default="mean,q10",
        help="Comma-separated list of aggregations to compare (order defines baseline).",
    )
    parser.add_argument(
        "--scores-dir",
        type=pathlib.Path,
        default=pathlib.Path("scores"),
        help="Directory containing cached score tensors.",
    )
    args = parser.parse_args()

    aggregates: List[str] = [agg.strip() for agg in args.aggregates.split(",") if agg.strip()]
    if not aggregates:
        raise ValueError("No aggregation methods provided. Use --aggregates mean,q10,...")

    print(f"\nLoading scores from '{args.scores_dir}' for aggregations: {', '.join(aggregates)}")

    score_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    stats_list = []
    for agg in aggregates:
        scores = load_scores(args.scores_dir, agg, args.timestamp)
        score_cache[agg] = scores
        stats = compute_stats(scores["in"], scores["out"], agg)
        stats_list.append(stats)

    sample_count_in = score_cache[aggregates[0]]["in"].numel()
    sample_count_out = score_cache[aggregates[0]]["out"].numel()
    print(f"✅ Loaded {sample_count_in} member and {sample_count_out} non-member samples per aggregation.\n")

    baseline_name = aggregates[0]
    baseline_stats = stats_list[0]
    baseline_scores = score_cache[baseline_name]

    # Print comparison table
    print("=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    delta_header = f"Δmean_in/out vs {baseline_name}"
    print(
        f"{'Method':<12} {'mean_in':<12} {'mean_out':<12} {'gap':<12} "
        f"{'ratio':<10} {delta_header:<26}"
    )
    print("-" * 100)
    for stats in stats_list:
        if stats["name"] == baseline_name:
            delta_text = "baseline"
        else:
            pct_in = ((stats["mean_in"] - baseline_stats["mean_in"]) / baseline_stats["mean_in"]) * 100
            pct_out = ((stats["mean_out"] - baseline_stats["mean_out"]) / baseline_stats["mean_out"]) * 100
            delta_text = f"{pct_in:+.2f}% / {pct_out:+.2f}%"
        print(
            f"{stats['name']:<12} "
            f"{stats['mean_in']:<12.6f} "
            f"{stats['mean_out']:<12.6f} "
            f"{stats['gap']:<12.6f} "
            f"{stats['ratio']:<10.3f} "
            f"{delta_text:<26}"
        )
    print()

    # Identity checks vs baseline
    print("=" * 100)
    print("IDENTITY CHECKS VS BASELINE")
    print("=" * 100)
    identity_lines = []
    for stats in stats_list[1:]:
        name = stats["name"]
        same_in = torch.allclose(score_cache[name]["in"], baseline_scores["in"], rtol=1e-5, atol=1e-8)
        same_out = torch.allclose(score_cache[name]["out"], baseline_scores["out"], rtol=1e-5, atol=1e-8)
        diff_in = (score_cache[name]["in"] - baseline_scores["in"]).abs().mean().item()
        diff_out = (score_cache[name]["out"] - baseline_scores["out"]).abs().mean().item()
        verdict = "IDENTICAL ⚠️" if same_in and same_out else "DIFFERENT ✅"
        print(
            f"{name:<12} -> {verdict} | "
            f"In abs diff: {diff_in:.6f}, Out abs diff: {diff_out:.6f}"
        )
        identity_lines.append(
            f"| {name} | {'YES ⚠️' if same_in else 'NO ✅'} | "
            f"{'YES ⚠️' if same_out else 'NO ✅'} | {diff_in:.6f} | {diff_out:.6f} |"
        )
    if len(stats_list) == 1:
        print("Only one aggregation provided; skipping identity checks.")
    print()

    # Detailed statistics
    print("=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)
    for stats in stats_list:
        name = stats["name"]
        print(f"\n{name} (In-set):")
        print(f"  Range : [{stats['min_in']:.6f}, {stats['max_in']:.6f}]")
        print(f"  Mean  : {stats['mean_in']:.6f} ± {stats['std_in']:.6f}")
        print(f"  Median: {stats['median_in']:.6f}")
        print(f"{name} (Out-set):")
        print(f"  Range : [{stats['min_out']:.6f}, {stats['max_out']:.6f}]")
        print(f"  Mean  : {stats['mean_out']:.6f} ± {stats['std_out']:.6f}")
        print(f"  Median: {stats['median_out']:.6f}")
    print()

    # Build markdown report
    report_lines = [
        "# Aggregation Comparison Report",
        "",
        f"**Generated:** {args.timestamp}",
        f"**Checkpoint:** {args.ckpt}",
        f"**Methods compared ({len(aggregates)}):** {', '.join(aggregates)}",
        "",
        "---",
        "",
        "## Summary Results",
        "",
        "| Method | mean_in | mean_out | gap | ratio | Δmean_in/out vs baseline | Δratio |",
        "|--------|---------|----------|-----|-------|--------------------------|--------|",
    ]
    for stats in stats_list:
        if stats["name"] == baseline_name:
            delta_means = "baseline"
            delta_ratio = "baseline"
        else:
            pct_in = ((stats["mean_in"] - baseline_stats["mean_in"]) / baseline_stats["mean_in"]) * 100
            pct_out = ((stats["mean_out"] - baseline_stats["mean_out"]) / baseline_stats["mean_out"]) * 100
            ratio_pct = ((stats["ratio"] - baseline_stats["ratio"]) / baseline_stats["ratio"]) * 100
            delta_means = f"{pct_in:+.2f}% / {pct_out:+.2f}%"
            delta_ratio = format_pct(ratio_pct)

        report_lines.append(
            f"| {stats['name']} | {stats['mean_in']:.6f} | {stats['mean_out']:.6f} | "
            f"{stats['gap']:.6f} | {stats['ratio']:.3f} | {delta_means} | {delta_ratio} |"
        )

    if identity_lines:
        report_lines.extend(
            [
                "",
                "## Identity Check vs Baseline",
                "",
                "| Method | In identical? | Out identical? | Mean abs diff (In) | Mean abs diff (Out) |",
                "|--------|---------------|----------------|--------------------|---------------------|",
            ]
        )
        report_lines.extend(identity_lines)

    report_lines.append("")
    report_lines.append("## Detailed Statistics")
    report_lines.append("")
    for stats in stats_list:
        report_lines.extend(
            [
                f"### {stats['name']}",
                "",
                f"- In-set range: [{stats['min_in']:.6f}, {stats['max_in']:.6f}]",
                f"- In-set mean ± std: {stats['mean_in']:.6f} ± {stats['std_in']:.6f}",
                f"- In-set median: {stats['median_in']:.6f}",
                f"- Out-set range: [{stats['min_out']:.6f}, {stats['max_out']:.6f}]",
                f"- Out-set mean ± std: {stats['mean_out']:.6f} ± {stats['std_out']:.6f}",
                f"- Out-set median: {stats['median_out']:.6f}",
                "",
            ]
        )

    report_lines.extend(
        [
            "---",
            "",
            "**Files analyzed:**",
        ]
    )
    for agg in aggregates:
        report_lines.append(
            f"- `scores/compare_{agg}_{args.timestamp}_eval_in.pt` / "
            f"`scores/compare_{agg}_{args.timestamp}_eval_out.pt`"
        )

    report_path = pathlib.Path(f"comparison_report_{args.timestamp}.md")
    report_path.write_text("\n".join(report_lines))
    print(f"📄 Full report saved to: {report_path}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
