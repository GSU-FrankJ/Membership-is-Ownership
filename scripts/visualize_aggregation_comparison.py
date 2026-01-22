#!/usr/bin/env python3
"""Visualize score distributions for multiple aggregation strategies."""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_stats(scores_in: torch.Tensor, scores_out: torch.Tensor) -> Dict[str, float]:
    mean_in = scores_in.mean().item()
    mean_out = scores_out.mean().item()
    gap = mean_out - mean_in
    ratio = mean_out / mean_in if mean_in > 0 else float("inf")
    return {"ratio": ratio, "gap": gap}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize aggregation comparison results.")
    parser.add_argument("--timestamp", required=True, help="Timestamp matching the comparison run.")
    parser.add_argument(
        "--aggregates",
        default="mean,q10",
        help="Comma-separated list of aggregations to plot (order matches analysis).",
    )
    parser.add_argument(
        "--scores-dir",
        type=pathlib.Path,
        default=pathlib.Path("scores"),
        help="Directory containing compare_<agg>_<timestamp> score tensors.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output PNG path (defaults to aggregation_comparison_<timestamp>.png).",
    )
    args = parser.parse_args()

    aggregates: List[str] = [agg.strip() for agg in args.aggregates.split(",") if agg.strip()]
    if not aggregates:
        raise ValueError("No aggregations provided for visualization.")

    scores: Dict[str, Dict[str, torch.Tensor]] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for agg in aggregates:
        in_path = args.scores_dir / f"compare_{agg}_{args.timestamp}_eval_in.pt"
        out_path = args.scores_dir / f"compare_{agg}_{args.timestamp}_eval_out.pt"
        scores_in = torch.load(in_path, map_location="cpu")["per_sample"]
        scores_out = torch.load(out_path, map_location="cpu")["per_sample"]
        scores[agg] = {"in": scores_in, "out": scores_out}
        stats[agg] = compute_stats(scores_in, scores_out)

    cols = min(3, len(aggregates))
    rows_hist = math.ceil(len(aggregates) / cols)
    total_rows = rows_hist + 1  # reserve last row for summary chart

    fig = plt.figure(figsize=(6 * cols, 3.8 * total_rows))
    gs = fig.add_gridspec(total_rows, cols, hspace=0.4, wspace=0.3)
    fig.suptitle(
        f"Aggregation Comparison ({', '.join(aggregates)})",
        fontsize=16,
        fontweight="bold",
    )

    # Histograms per aggregation
    for idx, agg in enumerate(aggregates):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        members = scores[agg]["in"].cpu().numpy()
        non_members = scores[agg]["out"].cpu().numpy()
        ax.hist(members, bins=50, alpha=0.6, label="Members", color="steelblue", density=True)
        ax.hist(non_members, bins=50, alpha=0.6, label="Non-members", color="indianred", density=True)
        ax.axvline(members.mean(), color="steelblue", linestyle="--", linewidth=1.8)
        ax.axvline(non_members.mean(), color="indianred", linestyle="--", linewidth=1.8)
        ax.set_title(f"{agg} aggregation", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-error score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.text(
            0.97,
            0.95,
            f"ratio={stats[agg]['ratio']:.3f}\ngap={stats[agg]['gap']:.6f}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Summary axis (ratio bars + gap line)
    summary_ax = fig.add_subplot(gs[rows_hist, :])
    x = np.arange(len(aggregates))
    ratios = [stats[agg]["ratio"] for agg in aggregates]
    gaps = [stats[agg]["gap"] for agg in aggregates]

    bar = summary_ax.bar(x, ratios, width=0.4, color="mediumseagreen", label="ratio (out/in)")
    summary_ax.set_ylabel("Ratio", color="mediumseagreen")
    summary_ax.tick_params(axis="y", labelcolor="mediumseagreen")
    summary_ax.set_xticks(x)
    summary_ax.set_xticklabels(aggregates, rotation=15)
    summary_ax.set_xlabel("Aggregation method")

    for rect in bar:
        height = rect.get_height()
        summary_ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    gap_ax = summary_ax.twinx()
    gap_ax.plot(x, gaps, color="darkorange", marker="o", linewidth=2, label="gap (out - in)")
    gap_ax.set_ylabel("Gap", color="darkorange")
    gap_ax.tick_params(axis="y", labelcolor="darkorange")

    lines, labels = summary_ax.get_legend_handles_labels()
    lines2, labels2 = gap_ax.get_legend_handles_labels()
    summary_ax.legend(lines + lines2, labels + labels2, loc="upper left")
    summary_ax.set_title("Ratio & Gap Overview", fontweight="bold")
    summary_ax.grid(True, axis="y", alpha=0.3)

    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.06, right=0.98, hspace=0.45, wspace=0.3)

    output_path = args.output or pathlib.Path(f"aggregation_comparison_{args.timestamp}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Visualization saved to: {output_path}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for agg in aggregates:
        print(
            f"{agg:<10} -> ratio={stats[agg]['ratio']:.3f}, "
            f"gap={stats[agg]['gap']:.6f} "
            f"(mean_in={scores[agg]['in'].mean():.6f}, mean_out={scores[agg]['out'].mean():.6f})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
