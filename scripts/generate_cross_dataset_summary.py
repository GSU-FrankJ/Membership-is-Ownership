#!/usr/bin/env python3
"""
Cross-Dataset Summary Generator.

Aggregates results from all dataset evaluations into a unified summary.

Outputs:
- summary_all_datasets.csv: Tabular summary
- summary_all_datasets.json: Full JSON with all details
- summary_report.md: Human-readable markdown report

Usage:
    python scripts/generate_cross_dataset_summary.py \
        --reports-dir runs/attack_qr/reports/ \
        --output runs/attack_qr/reports/summary_all_datasets.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger

LOGGER = get_winston_logger(__name__)


DATASETS = ["cifar10", "cifar100", "stl10", "celeba"]


def load_json(path: pathlib.Path) -> Optional[Dict]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_summary_row(dataset: str, report: Dict, split: str = "watermark_private") -> List[Dict]:
    """Extract summary rows from a dataset report."""
    rows = []
    
    stats = report.get("statistics", {})
    tests = report.get("tests", {})
    criteria = report.get("ownership_criteria", {})
    report_split = report.get("split", split)  # Use split from report if available
    
    for model_name, model_stats in stats.items():
        row = {
            "dataset": dataset,
            "split": report_split,
            "model": model_name,
            "t_error_mean": model_stats.get("mean"),
            "t_error_std": model_stats.get("std"),
            "t_error_q25": model_stats.get("q25"),
            "t_error_median": model_stats.get("median"),
        }
        
        # Find comparison with best baseline
        baseline_comparisons = [
            (k, v) for k, v in tests.items()
            if model_name in k and any(b in k for b in ["ddpm-", "ldm-", "baseline"])
        ]
        
        if baseline_comparisons:
            # Get comparison where this model is first (model vs baseline)
            for comp_key, comp in baseline_comparisons:
                if comp_key.startswith(f"{model_name}_vs_"):
                    row["p_value_ttest"] = comp["t_test"]["p_value_one_sided"]
                    row["p_value_mannwhitney"] = comp["mann_whitney"]["p_value_one_sided"]
                    row["cohens_d"] = comp["effect_size"]["cohens_d"]
                    row["ratio_vs_baseline"] = comp.get("ratio", None)
                    break
        
        row["ownership_verified"] = criteria.get("ownership_verified", None)
        rows.append(row)
    
    return rows


def generate_markdown_report(
    summaries: List[Dict],
    output_path: pathlib.Path,
) -> None:
    """Generate markdown summary report."""
    
    lines = [
        "# Cross-Dataset Ownership Verification Summary",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Overview",
        "",
    ]
    
    # Count verified datasets
    datasets_tested = set(s["dataset"] for s in summaries)
    datasets_verified = set(
        s["dataset"] for s in summaries 
        if s.get("ownership_verified") == True
    )
    
    lines.extend([
        f"- **Datasets evaluated:** {len(datasets_tested)}",
        f"- **Ownership verified:** {len(datasets_verified)}/{len(datasets_tested)}",
        "",
        "## Results by Dataset",
        "",
    ])
    
    # Group by dataset
    by_dataset = {}
    for s in summaries:
        ds = s["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(s)
    
    for dataset, rows in by_dataset.items():
        lines.append(f"### {dataset.upper()}")
        lines.append("")
        lines.append("| Model | Mean T-Error | Q25 | Cohen's d | Verified |")
        lines.append("|-------|--------------|-----|-----------|----------|")
        
        for row in rows:
            verified = "Yes" if row.get("ownership_verified") else "No"
            cohens_d = f"{row.get('cohens_d', 'N/A'):.2f}" if row.get('cohens_d') else "N/A"
            lines.append(
                f"| {row['model']} | {row['t_error_mean']:.4f} | {row['t_error_q25']:.4f} | {cohens_d} | {verified} |"
            )
        lines.append("")
    
    # Acceptance criteria summary
    lines.extend([
        "## Acceptance Criteria",
        "",
        "For each dataset, ownership is verified if:",
        "",
        "1. **Consistency:** Model A and Model B have similar t-error (p > 0.05)",
        "2. **Separation:** Model B vs Baseline shows p < 1e-6 and |Cohen's d| > 2.0",
        "3. **Ratio:** Baseline t-error / Model B t-error > 5.0",
        "",
    ])
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    LOGGER.info(f"Markdown report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate cross-dataset summary")
    parser.add_argument(
        "--reports-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "runs" / "attack_qr" / "reports",
        help="Directory containing per-dataset reports"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output CSV path"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to include in summary"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["watermark_private", "eval_nonmember"],
        help="Splits to include in summary"
    )
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.reports_dir / "summary_all_datasets.csv"
    
    LOGGER.info(f"Scanning reports in: {args.reports_dir}")
    
    # Collect all summaries
    all_summaries = []
    missing_reports = []
    
    for dataset in args.datasets:
        for split in args.splits:
            # Try new naming convention first (with split)
            report_path = args.reports_dir / dataset / f"baseline_comparison_{dataset}_{split}.json"
            if not report_path.exists():
                # Fallback to old naming convention (without split, assume watermark_private)
                if split == "watermark_private":
                    report_path = args.reports_dir / dataset / f"baseline_comparison_{dataset}.json"
            
            report = load_json(report_path)
            
            if report is None:
                LOGGER.warning(f"Report not found for {dataset}/{split}: {report_path}")
                missing_reports.append(f"{dataset}/{split}")
                continue
            
            rows = extract_summary_row(dataset, report, split)
            all_summaries.extend(rows)
            LOGGER.info(f"Loaded {len(rows)} entries from {dataset}/{split}")
    
    if not all_summaries:
        LOGGER.error("No reports found to summarize!")
        return
    
    # Backward compatibility: use missing_datasets alias
    missing_datasets = [m.split("/")[0] for m in missing_reports]
    
    # Write CSV
    fieldnames = [
        "dataset", "split", "model", "t_error_mean", "t_error_std", "t_error_q25",
        "t_error_median", "p_value_ttest", "p_value_mannwhitney",
        "cohens_d", "ratio_vs_baseline", "ownership_verified"
    ]
    
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_summaries)
    
    LOGGER.info(f"CSV summary saved: {args.output}")
    
    # Write JSON
    json_output = args.output.with_suffix(".json")
    with json_output.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "datasets_included": [d for d in args.datasets if d not in missing_datasets],
            "datasets_missing": missing_datasets,
            "entries": all_summaries,
        }, f, indent=2)
    LOGGER.info(f"JSON summary saved: {json_output}")
    
    # Write Markdown
    md_output = args.output.with_name("summary_report.md")
    generate_markdown_report(all_summaries, md_output)
    
    # Print summary table
    print("\n" + "=" * 110)
    print("CROSS-DATASET OWNERSHIP VERIFICATION SUMMARY")
    print("=" * 110)
    print(f"{'Dataset':<12} {'Split':<20} {'Model':<20} {'Mean':>10} {'Q25':>10} {'d':>8} {'Verified':>10}")
    print("-" * 110)
    
    for row in all_summaries:
        d = row.get('cohens_d')
        d_str = f"{d:.2f}" if d else "N/A"
        verified = "Yes" if row.get('ownership_verified') else "No"
        split = row.get('split', 'watermark_private')[:18]  # Truncate for display
        print(
            f"{row['dataset']:<12} {split:<20} {row['model']:<20} "
            f"{row['t_error_mean']:>10.4f} {row['t_error_q25']:>10.4f} "
            f"{d_str:>8} {verified:>10}"
        )
    
    print("-" * 110)
    verified_count = sum(1 for r in all_summaries if r.get('ownership_verified'))
    total_datasets = len(set(r['dataset'] for r in all_summaries))
    total_splits = len(set((r['dataset'], r.get('split', '')) for r in all_summaries))
    print(f"Ownership verified: {verified_count} entries across {total_datasets} datasets, {total_splits} splits")
    if missing_reports:
        print(f"Missing reports: {', '.join(missing_reports)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
