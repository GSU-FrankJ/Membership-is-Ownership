#!/usr/bin/env python3

"""Package an ownership evidence bundle for a QR-MIA evaluation run."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import tempfile
import zipfile
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_best_tau(results: List[Dict]) -> Dict:
    """Select the best tau entry using the sweep tie-break rules."""

    def key(entry: Dict) -> Tuple:
        metrics = entry["metrics"]
        by_fpr = {m["target_fpr"]: m for m in metrics}
        fpr_1e4 = by_fpr.get(1e-4) or by_fpr.get(0.0001)
        fpr_1e3 = by_fpr.get(1e-3) or by_fpr.get(0.001)
        if fpr_1e4 is None or fpr_1e3 is None:
            return (0, 0, 0, 0, 0)
        return (
            -fpr_1e4["tpr"],
            -fpr_1e3["tpr"],
            abs(fpr_1e4["fpr_calibration_error"]),
            -entry["roc_auc"],
            entry.get("threshold_variance", {}).get("mean", 0.0),
        )

    return sorted(results, key=key)[0]


def build_scores_tables(
    report_dir: pathlib.Path,
    tau: float,
    output_dir: pathlib.Path,
) -> Dict[str, pathlib.Path]:
    """Create `scores_<split>.parquet` files for the chosen tau."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, pathlib.Path] = {}
    for split in ("aux", "in", "out"):
        diagnostics_path = report_dir / f"diagnostics_{split}_tau_{tau}.parquet"
        if not diagnostics_path.exists():
            raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")
        df = pd.read_parquet(diagnostics_path)
        columns = [
            "score_raw",
            "score_log",
            "thresholds_per_model_raw",
            "thresholds_per_model_log",
            "final_decision",
            "member_label",
        ]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Diagnostics parquet {diagnostics_path} missing columns {missing}")
        filtered = df[columns].copy()
        if split == "in":
            split_name = "eval_in"
        elif split == "out":
            split_name = "eval_out"
        else:
            split_name = split
        out_path = output_dir / f"scores_{split_name}.parquet"
        filtered.to_parquet(out_path)
        outputs[split_name] = out_path
    return outputs


def load_yaml(path: pathlib.Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_split_hashes(config_paths: List[pathlib.Path]) -> Dict[str, str]:
    for cfg_path in config_paths:
        cfg = load_yaml(cfg_path)
        if "splits" in cfg:
            paths = cfg.get("splits", {}).get("paths", {})
            break
    else:
        return {}
    hashes = {}
    for name, rel in paths.items():
        split_path = pathlib.Path(rel)
        if not split_path.is_absolute():
            split_path = PROJECT_ROOT / split_path
        split_path = split_path.resolve()
        hashes[name] = sha256(split_path)
    return hashes


def render_bundle_readme(
    run_id: str,
    tag: str | None,
    tau_entry: Dict,
    diffusion_sha: str,
    ensemble_sha: str,
    qr_hash: str,
    split_hashes: Dict[str, str],
    configs: List[pathlib.Path],
    reproduction_cmd: str,
) -> str:
    lines = [
        "# Ownership Bundle",
        "",
        f"- Report ID: `{run_id}`",
        f"- Score tag: `{tag or 'default'}`",
        f"- Selected tau: `{tau_entry['tau']}`",
        f"- Diffusion checkpoint SHA256: `{diffusion_sha}`",
        f"- Ensemble SHA256: `{ensemble_sha}`",
        f"- Scores ensemble SHA256 (bagging): `{qr_hash}`",
        "",
        "## Metrics",
        "",
        "| Target FPR | Achieved FPR | FPR Error | TPR | Precision | TP | FP | TN | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    metrics = {m["target_fpr"]: m for m in tau_entry["metrics"]}
    for target in sorted(metrics.keys()):
        metric = metrics[target]
        lines.append(
            f"| {target:.5f} | {metric['achieved_fpr']:.6f} | {metric['fpr_calibration_error']:.6f} "
            f"| {metric['tpr']:.6f} | {metric['precision']:.6f} | {metric['tp']} | {metric['fp']} "
            f"| {metric['tn']} | {metric['fn']} |"
        )
    lines.extend(
        [
            "",
            f"ROC-AUC: **{tau_entry['roc_auc']:.6f}**",
            "",
            "## Split Hashes",
            "",
        ]
    )
    for name, digest in split_hashes.items():
        lines.append(f"- `{name}` → `{digest}`")
    lines.extend(
        [
            "",
            "## Configurations",
            "",
        ]
    )
    for cfg in configs:
        lines.append(f"- `{cfg}`")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            reproduction_cmd,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def create_manifest(mapping: Dict[str, str]) -> str:
    return json.dumps(mapping, indent=2)


def make_bundle(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.resolve()
    if not (report_dir / "report.json").exists():
        raise FileNotFoundError(f"{report_dir}/report.json not found")

    report = load_json(report_dir / "report.json")
    best_tau = select_best_tau(report["results"])
    tau_value = best_tau["tau"]

    configs = [pathlib.Path(cfg).resolve() for cfg in args.config_configs]

    ensemble_path = pathlib.Path(report.get("ensemble_path", "")).expanduser().resolve()
    ensemble_hash = sha256(ensemble_path)
    run_json_path = ensemble_path.parent.parent / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"Expected run metadata at {run_json_path}")

    ckpt_path = args.ckpt.resolve()
    diffusion_sha = sha256(ckpt_path)
    qr_hash = ensemble_hash

    split_hashes = compute_split_hashes(configs)
    score_tag = report["scores"].get("tag") or "default"
    ensemble_arg = shlex.quote(str(ensemble_path))
    reproduction_cmd = (
        "bash scripts/eval_qr.sh "
        f"--config configs/attack_qr.yaml --scores-tag {score_tag} "
        f"--ensemble {ensemble_arg}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        scores_output = tmp_path / "scores"
        score_tables = build_scores_tables(report_dir, tau_value, scores_output)

        bundle_readme = render_bundle_readme(
            run_id=report_dir.name,
            tag=report["scores"].get("tag"),
            tau_entry=best_tau,
            diffusion_sha=diffusion_sha,
            ensemble_sha=ensemble_hash,
            qr_hash=qr_hash,
            split_hashes=split_hashes,
            configs=configs,
            reproduction_cmd=reproduction_cmd,
        )
        bundle_readme_path = tmp_path / "BUNDLE_README.md"
        bundle_readme_path.write_text(bundle_readme, encoding="utf-8")

        manifest_entries: Dict[str, str] = {}
        artifacts: List[Tuple[pathlib.Path, str]] = []

        for cfg in configs:
            manifest_entries[f"configs/{cfg.name}"] = sha256(cfg)
            artifacts.append((cfg, f"configs/{cfg.name}"))

        manifest_entries["report.json"] = sha256(report_dir / "report.json")
        manifest_entries["report.md"] = sha256(report_dir / "report.md")
        artifacts.append((report_dir / "report.json", "report.json"))
        artifacts.append((report_dir / "report.md", "report.md"))

        for png in sorted(report_dir.glob("*.png")):
            arcname = f"plots/{png.name}"
            manifest_entries[arcname] = sha256(png)
            artifacts.append((png, arcname))

        manifest_entries["run.json"] = sha256(run_json_path)
        artifacts.append((run_json_path, "run.json"))

        for split, path in score_tables.items():
            arcname = f"scores/scores_{split}.parquet"
            manifest_entries[arcname] = sha256(path)
            artifacts.append((path, arcname))

        manifest_entries["BUNDLE_README.md"] = sha256(bundle_readme_path)
        artifacts.append((bundle_readme_path, "BUNDLE_README.md"))

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(create_manifest(manifest_entries), encoding="utf-8")
        artifacts.append((manifest_path, "manifest.json"))

        with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, arc in artifacts:
                zf.write(src, arc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ownership evidence bundle.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--config-configs", nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_bundle(args)


if __name__ == "__main__":
    main()
