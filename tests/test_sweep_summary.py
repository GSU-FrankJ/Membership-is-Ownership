import csv
import json
import pathlib
import sys

from attacks.eval import aggregate_sweep


def _write_report(path: pathlib.Path, tag: str, tau_metrics: list[dict]) -> None:
    payload = {
        "scores": {"tag": tag},
        "results": tau_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_aggregate_sweep_csv(tmp_path: pathlib.Path, monkeypatch) -> None:
    reports_root = tmp_path / "reports"

    tau_a = {
        "tau": 0.001,
        "roc_auc": 0.9,
        "threshold_variance": {"mean": 0.1, "std": 0.01},
        "metrics": [
            {
                "target_fpr": 0.0001,
                "tpr": 0.5,
                "precision": 0.8,
                "achieved_fpr": 0.00011,
                "fpr_calibration_error": 0.00001,
                "tp": 50,
                "fp": 1,
                "tn": 999,
                "fn": 50,
            },
            {
                "target_fpr": 0.001,
                "tpr": 0.6,
                "precision": 0.75,
                "achieved_fpr": 0.00105,
                "fpr_calibration_error": 0.00005,
                "tp": 60,
                "fp": 10,
                "tn": 990,
                "fn": 40,
            },
        ],
    }
    tau_b = {
        "tau": 0.002,
        "roc_auc": 0.88,
        "threshold_variance": {"mean": 0.2, "std": 0.02},
        "metrics": [
            {
                "target_fpr": 0.0001,
                "tpr": 0.45,
                "precision": 0.7,
                "achieved_fpr": 0.00009,
                "fpr_calibration_error": -0.00001,
                "tp": 45,
                "fp": 0,
                "tn": 1000,
                "fn": 55,
            },
            {
                "target_fpr": 0.001,
                "tpr": 0.55,
                "precision": 0.72,
                "achieved_fpr": 0.00095,
                "fpr_calibration_error": -0.00005,
                "tp": 55,
                "fp": 9,
                "tn": 991,
                "fn": 45,
            },
        ],
    }

    _write_report(reports_root / "run_a" / "report.json", "run_a", [tau_a])
    _write_report(reports_root / "run_b" / "report.json", "run_b", [tau_b])

    out_csv = tmp_path / "summary.csv"

    argv = [
        "aggregate_sweep",
        "--reports-root",
        str(reports_root),
        "--out",
        str(out_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    aggregate_sweep.main()

    with out_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 2
    tags = {row["tag"] for row in rows}
    assert tags == {"run_a", "run_b"}
    row_a = next(row for row in rows if row["tag"] == "run_a")
    assert float(row_a["tpr_at_fpr_1e-4"]) == 0.5
    assert float(row_a["threshold_var_mean"]) == 0.1
