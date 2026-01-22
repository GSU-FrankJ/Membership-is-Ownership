import pathlib

import pandas as pd  # type: ignore

from scripts.make_ownership_bundle import build_scores_tables


def _write_diagnostics(path: pathlib.Path, member_label: int) -> None:
    data = {
        "score_raw": [0.1, 0.2],
        "score_log": [0.01, 0.02],
        "margin_log": [0.5, 0.4],
        "final_decision": [1, 0],
        "member_label": [member_label, member_label],
        "thresholds_per_model_raw": [[0.1, 0.2], [0.3, 0.4]],
        "thresholds_per_model_log": [[0.01, 0.02], [0.03, 0.04]],
    }
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_build_scores_tables_schema(tmp_path: pathlib.Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    tau = 0.001
    _write_diagnostics(report_dir / f"diagnostics_aux_tau_{tau}.parquet", member_label=1)
    _write_diagnostics(report_dir / f"diagnostics_in_tau_{tau}.parquet", member_label=1)
    _write_diagnostics(report_dir / f"diagnostics_out_tau_{tau}.parquet", member_label=0)

    out_dir = tmp_path / "scores"
    tables = build_scores_tables(report_dir, tau, out_dir)

    expected_columns = [
        "score_raw",
        "score_log",
        "thresholds_per_model_raw",
        "thresholds_per_model_log",
        "final_decision",
        "member_label",
    ]

    for split, parquet_path in tables.items():
        df = pd.read_parquet(parquet_path)
        assert list(df.columns) == expected_columns, f"{split} columns mismatch"
        assert pd.api.types.is_float_dtype(df["score_raw"])
        assert pd.api.types.is_float_dtype(df["score_log"])
        assert pd.api.types.is_integer_dtype(df["final_decision"])
        assert pd.api.types.is_integer_dtype(df["member_label"])
        assert list(df["thresholds_per_model_raw"].iloc[0]) == [0.1, 0.2]
        assert list(df["thresholds_per_model_log"].iloc[0]) == [0.01, 0.02]
