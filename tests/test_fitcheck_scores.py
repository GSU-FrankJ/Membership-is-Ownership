import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fitcheck_detects_gap(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()

    torch.save(
        {"per_sample": torch.abs(torch.randn(1000)) * 0.5},
        scores_dir / "eval_in.pt",
    )
    torch.save(
        {"per_sample": torch.abs(torch.randn(1000)) * 1.0 + 0.2},
        scores_dir / "eval_out.pt",
    )

    outdir = tmp_path / "fitcheck_out"
    cmd = [
        sys.executable,
        "tools/fitcheck.py",
        "--ckpt",
        "dummy.ckpt",
        "--outdir",
        str(outdir),
        "--scores-root",
        str(scores_dir),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["ratio"] is not None
    assert summary["ratio"] > 1.1
    assert summary["decision"]["recon_gap_strong"] is True
