"""
MiO (Membership is Ownership) adapter for the unified eval harness.

Wraps existing t-error + ownership eval logic to conform to the
eval_baselines.py interface.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines import register_method
from src.attacks.baselines.t_error_hf import compute_baseline_scores
from src.ddpm_ddim.models.unet import build_unet
from src.ddpm_ddim.schedulers.betas import build_cosine_schedule


def load_mio_model(checkpoint_path: pathlib.Path, model_cfg: dict, device: str):
    """Load a MiO DDIM model and its alpha_bar schedule."""
    model = build_unet(model_cfg.get("model", {}))

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    T = model_cfg.get("diffusion", {}).get("timesteps", 1000)
    _, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    return model, alphas_bar


def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def three_point_check(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> Tuple[bool, Dict]:
    """Check the three-point ownership criteria.

    1. consistency: t-test p > 0.05 between member and nonmember on the
       *owner's* model (skipped here -- handled at harness level)
    2. separation: large Cohen's d between member/nonmember
    3. ratio: nonmember mean / member mean > 5
    """
    d = compute_cohens_d(member_scores, nonmember_scores)
    ratio = float(np.mean(nonmember_scores) / np.mean(member_scores)) if np.mean(member_scores) > 0 else float("inf")

    _, p_val = stats.ttest_ind(member_scores, nonmember_scores)

    separation = abs(d) > 2.0 and p_val < 1e-6
    ratio_ok = ratio > 5.0

    passed = separation and ratio_ok

    return passed, {
        "cohens_d": d,
        "ratio": ratio,
        "t_test_p": float(p_val),
        "separation": separation,
        "ratio_check": ratio_ok,
    }


@register_method("mio")
class MiOAdapter:
    """Adapter that wraps MiO t-error computation."""

    def __init__(self, checkpoint: pathlib.Path, model_cfg: dict, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self.model, self.alphas_bar = load_mio_model(checkpoint, model_cfg, device)

    def compute_scores(
        self,
        dataloader: DataLoader,
        k_timesteps: int = 50,
        agg: str = "q25",
    ) -> torch.Tensor:
        return compute_baseline_scores(
            dataloader=dataloader,
            model=self.model,
            alphas_bar=self.alphas_bar,
            T=1000,
            k=k_timesteps,
            agg=agg,
            device=self.device,
            desc="mio-scores",
        )

    def native_verify(self, member_scores: np.ndarray, nonmember_scores: np.ndarray) -> Dict:
        """MiO native verification is the t-error ownership test itself."""
        passed, details = three_point_check(member_scores, nonmember_scores)
        return {"pass": passed, **details}
