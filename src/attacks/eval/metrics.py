"""Shared evaluation metrics with unified semantics: larger score = member-like."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _to_numpy(scores) -> np.ndarray:
    """Convert torch or array-like scores to a flattened numpy array."""
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().float().view(-1).numpy()
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def _summary_stats(samples: np.ndarray) -> Dict[str, float]:
    if samples.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    ci_low = float(np.quantile(samples, 0.025))
    ci_high = float(np.quantile(samples, 0.975))
    return {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high}


def roc_auc(scores_in, scores_out) -> float:
    """Compute ROC-AUC given member (in) and non-member (out) scores.
    
    Semantics: larger score => more likely member.
    """
    in_np = _to_numpy(scores_in)
    out_np = _to_numpy(scores_out)
    if in_np.size == 0 or out_np.size == 0:
        return 0.0
    labels = np.concatenate([np.ones_like(in_np), np.zeros_like(out_np)])
    scores = np.concatenate([in_np, out_np])
    return float(roc_auc_score(labels, scores))


def tpr_precision_at_fpr(
    scores_in,
    scores_out,
    target_fpr: float,
    num_bootstrap: int = 0,
    seed: int = 0,
) -> Dict:
    """Compute TPR/precision at a target FPR.
    
    Semantics: larger score => more likely member.
    """
    in_np = _to_numpy(scores_in)
    out_np = _to_numpy(scores_out)

    if target_fpr <= 0.0 or out_np.size == 0:
        threshold = np.inf
    else:
        # Upper-tail quantile: ensure FPR ~ target_fpr
        quantile_level = 1.0 - target_fpr
        try:
            threshold = float(np.quantile(out_np, quantile_level, method="linear"))
        except TypeError:
            # Fallback for older NumPy where `method` is not available
            threshold = float(np.quantile(out_np, quantile_level, interpolation="linear"))

    tp = int((in_np > threshold).sum())
    fp = int((out_np > threshold).sum())
    fn = int(in_np.size - tp)
    tn = int(out_np.size - fp)

    tpr = tp / in_np.size if in_np.size else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    achieved_fpr = fp / out_np.size if out_np.size else 0.0

    result: Dict[str, object] = {
        "threshold": threshold,
        "tpr": float(tpr),
        "precision": float(precision),
        "achieved_fpr": float(achieved_fpr),
        "fpr_error": float(achieved_fpr - target_fpr),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }

    if num_bootstrap and (in_np.size + out_np.size) > 0:
        rng = np.random.default_rng(seed)
        idx_in = np.arange(in_np.size)
        idx_out = np.arange(out_np.size)
        boot_tpr = []
        boot_precision = []
        for _ in range(num_bootstrap):
            sample_in = in_np[rng.choice(idx_in, size=in_np.size, replace=True)] if in_np.size else in_np
            sample_out = out_np[rng.choice(idx_out, size=out_np.size, replace=True)] if out_np.size else out_np
            stats = tpr_precision_at_fpr(sample_in, sample_out, target_fpr, num_bootstrap=0)
            boot_tpr.append(stats["tpr"])
            boot_precision.append(stats["precision"])
        result["bootstrap"] = {
            "tpr": _summary_stats(np.array(boot_tpr)),
            "precision": _summary_stats(np.array(boot_precision)),
        }

    return result


__all__ = ["roc_auc", "tpr_precision_at_fpr"]

