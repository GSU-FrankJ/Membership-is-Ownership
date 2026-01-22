"""Strict unbiased MMD^2 with cubic polynomial kernel."""

from __future__ import annotations

import torch


def cubic_kernel(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Cubic polynomial kernel k(u,v) = ((u·v)/d + 1)^3."""

    U = U.float()
    V = V.float()
    d = U.shape[-1]
    scale = 1.0 / float(d)
    return torch.pow(torch.matmul(U, V.T) * scale + 1.0, 3)


def mean_offdiag(mat: torch.Tensor) -> torch.Tensor:
    """Mean of off-diagonal elements."""

    n = mat.shape[0]
    if n <= 1:
        return mat.new_tensor(0.0)
    total = mat.sum()
    diag_sum = torch.diagonal(mat).sum()
    return (total - diag_sum) / (n * (n - 1))


def _unbiased_component(mat: torch.Tensor) -> torch.Tensor:
    """Unbiased estimator component (sum minus diagonal)."""

    n = mat.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples for unbiased estimate, got {n}")
    total = mat.sum()
    diag_sum = torch.diagonal(mat).sum()
    return (total - diag_sum) / (n * (n - 1))


def mmd2_components(Kxx: torch.Tensor, Kyy: torch.Tensor, Kxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mmd2, Exx, Eyy, Exy) using unbiased estimators."""

    Kxx = Kxx.float()
    Kyy = Kyy.float()
    Kxy = Kxy.float()

    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"Need n>=2 and m>=2 (got {n}, {m}) for unbiased MMD")

    Exx = _unbiased_component(Kxx)
    Eyy = _unbiased_component(Kyy)
    Exy = Kxy.mean()
    mmd2 = Exx + Eyy - 2.0 * Exy
    return mmd2, Exx, Eyy, Exy


def mmd2_unbiased(Kxx: torch.Tensor, Kyy: torch.Tensor, Kxy: torch.Tensor) -> torch.Tensor:
    """Strict unbiased MMD^2 estimator including xx, yy, xy terms."""

    mmd2, _, _, _ = mmd2_components(Kxx, Kyy, Kxy)
    return mmd2.float()


def mmd2_from_features(fx: torch.Tensor, fy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience helper: compute kernels + components from features."""

    fx = fx.float()
    fy = fy.float()
    Kxx = cubic_kernel(fx, fx)
    Kyy = cubic_kernel(fy, fy)
    Kxy = cubic_kernel(fx, fy)
    mmd2, Exx, Eyy, Exy = mmd2_components(Kxx, Kyy, Kxy)
    return mmd2, Exx, Eyy, Exy, Kxx, Kyy, Kxy



__all__ = ["cubic_kernel", "mmd2_unbiased", "mmd2_components", "mmd2_from_features", "mean_offdiag"]
