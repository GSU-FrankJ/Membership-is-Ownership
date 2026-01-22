import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpm_ddim.mmd_loss import cubic_kernel, mmd2_components, mmd2_from_features, mmd2_unbiased


def test_mmd_gradients():
    torch.manual_seed(0)
    fx = torch.randn(8, 4, requires_grad=True)
    fy = fx.detach().clone()
    Kxx = cubic_kernel(fx, fx)
    Kyy = cubic_kernel(fy, fy)
    Kxy = cubic_kernel(fx, fy)
    loss = mmd2_unbiased(Kxx, Kyy, Kxy)
    loss.backward()
    assert torch.isfinite(fx.grad).all()


def test_mmd_components_identical():
    torch.manual_seed(0)
    fx = torch.randn(24, 3)
    fy = torch.randn(24, 3)  # same distribution, independent samples
    mmd2, Exx, Eyy, Exy, Kxx, Kyy, Kxy = mmd2_from_features(fx, fy)
    assert torch.isfinite(mmd2)
    assert mmd2.item() > -1.0  # should hover near zero for same distribution
    assert torch.isfinite(Exx + Eyy - 2 * Exy)
    assert Kxx.shape == (fx.shape[0], fx.shape[0])
    assert Kyy.shape == (fy.shape[0], fy.shape[0])
    assert Kxy.shape == (fx.shape[0], fy.shape[0])


def test_mmd_shape_and_type():
    fx = torch.randn(6, 3)
    fy = torch.randn(6, 3)
    Kxx = cubic_kernel(fx, fx)
    Kyy = cubic_kernel(fy, fy)
    Kxy = cubic_kernel(fx, fy)
    loss = mmd2_unbiased(Kxx, Kyy, Kxy)
    assert loss.dim() == 0
    assert loss.dtype == torch.float32
