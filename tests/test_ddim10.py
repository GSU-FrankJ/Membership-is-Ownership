import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpm_ddim.samplers.ddim10 import build_linear_timesteps, ddim_sample_differentiable


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):  # type: ignore[override]
        return torch.zeros_like(x)


class NaNModel(torch.nn.Module):
    def forward(self, x, t):  # type: ignore[override]
        return torch.full_like(x, float("nan"))


def test_build_linear_timesteps():
    steps = build_linear_timesteps(1000, 10)
    assert len(steps) == 11
    assert steps[0] == 999
    assert steps[-1] == 0
    assert all(s1 > s2 for s1, s2 in zip(steps, steps[1:]))
    assert all(0 <= s < 1000 for s in steps)


def test_ddim_determinism():
    torch.manual_seed(0)
    alphas_bar = torch.linspace(0.1, 0.99, steps=1000)
    timesteps = build_linear_timesteps(1000, 10)
    noise = torch.randn(2, 3, 4, 4)
    model = ZeroModel()

    out1 = ddim_sample_differentiable(
        model,
        alphas_bar,
        noise.shape,
        timesteps,
        device=torch.device("cpu"),
        use_checkpoint=False,
        noise=noise,
    )
    out2 = ddim_sample_differentiable(
        model,
        alphas_bar,
        noise.shape,
        timesteps,
        device=torch.device("cpu"),
        use_checkpoint=False,
        noise=noise,
    )
    assert torch.allclose(out1, out2)
    assert out1.shape == noise.shape


def test_ddim_debug_dump(tmp_path):
    alphas_bar = torch.linspace(0.1, 0.9, steps=20)
    timesteps = build_linear_timesteps(20, 5)
    noise = torch.randn(2, 3, 4, 4)
    model = NaNModel()

    with pytest.raises(RuntimeError):
        ddim_sample_differentiable(
            model,
            alphas_bar,
            noise.shape,
            timesteps,
            device=torch.device("cpu"),
            use_checkpoint=False,
            noise=noise,
            debug_ddim=True,
            debug_dir=tmp_path,
        )

    assert list(tmp_path.glob("ddim_*_step*.pt"))


def test_ddim_scale_invariant():
    torch.manual_seed(0)
    alphas_bar = torch.linspace(0.5, 0.99, steps=20)
    timesteps = build_linear_timesteps(20, 5)
    noise = torch.randn(2, 3, 4, 4)
    model = ZeroModel()

    out = ddim_sample_differentiable(
        model,
        alphas_bar,
        noise.shape,
        timesteps,
        device=torch.device("cpu"),
        use_checkpoint=False,
        noise=noise,
        debug_scale=True,
        scale_threshold=50.0,
        fail_on_explode=True,
    )
    assert out.abs().max().item() < 50.0
