"""
WDM (Watermark Diffusion Model) adapter for the unified eval harness.

Wraps WDM's UNet and diffusion for t-error computation and native
watermark extraction verification.

Reference: "Intellectual Property Protection of Diffusion Models via
the Watermark Diffusion Process" (arXiv:2306.03436)
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
if str(WDM_REPO) not in sys.path:
    sys.path.insert(0, str(WDM_REPO))

from scripts.baselines import register_method
from scripts.baselines.mio_adapter import compute_cohens_d, three_point_check
from src.attacks.scores.t_error import uniform_timesteps

# WDM imports
from wdm.script_util import create_model_and_diffusion


# ---------------------------------------------------------------------------
# WDM CIFAR-10 config (matches configs/train_mse_wdp.yaml)
# ---------------------------------------------------------------------------
WDM_CIFAR10_CONFIG = dict(
    image_size=32,
    num_channels=128,
    num_res_blocks=3,
    num_heads=4,
    num_heads_upsample=-1,
    attention_resolutions="16,8",
    dropout=0.1,
    learn_sigma=False,
    sigma_small=False,
    class_cond=False,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    use_checkpoint=False,
    use_scale_shift_norm=False,
)


class WDMModelWrapper(torch.nn.Module):
    """Wrap WDM's UNetModel to present a (x_t, t) -> eps interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_t, t):
        return self.model(x_t, t)


def load_wdm_model(checkpoint_path: pathlib.Path, device: str = "cuda"):
    """Load a WDM checkpoint and return (wrapped_model, alphas_bar).

    WDM checkpoints are plain state_dict files saved by train_util.py.
    WDM uses a LINEAR beta schedule, so we compute alphas_bar accordingly.
    """
    model, diffusion = create_model_and_diffusion(**WDM_CIFAR10_CONFIG)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # WDM uses linear schedule: betas in [0.0001, 0.02] over 1000 steps
    alphas_bar = torch.tensor(
        diffusion.alphas_cumprod, dtype=torch.float32, device=device
    )

    wrapped = WDMModelWrapper(model).to(device)
    wrapped.eval()
    return wrapped, alphas_bar, diffusion


def wdp_extract_watermark(
    diffusion,
    model,
    trigger: np.ndarray,
    gamma1: float = 0.8,
    num_samples: int = 16,
    device: str = "cuda",
):
    """Run WDP reverse process to extract watermark images.

    Returns extracted images as numpy array [N, 3, 32, 32] in [-1, 1].
    """
    shape = (num_samples, 3, 32, 32)
    model_fn = model.model if isinstance(model, WDMModelWrapper) else model

    samples = diffusion.wdp_p_sample_loop(
        model_fn,
        shape,
        wdp_gamma1=gamma1,
        wdp_trigger=trigger,
        clip_denoised=True,
        device=device,
        progress=True,
    )
    return samples.cpu().numpy()


@register_method("wdm")
class WDMAdapter:
    """Adapter wrapping WDM for the eval harness."""

    def __init__(
        self,
        checkpoint: pathlib.Path,
        model_cfg: dict | None = None,
        device: str = "cuda",
        trigger_path: str | None = None,
        gamma1: float = 0.8,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.gamma1 = gamma1

        self.model, self.alphas_bar, self.diffusion = load_wdm_model(
            checkpoint, device
        )

        # Load trigger for native verification
        self.trigger = None
        if trigger_path is not None:
            self.trigger = self._load_trigger(trigger_path)

    @staticmethod
    def _load_trigger(path: str) -> np.ndarray:
        """Load trigger image and normalise to [-1, 1], shape (3, 32, 32)."""
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((32, 32), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def compute_scores(
        self,
        dataloader: DataLoader,
        k_timesteps: int = 50,
        agg: str = "q25",
    ) -> torch.Tensor:
        """Compute per-sample t-error scores (same logic as MiO)."""
        from src.attacks.baselines.t_error_hf import compute_baseline_scores

        return compute_baseline_scores(
            dataloader=dataloader,
            model=self.model,
            alphas_bar=self.alphas_bar,
            T=1000,
            k=k_timesteps,
            agg=agg,
            device=self.device,
            desc="wdm-scores",
        )

    def native_verify(
        self, member_scores: np.ndarray, nonmember_scores: np.ndarray
    ) -> Dict:
        """WDM native verification = watermark extraction + three-point check.

        The watermark extraction requires running the WDP reverse process,
        which is done separately (see ``wdp_extract_watermark``).
        Here we report t-error-based ownership stats.
        """
        passed, details = three_point_check(member_scores, nonmember_scores)
        return {"pass": passed, **details}
