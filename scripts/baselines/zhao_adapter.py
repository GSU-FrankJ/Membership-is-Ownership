"""
Zhao et al. (WatermarkDM) adapter for the unified eval harness.

Wraps the EDM-based WatermarkDM pipeline for t-error computation
and native watermark bit-accuracy verification.

Reference: "A Recipe for Watermarking Diffusion Models" (arXiv:2303.10137)

Two-stage pipeline:
  Stage 1: StegaStamp encoder/decoder embeds 64-bit string into images
  Stage 2: EDM (SongUNet, ddpmpp) trained on watermarked images

EDM checkpoint format: pickle dict with key 'ema' containing full
EDMPrecond wrapper. Call signature: model(x_noisy, sigma) -> x_hat_0
where x_noisy = x0 + sigma * eps (EDM noise convention), images in [-1,1].
"""

from __future__ import annotations

import pathlib
import pickle
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WATERMARKDM_REPO = (
    PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo"
)
EDM_DIR = WATERMARKDM_REPO / "edm"
if str(EDM_DIR) not in sys.path:
    sys.path.insert(0, str(EDM_DIR))

from scripts.baselines import register_method
from scripts.baselines.mio_adapter import compute_cohens_d, three_point_check
from scripts.baselines.edm_sigma_mapping import (
    alpha_bar_cosine,
    sigma_from_alpha_bar,
)


def load_edm_model(checkpoint_path: pathlib.Path, device: str = "cuda"):
    """Load an EDM checkpoint and return the full EDMPrecond network in eval mode.

    EDM checkpoints are pickle files containing a dict with key 'ema'
    which is the full EDMPrecond wrapper (includes c_skip/c_out/c_in/c_noise
    preconditioning around the SongUNet backbone).
    """
    checkpoint_path = pathlib.Path(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    net = data["ema"]
    net = net.to(device)
    net.eval()
    return net


def compute_edm_t_error(
    edm_model,
    x0: torch.Tensor,
    K: int = 50,
    seed: int = 42,
    device: str = "cuda",
) -> np.ndarray:
    """Compute per-sample Q25 t-error scores using sigma-mapping.

    For each sample and K random timesteps from our cosine schedule:
      1. Convert timestep t -> alpha_bar -> sigma via sigma = sqrt((1-abar)/abar)
      2. Forward noise in EDM convention: x_sigma = x0 + sigma * eps
      3. Denoise via full preconditioned network: x_hat_0 = edm_model(x_sigma, sigma)
      4. Compute MSE: ||x_hat_0 - x0||^2 per sample, averaged over spatial dims

    Returns Q25 (25th percentile) of MSE across K noise levels per sample.

    Args:
        edm_model: EDMPrecond network in eval mode.
        x0: Clean images, shape (N, C, H, W), expected in [-1, 1].
        K: Number of random timesteps to sample.
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        scores: numpy array of shape (N,) with Q25 t-error per sample.
    """
    rng = np.random.RandomState(seed)
    timesteps = rng.randint(0, 1000, size=K)

    # Precompute sigmas for chosen timesteps
    sigmas = []
    for t in timesteps:
        ab = alpha_bar_cosine(t)
        sig = sigma_from_alpha_bar(ab)
        sigmas.append(sig)

    x0 = x0.to(device).float()
    N = x0.shape[0]
    all_mse = torch.zeros(N, K, device=device)

    with torch.no_grad():
        for k_idx, (t, sig) in enumerate(zip(timesteps, sigmas)):
            sigma_tensor = torch.full((N,), sig, device=device, dtype=torch.float32)

            # EDM noise convention: x_noisy = x0 + sigma * eps
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma_tensor.view(-1, 1, 1, 1) * eps

            # Denoise: EDMPrecond outputs denoised x_hat_0 directly
            x_hat_0 = edm_model(x_noisy, sigma_tensor).float()

            # Per-sample MSE averaged over (C, H, W)
            mse = ((x_hat_0 - x0) ** 2).mean(dim=(1, 2, 3))
            all_mse[:, k_idx] = mse

    # Q25 aggregation across K timesteps
    scores = torch.quantile(all_mse, 0.25, dim=1).cpu().numpy()
    return scores


@register_method("zhao")
class ZhaoAdapter:
    """Adapter wrapping Zhao et al. (WatermarkDM) for the eval harness."""

    def __init__(
        self,
        checkpoint: pathlib.Path,
        model_cfg: dict | None = None,
        device: str = "cuda",
        decoder_path: str | None = None,
        bit_length: int = 64,
    ):
        self.checkpoint = pathlib.Path(checkpoint)
        self.device = device
        self.bit_length = bit_length

        self.edm_model = load_edm_model(self.checkpoint, device)

        # Optionally load StegaStamp decoder for native verification
        self.decoder = None
        if decoder_path is not None:
            self.decoder = self._load_stegastamp_decoder(decoder_path, bit_length)

    def _load_stegastamp_decoder(self, decoder_path: str, bit_length: int):
        """Load StegaStamp decoder for watermark bit extraction."""
        string2img_dir = WATERMARKDM_REPO / "string2img"
        if str(string2img_dir) not in sys.path:
            sys.path.insert(0, str(string2img_dir))

        from models import StegaStampDecoder

        state_dict = torch.load(decoder_path, map_location="cpu")
        decoder = StegaStampDecoder(
            resolution=32, IMAGE_CHANNELS=3, fingerprint_size=bit_length
        )
        decoder.load_state_dict(state_dict)
        decoder.to(self.device)
        decoder.eval()
        return decoder

    def compute_scores(
        self,
        dataloader: DataLoader,
        k_timesteps: int = 50,
        agg: str = "q25",
    ) -> torch.Tensor:
        """Compute per-sample t-error scores using sigma-mapped EDM denoising."""
        all_scores = []
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            # Ensure images are in [-1, 1] for EDM
            if images.min() >= 0:
                images = images * 2.0 - 1.0
            scores = compute_edm_t_error(
                self.edm_model,
                images,
                K=k_timesteps,
                seed=42,
                device=self.device,
            )
            all_scores.append(scores)
        return torch.tensor(np.concatenate(all_scores))

    def native_verify(
        self, member_scores: np.ndarray, nonmember_scores: np.ndarray
    ) -> Dict:
        """Zhao native verification = bit accuracy + three-point check.

        Bit accuracy from generated images requires separate invocation
        of the StegaStamp decoder on EDM-generated samples. Here we
        report t-error-based ownership stats.
        """
        passed, details = three_point_check(member_scores, nonmember_scores)
        return {"pass": passed, **details}

    def extract_bits(self, images: torch.Tensor) -> torch.Tensor:
        """Extract watermark bits from images using StegaStamp decoder.

        Args:
            images: tensor (N, 3, 32, 32) in [0, 1].

        Returns:
            bits: tensor (N, bit_length) of 0/1 predictions.
        """
        if self.decoder is None:
            raise RuntimeError("No decoder loaded -- provide decoder_path")
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.decoder(images)
            bits = (logits > 0).long()
        return bits.cpu()
