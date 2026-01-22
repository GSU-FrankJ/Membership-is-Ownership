"""CLIP feature extraction utilities with fallbacks."""

from __future__ import annotations

import functools
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)

# Official CLIP normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
CLIP_RESOLUTION = 224


class ClipUnavailable(RuntimeError):
    """Raised when no CLIP backend can be imported."""


def _try_open_clip(model_name: str, device: torch.device):
    try:
        import open_clip  # type: ignore
    except Exception:
        return None

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOGGER.info("Loaded CLIP via open_clip (%s)", model_name)
    return {"backend": "open_clip", "model": model}


def _try_transformers_clip(model_name: str, device: torch.device):
    try:
        from transformers import CLIPVisionModelWithProjection  # type: ignore
    except Exception:
        return None

    model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOGGER.info("Loaded CLIP vision tower via transformers (%s)", model_name)
    return {"backend": "transformers", "model": model}


def _try_clip_pkg(model_name: str, device: torch.device):
    try:
        import clip  # type: ignore
    except Exception:
        return None
    model, _ = clip.load(model_name, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOGGER.info("Loaded CLIP via clip package (%s)", model_name)
    return {"backend": "clip", "model": model}


@functools.lru_cache(maxsize=1)
def load_clip(model_name: str = "ViT-B-32", device: Optional[torch.device] = None):
    """Load a CLIP model; fallback order: open_clip -> transformers -> clip."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = (
        _try_open_clip(model_name, device)
        or _try_transformers_clip("openai/clip-vit-base-patch32", device)
        or _try_clip_pkg(model_name, device)
    )
    if bundle is None:
        raise ClipUnavailable("No CLIP backend available; install open_clip_torch or transformers.")
    bundle["device"] = device
    return bundle


def _tensor_preprocess(
    images: torch.Tensor,
    device: torch.device,
    data_mean: Optional[Sequence[float]] = None,
    data_std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Differentiable preprocessing matching CLIP defaults."""

    if data_mean is not None and data_std is not None:
        mean = torch.tensor(data_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        img_01 = torch.clamp(images * std + mean, 0.0, 1.0)
    else:
        img_01 = torch.clamp((images + 1.0) * 0.5, 0.0, 1.0)
    if img_01.shape[-1] != CLIP_RESOLUTION or img_01.shape[-2] != CLIP_RESOLUTION:
        img_01 = F.interpolate(img_01, size=(CLIP_RESOLUTION, CLIP_RESOLUTION), mode="bicubic", align_corners=False)
    mean = CLIP_MEAN.to(device, dtype=img_01.dtype)
    std = CLIP_STD.to(device, dtype=img_01.dtype)
    return (img_01.to(device) - mean) / std


def extract_clip_features(
    images: torch.Tensor,
    clip_bundle: dict,
    device: torch.device,
    enable_grad: bool = True,
    data_mean: Optional[Sequence[float]] = None,
    data_std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Encode images (diffusion space) into CLIP features."""

    backend = clip_bundle["backend"]
    model = clip_bundle["model"]
    processed = _tensor_preprocess(images, device, data_mean=data_mean, data_std=data_std)

    ctx = torch.enable_grad if enable_grad else torch.no_grad
    with ctx():
        if backend == "open_clip":
            feats = model.encode_image(processed)
        elif backend == "transformers":
            outputs = model(pixel_values=processed)
            feats = outputs.image_embeds
        else:  # clip package
            feats = model.encode_image(processed)

    feats = feats.float()
    return feats


__all__ = ["load_clip", "extract_clip_features", "ClipUnavailable", "CLIP_MEAN", "CLIP_STD", "CLIP_RESOLUTION"]
