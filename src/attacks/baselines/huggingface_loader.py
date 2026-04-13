"""
HuggingFace model loader for baseline comparison.

This module provides utilities to load pretrained diffusion models from HuggingFace
and wrap them in a compatible interface for t-error computation.

Supports:
- Multi-resolution models (32x32, 64x64, 96x96, 256x256)
- Automatic input resizing for resolution mismatch
- Multiple baseline models per dataset

Key difference from our models:
- HuggingFace DDPM expects input in [-1, 1] range
- Our models expect dataset-normalized input (mean/std normalized)
- We handle this conversion and resizing in the wrapper

Usage:
    # Load from registry
    model, alphas_bar = load_baseline_from_registry("ddpm-cifar10", "cifar10")
    
    # Load directly
    model, alphas_bar = load_hf_ddpm_cifar10()
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


# Model registry - maps model names to HuggingFace IDs
BASELINE_MODELS = {
    "ddpm-cifar10": {
        "model_id": "google/ddpm-cifar10-32",
        "resolution": 32,
        "type": "ddpm",
    },
    "ddpm-celebahq": {
        "model_id": "google/ddpm-celebahq-256",
        "resolution": 256,
        "type": "ddpm",
    },
    "ddpm-bedroom": {
        "model_id": "google/ddpm-ema-bedroom-256",
        "resolution": 256,
        "type": "ddpm",
    },
    "ldm-celebahq": {
        "model_id": "CompVis/ldm-celebahq-256",
        "resolution": 256,
        "type": "ldm",
    },
    "ddpm-church": {
        "model_id": "google/ddpm-ema-church-256",
        "resolution": 256,
        "type": "ddpm",
    },
}


def load_baselines_config(config_path: pathlib.Path) -> Dict:
    """Load baseline configuration from YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_baselines_for_dataset(dataset: str, config_path: Optional[pathlib.Path] = None) -> List[Dict]:
    """
    Get list of baseline models for a dataset.
    
    Args:
        dataset: Dataset name (cifar10, cifar100, stl10, celeba)
        config_path: Optional path to baselines config YAML
    
    Returns:
        List of baseline model configs
    """
    if config_path and config_path.exists():
        config = load_baselines_config(config_path)
        return config.get(dataset, [])
    
    # Default fallback
    defaults = {
        "cifar10": [BASELINE_MODELS["ddpm-cifar10"], BASELINE_MODELS["ddpm-bedroom"]],
        "cifar100": [BASELINE_MODELS["ddpm-cifar10"], BASELINE_MODELS["ddpm-bedroom"]],
        "stl10": [BASELINE_MODELS["ddpm-cifar10"], BASELINE_MODELS["ddpm-church"], BASELINE_MODELS["ddpm-bedroom"]],
        "celeba": [BASELINE_MODELS["ddpm-celebahq"], BASELINE_MODELS["ldm-celebahq"], BASELINE_MODELS["ddpm-bedroom"]],
    }
    return defaults.get(dataset, [])


class HFModelWrapper(nn.Module):
    """
    Wrapper for HuggingFace UNet2DModel to match our model interface.
    
    This wrapper handles:
    1. Input normalization conversion (dataset mean/std -> [-1, 1])
    2. Resolution adaptation (resize input to model's expected size)
    3. Output format matching (just return sample, not full dict)
    4. Timestep handling (HF expects 1D tensor)
    
    Args:
        hf_unet: HuggingFace UNet2DModel
        input_mean: Dataset channel means for input normalization
        input_std: Dataset channel stds for input normalization
        model_resolution: Model's expected input resolution
        input_resolution: Input data's resolution (for resizing)
    """
    
    def __init__(
        self,
        hf_unet: nn.Module,
        input_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        input_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        model_resolution: int = 32,
        input_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.hf_unet = hf_unet
        self.model_resolution = model_resolution
        self.input_resolution = input_resolution or model_resolution
        
        # Register normalization parameters as buffers (move with model)
        self.register_buffer(
            "input_mean", 
            torch.tensor(input_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std",
            torch.tensor(input_std).view(1, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching our model interface.
        
        Args:
            x: Input tensor in dataset normalized space [B, 3, H, W]
            t: Timestep tensor [B]
        
        Returns:
            Predicted noise [B, 3, H, W] (same size as input)
        """
        original_size = x.shape[-1]
        
        # Convert from dataset normalized to [-1, 1] range
        # x_norm = (x_raw - mean) / std
        # x_raw = x_norm * std + mean  (recover [0, 1])
        # x_hf = x_raw * 2 - 1  (convert to [-1, 1])
        x_01 = x * self.input_std + self.input_mean  # [0, 1] range
        x_hf = x_01 * 2.0 - 1.0  # [-1, 1] range
        
        # Resize to model's expected resolution if needed
        if original_size != self.model_resolution:
            x_hf = F.interpolate(
                x_hf, 
                size=self.model_resolution, 
                mode='bilinear', 
                align_corners=False
            )
        
        # HuggingFace UNet expects timestep as 1D tensor
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Forward through HF model
        # HF UNet returns a dict-like object with 'sample' key
        output = self.hf_unet(x_hf, t, return_dict=True)
        noise_pred = output.sample
        
        # Resize back to original resolution if needed
        if original_size != self.model_resolution:
            noise_pred = F.interpolate(
                noise_pred, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return noise_pred


# Backward compatibility alias
def _make_cifar_wrapper(hf_unet, cifar_mean, cifar_std):
    """Create wrapper with CIFAR-style normalization for backward compatibility."""
    return HFModelWrapper(
        hf_unet,
        input_mean=cifar_mean,
        input_std=cifar_std,
        model_resolution=32,
        input_resolution=32,
    )


def load_hf_baseline(
    model_name: str,
    device: str = "cuda",
    input_mean: Optional[Tuple[float, ...]] = None,
    input_std: Optional[Tuple[float, ...]] = None,
    input_resolution: Optional[int] = None,
) -> Tuple[HFModelWrapper, torch.Tensor]:
    """
    Load a HuggingFace baseline model by name from the registry.
    
    Supports automatic resolution adaptation - input images will be resized
    to match the model's expected resolution, then outputs resized back.
    
    Args:
        model_name: Model name from BASELINE_MODELS (e.g., "ddpm-cifar10", "ddpm-celebahq")
        device: Device to load model on
        input_mean: Dataset normalization means (default: [0.5, 0.5, 0.5])
        input_std: Dataset normalization stds (default: [0.5, 0.5, 0.5])
        input_resolution: Resolution of input data (for resize handling)
    
    Returns:
        Tuple of (wrapped_model, alphas_bar)
    
    Example:
        >>> # Load CelebAHQ model for 64x64 CelebA images
        >>> model, alphas_bar = load_hf_baseline(
        ...     "ddpm-celebahq",
        ...     input_mean=(0.5, 0.5, 0.5),
        ...     input_std=(0.5, 0.5, 0.5),
        ...     input_resolution=64
        ... )
    """
    if model_name not in BASELINE_MODELS:
        available = ", ".join(BASELINE_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_info = BASELINE_MODELS[model_name]
    model_id = model_info["model_id"]
    model_resolution = model_info["resolution"]
    model_type = model_info["type"]
    
    # LDM models need special handling
    if model_type == "ldm":
        from src.attacks.baselines.ldm_loader import load_ldm_baseline
        return load_ldm_baseline(
            model_id=model_id,
            device=device,
            input_mean=input_mean,
            input_std=input_std,
            input_resolution=input_resolution,
        )
    
    # Load DDPM model
    try:
        from diffusers import DDPMPipeline
    except ImportError:
        raise ImportError("diffusers package required. Install with: pip install diffusers")
    
    logger.info(f"Loading HuggingFace model: {model_id}")
    
    pipe = DDPMPipeline.from_pretrained(model_id)
    hf_unet = pipe.unet
    scheduler = pipe.scheduler
    alphas_bar = scheduler.alphas_cumprod.clone()
    
    logger.info(f"Loaded model with {sum(p.numel() for p in hf_unet.parameters()):,} parameters")
    logger.info(
        f"Model resolution: {model_resolution}x{model_resolution}, "
        f"Input resolution: {input_resolution or 'same'}"
    )
    
    # Default normalization if not provided
    if input_mean is None:
        input_mean = (0.5, 0.5, 0.5)
    if input_std is None:
        input_std = (0.5, 0.5, 0.5)
    
    wrapped_model = HFModelWrapper(
        hf_unet,
        input_mean=input_mean,
        input_std=input_std,
        model_resolution=model_resolution,
        input_resolution=input_resolution,
    )
    
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    alphas_bar = alphas_bar.to(device)
    
    return wrapped_model, alphas_bar


def load_hf_ddpm_cifar10(
    model_id: str = "google/ddpm-cifar10-32",
    device: str = "cuda",
    cifar_mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
    cifar_std: Tuple[float, ...] = (0.247, 0.2435, 0.2616),
) -> Tuple[HFModelWrapper, torch.Tensor]:
    """
    Load a pretrained DDPM model from HuggingFace for CIFAR-10.
    
    This is the original function kept for backward compatibility.
    For new code, prefer load_hf_baseline() with explicit parameters.
    
    Args:
        model_id: HuggingFace model identifier (default: google/ddpm-cifar10-32)
        device: Device to load model on
        cifar_mean: CIFAR-10 normalization means
        cifar_std: CIFAR-10 normalization stds
    
    Returns:
        Tuple of (wrapped_model, alphas_bar)
    """
    try:
        from diffusers import DDPMPipeline
    except ImportError:
        raise ImportError(
            "diffusers package required. Install with: pip install diffusers"
        )
    
    logger.info(f"Loading HuggingFace model: {model_id}")
    
    pipe = DDPMPipeline.from_pretrained(model_id)
    hf_unet = pipe.unet
    scheduler = pipe.scheduler
    alphas_bar = scheduler.alphas_cumprod.clone()
    
    logger.info(
        f"Loaded model with {sum(p.numel() for p in hf_unet.parameters()):,} parameters"
    )
    logger.info(
        f"Scheduler: T={len(alphas_bar)}, "
        f"alpha_bar[0]={alphas_bar[0]:.4f}, "
        f"alpha_bar[-1]={alphas_bar[-1]:.6f}"
    )
    
    # Use backward-compatible wrapper
    wrapped_model = HFModelWrapper(
        hf_unet,
        input_mean=cifar_mean,
        input_std=cifar_std,
        model_resolution=32,
        input_resolution=32,
    )
    
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    alphas_bar = alphas_bar.to(device)
    
    return wrapped_model, alphas_bar


def load_baseline_from_registry(
    baseline_name: str,
    dataset: str,
    device: str = "cuda",
    config_path: Optional[pathlib.Path] = None,
) -> Tuple[HFModelWrapper, torch.Tensor]:
    """
    Load a baseline model using the dataset registry configuration.
    
    This is the recommended way to load baselines - it automatically
    handles dataset-specific normalization and resolution.
    
    Args:
        baseline_name: Name of baseline (e.g., "ddpm-cifar10", "ddpm-celebahq")
        dataset: Dataset name for normalization parameters
        device: Device to load model on
        config_path: Optional path to baselines_by_dataset.yaml
    
    Returns:
        Tuple of (wrapped_model, alphas_bar)
    """
    # Dataset-specific normalization
    DATASET_NORMS = {
        "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.247, 0.2435, 0.2616), "res": 32},
        "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761), "res": 32},
        "stl10": {"mean": (0.4467, 0.4398, 0.4066), "std": (0.2603, 0.2566, 0.2713), "res": 96},
        "celeba": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "res": 64},
    }
    
    if dataset not in DATASET_NORMS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    norm = DATASET_NORMS[dataset]
    
    return load_hf_baseline(
        model_name=baseline_name,
        device=device,
        input_mean=norm["mean"],
        input_std=norm["std"],
        input_resolution=norm["res"],
    )


def load_random_baseline(
    device: str = "cuda",
    input_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    input_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
    resolution: int = 32,
) -> Tuple[HFModelWrapper, torch.Tensor]:
    """
    Load a randomly initialized DDPM model as a baseline.
    
    This provides a lower bound - a random model should have maximum t-error
    since it has learned nothing about any data distribution.
    
    Args:
        device: Device to load model on
        input_mean: Dataset normalization means
        input_std: Dataset normalization stds
        resolution: Model resolution (32, 64, 96, or 256)
    
    Returns:
        Tuple of wrapped model and alpha_bar schedule
    """
    try:
        from diffusers import UNet2DModel, DDPMScheduler
    except ImportError:
        raise ImportError(
            "diffusers package required. Install with: pip install diffusers"
        )
    
    logger.info(f"Creating randomly initialized DDPM baseline ({resolution}x{resolution})")
    
    # Scale architecture based on resolution
    if resolution <= 32:
        block_out_channels = (128, 128, 256, 256)
        down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
        up_block_types = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    elif resolution <= 64:
        block_out_channels = (128, 128, 256, 256, 512)
        down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    else:
        block_out_channels = (128, 256, 256, 512, 512)
        down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    
    hf_unet = UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    alphas_bar = scheduler.alphas_cumprod.clone()
    
    logger.info(f"Random model with {sum(p.numel() for p in hf_unet.parameters()):,} parameters")
    
    wrapped_model = HFModelWrapper(
        hf_unet,
        input_mean=input_mean,
        input_std=input_std,
        model_resolution=resolution,
        input_resolution=resolution,
    )
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    alphas_bar = alphas_bar.to(device)
    
    return wrapped_model, alphas_bar
