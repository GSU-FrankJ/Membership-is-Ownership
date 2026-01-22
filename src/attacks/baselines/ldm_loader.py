"""
Latent Diffusion Model (LDM) loader for baseline comparison.

LDMs operate in a compressed latent space using a VAE:
1. Images are encoded to latent representations via VAE encoder
2. Diffusion operates in latent space
3. Latents are decoded back to images via VAE decoder

For t-error computation with LDMs, we:
1. Encode input images to latent space
2. Apply forward diffusion in latent space
3. Predict noise with UNet in latent space
4. Compute reconstruction error in latent space

Supported Models:
- CompVis/ldm-celebahq-256 (256x256 CelebA faces)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LDMWrapper(nn.Module):
    """
    Wrapper for Latent Diffusion Model to match our model interface.
    
    For t-error computation, we operate entirely in latent space since
    that's where the diffusion happens.
    
    Args:
        vae: VAE encoder/decoder module
        unet: UNet for latent denoising
        input_mean: Dataset channel means for input normalization
        input_std: Dataset channel stds for input normalization
        model_resolution: Model's expected input resolution (256 for LDM)
        input_resolution: Actual input data resolution
        latent_scale_factor: VAE downscale factor (typically 8)
    """
    
    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        input_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        input_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        model_resolution: int = 256,
        input_resolution: Optional[int] = None,
        latent_scale_factor: int = 8,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.model_resolution = model_resolution
        self.input_resolution = input_resolution or model_resolution
        self.latent_scale_factor = latent_scale_factor
        
        # Freeze VAE and UNet - we don't train them
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # Register normalization parameters
        self.register_buffer(
            "input_mean",
            torch.tensor(input_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std",
            torch.tensor(input_std).view(1, 3, 1, 1)
        )
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            x: Images in dataset normalized space [B, 3, H, W]
        
        Returns:
            Latent representations [B, C_latent, H/scale, W/scale]
        """
        # Convert to [-1, 1] range for VAE
        x_01 = x * self.input_std + self.input_mean
        x_vae = x_01 * 2.0 - 1.0
        
        # Resize to model resolution if needed
        if x.shape[-1] != self.model_resolution:
            x_vae = F.interpolate(
                x_vae,
                size=self.model_resolution,
                mode='bilinear',
                align_corners=False
            )
        
        # Encode using VAE
        with torch.no_grad():
            posterior = self.vae.encode(x_vae)
            if hasattr(posterior, 'latent_dist'):
                # diffusers style
                latent = posterior.latent_dist.sample()
            elif hasattr(posterior, 'sample'):
                latent = posterior.sample()
            else:
                latent = posterior
            
            # Scale latent by VAE scaling factor if present
            if hasattr(self.vae.config, 'scaling_factor'):
                latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - operates in latent space for LDM.
        
        For t-error computation, x is expected to already be in latent space.
        Use encode_to_latent() first to convert images to latents.
        
        Args:
            x: Latent tensor [B, C_latent, H_lat, W_lat]
            t: Timestep tensor [B]
        
        Returns:
            Predicted noise in latent space [B, C_latent, H_lat, W_lat]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        output = self.unet(x, t, return_dict=True)
        return output.sample
    
    def forward_from_images(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convenience method that handles encoding automatically.
        
        Note: For t-error computation, you typically want to use the
        latent-space version directly to avoid VAE overhead.
        
        Args:
            x: Input images in dataset normalized space [B, 3, H, W]
            t: Timestep tensor [B]
        
        Returns:
            Predicted noise in latent space [B, C_latent, H_lat, W_lat]
        """
        latent = self.encode_to_latent(x)
        return self.forward(latent, t)


def load_ldm_baseline(
    model_id: str = "CompVis/ldm-celebahq-256",
    device: str = "cuda",
    input_mean: Optional[Tuple[float, ...]] = None,
    input_std: Optional[Tuple[float, ...]] = None,
    input_resolution: Optional[int] = None,
) -> Tuple[LDMWrapper, torch.Tensor]:
    """
    Load a Latent Diffusion Model from HuggingFace.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
        input_mean: Dataset normalization means
        input_std: Dataset normalization stds
        input_resolution: Input data resolution for resizing
    
    Returns:
        Tuple of (wrapped_model, alphas_bar)
    
    Note:
        For t-error computation with LDM, you need to:
        1. Encode images to latent space: latent = model.encode_to_latent(images)
        2. Apply forward diffusion in latent space
        3. Use model.forward(latent, t) for noise prediction
        4. Compute error in latent space
    """
    try:
        from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
    except ImportError:
        raise ImportError("diffusers package required. Install with: pip install diffusers")
    
    logger.info(f"Loading LDM model: {model_id}")
    
    # Try to load VAE
    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        logger.info("Loaded VAE from subfolder 'vae'")
    except Exception as e:
        logger.warning(f"Failed to load VAE from subfolder: {e}")
        try:
            vae = AutoencoderKL.from_pretrained(model_id)
            logger.info("Loaded VAE from model root")
        except Exception:
            raise ValueError(f"Could not load VAE from {model_id}")
    
    # Try to load UNet
    try:
        unet = UNet2DModel.from_pretrained(model_id, subfolder="unet")
        logger.info("Loaded UNet from subfolder 'unet'")
    except Exception as e:
        logger.warning(f"Failed to load UNet from subfolder: {e}")
        raise ValueError(f"Could not load UNet from {model_id}")
    
    # Create scheduler for alpha_bar
    try:
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    except Exception:
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    alphas_bar = scheduler.alphas_cumprod.clone()
    
    # Log model info
    vae_params = sum(p.numel() for p in vae.parameters())
    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"Loaded VAE with {vae_params:,} parameters")
    logger.info(f"Loaded UNet with {unet_params:,} parameters")
    logger.info(
        f"Scheduler: T={len(alphas_bar)}, "
        f"alpha_bar[0]={alphas_bar[0]:.4f}, "
        f"alpha_bar[-1]={alphas_bar[-1]:.6f}"
    )
    
    # Determine latent scale factor
    if hasattr(vae.config, 'sample_size') and hasattr(unet.config, 'sample_size'):
        latent_scale = vae.config.sample_size // unet.config.sample_size
    else:
        latent_scale = 8  # Default for most LDMs
    
    logger.info(f"Latent scale factor: {latent_scale}")
    
    # Default normalization
    if input_mean is None:
        input_mean = (0.5, 0.5, 0.5)
    if input_std is None:
        input_std = (0.5, 0.5, 0.5)
    
    wrapped_model = LDMWrapper(
        vae=vae,
        unet=unet,
        input_mean=input_mean,
        input_std=input_std,
        model_resolution=256,
        input_resolution=input_resolution,
        latent_scale_factor=latent_scale,
    )
    
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    alphas_bar = alphas_bar.to(device)
    
    return wrapped_model, alphas_bar


def compute_ldm_t_error(
    images: torch.Tensor,
    timesteps: list,
    model: LDMWrapper,
    alphas_bar: torch.Tensor,
    agg: str = "q25",
) -> torch.Tensor:
    """
    Compute t-error for LDM in latent space.
    
    This is a specialized t-error computation for LDMs that operates
    entirely in latent space.
    
    Args:
        images: Batch of images [B, 3, H, W] in dataset normalized space
        timesteps: List of timesteps to evaluate
        model: LDMWrapper model
        alphas_bar: Alpha_bar schedule
        agg: Aggregation method ("q25", "mean", "median")
    
    Returns:
        Aggregated t-error scores [B]
    """
    model.eval()
    device = images.device
    batch_size = images.size(0)
    
    # Encode to latent space
    with torch.no_grad():
        latent = model.encode_to_latent(images)
    
    # Compute t-error at each timestep in latent space
    all_errors = []
    for t in timesteps:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        alpha_bar_t = alphas_bar[t]
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus = (1 - alpha_bar_t).sqrt()
        
        # Forward diffusion in latent space
        noise = torch.randn_like(latent)
        latent_t = sqrt_alpha_bar * latent + sqrt_one_minus * noise
        
        # Predict noise
        with torch.no_grad():
            eps_pred = model.forward(latent_t, t_tensor)
        
        # Reconstruct latent
        latent_hat = (latent_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar.clamp(min=1e-6)
        
        # Compute L2 error in latent space
        diff = latent_hat - latent
        error = (diff ** 2).view(batch_size, -1).sum(dim=1)
        all_errors.append(error)
    
    # Stack and aggregate
    errors = torch.stack(all_errors, dim=0).T  # [B, T]
    
    if agg == "mean":
        return errors.mean(dim=1)
    elif agg.startswith("q"):
        q_val = int(agg[1:]) / 100.0
        return torch.quantile(errors, q_val, dim=1)
    elif agg == "median":
        return torch.median(errors, dim=1).values
    else:
        raise ValueError(f"Unknown aggregation: {agg}")
