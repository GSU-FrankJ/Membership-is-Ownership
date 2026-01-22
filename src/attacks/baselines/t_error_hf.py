"""
T-error computation for HuggingFace baseline models.

This module adapts our t-error computation to work with HuggingFace models,
providing utilities for baseline comparison in ownership verification.

The HFModelWrapper from huggingface_loader.py handles normalization differences,
so this module can use the same t_error logic with minimal adaptation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks.scores.t_error import uniform_timesteps

logger = logging.getLogger(__name__)


def _t_error_single_step(
    x0: torch.Tensor,
    t: int,
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute t-error for a single timestep (compatible with any wrapped model).
    
    This is a copy of the core logic from attacks/scores/t_error.py,
    kept here to avoid circular imports and ensure compatibility.
    
    Args:
        x0: Clean images [B, C, H, W] (CIFAR-normalized)
        t: Timestep (int)
        model: Model that accepts (x, t) and returns noise prediction
        alphas_bar: Alpha_bar schedule [T]
    
    Returns:
        Per-sample L2 reconstruction error [B]
    """
    device = x0.device
    batch_size = x0.size(0)
    
    # Create timestep tensor
    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
    
    # Get alpha_bar for this timestep
    alpha_bar_t = alphas_bar[t]
    sqrt_alpha_bar = alpha_bar_t.sqrt()
    sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
    
    # Forward diffusion: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
    noise = torch.randn_like(x0)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    
    # Predict noise
    with torch.no_grad():
        eps_pred = model(x_t, t_tensor)
    
    # Reconstruct x0
    sqrt_alpha_bar_clamped = sqrt_alpha_bar.clamp(min=1e-6)
    x0_hat = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar_clamped
    
    # Compute L2 error
    diff = x0_hat - x0
    error = (diff ** 2).view(batch_size, -1).sum(dim=1)
    
    return error


def compute_t_error_hf(
    images: torch.Tensor,
    timesteps: Sequence[int],
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
    agg: str = "q25",
) -> torch.Tensor:
    """
    Compute aggregated t-error scores using a HuggingFace model.
    
    Args:
        images: Batch of CIFAR-normalized images [B, 3, 32, 32]
        timesteps: List of timesteps to evaluate
        model: HFModelWrapper or compatible model
        alphas_bar: Alpha_bar schedule [T]
        agg: Aggregation method ("q25", "mean", "median")
    
    Returns:
        Aggregated t-error scores [B]
    """
    model.eval()
    batch_size = images.size(0)
    
    # Collect errors for each timestep
    all_errors = []
    for t in timesteps:
        error = _t_error_single_step(images, t, model, alphas_bar)
        all_errors.append(error)
    
    # Stack: [T, B] -> [B, T]
    errors_per_sample = torch.stack(all_errors, dim=0).T
    
    # Aggregate
    if agg == "mean":
        return errors_per_sample.mean(dim=1)
    elif agg.startswith("q"):
        q_val = int(agg[1:]) / 100.0
        return torch.quantile(errors_per_sample, q_val, dim=1)
    elif agg == "median":
        return torch.median(errors_per_sample, dim=1).values
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


def compute_baseline_scores(
    dataloader: DataLoader,
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
    T: int = 1000,
    k: int = 50,
    agg: str = "q25",
    device: str = "cuda",
    desc: str = "Computing scores",
) -> torch.Tensor:
    """
    Compute t-error scores for a dataset using the given model.
    
    This is the main entry point for computing baseline scores.
    
    Args:
        dataloader: DataLoader yielding (images, labels) or just images
        model: Wrapped model (HFModelWrapper or our UNet)
        alphas_bar: Alpha_bar schedule
        T: Total diffusion timesteps (default: 1000)
        k: Number of timesteps to sample (default: 50)
        agg: Aggregation method (default: "q25")
        device: Compute device
        desc: Progress bar description
    
    Returns:
        All scores concatenated [N]
    """
    # Generate timesteps (drop last one for numerical stability)
    timesteps = uniform_timesteps(T, k + 1)[:-1]
    logger.info(f"Using {len(timesteps)} timesteps for evaluation")
    
    model.eval()
    all_scores = []
    
    for batch in tqdm(dataloader, desc=desc):
        # Handle both (images, labels) and just images
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        
        images = images.to(device)
        
        # Compute t-error scores
        scores = compute_t_error_hf(images, timesteps, model, alphas_bar, agg=agg)
        all_scores.append(scores.cpu())
    
    return torch.cat(all_scores, dim=0)


def compare_models_on_split(
    dataloader: DataLoader,
    models: Dict[str, Tuple[torch.nn.Module, torch.Tensor]],
    T: int = 1000,
    k: int = 50,
    agg: str = "q25",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Compare multiple models on the same data split.
    
    Args:
        dataloader: DataLoader for evaluation data
        models: Dict mapping model_name -> (model, alphas_bar)
        T: Total timesteps
        k: Number of timesteps to sample
        agg: Aggregation method
        device: Compute device
    
    Returns:
        Dict mapping model_name -> scores tensor [N]
    """
    results = {}
    
    for name, (model, alphas_bar) in models.items():
        logger.info(f"Computing scores for: {name}")
        scores = compute_baseline_scores(
            dataloader=dataloader,
            model=model,
            alphas_bar=alphas_bar,
            T=T,
            k=k,
            agg=agg,
            device=device,
            desc=f"scores-{name}",
        )
        results[name] = scores
        logger.info(f"  {name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    return results
