"""
Baseline models for MIA comparison.

This module provides utilities for loading and using public baseline models
(e.g., HuggingFace pretrained diffusion models) to compare against our trained
models for ownership verification.

Supports:
- Multi-resolution models (32x32, 64x64, 96x96, 256x256)
- Automatic input resizing for resolution mismatch
- Multiple baseline models per dataset
- Latent Diffusion Models (LDM)

Usage:
    from src.attacks.baselines import load_hf_baseline, load_baseline_from_registry
    
    # Load by model name
    model, alphas_bar = load_hf_baseline("ddpm-celebahq", input_resolution=64)
    
    # Load with dataset-specific normalization
    model, alphas_bar = load_baseline_from_registry("ddpm-cifar10", "cifar10")
"""

from src.attacks.baselines.huggingface_loader import (
    load_hf_ddpm_cifar10,
    load_hf_baseline,
    load_baseline_from_registry,
    load_random_baseline,
    list_baselines_for_dataset,
    HFModelWrapper,
    BASELINE_MODELS,
)
from src.attacks.baselines.t_error_hf import (
    compute_t_error_hf,
    compute_baseline_scores,
)

__all__ = [
    # Main loading functions
    "load_hf_baseline",
    "load_baseline_from_registry",
    "load_hf_ddpm_cifar10",  # Backward compatibility
    "load_random_baseline",
    "list_baselines_for_dataset",
    # Model wrapper
    "HFModelWrapper",
    # Registry
    "BASELINE_MODELS",
    # T-error computation
    "compute_t_error_hf",
    "compute_baseline_scores",
]
