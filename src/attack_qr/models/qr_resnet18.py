"""ResNet18-based quantile regression model for membership inference.

This module provides the ResNet18QR model that takes (image, stats) as input
and predicts quantiles of the t-error distribution for membership inference.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ResNet18QR(nn.Module):
    """ResNet18 backbone with stats fusion for quantile regression.
    
    The model takes two inputs:
    - images: CIFAR-10 images [B, 3, 32, 32]
    - stats: Summary statistics from t-error sequence [B, D]
    
    For the q25 workflow, stats contains [mean_error, std_error, l2_error] (D=3).
    The stats do NOT include the q25 score itself to avoid information leakage.
    
    Architecture:
    1. ResNet18 backbone extracts image features [B, 512]
    2. Stats are concatenated: [B, 512 + D]
    3. MLP head predicts quantiles: [B, num_outputs]
    
    Attributes:
        stats_dim: Dimension of the stats input (default: 3 for q25 workflow)
    """
    
    # Default stats dimension for q25 workflow (mean_error, std_error, l2_error)
    DEFAULT_STATS_DIM = 3
    
    def __init__(self, num_outputs: int, stats_dim: int = 3, dropout: float = 0.0):
        """Initialize the model.
        
        Args:
            num_outputs: Number of quantile outputs (e.g., 2 for tau=0.01 and tau=0.001)
            stats_dim: Dimension of stats input. Default 3 for q25 workflow
                       (mean_error, std_error, l2_error). Use 4 for legacy pairs workflow.
            dropout: Dropout rate for regularization (default: 0.0)
        """
        super().__init__()
        if stats_dim <= 0:
            raise ValueError("stats_dim must be positive")
        
        self.stats_dim = stats_dim
        
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        hidden_features = in_features // 2  # 256
        layers = [
            nn.Linear(in_features + stats_dim, in_features),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_features, num_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Input images [B, 3, 32, 32]
            stats: Summary stats [B, D] where D = stats_dim
        
        Returns:
            Quantile predictions [B, num_outputs]
        """
        features = self.backbone(images)  # [B, 512]
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        combined = torch.cat([features, stats], dim=1)  # [B, 512 + D]
        return self.head(combined)  # [B, num_outputs]


class ResNet18GaussianQR(nn.Module):
    """ResNet18 + Gaussian head for conditional score distribution modeling.
    
    The network predicts the mean and log-std of a Gaussian over
    y = log1p(score), enabling closed-form quantile computation for any tau.
    """

    def __init__(self, stats_dim: int = 3, hidden_dim: int = 256) -> None:
        """Initialize Gaussian QR with shared backbone.
        
        Args:
            stats_dim: Dimensionality of stats input (e.g., mean/std/L2). Default 3.
            hidden_dim: Width of fusion MLP before splitting into mu/log_sigma.
        """
        super().__init__()
        if stats_dim <= 0:
            raise ValueError("stats_dim must be positive")

        self.stats_dim = stats_dim

        # Backbone: keep classification layers to produce a compact 512-d feature.
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Linear(in_features, 512)
        self.backbone = backbone

        # Fusion head: combine image features and stats to regress (mu, log_sigma).
        self.head = nn.Sequential(
            nn.Linear(512 + stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing Gaussian parameters for log-space scores."""
        img_features = self.backbone(images)  # [B, 512]
        combined = torch.cat([img_features, stats], dim=1)  # [B, 512 + stats_dim]
        gaussian_out = self.head(combined)  # [B, 2]
        mu = gaussian_out[:, 0]
        log_sigma = gaussian_out[:, 1]
        return mu, log_sigma
