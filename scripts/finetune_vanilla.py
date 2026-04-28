#!/usr/bin/env python3
"""Vanilla SGD fine-tuning with standard diffusion loss (noise prediction MSE).

No MMD, no CLIP features — just continues training from a checkpoint with
the standard denoising objective. Used for robustness evaluation (Table 6).
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule

LOGGER = get_winston_logger(__name__)


class EMA:
    """Simple EMA tracker using deepcopy (no model.config dependency)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def to(self, device) -> None:
        self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_indices(path):
    import numpy as np
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return list(data)


def main():
    parser = argparse.ArgumentParser(description="Vanilla SGD fine-tuning")
    parser.add_argument("--base-checkpoint", type=pathlib.Path, required=True,
                       help="Path to Model A checkpoint")
    parser.add_argument("--model-config", type=pathlib.Path, required=True)
    parser.add_argument("--data-config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ckpt-interval", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load configs
    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)

    T = model_cfg["diffusion"]["timesteps"]
    image_shape = data_cfg["dataset"].get("image_shape", [3, 32, 32])
    image_size = image_shape[-1]

    # Build model and load checkpoint
    model = build_unet(model_cfg["model"])
    ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.train()
    LOGGER.info("Loaded base checkpoint: %s", args.base_checkpoint)

    # EMA
    ema = EMA(model, decay=args.ema_decay)
    ema.to(device)

    # Diffusion schedule
    _, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    # Optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    LOGGER.info("Optimizer: %s (lr=%s)", args.optimizer, args.lr)

    # Build dataloader (reuse finetune_mmd_ddm's MultiDatasetSubset)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from finetune_mmd_ddm import MultiDatasetSubset, load_indices as load_idx

    indices_path = pathlib.Path(data_cfg["splits"]["paths"]["member_train"])
    member_indices = load_idx(indices_path)
    dataset_name = data_cfg["dataset"].get("name", "cifar10")
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])

    dataset = MultiDatasetSubset(
        dataset_name=dataset_name, root=root, indices=member_indices,
        mean=mean, std=std, image_size=image_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    data_iter = iter(dataloader)
    LOGGER.info("Dataset: %s, %d samples, size=%dx%d", dataset_name, len(dataset), image_size, image_size)

    # Training loop — standard diffusion loss
    for step in range(1, args.iterations + 1):
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x0, _ = next(data_iter)
        x0 = x0.to(device)

        # Sample random timesteps
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)

        # Forward diffusion
        sqrt_alpha = alphas_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - alphas_bar[t]).sqrt().view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise

        # Predict noise
        eps_pred = model(x_t, t)
        loss = torch.nn.functional.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        if step % args.log_interval == 0:
            LOGGER.info("Step %d/%d  loss=%.6f", step, args.iterations, loss.item())

        if step % args.ckpt_interval == 0 or step == args.iterations:
            ckpt_path = args.output / f"ckpt_{step:04d}_ema.pt"
            ema_state = ema.ema_model.state_dict()
            torch.save({"model": ema_state}, ckpt_path)
            LOGGER.info("Saved EMA checkpoint: %s", ckpt_path)

    LOGGER.info("Fine-tuning complete. Final checkpoint at %s", args.output)


if __name__ == "__main__":
    main()
