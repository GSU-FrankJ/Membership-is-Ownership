#!/usr/bin/env python3
"""SGD vanilla fine-tuning adapter for WDM (Peng et al. 2023).

Mirrors MiO's `scripts/finetune_vanilla.py` protocol but targets WDM's
own UNetModel + linear β schedule (not MiO's UNet + cosine). Produces a
single attacked checkpoint at the 500-iter mark.

Protocol (matches MiO SGD-FT):
    data        = clean CIFAR-10 train split (50K images)
    loss        = standard DDPM ε-prediction MSE  F.mse_loss(eps_pred, noise)
    optimizer   = SGD(lr=5e-6, momentum=0.9)
    iterations  = 500 (exact)
    batch size  = 128
    ema decay   = 0.9999
    normalize   = (0.5, 0.5, 0.5) so x0 ∈ [-1, 1]

Only the two backbone-dependent primitives change vs MiO SGD-FT:
    (a) model factory : wdm.script_util.create_model_and_diffusion(**WDM_CIFAR10_CONFIG)
    (b) schedule      : wdm.gaussian_diffusion.get_named_beta_schedule("linear", T)
                        → alphas_bar = cumprod(1 - betas)

All other code is a faithful reimplementation of
scripts/finetune_vanilla.py:110–152 with the above two swaps.

Output:
    experiments/baseline_comparison/results/robustness_attacks/cifar10/
        sgd_ft/wdm_sgd_ft_500.pt
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
if str(WDM_REPO) not in sys.path:
    sys.path.insert(0, str(WDM_REPO))

# Reuse WDM-adapter primitives from Stage 1
from scripts.finetune_mmd_wdm import (
    WDM_CIFAR10_CONFIG,
    DEFAULT_CHECKPOINT,
    build_wdm_model,
    build_wdm_linear_alphas_bar,
    load_wdm_checkpoint,
    sha256_first_n_params,
)


T_STEPS = 1000
DEFAULT_OUTPUT = pathlib.Path(
    "experiments/baseline_comparison/results/robustness_attacks/cifar10/"
    "sgd_ft/wdm_sgd_ft_500.pt"
)
ITERATIONS = 500
BATCH_SIZE = 128
LR = 5e-6
MOMENTUM = 0.9
EMA_DECAY = 0.9999
SEED = 42


class EMA:
    """Simple EMA tracker using deepcopy (matches finetune_vanilla.py:30–46)."""

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


def build_clean_cifar10_dataloader(batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    root = PROJECT_ROOT / "data" / "cifar-10"
    dataset = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- Load WDM UNet + clean EMA checkpoint ---
    print(f"[sgd] Loading WDM UNet + checkpoint {args.checkpoint}")
    model = build_wdm_model()
    info = load_wdm_checkpoint(model, args.checkpoint)
    src_sha = sha256_first_n_params(info["state_dict"], n=10)
    print(f"[sgd] source SHA-256(first 10 params) = {src_sha}")
    print(f"[sgd] UNet params = {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device).train()

    ema = EMA(model, decay=EMA_DECAY)
    ema.to(device)

    # --- Schedule (WDM linear β) ---
    alphas_bar = build_wdm_linear_alphas_bar(T_STEPS, device=device)
    print(f"[sgd] linear α_bar: range=[{alphas_bar[-1].item():.4e}, {alphas_bar[0].item():.4e}]")

    # --- Optimizer: SGD with momentum (matches finetune_vanilla.py) ---
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print(f"[sgd] optimizer = SGD(lr={args.lr}, momentum={args.momentum})")

    # --- Clean CIFAR-10 dataloader ---
    loader = build_clean_cifar10_dataloader(args.batch_size)
    print(f"[sgd] Loaded {len(loader.dataset)} clean CIFAR-10 train samples, batch={args.batch_size}")
    data_iter = iter(loader)

    initial_loss, last_loss = None, None
    t0 = time.time()
    for step in range(1, args.iterations + 1):
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, _ = next(data_iter)
        x0 = x0.to(device)

        t = torch.randint(0, T_STEPS, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        sqrt_ab = alphas_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1mab = (1 - alphas_bar[t]).sqrt().view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise

        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        assert torch.isfinite(loss).item(), f"NaN loss at step {step}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)

        if initial_loss is None:
            initial_loss = loss.item()
        last_loss = loss.item()

        if step == 1 or step % args.log_interval == 0 or step == args.iterations:
            elapsed = time.time() - t0
            print(f"[sgd] step={step}/{args.iterations} loss={loss.item():.6f} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[sgd] done. initial_loss={initial_loss:.6f} final_loss={last_loss:.6f} elapsed={elapsed:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_sha = sha256_first_n_params(ema.ema_model.state_dict(), n=10)
    torch.save({
        "state_dict": ema.ema_model.state_dict(),
        "wdm_config": WDM_CIFAR10_CONFIG,
        "training": {
            "source_checkpoint": str(args.checkpoint),
            "source_sha256_first10": src_sha,
            "output_sha256_first10": out_sha,
            "protocol": "sgd-ft / standard DDPM eps-MSE / WDM linear beta schedule",
            "iterations": args.iterations,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "ema_decay": EMA_DECAY,
            "seed": args.seed,
            "initial_loss": initial_loss,
            "final_loss": last_loss,
            "wall_time_sec": elapsed,
        },
    }, args.output)
    print(f"[sgd] saved → {args.output}")
    print(f"[sgd] output SHA-256(first 10 params) = {out_sha}")


if __name__ == "__main__":
    main()
