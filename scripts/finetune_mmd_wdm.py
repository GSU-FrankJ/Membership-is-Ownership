#!/usr/bin/env python3
"""MMD fine-tuning adapter for the WDM (Peng et al. 2023) watermarked DDPM.

This adapter reuses MiO's MMD-FT primitives
(scripts/finetune_mmd_ddm.py + src/ddpm_ddim/*) but swaps two components so
the loop operates on WDM's native model rather than MiO's DDIM:

    (1) Model factory
        MiO:  src.ddpm_ddim.models.build_unet(config_dict)
        WDM:  wdm.script_util.create_model_and_diffusion(**WDM_CIFAR10_CONFIG)

        WDM's UNetModel forward signature is model(x, timesteps, y=None),
        which is call-compatible with MiO's sampler's model(x, t_batch_long)
        for class-unconditional (y=None) use.

    (2) Noise schedule
        MiO:  src.ddpm_ddim.schedulers.build_cosine_schedule(T)  -> cosine
        WDM:  wdm.gaussian_diffusion.get_named_beta_schedule("linear", T)
              -> linear, matching WDM's training-time schedule.

Everything else (10-step DDIM eta=0 sampling via MiO's differentiable
sampler, cubic-polynomial-kernel MMD on CLIP ViT-B/32 features, 500
iterations at lr=5e-6, EMA decay 0.9999) is unchanged.

Task mode flags:

    --sanity-check
        Load WDM's UNet, load the EMA checkpoint, run a single forward
        pass on a random noise tensor, verify output shape matches input
        shape, print SUCCESS / FAILURE and exit.  No training.

    --iterations N, --lr LR
        Run actual MMD fine-tuning for N iterations.

Output (when training):
    experiments/baseline_comparison/results/robustness_attacks/cifar10/
        mmd_ft/wdm_mmd_ft_500.pt   (final EMA state_dict)

Per Part 4B Stage 1 spec: this file is the adapter only; Stage 2 runs
the full 500-iteration training.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
if str(WDM_REPO) not in sys.path:
    sys.path.insert(0, str(WDM_REPO))
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# WDM CIFAR-10 config — authoritative source: scripts/baselines/wdm_native_extract.py:31-52
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

DEFAULT_CHECKPOINT = pathlib.Path(
    "/data/short/fjiang4/experiments/baseline_comparison/"
    "wdm_repo/exps/exp_03-09-05-10/logs/ema_0.999_300000.pt"
)
DEFAULT_OUTPUT = pathlib.Path(
    "experiments/baseline_comparison/results/robustness_attacks/cifar10/"
    "mmd_ft/wdm_mmd_ft_500.pt"
)

T_STEPS = 1000
DDIM_STEPS = 10
LR = 5e-6
ITERATIONS = 500
BATCH_SIZE = 128
EMA_DECAY = 0.9999
SEED = 42


def build_wdm_model():
    """Load WDM's own UNetModel. Returns the model (diffusion obj discarded)."""
    from wdm.script_util import create_model_and_diffusion
    model, _diffusion = create_model_and_diffusion(**WDM_CIFAR10_CONFIG)
    return model


def build_wdm_linear_alphas_bar(T: int, device: str = "cuda") -> torch.Tensor:
    """Compute alpha_bar_t = cumprod(1 - beta_t) from WDM's linear schedule.

    WDM's linear schedule (from upstream get_named_beta_schedule):
        beta_start = 1000 / T * 1e-4
        beta_end   = 1000 / T * 2e-2
        betas = linspace(beta_start, beta_end, T)
    """
    from wdm.gaussian_diffusion import get_named_beta_schedule
    betas_np = get_named_beta_schedule("linear", T)
    alphas = 1.0 - betas_np
    alphas_bar_np = np.cumprod(alphas)
    alphas_bar = torch.from_numpy(alphas_bar_np).to(device=device, dtype=torch.float32)
    return alphas_bar


def load_wdm_checkpoint(model: torch.nn.Module, ckpt_path: pathlib.Path) -> dict:
    """Load WDM's raw state_dict (not wrapped in {"state_dict": ...})."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = model.load_state_dict(state, strict=True)
    return {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "state_dict": state,
    }


def sha256_first_n_params(state_dict: dict, n: int = 10) -> str:
    """SHA-256 of concatenated float bytes of first N parameters (sorted by key)."""
    keys = sorted(state_dict.keys())[:n]
    h = hashlib.sha256()
    for k in keys:
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Sanity check                                                                #
# --------------------------------------------------------------------------- #
def run_sanity_check(checkpoint: pathlib.Path, device: str) -> bool:
    print(f"[sanity] Loading WDM UNet via wdm.script_util.create_model_and_diffusion ...")
    model = build_wdm_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sanity] UNetModel built: {n_params:,} parameters")

    print(f"[sanity] Loading checkpoint: {checkpoint}")
    if not checkpoint.exists():
        print(f"[sanity] FAILURE: checkpoint not found: {checkpoint}")
        return False
    info = load_wdm_checkpoint(model, checkpoint)
    if info["missing"] or info["unexpected"]:
        print(
            f"[sanity] FAILURE: state_dict mismatch "
            f"missing={len(info['missing'])} unexpected={len(info['unexpected'])}"
        )
        return False
    print(f"[sanity] state_dict loaded strict=True, no missing/unexpected keys")

    sha = sha256_first_n_params(info["state_dict"], n=10)
    print(f"[sanity] SHA-256(first 10 params, key-sorted): {sha}")

    print(f"[sanity] Moving model to {device} and running single forward pass ...")
    model = model.to(device).eval()
    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.tensor([100, 500], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape, f"shape mismatch: input {x.shape} vs output {out.shape}"
    finite = torch.isfinite(out).all().item()
    print(
        f"[sanity] forward pass OK: input {tuple(x.shape)} -> output {tuple(out.shape)}, "
        f"all_finite={finite}, out_stats mean={out.mean().item():.4f} std={out.std().item():.4f}"
    )

    print(f"[sanity] Building WDM linear alpha_bar schedule ...")
    alphas_bar = build_wdm_linear_alphas_bar(T_STEPS, device=device)
    assert alphas_bar.shape == (T_STEPS,), f"alphas_bar shape {alphas_bar.shape}"
    assert torch.all(alphas_bar > 0), "alphas_bar has non-positive entries"
    assert torch.all(alphas_bar <= 1), "alphas_bar > 1"
    monotonic = bool(torch.all(alphas_bar[1:] <= alphas_bar[:-1]).item())
    print(
        f"[sanity] alphas_bar OK: shape={tuple(alphas_bar.shape)} "
        f"range=[{alphas_bar[-1].item():.4e}, {alphas_bar[0].item():.4e}] "
        f"monotone_nonincreasing={monotonic}"
    )

    print(f"[sanity] MiO sampler import check ...")
    from src.ddpm_ddim.samplers.ddim10 import build_linear_timesteps
    ts = build_linear_timesteps(T_STEPS, DDIM_STEPS, start=T_STEPS - 1)
    print(f"[sanity] build_linear_timesteps OK: len={len(ts)} timesteps={ts}")

    print(f"[sanity] SUCCESS: WDM MMD-FT adapter is ready for Stage 2 execution")
    return True


# --------------------------------------------------------------------------- #
# Training loop (used in Stage 2, not Stage 1)                                #
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self) -> dict:
        return self.shadow


def build_cifar10_dataloader(batch_size: int, num_workers: int = 2) -> DataLoader:
    # Matches MiO's normalization: ToTensor + Normalize(0.5, 0.5) for [-1, 1]
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    root = PROJECT_ROOT / "data" / "cifar-10"
    # Use torchvision's canonical CIFAR-10 path
    dataset = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def run_training(
    checkpoint: pathlib.Path,
    output: pathlib.Path,
    iterations: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    from src.ddpm_ddim.samplers.ddim10 import (
        build_linear_timesteps, ddim_sample_differentiable,
    )
    from src.ddpm_ddim.clip_features import extract_clip_features, load_clip
    from src.ddpm_ddim.mmd_loss import cubic_kernel, mmd2_components

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[train] Device: {device}, iterations={iterations}, lr={lr}, batch={batch_size}")
    model = build_wdm_model()
    info = load_wdm_checkpoint(model, checkpoint)
    sha = sha256_first_n_params(info["state_dict"], n=10)
    print(f"[train] Loaded checkpoint {checkpoint.name}, SHA-256(first 10)={sha}")
    model = model.to(device).train()

    ema = EMA(model, decay=EMA_DECAY)

    alphas_bar = build_wdm_linear_alphas_bar(T_STEPS, device=device)
    timesteps = build_linear_timesteps(T_STEPS, DDIM_STEPS, start=T_STEPS - 1)
    print(f"[train] Linear schedule + {DDIM_STEPS}-step DDIM timesteps prepared")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    print(f"[train] Loading CIFAR-10 train loader ...")
    loader = build_cifar10_dataloader(batch_size)
    data_iter = iter(loader)

    print(f"[train] Loading CLIP ViT-B/32 ...")
    clip_bundle = load_clip(device=device)
    data_mean = [0.5, 0.5, 0.5]
    data_std = [0.5, 0.5, 0.5]

    initial_loss = None
    last_loss = None
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, iterations + 1):
        try:
            real_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_batch, _ = next(data_iter)
        real_batch = real_batch.to(device)
        noise = torch.randn(batch_size, 3, 32, 32, device=device)

        gen_batch = ddim_sample_differentiable(
            model=model,
            alphas_bar=alphas_bar,
            shape=noise.shape,
            timesteps=timesteps,
            device=device,
            use_checkpoint=True,
            noise=noise,
            debug_ddim=False,
            debug_dir=None,
            ddim_fp32=True,
            debug_scale=False,
        )
        x0_for_clip = torch.tanh(gen_batch / 3.0)
        fake_feats = extract_clip_features(
            x0_for_clip, clip_bundle, device, enable_grad=True,
            data_mean=data_mean, data_std=data_std,
        )
        real_feats = extract_clip_features(
            real_batch, clip_bundle, device, enable_grad=False,
            data_mean=data_mean, data_std=data_std,
        )
        Kxx = cubic_kernel(fake_feats, fake_feats)
        Kyy = cubic_kernel(real_feats, real_feats)
        Kxy = cubic_kernel(fake_feats, real_feats)
        mmd2, Exx, Eyy, Exy = mmd2_components(Kxx, Kyy, Kxy)
        loss = mmd2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)

        if initial_loss is None:
            initial_loss = loss.item()
        last_loss = loss.item()

        if step == 1 or step % 10 == 0 or step == iterations:
            elapsed = time.time() - t0
            print(
                f"[train] step={step}/{iterations} loss={loss.item():.6f} "
                f"elapsed={elapsed:.1f}s"
            )

    print(f"[train] Done. initial_loss={initial_loss:.6f} final_loss={last_loss:.6f}")
    state = {
        "state_dict": ema.state_dict(),
        "wdm_config": WDM_CIFAR10_CONFIG,
        "training": {
            "source_checkpoint": str(checkpoint),
            "iterations": iterations,
            "lr": lr,
            "batch_size": batch_size,
            "initial_loss": initial_loss,
            "final_loss": last_loss,
            "seed": SEED,
        },
    }
    torch.save(state, output)
    print(f"[train] Saved attacked checkpoint -> {output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sanity-check", action="store_true",
                   help="Load model + checkpoint + run single forward pass; no training")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sanity_check:
        ok = run_sanity_check(args.checkpoint, args.device)
        sys.exit(0 if ok else 1)
    run_training(
        checkpoint=args.checkpoint,
        output=args.output,
        iterations=args.iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
