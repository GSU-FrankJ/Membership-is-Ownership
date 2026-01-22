#!/usr/bin/env python3

"""Evaluate FID and FID_CLIP for 10-step DDIM finetuned models."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import inception_v3, Inception_V3_Weights

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule
from src.ddpm_ddim.samplers.ddim10 import build_linear_timesteps, ddim_sample_differentiable
from src.ddpm_ddim.clip_features import extract_clip_features, load_clip


LOGGER = get_winston_logger(__name__)


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_indices(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _denormalize(images: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return torch.clamp(images * std_t + mean_t, 0.0, 1.0)


def frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    eigvals, eigvecs = torch.linalg.eigh(cov_prod)
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_cov = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    trace_term = torch.trace(sigma1 + sigma2 - 2.0 * sqrt_cov)
    fid = diff.dot(diff) + trace_term
    return float(fid.real.item())


def compute_stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = features.mean(dim=0)
    diff = features - mu
    cov = diff.T @ diff / (features.shape[0] - 1)
    return mu, cov


def build_real_loader(data_cfg: Dict, indices, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=data_cfg["dataset"]["normalization"]["mean"], std=data_cfg["dataset"]["normalization"]["std"])])
    dataset = CIFAR10(
        root=data_cfg["dataset"]["root"],
        train=True,
        download=False,
        transform=transform,
    )
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def prepare_inception(device: torch.device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    meta_mean = torch.tensor(weights.meta["mean"], device=device).view(1, 3, 1, 1)
    meta_std = torch.tensor(weights.meta["std"], device=device).view(1, 3, 1, 1)
    return model, meta_mean, meta_std


def inception_features(model, meta_mean, meta_std, images: torch.Tensor, data_mean, data_std) -> torch.Tensor:
    imgs = _denormalize(images, data_mean, data_std)
    if imgs.shape[-1] != 299 or imgs.shape[-2] != 299:
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
    imgs = (imgs - meta_mean) / meta_std
    with torch.no_grad():
        feats = model(imgs)
    return feats


def accumulate_features(loader, feature_fn, limit: int) -> torch.Tensor:
    feats = []
    seen = 0
    for batch, _ in loader:
        f = feature_fn(batch)
        feats.append(f.cpu())
        seen += batch.shape[0]
        if seen >= limit:
            break
    concat = torch.cat(feats, dim=0)
    return concat[:limit]


def accumulate_fake_features(
    feature_fn,
    num_samples: int,
    batch_size: int,
) -> torch.Tensor:
    feats = []
    remaining = num_samples
    while remaining > 0:
        f = feature_fn(min(batch_size, remaining))
        feats.append(f.cpu())
        remaining -= f.shape[0]
    return torch.cat(feats, dim=0)[:num_samples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and FID_CLIP for 10-step DDIM models.")
    parser.add_argument("--model-config", type=pathlib.Path, default=pathlib.Path("configs/model_ddim.yaml"))
    parser.add_argument("--data-config", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="EMA finetune checkpoint path")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/mmd_eval"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--grad-checkpoint", action="store_true", help="Use gradient checkpointing in sampler")
    parser.add_argument("--t-start", type=int, default=None, help="Optional starting timestep for DDIM10 (default: T-1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet(model_cfg["model"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    T = model_cfg["diffusion"]["timesteps"]
    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)
    t_start = args.t_start if args.t_start is not None else T - 1
    timesteps = build_linear_timesteps(T, 10, start=t_start)

    data_mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    data_std = tuple(data_cfg["dataset"]["normalization"]["std"])

    member_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["member_train"]))
    real_loader = build_real_loader(data_cfg, member_indices, args.batch_size, args.num_workers)

    inception_model, inc_mean, inc_std = prepare_inception(device)
    clip_bundle = load_clip(device=device)

    def real_inception_fn(batch):
        return inception_features(inception_model, inc_mean, inc_std, batch.to(device), data_mean, data_std)

    def real_clip_fn(batch):
        return extract_clip_features(batch.to(device), clip_bundle, device, enable_grad=False, data_mean=data_mean, data_std=data_std)

    LOGGER.info("Accumulating real features...")
    real_inc = accumulate_features(real_loader, real_inception_fn, args.num_samples)
    real_clip = accumulate_features(real_loader, real_clip_fn, args.num_samples)

    def fake_inception_fn(curr_batch: int):
        noise = torch.randn(curr_batch, 3, 32, 32, device=device)
        with torch.no_grad():
            samples = ddim_sample_differentiable(
                model=model,
                alphas_bar=alphas_bar,
                shape=noise.shape,
                timesteps=timesteps,
                device=device,
                use_checkpoint=args.grad_checkpoint,
                noise=noise,
            )
        return inception_features(inception_model, inc_mean, inc_std, samples, data_mean, data_std)

    def fake_clip_fn(curr_batch: int):
        noise = torch.randn(curr_batch, 3, 32, 32, device=device)
        with torch.no_grad():
            samples = ddim_sample_differentiable(
                model=model,
                alphas_bar=alphas_bar,
                shape=noise.shape,
                timesteps=timesteps,
                device=device,
                use_checkpoint=args.grad_checkpoint,
                noise=noise,
            )
        return extract_clip_features(samples, clip_bundle, device, enable_grad=False, data_mean=data_mean, data_std=data_std)

    LOGGER.info("Accumulating fake features...")
    fake_inc = accumulate_fake_features(fake_inception_fn, args.num_samples, args.batch_size)
    fake_clip = accumulate_fake_features(fake_clip_fn, args.num_samples, args.batch_size)

    mu_r_inc, cov_r_inc = compute_stats(real_inc)
    mu_f_inc, cov_f_inc = compute_stats(fake_inc)
    fid = frechet_distance(mu_r_inc, cov_r_inc, mu_f_inc, cov_f_inc)

    mu_r_clip, cov_r_clip = compute_stats(real_clip)
    mu_f_clip, cov_f_clip = compute_stats(fake_clip)
    fid_clip = frechet_distance(mu_r_clip, cov_r_clip, mu_f_clip, cov_f_clip)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {"fid": fid, "fid_clip": fid_clip, "num_samples": args.num_samples}
    out_path = args.out / "metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("FID=%.4f FID_CLIP=%.4f (saved to %s)", fid, fid_clip, out_path)


if __name__ == "__main__":
    main()
