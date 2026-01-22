#!/usr/bin/env python3

"""Full MMD finetuning with 10-step DDIM (η=0) - Multi-dataset support.

Supports:
- CIFAR-10 (32x32)
- CIFAR-100 (32x32)
- STL-10 (96x96)
- CelebA (64x64)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule
from src.ddpm_ddim.samplers.ddim10 import build_linear_timesteps, ddim_sample_differentiable
from src.ddpm_ddim.clip_features import extract_clip_features, load_clip
from src.ddpm_ddim.mmd_loss import cubic_kernel, mmd2_components, mmd2_from_features, mmd2_unbiased


LOGGER = get_winston_logger(__name__)


class MultiDatasetSubset(Dataset):
    """Generic dataset subset for multi-dataset support."""

    def __init__(
        self,
        dataset_name: str,
        root: pathlib.Path,
        indices: List[int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        image_size: int = 32,
    ):
        """
        Args:
            dataset_name: One of 'cifar10', 'cifar100', 'stl10', 'celeba'
            root: Dataset root directory
            indices: Indices to include in subset
            mean: Normalization mean
            std: Normalization std
            image_size: Target image size (for resize)
        """
        self.dataset_name = dataset_name.lower()
        self.image_size = image_size
        
        # Build transforms
        transform_list = []
        if self.dataset_name == "stl10" and image_size != 96:
            transform_list.append(transforms.Resize(image_size))
        elif self.dataset_name == "celeba":
            transform_list.extend([
                transforms.CenterCrop(178),
                transforms.Resize(image_size),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform = transforms.Compose(transform_list)
        
        # Load base dataset
        if self.dataset_name == "cifar10":
            base = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
        elif self.dataset_name == "cifar100":
            base = datasets.CIFAR100(root=str(root), train=True, download=False, transform=transform)
        elif self.dataset_name == "stl10":
            base = datasets.STL10(root=str(root), split="train", download=False, transform=transform)
        elif self.dataset_name == "celeba":
            base = datasets.CelebA(root=str(root), split="train", download=False, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.subset = Subset(base, indices)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        return self.subset[idx]


# Backward compatibility alias
class CIFARSubset(MultiDatasetSubset):
    """CIFAR-10 subset with deterministic normalization - backward compatibility."""

    def __init__(self, root: pathlib.Path, indices, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        super().__init__(
            dataset_name="cifar10",
            root=root,
            indices=list(indices) if not isinstance(indices, list) else indices,
            mean=mean,
            std=std,
            image_size=32,
        )


class EMA:
    """Simple EMA tracker."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.ema_model = build_unet(model.config.__dict__)  # type: ignore[arg-type]
        self.ema_model.load_state_dict(model.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def to(self, device: torch.device) -> None:
        self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_indices(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(state: Dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    LOGGER.info("Saved checkpoint to %s", path)


def _kernel_stats(K: torch.Tensor) -> Dict[str, float]:
    Kf = K.float()
    diag = torch.diagonal(Kf)
    return {
        "mean": Kf.mean().item(),
        "max": Kf.max().item(),
        "min": Kf.min().item(),
        "diag_mean": diag.mean().item(),
    }


def _maybe_dump_mmd(debug_dir: pathlib.Path, step: int, fx: torch.Tensor, fy: torch.Tensor, Kxx, Kyy, Kxy) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"mmd_debug_step{step:04d}.pt"
    torch.save(
        {
            "fx": fx.detach().cpu(),
            "fy": fy.detach().cpu(),
            "Kxx": Kxx.detach().cpu(),
            "Kyy": Kyy.detach().cpu(),
            "Kxy": Kxy.detach().cpu(),
        },
        path,
    )
    LOGGER.warning("Saved MMD debug bundle to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMD finetuning with 10-step DDIM (η=0)")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/mmd_finetune_cifar10_ddim10.yaml"))
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g., cuda:0")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Override output directory")
    parser.add_argument("--iters", type=int, default=None, help="Override number of finetune iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if CUDA is available")
    parser.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing in DDIM steps")
    parser.add_argument("--debug-ddim", action="store_true", help="Enable detailed DDIM per-step checks and dumps")
    parser.add_argument("--debug-scale", action="store_true", help="Enable DDIM scale instrumentation and dumps")
    parser.add_argument("--debug-dir", type=pathlib.Path, default=None, help="Optional debug directory for DDIM/MMD dumps")
    parser.add_argument(
        "--ddim-fp32",
        dest="ddim_fp32",
        action="store_true",
        default=True,
        help="Force DDIM algebra/model to run in float32 (default: on)",
    )
    parser.add_argument(
        "--no-ddim-fp32",
        dest="ddim_fp32",
        action="store_false",
        help="Allow DDIM to follow outer autocast instead of forcing float32",
    )
    parser.add_argument(
        "--fail-on-explode",
        action="store_true",
        help="Raise if |x0| or |x0_hat| exceeds the scale threshold during DDIM sampling",
    )
    parser.add_argument(
        "--scale-threshold",
        type=float,
        default=50.0,
        help="Absolute value threshold to treat DDIM outputs as exploded",
    )
    parser.add_argument("--t-start", type=int, default=None, help="Optional starting timestep for DDIM10 (default: T-1)")
    parser.add_argument(
        "--mmd-mode",
        type=str,
        default="strict",
        choices=["strict", "drop_yy"],
        help="MMD formulation: strict uses Exx+Eyy-2Exy; drop_yy omits Eyy",
    )
    parser.add_argument("--self-test-mmd", action="store_true", help="Run MMD self-test and exit")
    parser.add_argument("--check-grad-sign", action="store_true", help="Compare gradient sign vs negated loss once")
    return parser.parse_args()


def build_dataloader(data_cfg: Dict, indices, batch_size: int, num_workers: int) -> DataLoader:
    """Build dataloader for any supported dataset."""
    dataset_name = data_cfg["dataset"].get("name", "cifar10")
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])  # type: ignore[assignment]
    std = tuple(data_cfg["dataset"]["normalization"]["std"])  # type: ignore[assignment]
    image_shape = data_cfg["dataset"].get("image_shape", [3, 32, 32])
    image_size = image_shape[-1]
    
    dataset = MultiDatasetSubset(
        dataset_name=dataset_name,
        root=root,
        indices=list(indices) if not isinstance(indices, list) else indices,
        mean=mean,
        std=std,
        image_size=image_size,
    )
    
    LOGGER.info(f"Built dataloader for {dataset_name} with {len(dataset)} samples, size={image_size}x{image_size}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_cfg = load_yaml(pathlib.Path(cfg["base"]["model_config"]))
    data_cfg = load_yaml(pathlib.Path(cfg["base"]["data_config"]))

    # Get dataset info
    dataset_name = data_cfg["dataset"].get("name", "cifar10")
    image_shape = data_cfg["dataset"].get("image_shape", [3, 32, 32])
    image_size = image_shape[-1]
    LOGGER.info(f"Dataset: {dataset_name}, resolution: {image_size}x{image_size}")

    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda" and cfg["finetune"].get("amp", True)

    run_dir = args.out or pathlib.Path(cfg.get("output_dir", "runs/mmd_finetune")) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.debug_dir or (run_dir / "debug")
    if args.debug_ddim or args.debug_scale or args.fail_on_explode:
        debug_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("DDIM debug/scale mode enabled; artifacts will be saved to %s", debug_dir)
    (run_dir / "configs").mkdir(exist_ok=True)
    with (run_dir / "configs" / "mmd_finetune.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)

    base_ckpt = pathlib.Path(cfg["base"]["checkpoint"])
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")

    T = model_cfg["diffusion"]["timesteps"]
    batch_size = args.batch_size or cfg["finetune"]["batch_size"]
    iterations = args.iters or cfg["finetune"]["iterations"]
    t_start = args.t_start if args.t_start is not None else cfg["sampler"].get("t_start", T - 1)
    timesteps = build_linear_timesteps(T, cfg["sampler"].get("steps", 10), start=t_start)
    LOGGER.info("Using timesteps (len=%d, start=%d): %s", len(timesteps), t_start, timesteps)

    LOGGER.info("Building model and loading EMA weights from %s", base_ckpt)
    model = build_unet(model_cfg["model"])
    state = torch.load(base_ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.train()

    ema_decay = cfg["finetune"].get("ema_decay", model_cfg["training"].get("ema_decay", 0.9999))
    ema_ft = EMA(model, decay=ema_decay)
    ema_ft.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["finetune"]["lr"],
        betas=tuple(cfg["finetune"]["betas"]),
        eps=cfg["finetune"]["eps"],
    )
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    indices_path = pathlib.Path(data_cfg["splits"]["paths"]["member_train"])
    member_indices = load_indices(indices_path)
    dataloader = build_dataloader(data_cfg, member_indices, batch_size, args.num_workers)
    data_iter = iter(dataloader)

    clip_bundle = load_clip(device=device)
    data_mean = data_cfg["dataset"]["normalization"]["mean"]
    data_std = data_cfg["dataset"]["normalization"]["std"]
    LOGGER.info("MMD mode: %s", args.mmd_mode)

    log_interval = cfg["finetune"].get("log_interval", 10)
    ckpt_interval = cfg["finetune"].get("checkpoint_interval", 50)
    grad_clip = cfg["finetune"].get("grad_clip", None)

    def run_mmd_self_test() -> None:
        LOGGER.info("Running MMD self-test...")
        torch.manual_seed(0)
        fx = torch.randn(64, 128, device=device)
        fy_same = torch.randn(64, 128, device=device)
        fy_diff = torch.randn(64, 128, device=device)
        mmd_same, exx_s, eyy_s, exy_s, *_ = mmd2_from_features(fx, fy_same)
        mmd_diff, exx_d, eyy_d, exy_d, *_ = mmd2_from_features(fx, fy_diff)
        LOGGER.info(
            "MMD self-test (same dist): mmd2=%.6f exx=%.6f eyy=%.6f exy=%.6f",
            mmd_same.item(),
            exx_s.item(),
            eyy_s.item(),
            exy_s.item(),
        )
        LOGGER.info(
            "MMD self-test (diff dist): mmd2=%.6f exx=%.6f eyy=%.6f exy=%.6f",
            mmd_diff.item(),
            exx_d.item(),
            eyy_d.item(),
            exy_d.item(),
        )
        if mmd_same < -1e-2:
            LOGGER.warning("MMD self-test near-zero case is negative: %.6f", mmd_same.item())
        LOGGER.info("Self-test complete.")

    if args.self_test_mmd:
        run_mmd_self_test()
        return

    def check_grad_sign_once(real_batch: torch.Tensor) -> None:
        LOGGER.info("Running gradient sign check...")
        param = next(p for p in model.parameters() if p.requires_grad)
        noise = torch.randn(real_batch.shape[0], 3, 32, 32, device=device)

        def compute_loss(sign: float) -> torch.Tensor:
            gen_batch = ddim_sample_differentiable(
                model=model,
                alphas_bar=alphas_bar,
                shape=noise.shape,
                timesteps=timesteps,
                device=device,
                use_checkpoint=not args.no_grad_checkpoint,
                noise=noise,
                debug_ddim=False,
                debug_dir=None,
                ddim_fp32=args.ddim_fp32,
                debug_scale=False,
            )
            x0_for_clip = torch.tanh(gen_batch / 3.0)
            fx = extract_clip_features(x0_for_clip, clip_bundle, device, enable_grad=True, data_mean=data_mean, data_std=data_std)
            fy = extract_clip_features(real_batch, clip_bundle, device, enable_grad=False, data_mean=data_mean, data_std=data_std)
            Kxx = cubic_kernel(fx, fx)
            Kyy = cubic_kernel(fy, fy)
            Kxy = cubic_kernel(fx, fy)
            mmd2, Exx, Eyy, Exy = mmd2_components(Kxx, Kyy, Kxy)
            if args.mmd_mode == "drop_yy":
                mmd2 = Exx - 2 * Exy
            return sign * mmd2

        optimizer.zero_grad(set_to_none=True)
        loss_pos = compute_loss(1.0)
        loss_pos.backward()
        g1 = param.grad.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        loss_neg = compute_loss(-1.0)
        loss_neg.backward()
        g2 = param.grad.detach().clone()

        dot = torch.dot(g1.flatten(), g2.flatten())
        cosine = torch.nn.functional.cosine_similarity(g1.flatten(), g2.flatten(), dim=0)
        LOGGER.info("Grad sign check: dot=%.6f cosine=%.6f (expect ~-1)", dot.item(), cosine.item())
        optimizer.zero_grad(set_to_none=True)

    if args.check_grad_sign:
        try:
            real_batch, _ = next(iter(dataloader))
        except StopIteration:
            raise RuntimeError("Empty dataloader for grad sign check")
        real_batch = real_batch.to(device)
        check_grad_sign_once(real_batch)

    for step in range(1, iterations + 1):
        try:
            real_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_batch, _ = next(data_iter)
        real_batch = real_batch.to(device)
        noise = torch.randn(batch_size, 3, image_size, image_size, device=device)

        with autocast(enabled=use_amp):
            gen_batch = ddim_sample_differentiable(
                model=model,
                alphas_bar=alphas_bar,
                shape=noise.shape,
                timesteps=timesteps,
                device=device,
                use_checkpoint=not args.no_grad_checkpoint,
                noise=noise,
                debug_ddim=args.debug_ddim,
                debug_dir=debug_dir if (args.debug_ddim or args.debug_scale or args.fail_on_explode) else None,
                ddim_fp32=args.ddim_fp32,
                debug_scale=args.debug_scale,
                scale_threshold=args.scale_threshold,
                fail_on_explode=args.fail_on_explode,
            )
        assert torch.isfinite(gen_batch).all(), "NaN/Inf in x0 after DDIM"
        absmax = gen_batch.abs().max().item()
        sat_ratio = (gen_batch.abs() > 1.0).float().mean().item()
        LOGGER.info("x0 range: [%.3f, %.3f] absmax=%.3f sat_ratio=%.3f", gen_batch.min().item(), gen_batch.max().item(), absmax, sat_ratio)
        if args.fail_on_explode and absmax > args.scale_threshold:
            raise RuntimeError(f"x0 absmax {absmax:.2f} exceeds threshold {args.scale_threshold}")

        # Smooth clipping into [-1, 1] before CLIP to avoid hard clamp while keeping gradients.
        x0_for_clip = torch.tanh(gen_batch / 3.0)
        check_finite_x01 = (x0_for_clip + 1.0) * 0.5
        assert torch.isfinite(x0_for_clip).all(), "NaN/Inf in x0_for_clip"
        assert torch.isfinite(check_finite_x01).all(), "NaN/Inf in x01"

        with autocast(enabled=False):
            fake_feats = extract_clip_features(x0_for_clip, clip_bundle, device, enable_grad=True, data_mean=data_mean, data_std=data_std)
            real_feats = extract_clip_features(real_batch, clip_bundle, device, enable_grad=False, data_mean=data_mean, data_std=data_std)
            assert torch.isfinite(fake_feats).all(), "NaN/Inf in CLIP features fx"
            assert torch.isfinite(real_feats).all(), "NaN/Inf in CLIP features fy"
            LOGGER.info("fx norm mean=%.3f fy norm mean=%.3f", fake_feats.norm(dim=1).mean().item(), real_feats.norm(dim=1).mean().item())

            Kxx = cubic_kernel(fake_feats, fake_feats)
            Kyy = cubic_kernel(real_feats, real_feats)
            Kxy = cubic_kernel(fake_feats, real_feats)
            assert torch.isfinite(Kxx).all(), "NaN/Inf in Kxx"
            assert torch.isfinite(Kyy).all(), "NaN/Inf in Kyy"
            assert torch.isfinite(Kxy).all(), "NaN/Inf in Kxy"
            LOGGER.info(
                "Kernel stats: Kxx[min=%.3e,max=%.3e,diag=%.3e], Kyy[min=%.3e,max=%.3e,diag=%.3e], Kxy[min=%.3e,max=%.3e]",
                Kxx.min().item(),
                Kxx.max().item(),
                torch.diagonal(Kxx).mean().item(),
                Kyy.min().item(),
                Kyy.max().item(),
                torch.diagonal(Kyy).mean().item(),
                Kxy.min().item(),
                Kxy.max().item(),
            )

            mmd2, Exx, Eyy, Exy = mmd2_components(Kxx, Kyy, Kxy)
            if args.mmd_mode == "drop_yy":
                mmd2 = Exx - 2 * Exy
            LOGGER.info("MMD components: Exx=%.6f Eyy=%.6f Exy=%.6f MMD2=%.6f", Exx.item(), Eyy.item(), Exy.item(), mmd2.item())
            if mmd2 < -1e-4:
                LOGGER.warning("MMD2 is negative (%.6f); dumping debug tensors", mmd2.item())
                _maybe_dump_mmd(debug_dir, step, fake_feats, real_feats, Kxx, Kyy, Kxy)
            loss = mmd2
            assert torch.isfinite(loss).all(), "NaN/Inf in loss"

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        ema_ft.update(model)

        if step % log_interval == 0 or step == 1:
            grad_norm = 0.0
            with torch.no_grad():
                total = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
            LOGGER.info("step=%d loss=%.6f grad_norm=%.4f", step, loss.item(), grad_norm)

        if step % ckpt_interval == 0 or step == iterations:
            save_checkpoint(
                {
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "config": cfg,
                },
                run_dir / f"ckpt_{step:04d}_raw.pt",
            )
            save_checkpoint(
                {
                    "step": step,
                    "state_dict": ema_ft.ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "config": cfg,
                },
                run_dir / f"ckpt_{step:04d}_ema.pt",
            )

    LOGGER.info("Finetuning complete. Artifacts stored in %s", run_dir)


if __name__ == "__main__":
    main()
