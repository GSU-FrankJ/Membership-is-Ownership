#!/usr/bin/env python3

"""Standalone reproducer for DDIM10 NaN/Inf issues."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule
from src.ddpm_ddim.samplers.ddim10 import build_linear_timesteps, ddim_sample_differentiable


LOGGER = get_winston_logger(__name__)


def load_yaml(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug DDIM10 sampling stability (NaN/Inf hunting).")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="EMA checkpoint path")
    parser.add_argument("--model-config", type=pathlib.Path, default=pathlib.Path("configs/model_ddim.yaml"))
    parser.add_argument("--steps", type=int, default=10, help="Number of DDIM steps (K)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size of pure noise samples")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--debug-ddim", action="store_true", help="Enable per-step DDIM validation/logging")
    parser.add_argument("--debug-scale", action="store_true", help="Enable DDIM scale instrumentation")
    parser.add_argument(
        "--debug-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to dump debug artifacts (default: runs/debug_ddim/<timestamp>)",
    )
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
    parser.add_argument("--fail-on-explode", action="store_true", help="Raise if |x0| exceeds the scale threshold")
    parser.add_argument("--scale-threshold", type=float, default=50.0, help="Absolute threshold for explosion detection")
    parser.add_argument("--t-start", type=int, default=None, help="Optional starting timestep for DDIM10 (default: T-1)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = load_yaml(args.model_config)

    LOGGER.info("Loading EMA checkpoint from %s", args.checkpoint)
    model = build_unet(model_cfg["model"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    T = model_cfg["diffusion"]["timesteps"]
    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)
    t_start = args.t_start if args.t_start is not None else T - 1
    timesteps = build_linear_timesteps(T, args.steps, start=t_start)

    debug_dir = args.debug_dir
    if debug_dir is None and (args.debug_ddim or args.debug_scale or args.fail_on_explode):
        debug_dir = pathlib.Path("runs") / "debug_ddim" / time.strftime("%Y%m%d-%H%M%S")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    noise = torch.randn(args.batch, 3, 32, 32, device=device)

    LOGGER.info(
        "Running DDIM10 debug: steps=%d (len=%d), start=%d, device=%s, debug_dir=%s, fp32=%s, debug_scale=%s",
        args.steps,
        len(timesteps),
        t_start,
        device,
        debug_dir if debug_dir is not None else None,
        args.ddim_fp32,
        args.debug_scale,
    )

    try:
        with torch.no_grad():
            x0 = ddim_sample_differentiable(
                model=model,
                alphas_bar=alphas_bar,
                shape=noise.shape,
                timesteps=timesteps,
                device=device,
                use_checkpoint=False,
                noise=noise,
                debug_ddim=args.debug_ddim,
                debug_dir=debug_dir,
                ddim_fp32=args.ddim_fp32,
                debug_scale=args.debug_scale,
                scale_threshold=args.scale_threshold,
                fail_on_explode=args.fail_on_explode,
            )
        if not torch.isfinite(x0).all():
            raise RuntimeError("NaN/Inf detected in final x0")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("DDIM debug failed: %s", exc)
        return 1

    sat_ratio = (x0.abs() > 1.0).float().mean().item()
    absmax = x0.abs().max().item()
    LOGGER.info(
        "DDIM debug succeeded: x0 range [%.4f, %.4f], mean=%.4f, absmax=%.4f, sat_ratio=%.4f",
        x0.min().item(),
        x0.max().item(),
        x0.mean().item(),
        absmax,
        sat_ratio,
    )
    if absmax > args.scale_threshold and args.fail_on_explode:
        LOGGER.error("Final x0 absmax %.2f exceeds threshold %.2f", absmax, args.scale_threshold)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
