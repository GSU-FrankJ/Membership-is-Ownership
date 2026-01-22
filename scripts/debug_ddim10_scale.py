#!/usr/bin/env python3

"""Scale-focused DDIM10 smoke test to detect magnitude explosions."""

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
    parser = argparse.ArgumentParser(description="Debug DDIM10 scale behavior (detect |x0| explosions).")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="EMA checkpoint path")
    parser.add_argument("--model-config", type=pathlib.Path, default=pathlib.Path("configs/model_ddim.yaml"))
    parser.add_argument("--steps", type=int, default=10, help="Number of DDIM steps (K)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size of pure noise samples")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--scale-threshold", type=float, default=50.0, help="Absolute threshold that triggers failure")
    parser.add_argument("--debug-dir", type=pathlib.Path, default=None, help="Optional directory for debug dumps")
    parser.add_argument("--t-start", type=int, default=None, help="Override the starting timestep (default T-1)")
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
    if debug_dir is None:
        debug_dir = pathlib.Path("runs") / "debug_ddim_scale" / time.strftime("%Y%m%d-%H%M%S")
    debug_dir.mkdir(parents=True, exist_ok=True)

    noise = torch.randn(args.batch, 3, 32, 32, device=device)
    LOGGER.info(
        "Running DDIM10 scale debug: steps=%d (len=%d), start=%d, device=%s, debug_dir=%s, threshold=%.2f",
        args.steps,
        len(timesteps),
        t_start,
        device,
        debug_dir,
        args.scale_threshold,
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
                debug_scale=True,
                scale_threshold=args.scale_threshold,
                fail_on_explode=True,
                debug_dir=debug_dir,
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("DDIM scale debug failed: %s", exc)
        return 1

    sat_ratio = (x0.abs() > 1.0).float().mean().item()
    absmax = x0.abs().max().item()
    LOGGER.info("DDIM scale debug passed: absmax=%.4f, sat_ratio=%.4f, range=[%.4f, %.4f]", absmax, sat_ratio, x0.min().item(), x0.max().item())
    if absmax > args.scale_threshold:
        LOGGER.error("Final x0 absmax %.2f exceeds threshold %.2f", absmax, args.scale_threshold)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
