#!/usr/bin/env python
"""Generate images from a trained WDM model.

Usage:
    python scripts/baselines/generate_wdm.py \
        --checkpoint <path_to_ema_checkpoint.pt> \
        --num-samples 50000 \
        --output-dir experiments/baseline_comparison/results/wdm/samples \
        --seed 42 --batch-size 256
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
if str(WDM_REPO) not in sys.path:
    sys.path.insert(0, str(WDM_REPO))

from wdm.script_util import create_model_and_diffusion
from scripts.baselines.wdm_adapter import WDM_CIFAR10_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Generate images from a WDM checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to WDM checkpoint (.pt)")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--use-ddim", action="store_true", help="Use DDIM sampling (faster)")
    p.add_argument("--ddim-steps", type=int, default=50, help="DDIM steps if --use-ddim")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model & diffusion
    cfg = dict(WDM_CIFAR10_CONFIG)
    if args.use_ddim:
        cfg["timestep_respacing"] = f"ddim{args.ddim_steps}"

    model, diffusion = create_model_and_diffusion(**cfg)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded WDM model from {args.checkpoint}")
    print(f"Generating {args.num_samples} images -> {out_dir}")

    generated = 0
    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating")
        while generated < args.num_samples:
            bs = min(args.batch_size, args.num_samples - generated)
            shape = (bs, 3, 32, 32)

            if args.use_ddim:
                samples = diffusion.ddim_sample_loop(
                    model, shape, clip_denoised=True, device=device, progress=False
                )
            else:
                samples = diffusion.p_sample_loop(
                    model, shape, clip_denoised=True, device=device, progress=False
                )

            # Convert [-1, 1] -> [0, 255]
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()  # NCHW -> NHWC

            for i in range(bs):
                img = Image.fromarray(samples[i])
                img.save(out_dir / f"{generated:06d}.png")
                generated += 1

            pbar.update(bs)
        pbar.close()

    print(f"Done. {generated} images saved to {out_dir}")


if __name__ == "__main__":
    main()
