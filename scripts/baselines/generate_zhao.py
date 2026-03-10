"""
Generate images using Zhao et al.'s EDM-based watermarked diffusion model.

Uses EDM's native Heun sampler (Algorithm 2 from Karras et al. 2022).
Saves individual PNG files.

Usage:
    python scripts/baselines/generate_zhao.py \
        --checkpoint path/to/network-snapshot-XXXXXX.pkl \
        --num-samples 1000 \
        --output-dir experiments/baseline_comparison/zhao/cifar10/generated \
        --seed 42 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import PIL.Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
EDM_DIR = (
    PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo" / "edm"
)
if str(EDM_DIR) not in sys.path:
    sys.path.insert(0, str(EDM_DIR))


def edm_sampler(
    net,
    latents,
    class_labels=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    """EDM sampler (Algorithm 2 from Karras et al. 2022).

    Heun's 2nd-order method with optional stochastic churn.
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = (
            x_cur
            + (t_hat**2 - t_cur**2).sqrt()
            * S_noise
            * torch.randn_like(x_cur)
        )

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def main():
    parser = argparse.ArgumentParser(description="Generate images from Zhao EDM model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to EDM .pkl checkpoint"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of images to generate"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for PNGs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-steps", type=int, default=18, help="Number of sampling steps"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(device)
    net.eval()

    print(
        f"Model: res={net.img_resolution}, channels={net.img_channels}, "
        f"labels={net.label_dim}"
    )

    num_generated = 0
    batch_idx = 0
    while num_generated < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - num_generated)

        # Deterministic latents per seed
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed + batch_idx)
        latents = torch.randn(
            batch_size,
            net.img_channels,
            net.img_resolution,
            net.img_resolution,
            generator=rng,
            device=device,
        )

        with torch.no_grad():
            images = edm_sampler(net, latents, num_steps=args.num_steps)

        # Convert to uint8 PNGs: [-1,1] -> [0,255]
        images_np = (
            (images * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        for i, img_np in enumerate(images_np):
            idx = num_generated + i
            path = os.path.join(args.output_dir, f"{idx:06d}.png")
            if img_np.shape[2] == 1:
                PIL.Image.fromarray(img_np[:, :, 0], "L").save(path)
            else:
                PIL.Image.fromarray(img_np, "RGB").save(path)

        num_generated += batch_size
        batch_idx += 1
        print(f"  Generated {num_generated}/{args.num_samples}")

    print(f"Done. Saved {num_generated} images to {args.output_dir}")


if __name__ == "__main__":
    main()
