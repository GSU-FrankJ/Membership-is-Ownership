#!/usr/bin/env python3
"""WDM weight-perturbation attack: add Gaussian noise N(0, σ²) to every
parameter tensor of WDM's UNet, then persist the attacked state_dict.

Per Stage 2 D39: this is **not** round-trip quantization. It is the
standard weight-perturbation attack `p ← p + randn_like(p) * sigma`.

Usage:
    python scripts/attack_noise_wdm.py --sigma 0.001 --seed 42 --output .../wdm_noise_001.pt
    python scripts/attack_noise_wdm.py --sigma 0.01  --seed 43 --output .../wdm_noise_01.pt

Output format:
    {
      "state_dict": attacked_state_dict,
      "wdm_config": WDM_CIFAR10_CONFIG,
      "attack": {
        "type": "gaussian_weight_noise",
        "sigma": float,
        "seed": int,
        "l2_perturbation_total": float,   # sum over params of ||delta||_2
        "linf_perturbation_max": float,   # max across all params of max |delta|
        "num_params_perturbed": int,
        "total_param_count": int,
      },
    }
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_mmd_wdm import (
    WDM_CIFAR10_CONFIG,
    DEFAULT_CHECKPOINT,
    build_wdm_model,
    load_wdm_checkpoint,
    sha256_first_n_params,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sigma", type=float, required=True, help="Gaussian noise standard deviation")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic seed for torch.manual_seed")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=str, default="cpu",
                        help="device for the manipulation (CPU fine — single pass, no training)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[noise] Loading WDM clean checkpoint {args.checkpoint}")
    model = build_wdm_model()
    info = load_wdm_checkpoint(model, args.checkpoint)
    src_sha = sha256_first_n_params(info["state_dict"], n=10)
    print(f"[noise] source SHA-256(first 10 params) = {src_sha}")
    model = model.to(device).eval()

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"[noise] UNet total params = {total_params_before:,}")

    # Deterministic perturbation
    torch.manual_seed(args.seed)
    l2_sum = 0.0
    linf_max = 0.0
    num_perturbed = 0
    with torch.no_grad():
        for p in model.parameters():
            delta = torch.randn_like(p) * args.sigma
            l2_sum += float(torch.linalg.norm(delta.flatten()).item())
            linf_max = max(linf_max, float(delta.abs().max().item()))
            p.add_(delta)
            num_perturbed += p.numel()
    print(f"[noise] applied σ={args.sigma} to {num_perturbed:,} / {total_params_before:,} params")
    print(f"[noise] total L2 perturbation = {l2_sum:.4f}")
    print(f"[noise] max |δ| (L∞)          = {linf_max:.6f}  (expected ~{3 * args.sigma:.6f} at 3σ)")

    # Sanity forward pass
    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.tensor([100, 500], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape, f"shape mismatch: {out.shape}"
    finite = bool(torch.isfinite(out).all().item())
    print(f"[noise] forward-pass sanity: out.shape={tuple(out.shape)} finite={finite} mean={out.mean().item():.4f} std={out.std().item():.4f}")
    if not finite:
        raise RuntimeError(f"Non-finite output after σ={args.sigma} perturbation; aborting.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    attacked_state = model.state_dict()
    out_sha = sha256_first_n_params(attacked_state, n=10)
    torch.save({
        "state_dict": attacked_state,
        "wdm_config": WDM_CIFAR10_CONFIG,
        "attack": {
            "type": "gaussian_weight_noise",
            "sigma": args.sigma,
            "seed": args.seed,
            "l2_perturbation_total": float(l2_sum),
            "linf_perturbation_max": float(linf_max),
            "num_params_perturbed": int(num_perturbed),
            "total_param_count": int(total_params_before),
            "source_checkpoint": str(args.checkpoint),
            "source_sha256_first10": src_sha,
            "output_sha256_first10": out_sha,
            "forward_sanity_finite": finite,
        },
    }, args.output)
    print(f"[noise] saved → {args.output}")
    print(f"[noise] output SHA-256(first 10 params) = {out_sha}")


if __name__ == "__main__":
    main()
