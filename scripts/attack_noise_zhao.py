#!/usr/bin/env python3
"""Zhao EDM weight-perturbation attack: add N(0, σ²) to every parameter of
net.model (the inner SongUNet inside EDMPrecond). Mirrors
scripts/attack_noise_wdm.py but respects Zhao's precond-wrapper scope.

Per Stage 3 D39: this is **not** round-trip quantization. It is the
standard weight-perturbation attack `p ← p + randn_like(p) * sigma`.

Scope: **net.model.parameters() only** (the SongUNet). EDMPrecond's
σ-conditional scalars (sigma_min, sigma_max, sigma_data) are Python
attributes, not nn.Parameter, and will not be touched by
model.parameters() iteration anyway. Asserted at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import pickle
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EDM_DIR = PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo" / "edm"
for p in (str(PROJECT_ROOT), str(EDM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_utils import persistence  # noqa: F401


DEFAULT_CHECKPOINT = pathlib.Path(
    "/data/short/fjiang4/experiments/baseline_comparison/zhao/cifar10/edm/"
    "00000-images-uncond-ddpmpp-edm-gpus1-batch512-fp32/"
    "network-snapshot-015053.pkl"
)


def sha256_first_n_params(net: torch.nn.Module, n: int = 10) -> str:
    params = dict(net.model.named_parameters())
    keys = sorted(params.keys())[:n]
    h = hashlib.sha256()
    for k in keys:
        v = params[k]
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[zhao-noise] Loading Zhao EDMPrecond {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        data = pickle.load(f)
    net = data["ema"]
    aux_keys = [k for k in data.keys() if k != "ema"]

    assert type(net).__name__ == "EDMPrecond", f"Expected EDMPrecond, got {type(net).__name__}"
    assert type(net.model).__name__ == "SongUNet", f"Expected SongUNet inner, got {type(net.model).__name__}"

    src_sha = sha256_first_n_params(net, n=10)
    print(f"[zhao-noise] source SHA-256(first 10 SongUNet params) = {src_sha}")

    # Confirm EDMPrecond scalars are Python attrs, NOT nn.Parameter.
    outer_params = {id(p): n for n, p in net.named_parameters()}
    inner_params = {id(p): n for n, p in net.model.named_parameters()}
    only_in_outer = set(outer_params) - set(inner_params)
    if only_in_outer:
        print(f"[zhao-noise] WARNING: {len(only_in_outer)} parameters on EDMPrecond live outside net.model:")
        for pid in only_in_outer:
            print(f"  - {outer_params[pid]}")
        print("[zhao-noise] These will NOT be perturbed (per spec). Continuing.")
    else:
        print(f"[zhao-noise] verified: all {len(outer_params)} parameters live in net.model (EDMPrecond scalars are Python attrs, not params)")

    net = net.to(device).eval()
    n_params_inner_before = sum(p.numel() for p in net.model.parameters())
    print(f"[zhao-noise] net.model total params = {n_params_inner_before:,}")

    # Deterministic perturbation
    torch.manual_seed(args.seed)
    l2_sum = 0.0
    linf_max = 0.0
    num_perturbed = 0
    with torch.no_grad():
        for p in net.model.parameters():
            delta = torch.randn_like(p) * args.sigma
            l2_sum += float(torch.linalg.norm(delta.flatten()).item())
            linf_max = max(linf_max, float(delta.abs().max().item()))
            p.add_(delta)
            num_perturbed += p.numel()

    print(f"[zhao-noise] applied σ={args.sigma} to {num_perturbed:,} / {n_params_inner_before:,} params (SongUNet)")
    print(f"[zhao-noise] total L2 perturbation = {l2_sum:.4f}")
    print(f"[zhao-noise] max |δ| (L∞)          = {linf_max:.6f}  (expected ~{3 * args.sigma:.6f} at 3σ)")

    # Sanity forward pass
    x = torch.randn(2, 3, 32, 32, device=device)
    sigma = torch.tensor([1.0, 1.0], device=device).view(-1, 1, 1, 1)
    with torch.no_grad():
        out = net(x, sigma)
    assert out.shape == x.shape, f"shape mismatch: {out.shape}"
    finite = bool(torch.isfinite(out).all().item())
    print(f"[zhao-noise] forward-pass sanity: out.shape={tuple(out.shape)} finite={finite} mean={out.mean().item():.4f} std={out.std().item():.4f}")
    if not finite:
        raise RuntimeError(f"Non-finite output after σ={args.sigma} perturbation; aborting.")

    out_sha = sha256_first_n_params(net, n=10)
    if out_sha == src_sha:
        raise RuntimeError("Output SHA-256 matches source SHA-256; perturbation did not modify weights. Aborting.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "ema": net,
        "attack": {
            "type": "gaussian_weight_noise",
            "sigma": args.sigma,
            "seed": args.seed,
            "scope": "net.model (SongUNet) parameters only",
            "l2_perturbation_total": float(l2_sum),
            "linf_perturbation_max": float(linf_max),
            "n_params_perturbed": int(num_perturbed),
            "total_inner_param_count": int(n_params_inner_before),
            "source_checkpoint": str(args.checkpoint),
            "source_sha256_first10": src_sha,
            "output_sha256_first10": out_sha,
            "forward_sanity_finite": finite,
            "aux_keys_dropped_from_source_pkl": aux_keys,
        },
    }
    with open(args.output, "wb") as f:
        pickle.dump(save_payload, f)
    print(f"[zhao-noise] saved → {args.output}")
    print(f"[zhao-noise] output SHA-256(first 10 SongUNet params) = {out_sha}")


if __name__ == "__main__":
    main()
