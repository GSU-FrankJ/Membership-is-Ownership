#!/usr/bin/env python3
"""Stage 4.2 — Zhao (StegaStamp on EDM) generation + decoding + verdict.

Pipeline per checkpoint:
  1. Load Zhao EDMPrecond from pkl (data['ema']).
  2. Generate N=1000 images via EDM's Heun 18-step sampler
     (default sigma_min=0.002, sigma_max=80, rho=7, deterministic).
  3. Load StegaStampDecoder (64-bit head) and decode bits per image.
  4. Bit-accuracy = mean(predicted_bits == GT_64bit) averaged over N images.
  5. Threshold at 0.9 → ✅/❌ verdict.

Does NOT modify watermarkdm_repo/string2img/detect_watermark_cifar10.py.
The 4-bit "0100" hardcode in that file is bypassed by using a proper
64-bit GT string here — recovered from Stage 1.5 from the 50 copies of
embedded_fingerprints.txt under datasets/embedded/cifar10/note/.

Outputs to experiments/baseline_comparison/results/table6/zhao_<attack>.json
with: method, attack, bit_accuracy (mean/std/min/max), n_samples,
fingerprint_len, verdict, borderline, generation/decoding wall times,
and an inline quality snapshot (mean pixel value, std, fraction
saturated-to-black, fraction saturated-to-white) for post-hoc
diagnosis of attacked-generation degradation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import tempfile
import time

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo"
EDM_DIR = WDM_REPO / "edm"
STRING2IMG = WDM_REPO / "string2img"

for p in (str(PROJECT_ROOT), str(EDM_DIR), str(STRING2IMG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_utils import persistence  # noqa: F401 — side effect for pickle

# Import Zhao's EDM sampler + StegaStamp decoder
from scripts.baselines.generate_zhao import edm_sampler
from models import StegaStampDecoder


# Recovered in Stage 1.5 from 50 copies of embedded_fingerprints.txt
# under watermarkdm_repo/edm/datasets/embedded/cifar10/note/{00000..00049}/
GT_FINGERPRINT_64BIT = "0100010001000010111010111111110011101000001111101101010110000000"
FINGERPRINT_LEN = 64
DEFAULT_DECODER = pathlib.Path(
    "/data/short/fjiang4/experiments/baseline_comparison/zhao/cifar10/encoder/"
    "checkpoints/stegastamp_64_09032026_05:13:01_decoder.pth"
)
THRESHOLD = 0.9
IMAGE_RESOLUTION = 32
IMAGE_CHANNELS = 3


def generate_samples(
    ckpt_path: pathlib.Path,
    n_samples: int,
    batch_size: int,
    num_steps: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Generate n_samples via EDM Heun sampler. Returns tensor of shape
    [N, 3, 32, 32] in [0, 1] + timing/quality stats dict."""
    print(f"[zhao-gen] Loading EDMPrecond from {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(device).eval()
    print(
        f"[zhao-gen] net.img_resolution={net.img_resolution} "
        f"channels={net.img_channels} sigma_min={net.sigma_min:.4f} "
        f"sigma_max={net.sigma_max:.4f}"
    )

    all_images = []
    t0 = time.time()
    num_batches = (n_samples + batch_size - 1) // batch_size
    for bi in range(num_batches):
        cur = min(batch_size, n_samples - bi * batch_size)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed + bi)
        latents = torch.randn(
            cur, net.img_channels, net.img_resolution, net.img_resolution,
            generator=rng, device=device,
        )
        with torch.no_grad():
            x = edm_sampler(net, latents, class_labels=None, num_steps=num_steps)
        # x is in [-1, 1] range (EDM output); map to [0, 1]
        x = (x.clamp(-1, 1).to(torch.float32) + 1) * 0.5
        all_images.append(x.cpu())
        if bi == 0 or (bi + 1) % 5 == 0 or bi == num_batches - 1:
            elapsed = time.time() - t0
            print(f"[zhao-gen] batch {bi + 1}/{num_batches} generated, elapsed={elapsed:.1f}s")
    elapsed = time.time() - t0
    images = torch.cat(all_images, dim=0)
    print(f"[zhao-gen] produced {images.shape[0]} images in {elapsed:.1f}s "
          f"({elapsed / images.shape[0] * 1000:.1f} ms/img)")

    # Quality stats for post-hoc diagnosis
    mean_px = float(images.mean().item())
    std_px = float(images.std().item())
    sat_black = float((images < 0.01).float().mean().item())
    sat_white = float((images > 0.99).float().mean().item())
    finite_frac = float(torch.isfinite(images).float().mean().item())
    return images, {
        "gen_wall_time_sec": elapsed,
        "n_samples": int(images.shape[0]),
        "mean_pixel": mean_px,
        "std_pixel": std_px,
        "frac_saturated_black": sat_black,
        "frac_saturated_white": sat_white,
        "frac_finite": finite_frac,
    }


def decode_bits(
    images: torch.Tensor,
    decoder_path: pathlib.Path,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict]:
    """Run StegaStampDecoder on images [N, 3, 32, 32] in [0, 1].
    Returns per-image bit-accuracy vector (N,) + timing dict."""
    state = torch.load(decoder_path, map_location="cpu")
    dense2_shape = state["dense.2.weight"].shape
    assert dense2_shape[0] == FINGERPRINT_LEN, \
        f"decoder fingerprint size mismatch: {dense2_shape[0]} vs {FINGERPRINT_LEN}"

    decoder = StegaStampDecoder(IMAGE_RESOLUTION, IMAGE_CHANNELS, FINGERPRINT_LEN)
    decoder.load_state_dict(state)
    decoder = decoder.to(device).eval()

    gt_bits = torch.tensor(
        [int(b) for b in GT_FINGERPRINT_64BIT], dtype=torch.long, device=device,
    ).view(1, FINGERPRINT_LEN)
    assert gt_bits.shape == (1, 64), gt_bits.shape

    t0 = time.time()
    per_image_acc = np.zeros(images.shape[0], dtype=np.float64)
    idx = 0
    for bi in range(0, images.shape[0], batch_size):
        batch = images[bi : bi + batch_size].to(device)
        with torch.no_grad():
            logits = decoder(batch)                   # [B, 64]
            pred_bits = (logits > 0).long()           # {0, 1}
        match = (pred_bits == gt_bits).float().mean(dim=1)  # per-image accuracy
        n = match.shape[0]
        per_image_acc[idx : idx + n] = match.cpu().numpy()
        idx += n
    elapsed = time.time() - t0
    return per_image_acc, {
        "decode_wall_time_sec": elapsed,
        "decoder_fingerprint_len": int(dense2_shape[0]),
        "gt_fingerprint_len": FINGERPRINT_LEN,
        "gt_fingerprint": GT_FINGERPRINT_64BIT,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--attack-name", type=str, required=True)
    ap.add_argument(
        "--output-dir", type=pathlib.Path,
        default=pathlib.Path("experiments/baseline_comparison/results/table6"),
    )
    ap.add_argument("--decoder-path", type=pathlib.Path, default=DEFAULT_DECODER)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.decoder_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint missing: {args.decoder_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"EDM checkpoint missing: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate ----
    images, gen_stats = generate_samples(
        args.checkpoint, args.n_samples, args.batch_size, args.num_steps,
        args.seed, args.device,
    )
    if gen_stats["frac_finite"] < 1.0:
        raise RuntimeError(
            f"Non-finite generated pixels detected ({gen_stats['frac_finite']:.4f} finite). Aborting."
        )

    # ---- Decode ----
    per_image_acc, dec_stats = decode_bits(
        images, args.decoder_path, args.batch_size, args.device,
    )
    bit_acc_mean = float(np.mean(per_image_acc))
    bit_acc_std = float(np.std(per_image_acc))
    bit_acc_min = float(np.min(per_image_acc))
    bit_acc_max = float(np.max(per_image_acc))

    # Distribution bins (for sanity)
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9]
    hist, _ = np.histogram(per_image_acc, bins=bins)
    bin_labels = [f"<{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
    # Make 1.0 inclusive
    bin_labels[-1] = "≤1.00"

    verdict = "PASS" if bit_acc_mean >= THRESHOLD else "FAIL"
    borderline = abs(bit_acc_mean - THRESHOLD) < 0.02

    output = {
        "method": "Zhao",
        "attack": args.attack_name,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "threshold": THRESHOLD,
        "metric": "bit_accuracy",
        "bit_accuracy_mean": bit_acc_mean,
        "bit_accuracy_std": bit_acc_std,
        "bit_accuracy_min": bit_acc_min,
        "bit_accuracy_max": bit_acc_max,
        "bit_accuracy_histogram": {"edges": bins, "labels": bin_labels, "counts": hist.tolist()},
        "n_samples": args.n_samples,
        "fingerprint_len": FINGERPRINT_LEN,
        "gt_fingerprint": GT_FINGERPRINT_64BIT,
        "verdict": verdict,
        "borderline": borderline,
        "generation": gen_stats,
        "decoding": dec_stats,
        "sampler": {"num_steps": args.num_steps, "seed": args.seed, "batch_size": args.batch_size},
    }
    out_json = args.output_dir / f"zhao_{args.attack_name}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Zhao × {args.attack_name}: bit_acc = {bit_acc_mean:.4f} ± {bit_acc_std:.4f}  [threshold = {THRESHOLD}]  → {verdict}")
    print(f"  per-image acc: min={bit_acc_min:.4f}  max={bit_acc_max:.4f}")
    print(f"  histogram: {dict(zip(bin_labels, hist.tolist()))}")
    print(f"  gen  time: {gen_stats['gen_wall_time_sec']:.1f}s  (mean_px={gen_stats['mean_pixel']:.3f}, std={gen_stats['std_pixel']:.3f}, sat_black={gen_stats['frac_saturated_black']:.3f}, sat_white={gen_stats['frac_saturated_white']:.3f})")
    print(f"  dec  time: {dec_stats['decode_wall_time_sec']:.1f}s")
    print(f"  borderline: {borderline}")
    print(f"  JSON: {out_json}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
