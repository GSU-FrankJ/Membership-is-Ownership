#!/usr/bin/env python3
"""Stage 4.1 — WDM native watermark extraction + pass/fail verdict.

Wraps scripts/baselines/wdm_native_extract.py with:
  (1) Checkpoint-format auto-detection (bare state_dict OR wrapped
      {"state_dict": ..., ...} from our Stage 2 attacks).
  (2) 0.9 pixel-accuracy threshold → ✅/❌ verdict.
  (3) JSON output to experiments/baseline_comparison/results/table6/
      wdm_<attack>.json including the verdict, all three metrics, and
      the source checkpoint path for downstream audit.

Does NOT modify any checkpoint on disk. Does NOT modify
wdm_native_extract.py. Extraction parameters match
wdm_native_extract.py defaults (num_samples=16, gamma1=0.8,
wdp_key=1998, trigger_type=0, trigger/ref images from wdm_repo).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
sys.path.insert(0, str(WDM_REPO))
sys.path.insert(0, str(PROJECT_ROOT))

from wdm.script_util import create_model_and_diffusion
from wdm.wdp_util import generate_wdp_trigger

# Canonical WDM CIFAR-10 config — authoritative source: scripts/baselines/wdm_native_extract.py:31-52
WDM_CIFAR10_CONFIG = dict(
    image_size=32, num_channels=128, num_res_blocks=3, num_heads=4,
    num_heads_upsample=-1, attention_resolutions="16,8", dropout=0.1,
    learn_sigma=False, sigma_small=False, class_cond=False,
    diffusion_steps=1000, noise_schedule="linear", timestep_respacing="",
    use_kl=False, predict_xstart=False, rescale_timesteps=False,
    rescale_learned_sigmas=False, use_checkpoint=False, use_scale_shift_norm=False,
)

DEFAULT_TRIGGER = WDM_REPO / "datasets/imgs/trigger_sel.jpg"
DEFAULT_REF = WDM_REPO / "datasets/imgs/single_wm.jpg"
THRESHOLD = 0.9


def load_checkpoint_flexible(path: pathlib.Path) -> dict:
    """Return a bare state_dict from either a raw state_dict file or a
    wrapped {"state_dict": ..., ...} dict (as produced by our Stage 2
    attack scripts)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], "wrapped"
    return ckpt, "bare"


def run_extraction(checkpoint_path: pathlib.Path, device: str,
                   num_samples: int, gamma1: float, wdp_key: int,
                   trigger_type: int, trigger_path: str, ref_path: str):
    print(f"[extract] Loading checkpoint {checkpoint_path}")
    state, fmt = load_checkpoint_flexible(checkpoint_path)
    print(f"[extract] Format: {fmt}, state_dict keys = {len(state)}")

    model, diffusion = create_model_and_diffusion(**WDM_CIFAR10_CONFIG)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, \
        f"state dict mismatch: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    model.to(device)
    model.train()  # WDM intentionally uses train mode for extraction (per wdm_native_extract.py:79)

    # Generate trigger
    shape = (num_samples, 3, 32, 32)
    wdp_trigger_kwargs = {"wdp_trigger_path": trigger_path}
    wdp_trigger = generate_wdp_trigger(
        wdp_key, trigger_type, shape, wdp_trigger_kwargs=wdp_trigger_kwargs,
    )

    print(f"[extract] Running WDP reverse process ({num_samples} samples)...")
    t0 = time.time()
    with torch.no_grad():
        sample, _ = diffusion.wdp_p_sample_loop(
            model, shape, gamma1, wdp_trigger,
            clip_denoised=True, progress=False, demo=False,
        )
    elapsed = time.time() - t0
    print(f"[extract] WDP sampling took {elapsed:.1f}s")

    # Convert to numpy [0, 255]
    extracted = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    extracted_np = extracted.permute(0, 2, 3, 1).cpu().numpy()

    # Load reference watermark
    ref_img = Image.open(ref_path).convert("RGB").resize((32, 32), Image.BICUBIC)
    ref_np = np.array(ref_img, dtype=np.uint8)

    # Metrics
    pixel_accs, ssim_scores, mse_scores = [], [], []
    ref_bin = (ref_np > 128).astype(np.float32)
    for i in range(num_samples):
        ext = extracted_np[i]
        ext_bin = (ext > 128).astype(np.float32)
        pixel_accs.append(float(np.mean(ref_bin == ext_bin)))
        if HAS_SKIMAGE:
            ssim_scores.append(float(ssim(ref_np, ext, channel_axis=2, data_range=255)))
        mse_scores.append(float(np.mean((ref_np.astype(np.float32) - ext.astype(np.float32)) ** 2)))

    return {
        "num_samples": num_samples,
        "pixel_accuracy_mean": float(np.mean(pixel_accs)),
        "pixel_accuracy_std": float(np.std(pixel_accs)),
        "pixel_accuracy_min": float(np.min(pixel_accs)),
        "pixel_accuracy_max": float(np.max(pixel_accs)),
        "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else None,
        "ssim_std": float(np.std(ssim_scores)) if ssim_scores else None,
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "format": fmt,
        "wall_time_sampling_sec": elapsed,
        "extraction_params": {
            "num_samples": num_samples,
            "gamma1": gamma1,
            "wdp_key": wdp_key,
            "trigger_type": trigger_type,
            "trigger_path": trigger_path,
            "watermark_ref": ref_path,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--attack-name", type=str, required=True,
                    help="Short attack name, used in output JSON filename (e.g. clean, mmd_ft, sgd_ft, noise_001, noise_01)")
    ap.add_argument("--output-dir", type=pathlib.Path,
                    default=pathlib.Path("experiments/baseline_comparison/results/table6"))
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--gamma1", type=float, default=0.8)
    ap.add_argument("--wdp-key", type=int, default=1998)
    ap.add_argument("--trigger-type", type=int, default=0)
    ap.add_argument("--trigger-path", type=str, default=str(DEFAULT_TRIGGER))
    ap.add_argument("--watermark-ref", type=str, default=str(DEFAULT_REF))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(args.trigger_path).exists():
        raise FileNotFoundError(f"trigger file missing: {args.trigger_path}")
    if not pathlib.Path(args.watermark_ref).exists():
        raise FileNotFoundError(f"watermark ref missing: {args.watermark_ref}")

    metrics = run_extraction(
        args.checkpoint, args.device,
        args.num_samples, args.gamma1, args.wdp_key,
        args.trigger_type, args.trigger_path, args.watermark_ref,
    )

    pa = metrics["pixel_accuracy_mean"]
    verdict = "PASS" if pa >= THRESHOLD else "FAIL"
    borderline = abs(pa - THRESHOLD) < 0.02

    output = {
        "method": "WDM",
        "attack": args.attack_name,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "threshold": THRESHOLD,
        "metric": "pixel_accuracy",
        "pixel_accuracy": metrics["pixel_accuracy_mean"],
        "pixel_accuracy_std": metrics["pixel_accuracy_std"],
        "pixel_accuracy_min": metrics["pixel_accuracy_min"],
        "pixel_accuracy_max": metrics["pixel_accuracy_max"],
        "ssim": metrics["ssim_mean"],
        "ssim_std": metrics["ssim_std"],
        "mse": metrics["mse_mean"],
        "mse_std": metrics["mse_std"],
        "verdict": verdict,
        "borderline": borderline,
        "checkpoint_format": metrics["format"],
        "wall_time_sampling_sec": metrics["wall_time_sampling_sec"],
        "extraction_params": metrics["extraction_params"],
    }

    out_json = args.output_dir / f"wdm_{args.attack_name}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"WDM × {args.attack_name}: pixel_acc = {pa:.4f}  [threshold = {THRESHOLD}]  → {verdict}")
    print(f"  SSIM = {output['ssim']:.4f}, MSE = {output['mse']:.2f}")
    print(f"  borderline: {borderline}")
    print(f"  JSON: {out_json}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
