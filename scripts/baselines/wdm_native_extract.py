#!/usr/bin/env python3
"""
WDM Native Watermark Extraction (Step 6.1).

Runs the WDP reverse process to extract watermark images from the trained
WDM model, then computes pixel-level similarity to the original watermark logo.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
sys.path.insert(0, str(WDM_REPO))
sys.path.insert(0, str(PROJECT_ROOT))

from wdm.script_util import create_model_and_diffusion
from wdm.wdp_util import generate_wdp_trigger


WDM_CIFAR10_CONFIG = dict(
    image_size=32,
    num_channels=128,
    num_res_blocks=3,
    num_heads=4,
    num_heads_upsample=-1,
    attention_resolutions="16,8",
    dropout=0.1,
    learn_sigma=False,
    sigma_small=False,
    class_cond=False,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    use_checkpoint=False,
    use_scale_shift_norm=False,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trigger-path", type=str,
                        default=str(WDM_REPO / "datasets/imgs/trigger_sel.jpg"))
    parser.add_argument("--watermark-ref", type=str,
                        default=str(WDM_REPO / "datasets/imgs/single_wm.jpg"))
    parser.add_argument("--gamma1", type=float, default=0.8)
    parser.add_argument("--wdp-key", type=int, default=1998)
    parser.add_argument("--trigger-type", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = args.device
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model and diffusion
    print("Loading WDM model...")
    model, diffusion = create_model_and_diffusion(**WDM_CIFAR10_CONFIG)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()  # WDM intentionally uses train mode for extraction

    # Generate trigger
    print("Generating trigger...")
    shape = (args.num_samples, 3, 32, 32)
    wdp_trigger_kwargs = {"wdp_trigger_path": args.trigger_path}
    wdp_trigger = generate_wdp_trigger(
        args.wdp_key, args.trigger_type, shape,
        wdp_trigger_kwargs=wdp_trigger_kwargs,
    )

    # Run WDP reverse process
    print(f"Extracting watermark ({args.num_samples} samples)...")
    with torch.no_grad():
        sample, _ = diffusion.wdp_p_sample_loop(
            model, shape, args.gamma1, wdp_trigger,
            clip_denoised=True, progress=True, demo=False,
        )

    # Convert to numpy images [0, 255]
    extracted = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    extracted_np = extracted.permute(0, 2, 3, 1).cpu().numpy()  # NCHW -> NHWC

    # Save extracted images
    for i in range(min(args.num_samples, 16)):
        img = Image.fromarray(extracted_np[i])
        img.save(output_dir / f"extracted_{i:03d}.png")

    # Load reference watermark
    ref_img = Image.open(args.watermark_ref).convert("RGB").resize((32, 32), Image.BICUBIC)
    ref_np = np.array(ref_img, dtype=np.uint8)

    # Compute similarity metrics
    pixel_accs = []
    ssim_scores = []
    mse_scores = []
    for i in range(args.num_samples):
        ext = extracted_np[i]
        # Pixel accuracy (threshold at 128)
        ref_bin = (ref_np > 128).astype(np.float32)
        ext_bin = (ext > 128).astype(np.float32)
        pixel_acc = np.mean(ref_bin == ext_bin)
        pixel_accs.append(pixel_acc)

        # SSIM (optional)
        if HAS_SKIMAGE:
            s = ssim(ref_np, ext, channel_axis=2, data_range=255)
            ssim_scores.append(s)

        # MSE
        mse = np.mean((ref_np.astype(np.float32) - ext.astype(np.float32)) ** 2)
        mse_scores.append(mse)

    print(f"\n{'='*60}")
    print(f"WDM Native Watermark Extraction Results")
    print(f"{'='*60}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Pixel accuracy: {np.mean(pixel_accs):.4f} ± {np.std(pixel_accs):.4f}")
    if ssim_scores:
        print(f"  SSIM:           {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"  MSE:            {np.mean(mse_scores):.2f} ± {np.std(mse_scores):.2f}")
    print(f"{'='*60}")

    # Save results
    import json
    results = {
        "num_samples": args.num_samples,
        "pixel_accuracy_mean": float(np.mean(pixel_accs)),
        "pixel_accuracy_std": float(np.std(pixel_accs)),
        "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else None,
        "ssim_std": float(np.std(ssim_scores)) if ssim_scores else None,
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "checkpoint": args.checkpoint,
        "gamma1": args.gamma1,
        "wdp_key": args.wdp_key,
    }
    with open(output_dir / "native_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_dir / 'native_results.json'}")


if __name__ == "__main__":
    main()
