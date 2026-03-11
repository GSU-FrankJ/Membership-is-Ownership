#!/usr/bin/env python3
"""
Evaluate pruned model checkpoints for all three methods.

- MiO: t-error ownership test (pruned model vs HF baseline)
- WDM: native watermark extraction
- Zhao: generate images from pruned model, decode watermark bit accuracy

Usage:
    # MiO pruned eval
    python scripts/baselines/eval_pruned.py --method mio \
        --checkpoint experiments/baseline_comparison/robustness/mio/cifar10/prune_30/pruned_30pct.pt \
        --output-dir experiments/baseline_comparison/robustness/mio/cifar10/prune_30/eval/

    # WDM pruned eval
    python scripts/baselines/eval_pruned.py --method wdm \
        --checkpoint experiments/baseline_comparison/robustness/wdm/cifar10/prune_30/pruned_30pct.pt \
        --output-dir experiments/baseline_comparison/robustness/wdm/cifar10/prune_30/eval/

    # Zhao pruned eval
    python scripts/baselines/eval_pruned.py --method zhao \
        --checkpoint experiments/baseline_comparison/robustness/zhao/cifar10/prune_30/pruned_30pct.pkl \
        --output-dir experiments/baseline_comparison/robustness/zhao/cifar10/prune_30/eval/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
if str(WDM_REPO) not in sys.path:
    sys.path.insert(0, str(WDM_REPO))

EDM_DIR = PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo" / "edm"
if str(EDM_DIR) not in sys.path:
    sys.path.insert(0, str(EDM_DIR))


def eval_mio_pruned(checkpoint, output_dir, device="cuda"):
    """Evaluate pruned MiO model using t-error ownership test vs HF baseline."""
    import yaml
    from src.ddpm_ddim.models import build_unet
    from src.ddpm_ddim.schedulers import build_cosine_schedule
    from src.attacks.baselines.t_error_hf import compute_baseline_scores
    from torchvision import datasets, transforms

    # Load model config
    with open(PROJECT_ROOT / "configs/model_ddim_cifar10.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs/data_cifar10.yaml") as f:
        data_cfg = yaml.safe_load(f)

    # Build model
    model = build_unet(model_cfg["model"])
    ckpt = torch.load(checkpoint, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    T = model_cfg["diffusion"]["timesteps"]
    _, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    # Load watermark private (members) and eval nonmembers
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Load splits
    split_paths = data_cfg["splits"]["paths"]
    with open(split_paths["watermark_private"]) as f:
        member_indices = json.load(f)
    with open(split_paths["eval_nonmember"]) as f:
        nonmember_indices = json.load(f)

    # Load CIFAR-10
    full_ds = datasets.CIFAR10(
        root=data_cfg["dataset"]["root"], train=True, download=False, transform=transform
    )

    # Subsample for speed: 1000 members, 1000 nonmembers
    n_eval = 1000
    member_subset = torch.utils.data.Subset(full_ds, member_indices[:n_eval])
    nonmember_subset = torch.utils.data.Subset(full_ds, nonmember_indices[:n_eval])

    member_loader = torch.utils.data.DataLoader(member_subset, batch_size=64, shuffle=False)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_subset, batch_size=64, shuffle=False)

    # Compute t-error scores
    print("Computing member t-errors on pruned model...")
    member_scores = compute_baseline_scores(
        member_loader, model, alphas_bar, T, k=50, agg="q25", device=device, desc="pruned-members"
    ).cpu().numpy()

    print("Computing nonmember t-errors on pruned model...")
    nonmember_scores = compute_baseline_scores(
        nonmember_loader, model, alphas_bar, T, k=50, agg="q25", device=device, desc="pruned-nonmembers"
    ).cpu().numpy()

    # Also compute scores on HF baseline (google/ddpm-cifar10-32) for owner-vs-baseline comparison
    print("Computing baseline t-errors...")
    from src.attacks.baselines.t_error_hf import compute_baseline_scores as compute_hf_scores
    # Use precomputed baseline scores from Phase 06 if available
    baseline_path = PROJECT_ROOT / "experiments/baseline_comparison/results/retroactive_defense/baseline_wd_t_errors.npy"
    if baseline_path.exists():
        baseline_scores = np.load(str(baseline_path))[:n_eval]
        print(f"  Loaded precomputed baseline scores: mean={baseline_scores.mean():.2f}")
    else:
        print("  WARNING: No precomputed baseline scores found. Skipping baseline comparison.")
        baseline_scores = None

    # Compute Cohen's d (member on pruned model vs baseline)
    from scipy import stats
    m_mean = float(np.mean(member_scores))
    m_std = float(np.std(member_scores))
    nm_mean = float(np.mean(nonmember_scores))
    nm_std = float(np.std(nonmember_scores))

    # Owner vs baseline
    if baseline_scores is not None:
        b_mean = float(np.mean(baseline_scores))
        b_std = float(np.std(baseline_scores))
        pooled_std = np.sqrt((np.var(member_scores) + np.var(baseline_scores)) / 2)
        d_owner_vs_baseline = float((m_mean - b_mean) / pooled_std) if pooled_std > 0 else 0.0
        ratio_owner_baseline = b_mean / m_mean if m_mean > 0 else float("inf")
        _, p_val = stats.ttest_ind(member_scores, baseline_scores)
        three_point_pass = abs(d_owner_vs_baseline) > 2.0 and float(p_val) < 1e-6 and ratio_owner_baseline > 5.0
    else:
        d_owner_vs_baseline = None
        ratio_owner_baseline = None
        three_point_pass = None

    results = {
        "method": "mio",
        "attack": "pruning_30pct",
        "member_mean": m_mean,
        "member_std": m_std,
        "nonmember_mean": nm_mean,
        "nonmember_std": nm_std,
        "owner_vs_baseline_cohens_d": d_owner_vs_baseline,
        "owner_vs_baseline_ratio": ratio_owner_baseline,
        "three_point_pass": three_point_pass,
        "checkpoint": str(checkpoint),
    }
    print(f"\nMiO Pruned Results:")
    print(f"  Member mean t-error: {m_mean:.2f} (std {m_std:.2f})")
    print(f"  Nonmember mean t-error: {nm_mean:.2f} (std {nm_std:.2f})")
    if d_owner_vs_baseline is not None:
        print(f"  Owner vs baseline Cohen's d: {d_owner_vs_baseline:.2f}")
        print(f"  Owner vs baseline ratio: {ratio_owner_baseline:.2f}")
        print(f"  Three-point pass: {three_point_pass}")
    return results


def eval_wdm_pruned(checkpoint, output_dir, device="cuda"):
    """Evaluate pruned WDM model via native watermark extraction."""
    from wdm.script_util import create_model_and_diffusion
    from wdm.wdp_util import generate_wdp_trigger
    from scripts.baselines.wdm_adapter import WDM_CIFAR10_CONFIG
    from PIL import Image

    # Load model
    print("Loading pruned WDM model...")
    model, diffusion = create_model_and_diffusion(**WDM_CIFAR10_CONFIG)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()  # WDM uses train mode for extraction

    # Generate trigger
    num_samples = 16
    shape = (num_samples, 3, 32, 32)
    trigger_path = str(WDM_REPO / "datasets/imgs/trigger_sel.jpg")
    wdp_trigger = generate_wdp_trigger(
        1998, 0, shape, wdp_trigger_kwargs={"wdp_trigger_path": trigger_path}
    )

    # Run WDP reverse process
    print(f"Extracting watermark ({num_samples} samples)...")
    with torch.no_grad():
        sample, _ = diffusion.wdp_p_sample_loop(
            model, shape, 0.8, wdp_trigger,
            clip_denoised=True, progress=True, demo=False,
        )

    extracted = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    extracted_np = extracted.permute(0, 2, 3, 1).cpu().numpy()

    # Save extracted images
    for i in range(num_samples):
        img = Image.fromarray(extracted_np[i])
        img.save(output_dir / f"extracted_pruned_{i:03d}.png")

    # Load reference watermark
    ref_img = Image.open(WDM_REPO / "datasets/imgs/single_wm.jpg").convert("RGB").resize((32, 32), Image.BICUBIC)
    ref_np = np.array(ref_img, dtype=np.uint8)

    # Compute metrics
    pixel_accs = []
    mse_scores = []
    for i in range(num_samples):
        ext = extracted_np[i]
        ref_bin = (ref_np > 128).astype(np.float32)
        ext_bin = (ext > 128).astype(np.float32)
        pixel_accs.append(np.mean(ref_bin == ext_bin))
        mse_scores.append(np.mean((ref_np.astype(np.float32) - ext.astype(np.float32)) ** 2))

    results = {
        "method": "wdm",
        "attack": "pruning_30pct",
        "pixel_accuracy_mean": float(np.mean(pixel_accs)),
        "pixel_accuracy_std": float(np.std(pixel_accs)),
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "native_pass": float(np.mean(pixel_accs)) > 0.90,
        "checkpoint": str(checkpoint),
    }
    print(f"\nWDM Pruned Results:")
    print(f"  Pixel accuracy: {np.mean(pixel_accs):.4f}")
    print(f"  MSE: {np.mean(mse_scores):.2f}")
    print(f"  Native pass (>90%): {results['native_pass']}")
    return results


def eval_zhao_pruned(checkpoint, output_dir, device="cuda"):
    """Evaluate pruned Zhao EDM model via watermark bit accuracy."""
    # Load pruned EDM model
    print("Loading pruned Zhao EDM model...")
    with open(checkpoint, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(device)
    net.eval()

    # Generate images using EDM sampler
    from scripts.baselines.generate_zhao import edm_sampler

    num_samples = 256
    batch_size = 64
    seed = 42
    all_images = []

    for batch_idx in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - batch_idx)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed + batch_idx // batch_size)
        latents = torch.randn(bs, net.img_channels, net.img_resolution, net.img_resolution,
                              generator=rng, device=device)
        with torch.no_grad():
            images = edm_sampler(net, latents, num_steps=18)
        # Convert to [0, 1] for decoder
        images_01 = (images * 0.5 + 0.5).clamp(0, 1).float()
        all_images.append(images_01.cpu())
        print(f"  Generated {batch_idx + bs}/{num_samples}")

    all_images = torch.cat(all_images, dim=0)

    # Load StegaStamp decoder
    decoder_path = str(
        PROJECT_ROOT / "experiments/baseline_comparison/zhao/cifar10/encoder/checkpoints/"
        "stegastamp_64_09032026_05:13:01_decoder.pth"
    )
    stegastamp_dir = PROJECT_ROOT / "experiments/baseline_comparison/watermarkdm_repo/string2img"
    if str(stegastamp_dir) not in sys.path:
        sys.path.insert(0, str(stegastamp_dir))

    from models import StegaStampDecoder
    decoder = StegaStampDecoder(resolution=32, IMAGE_CHANNELS=3, fingerprint_size=64)
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=False))
    decoder.to(device)
    decoder.eval()

    # Load expected fingerprint
    rng_fp = np.random.RandomState(42)
    expected_fp = rng_fp.randint(0, 2, size=64).astype(np.float32)

    # Decode watermarks
    bit_accs = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = all_images[i:i + batch_size].to(device)
            decoded = decoder(batch)
            decoded_bits = (decoded > 0).float().cpu().numpy()
            for j in range(len(batch)):
                acc = np.mean(decoded_bits[j] == expected_fp)
                bit_accs.append(acc)

    results = {
        "method": "zhao",
        "attack": "pruning_30pct",
        "bit_accuracy_mean": float(np.mean(bit_accs)),
        "bit_accuracy_std": float(np.std(bit_accs)),
        "n_samples": num_samples,
        "native_pass": float(np.mean(bit_accs)) > 0.90,
        "checkpoint": str(checkpoint),
    }
    print(f"\nZhao Pruned Results:")
    print(f"  Bit accuracy: {np.mean(bit_accs):.4f}")
    print(f"  Native pass (>90%): {results['native_pass']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pruned model checkpoints")
    parser.add_argument("--method", type=str, required=True, choices=["mio", "wdm", "zhao"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = pathlib.Path(args.checkpoint)

    if args.method == "mio":
        results = eval_mio_pruned(checkpoint, output_dir, args.device)
    elif args.method == "wdm":
        results = eval_wdm_pruned(checkpoint, output_dir, args.device)
    elif args.method == "zhao":
        results = eval_zhao_pruned(checkpoint, output_dir, args.device)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
