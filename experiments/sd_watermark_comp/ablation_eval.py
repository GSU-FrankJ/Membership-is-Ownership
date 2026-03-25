"""
Quick evaluation for ablation runs.

Usage:
  python ablation_eval.py --split-file data/splits/eval_split_100.json \
    --lora-path models/sd-v1-4-lora-a1 --gpu 0 \
    --out-csv experiments/sd_watermark_comp/scores/ablation/a1_quick.csv

  python ablation_eval.py --split-file data/splits/eval_split_500.json \
    --full-model-path models/sd-v1-4-full-a4 --gpu 3 \
    --out-csv experiments/sd_watermark_comp/scores/ablation/a4_quick.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def uniform_timesteps(T, k):
    if k >= T:
        return list(range(T))
    return sorted(set(int(round(i)) for i in np.linspace(0, T - 1, k)))


def compute_t_error_sd(latent, timesteps, unet, alphas_bar, uncond_emb, agg="q25",
                       precomputed_noise=None):
    device = latent.device
    batch_size = latent.size(0)
    all_errors = []
    emb = uncond_emb.expand(batch_size, -1, -1)

    for idx, t in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        ab = alphas_bar[t]
        sqrt_ab = ab.sqrt()
        sqrt_1_ab = (1 - ab).sqrt()

        noise = precomputed_noise[idx]
        latent_t = sqrt_ab * latent + sqrt_1_ab * noise

        with torch.no_grad():
            eps_pred = unet(latent_t, t_tensor, encoder_hidden_states=emb, return_dict=True).sample

        latent_hat = (latent_t - sqrt_1_ab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        # ||z - z_hat||^2 / (H * W * C)  — paper Eq. 6, normalized
        error = (latent_hat - latent).pow(2).mean(dim=[1, 2, 3])
        all_errors.append(error)

    errors = torch.stack(all_errors, dim=1).float()

    if agg == "mean":
        return errors.mean(dim=1)
    elif agg.startswith("q"):
        q_val = int(agg[1:]) / 100.0
        return torch.quantile(errors, q_val, dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


def _pixel_rescore(latent, timesteps, unet, alphas_bar, emb, precomputed_noise,
                    vae, scaling_factor, images_t, agg):
    """Recompute t-error in pixel space: decode latent_hat → pixel, MSE vs original."""
    device = latent.device
    batch_size = latent.size(0)
    emb_expanded = emb.expand(batch_size, -1, -1) if emb.size(0) == 1 else emb
    all_errors = []
    for idx, t in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        ab = alphas_bar[t]
        sqrt_ab = ab.sqrt()
        sqrt_1_ab = (1 - ab).sqrt()
        noise = precomputed_noise[idx]
        latent_t = sqrt_ab * latent + sqrt_1_ab * noise
        with torch.no_grad():
            eps_pred = unet(latent_t, t_tensor, encoder_hidden_states=emb_expanded, return_dict=True).sample
        latent_hat = (latent_t - sqrt_1_ab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        with torch.no_grad():
            x_hat = vae.decode(latent_hat / scaling_factor).sample
        error = (x_hat - images_t).pow(2).mean(dim=[1, 2, 3])
        all_errors.append(error)
    errors = torch.stack(all_errors, dim=1).float()
    if agg == "mean":
        return errors.mean(dim=1)
    elif agg.startswith("q"):
        q_val = int(agg[1:]) / 100.0
        return torch.quantile(errors, q_val, dim=1)
    return errors.mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True, help="Path to eval split JSON")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA weights dir")
    parser.add_argument("--full-model-path", default=None, help="Path to full fine-tuned model dir")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--t-min", type=int, default=0, help="Minimum timestep (inclusive)")
    parser.add_argument("--t-max", type=int, default=999, help="Maximum timestep (inclusive)")
    parser.add_argument("--agg", type=str, default="q25")
    parser.add_argument("--error-space", choices=["latent", "pixel"], default="latent",
                        help="Compute t-error in latent (4x64x64) or pixel (3x512x512) space")
    parser.add_argument("--use-caption", action="store_true",
                        help="Use per-image COCO caption instead of empty prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    assert args.lora_path or args.full_model_path, "Must specify --lora-path or --full-model-path"

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_model_path = os.path.join(PROJECT_ROOT, "models/sd-v1-4")

    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler

    # Load reference model (base SD v1.4)
    print(f"Loading reference model...")
    ref_pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    ref_unet = ref_pipe.unet
    ref_unet.eval()

    # Load target model
    if args.lora_path:
        print(f"Loading LoRA target from {args.lora_path}...")
        tgt_pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        tgt_pipe.load_lora_weights(args.lora_path)
        tgt_unet = tgt_pipe.unet
    else:
        print(f"Loading full fine-tuned target from {args.full_model_path}...")
        tgt_unet = UNet2DConditionModel.from_pretrained(
            os.path.join(args.full_model_path, "unet"), torch_dtype=torch.float16
        ).to(device)
    tgt_unet.eval()

    # Shared VAE and scheduler
    vae = ref_pipe.vae
    vae.eval()
    scaling_factor = vae.config.scaling_factor

    scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    alphas_bar = scheduler.alphas_cumprod.to(device=device, dtype=torch.float16)
    T = len(alphas_bar)

    # Unconditional text embedding
    tokenizer = ref_pipe.tokenizer
    text_encoder = ref_pipe.text_encoder
    with torch.no_grad():
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).to(device)
        uncond_emb = text_encoder(uncond_input.input_ids)[0]
    if not args.use_caption:
        del text_encoder
        if args.lora_path:
            del tgt_pipe.text_encoder
        del ref_pipe.text_encoder
        torch.cuda.empty_cache()

    timesteps = uniform_timesteps(args.t_max - args.t_min + 1, args.K)
    timesteps = [t + args.t_min for t in timesteps]
    print(f"Timesteps (K={args.K}, range=[{args.t_min},{args.t_max}]): {timesteps}")

    # Load eval split
    with open(args.split_file) as f:
        split = json.load(f)

    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if args.use_caption:
        print("Keeping text encoder for per-image caption conditioning")
    else:
        print("Using empty prompt (unconditional)")

    work = []
    for entry in split["members"]:
        work.append({
            "image_id": entry["image_id"],
            "label": "member",
            "file_path": os.path.join(PROJECT_ROOT, "data/coco2014/train2014", entry["file_name"]),
            "caption": entry.get("caption", ""),
        })
    for entry in split["nonmembers"]:
        work.append({
            "image_id": entry["image_id"],
            "label": "nonmember",
            "file_path": os.path.join(PROJECT_ROOT, "data/coco2014/val2014", entry["file_name"]),
            "caption": entry.get("caption", ""),
        })

    print(f"Eval set: {sum(1 for w in work if w['label']=='member')} members + "
          f"{sum(1 for w in work if w['label']=='nonmember')} non-members = {len(work)} total")

    # Inference
    results = []
    t_start = time.time()
    fieldnames = ["image_id", "label", "score_ref", "score_tgt", "score"]

    for i in range(0, len(work), args.batch_size):
        batch_items = work[i:i + args.batch_size]
        images = []
        for item in batch_items:
            img = Image.open(item["file_path"]).convert("RGB")
            images.append(transform(img))

        images_t = torch.stack(images).to(device=device, dtype=torch.float16)

        with torch.no_grad():
            posterior = vae.encode(images_t)
            latent = posterior.latent_dist.mean * scaling_factor

        # Per-image caption or unconditional embedding
        if args.use_caption:
            captions = [item["caption"] for item in batch_items]
            with torch.no_grad():
                cap_input = tokenizer(
                    captions, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(device)
                batch_emb = text_encoder(cap_input.input_ids)[0]
        else:
            batch_emb = uncond_emb

        precomputed_noise = [torch.randn_like(latent) for _ in timesteps]

        score_ref = compute_t_error_sd(latent, timesteps, ref_unet, alphas_bar, batch_emb, args.agg,
                                       precomputed_noise=precomputed_noise)
        score_tgt = compute_t_error_sd(latent, timesteps, tgt_unet, alphas_bar, batch_emb, args.agg,
                                       precomputed_noise=precomputed_noise)

        # Pixel-space error: decode latent_hat → pixel, compute MSE
        if args.error_space == "pixel":
            score_ref = _pixel_rescore(latent, timesteps, ref_unet, alphas_bar, batch_emb,
                                       precomputed_noise, vae, scaling_factor, images_t, args.agg)
            score_tgt = _pixel_rescore(latent, timesteps, tgt_unet, alphas_bar, batch_emb,
                                       precomputed_noise, vae, scaling_factor, images_t, args.agg)

        score = score_tgt - score_ref

        for j, item in enumerate(batch_items):
            results.append({
                "image_id": item["image_id"],
                "label": item["label"],
                "score_ref": float(score_ref[j].cpu()),
                "score_tgt": float(score_tgt[j].cpu()),
                "score": float(score[j].cpu()),
            })

        if (i + len(batch_items)) % 100 < args.batch_size:
            elapsed = time.time() - t_start
            print(f"  [{i + len(batch_items)}/{len(work)}] {elapsed:.1f}s elapsed")

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    elapsed = time.time() - t_start

    # Metrics
    scores_m = [r["score"] for r in results if r["label"] == "member"]
    scores_n = [r["score"] for r in results if r["label"] == "nonmember"]

    mean_m, std_m = np.mean(scores_m), np.std(scores_m)
    mean_n, std_n = np.mean(scores_n), np.std(scores_n)
    pooled_std = np.sqrt((std_m**2 + std_n**2) / 2)
    cohens_d = (mean_n - mean_m) / pooled_std if pooled_std > 0 else 0

    from sklearn.metrics import roc_curve, auc
    y_true = [1] * len(scores_m) + [0] * len(scores_n)
    y_score = [-s for s in scores_m] + [-s for s in scores_n]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    tpr_1pct = tpr[np.searchsorted(fpr, 0.01)]
    tpr_01pct = tpr[np.searchsorted(fpr, 0.001)]

    print(f"\n{'='*50}")
    print(f"RESULTS ({args.out_csv})")
    print(f"{'='*50}")
    print(f"N members:    {len(scores_m)}")
    print(f"N non-members:{len(scores_n)}")
    print(f"Member mean:  {mean_m:.4f} (std={std_m:.4f})")
    print(f"Nonmem mean:  {mean_n:.4f} (std={std_n:.4f})")
    print(f"Cohen's d:    {cohens_d:.4f}")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR@1%FPR:    {tpr_1pct:.4f}")
    print(f"TPR@0.1%FPR:  {tpr_01pct:.4f}")
    print(f"Time:         {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
