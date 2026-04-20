#!/usr/bin/env python3
"""
Adversary fine-tuning: takes a stolen LoRA model and continues training on new data.

Simulates a thief who:
1. Steals the owner's model (base SD v1.4 + LoRA weights)
2. Fine-tunes it on their own data (disjoint from W)

Usage:
  # B1: domain shift (disjoint COCO)
  CUDA_VISIBLE_DEVICES=0 python adversary_finetune.py \
    --base-model models/sd-v1-4 \
    --lora-weights models/sd-v1-4-lora-a6 \
    --train-dir data/ablation_train_dir_b1 \
    --output-dir models/sd-v1-4-lora-b1 \
    --max-steps 2000 --lr 5e-5

  # B2: task shift (synthetic images)
  CUDA_VISIBLE_DEVICES=1 python adversary_finetune.py \
    --base-model models/sd-v1-4 \
    --lora-weights models/sd-v1-4-lora-a6 \
    --train-dir data/ablation_train_dir_b2 \
    --output-dir models/sd-v1-4-lora-b2 \
    --max-steps 2000 --lr 5e-5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class ImageTextDataset(Dataset):
    """Simple dataset reading metadata.jsonl + images from a directory."""

    def __init__(self, data_dir, transform, tokenizer):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.items = []
        meta_path = os.path.join(data_dir, "metadata.jsonl")
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        print(f"Loaded {len(self.items)} items from {meta_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        tokens = self.tokenizer(
            item["text"], padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze(0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Path to base SD v1.4")
    parser.add_argument("--lora-weights", default=None, help="Path to stolen LoRA weights dir")
    parser.add_argument("--full-model-path", default=None, help="Path to stolen full fine-tuned model dir")
    parser.add_argument("--train-dir", required=True, help="Adversary's training data dir")
    parser.add_argument("--output-dir", required=True, help="Output dir for fine-tuned model")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (only used if --full-ft not set)")
    parser.add_argument("--full-ft", action="store_true", help="Full fine-tune (unfreeze all UNet params) instead of LoRA")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.lora_weights or args.full_model_path, \
        "Must specify --lora-weights or --full-model-path"

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    # Step 1: Load the stolen model (merge any existing LoRA into base UNet)
    if args.full_model_path:
        print(f"Loading stolen full model from {args.full_model_path}...")
        # Detect if this is a complete pipeline dir (has model_index.json) or UNet-only
        if os.path.exists(os.path.join(args.full_model_path, "model_index.json")):
            pipe = StableDiffusionPipeline.from_pretrained(
                args.full_model_path, torch_dtype=torch.float32, safety_checker=None
            ).to(device)
        else:
            print(f"  No model_index.json — loading base pipeline + swapping UNet from {args.full_model_path}...")
            from diffusers import UNet2DConditionModel
            pipe = StableDiffusionPipeline.from_pretrained(
                args.base_model, torch_dtype=torch.float32, safety_checker=None
            ).to(device)
            unet_subdir = os.path.join(args.full_model_path, "unet") if os.path.isdir(os.path.join(args.full_model_path, "unet")) else args.full_model_path
            swapped_unet = UNet2DConditionModel.from_pretrained(unet_subdir, torch_dtype=torch.float32).to(device)
            pipe.unet = swapped_unet
        # Full FT model: UNet already contains all modifications, no merge needed
    else:
        print(f"Loading base model from {args.base_model}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model, torch_dtype=torch.float32, safety_checker=None
        ).to(device)
        print(f"Loading stolen LoRA weights from {args.lora_weights}...")
        pipe.load_lora_weights(args.lora_weights)
        # Merge LoRA into base UNet so we start from a clean merged model
        print("Merging LoRA into base UNet...")
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        print("LoRA merged and unloaded. UNet is now a plain model with baked-in weights.")

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # Step 2: Add fresh LoRA or unfreeze all UNet params for full FT
    if args.full_ft:
        print("Full FT mode: unfreezing all UNet parameters...")
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(True)
        train_params = [p for p in unet.parameters() if p.requires_grad]
    else:
        print(f"Adding fresh LoRA r={args.lora_rank} for adversary fine-tuning...")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        unet = get_peft_model(unet, lora_config)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        train_params = [p for p in unet.parameters() if p.requires_grad]

    trainable = sum(p.numel() for p in train_params)
    total = sum(p.numel() for p in unet.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} total ({100*trainable/total:.2f}%)")

    if trainable == 0:
        print("ERROR: No trainable parameters found!")
        sys.exit(1)

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-2)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageTextDataset(args.train_dir, transform, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    # Training loop
    unet.train()
    vae.eval()
    text_encoder.eval()

    step = 0
    t_start = time.time()
    losses = []

    print(f"\nStarting adversary fine-tuning: {args.max_steps} steps, lr={args.lr}")
    print(f"Training data: {len(dataset)} images from {args.train_dir}")

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Encode text
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states=encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                avg_loss = np.mean(losses[-args.log_every:])
                elapsed = time.time() - t_start
                eta = (args.max_steps - step) / (step / elapsed) if step > 0 else 0
                print(f"  Step {step}/{args.max_steps} | loss={avg_loss:.4f} | "
                      f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s | {step/elapsed:.1f} steps/s")

    # Save weights (full UNet for full-FT, LoRA-only otherwise)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.full_ft:
        unet_save_dir = os.path.join(args.output_dir, "unet")
        os.makedirs(unet_save_dir, exist_ok=True)
        unet.save_pretrained(unet_save_dir)
        print(f"Saved full UNet to {unet_save_dir}")
    else:
        lora_state_dict = {}
        for key, val in unet.state_dict().items():
            if "lora" in key.lower():
                lora_state_dict[key] = val.cpu()
        from safetensors.torch import save_file
        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(lora_state_dict, save_path)
        print(f"Saved {len(lora_state_dict)} LoRA tensors to {save_path}")

    elapsed = time.time() - t_start
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    print(f"\n{'='*50}")
    print(f"Adversary fine-tuning complete")
    print(f"Steps: {step}, Final loss (last 100): {final_loss:.4f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
