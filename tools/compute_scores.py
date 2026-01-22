#!/usr/bin/env python3

"""Compute or sanitize eval_in/out reconstruction scores with per-sample vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule
from src.attacks.scores import t_error_aggregate, uniform_timesteps, compute_error_stats

LOGGER = get_winston_logger(__name__)
PREFERRED_KEYS = ("per_sample", "scores", "errors", "t_error", "vals")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SplitDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        indices: List[int],
        train: bool,
        mean,
        std,
    ) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        base = CIFAR10(root=str(root), train=train, download=False, transform=transform)
        self.subset = Subset(base, indices)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        return self.subset[idx]


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_indices(path: pathlib.Path) -> List[int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_checkpoint(root: pathlib.Path) -> pathlib.Path:
    candidates = sorted(
        [p for p in root.glob("ckpt_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {root}")
    return candidates[-1]


def resolve_checkpoint_path(
    checkpoint_root: pathlib.Path,
    prefer_ema: bool,
) -> pathlib.Path:
    ckpt_dir = find_latest_checkpoint(checkpoint_root)
    filename = "ema.ckpt" if prefer_ema else "model.ckpt"
    return ckpt_dir / filename


def load_model_from_checkpoint(
    model_cfg_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    device: torch.device,
) -> torch.nn.Module:
    model_cfg = load_yaml(model_cfg_path)
    model = build_unet(model_cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    LOGGER.info("Loaded model weights from %s", checkpoint_path)
    return model


def build_loader(
    data_cfg: Dict,
    split_name: str,
    indices: List[int],
    train_flag: bool,
    batch_size: int,
    fastdev: bool,
) -> DataLoader:
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])
    if fastdev:
        indices = indices[: min(len(indices), 1024)]
    dataset = SplitDataset(root, indices, train_flag, mean, std)
    LOGGER.info("%s dataset size=%d", split_name, len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["dataset"].get("num_workers", 8),
        pin_memory=True,
    )


def sanitize_tensor(values: torch.Tensor) -> torch.Tensor:
    finite_mask = torch.isfinite(values)
    dropped = (~finite_mask).sum().item()
    if dropped:
        LOGGER.warning("Dropped %d non-finite values before saving.", dropped)
    sanitized = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized


def normalize_vector(values: torch.Tensor) -> torch.Tensor:
    values = sanitize_tensor(values)
    if values.numel() == 0:
        return values.float()
    median_mag = values.abs().median().item()
    if median_mag > 1e6:
        denom = max(values.numel(), 1)
        values = values / denom
        LOGGER.warning("Auto-normalized very large magnitudes by ÷%d.", denom)
    return values.float()


def compute_split_scores(
    name: str,
    loader: DataLoader,
    model: torch.nn.Module,
    timesteps: Sequence[int],
    alphas_bar: torch.Tensor,
    device: torch.device,
    agg: str = "q25",
) -> Dict[str, torch.Tensor]:
    """Compute t-error scores with specified aggregation method.
    
    For q25 aggregation, also computes summary stats (mean, std, L2) from
    the per-timestep t-error sequence for use as additional model inputs.
    
    Args:
        name: Split name for logging
        loader: DataLoader for the split
        model: Denoising model
        timesteps: Timesteps to sample
        alphas_bar: Diffusion schedule
        device: Compute device
        agg: Aggregation method (q25 default for best separability, or mean, q10, q20, median, etc.)
    
    Returns:
        Dictionary containing:
            - "scores": Tensor of aggregated t-error scores [N]
            - "stats": Tensor of summary stats [N, 3] (for q25 only)
                       Contains [mean_error, std_error, l2_error] per sample.
                       Note: stats do NOT include q25 itself to avoid information leakage.
    """
    scores_list = []
    stats_list = []
    
    # Determine if we need to compute stats (only for q25 aggregation)
    compute_stats = (agg == "q25")
    
    for batch, _ in tqdm(loader, desc=f"scores-{name}"):
        images = batch.to(device)
        pixels = max(float(images[0].numel()), 1.0)
        
        if compute_stats:
            # Get both aggregated score and raw errors for stats computation
            batch_scores, raw_errors = t_error_aggregate(
                images,
                timesteps,
                model,
                alphas_bar,
                agg=agg,
                return_raw=True,
            )
            # Compute summary stats from raw errors (before pixel normalization)
            # Note: raw_errors is [B, k] where k is number of timesteps
            batch_stats = compute_error_stats(raw_errors / pixels)
            batch_stats = sanitize_tensor(batch_stats)
            stats_list.append(batch_stats.detach().cpu())
        else:
            batch_scores = t_error_aggregate(
                images,
                timesteps,
                model,
                alphas_bar,
                agg=agg,
            )
        
        batch_scores = batch_scores / pixels
        batch_scores = sanitize_tensor(batch_scores)
        scores_list.append(batch_scores.detach().cpu())
    
    # Concatenate all batches
    concatenated_scores = normalize_vector(torch.cat(scores_list))
    LOGGER.info("%s (agg=%s) sanitized scores shape=%s", name, agg, tuple(concatenated_scores.shape))
    
    result = {"scores": concatenated_scores}
    
    if compute_stats and stats_list:
        concatenated_stats = torch.cat(stats_list, dim=0)
        # Ensure stats are finite and normalized
        concatenated_stats = sanitize_tensor(concatenated_stats)
        result["stats"] = concatenated_stats
        LOGGER.info("%s (agg=%s) stats shape=%s", name, agg, tuple(concatenated_stats.shape))
    
    return result


def histogram(payload: torch.Tensor) -> Dict[str, torch.Tensor]:
    arr = payload.detach().cpu().numpy()
    counts, bins = np.histogram(arr, bins=40)
    return {
        "hist_counts": torch.from_numpy(counts.astype(np.int64)),
        "hist_bins": torch.from_numpy(bins.astype(np.float32)),
    }


def pick_existing_vector(payload) -> Optional[torch.Tensor]:
    if isinstance(payload, dict):
        for key in PREFERRED_KEYS:
            if key in payload:
                return torch.as_tensor(payload[key])
        for value in payload.values():
            if isinstance(value, (list, tuple)):
                return torch.as_tensor(value)
    elif isinstance(payload, torch.Tensor):
        return payload
    return None


def refresh_existing_file(path: pathlib.Path) -> None:
    payload = torch.load(path, map_location="cpu")
    vector = pick_existing_vector(payload)
    if vector is None:
        LOGGER.warning("Unable to refresh %s (no suitable tensor).", path)
        return
    sanitized = normalize_vector(vector.float())
    hist = histogram(sanitized)
    payload["per_sample"] = sanitized.clone()
    payload["scores"] = sanitized.clone()
    payload.update(hist)
    torch.save(payload, path)
    LOGGER.info("Refreshed per-sample vector in %s", path)


def save_scores(
    out_path: pathlib.Path,
    per_sample: torch.Tensor,
    metadata: Dict,
    agg: str = "q25",
    stats: Optional[torch.Tensor] = None,
) -> None:
    """Save computed scores with metadata and optional stats.
    
    Args:
        out_path: Output file path
        per_sample: Per-sample scores tensor [N]
        metadata: Metadata dictionary
        agg: Aggregation method used (q25 default for best separability)
        stats: Optional summary stats tensor [N, D] where D is typically 3
               (mean_error, std_error, l2_error). Only provided for q25 aggregation.
               Note: stats do NOT include q25 itself to avoid information leakage.
    """
    hist = histogram(per_sample)
    payload = {
        "per_sample": per_sample,
        "scores": per_sample.clone(),
        "aggregate": agg,
        "hist_bins": hist["hist_bins"],
        "hist_counts": hist["hist_counts"],
        "metadata": metadata,
    }
    
    # Add stats field for q25 aggregation (used as QR model input features)
    if stats is not None:
        payload["stats"] = stats
        LOGGER.info("Including stats tensor shape=%s in scores file", tuple(stats.shape))
    
    torch.save(payload, out_path)
    LOGGER.info("Saved sanitized scores (agg=%s) to %s", agg, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sanitized reconstruction scores with configurable aggregation.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/attack_qr.yaml"))
    parser.add_argument("--data-config", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fastdev", action="store_true", help="Use a tiny subset for quick checks.")
    parser.add_argument("--checkpoint-root", type=pathlib.Path, help="Override diffusion checkpoint root.")
    parser.add_argument("--ckpt", type=pathlib.Path, help="Load this specific EMA checkpoint.")
    parser.add_argument("--tag", type=str, default="q25", help="Tag suffix for cache files (default: q25).")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached files exist.")
    parser.add_argument("--aggregate", type=str, default=None, help="Aggregation method (mean, q10, q20, median). Overrides config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    attack_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    raw_timesteps = uniform_timesteps(
        attack_cfg["t_error"]["T"],
        5 if args.fastdev else attack_cfg["t_error"]["k_uniform"],
    )

    _betas, alphas_bar = build_cosine_schedule(attack_cfg["t_error"]["T"])
    alphas_bar = alphas_bar.to(device)

    timesteps = [
        t for t in raw_timesteps if float(alphas_bar[t].item()) > 1e-8
    ]
    dropped = len(raw_timesteps) - len(timesteps)
    if dropped:
        LOGGER.warning("Dropped %d unstable timesteps near T.", dropped)
    LOGGER.info("Timesteps: %s", timesteps)

    if args.ckpt is not None:
        checkpoint_path = args.ckpt.expanduser().resolve()
    else:
        checkpoint_root = (
            pathlib.Path(args.checkpoint_root)
            if args.checkpoint_root is not None
            else pathlib.Path(attack_cfg["model"]["checkpoint_root"])
        )
        if args.fastdev:
            fastdev_root = checkpoint_root.parent / "fastdev"
            if fastdev_root.exists():
                checkpoint_root = fastdev_root
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_root,
            attack_cfg["model"].get("prefer_ema", True),
        ).expanduser().resolve()

    model = load_model_from_checkpoint(
        pathlib.Path(attack_cfg["model"]["config"]),
        checkpoint_path,
        device,
    )

    cache_dir = pathlib.Path(attack_cfg["t_error"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    batch_size = attack_cfg["train"]["batch_size"]

    splits = {
        "aux": ("aux", False),
        "eval_in": ("eval_in", True),
        "eval_out": ("eval_out", True),
    }
    indices = {
        name: load_indices(pathlib.Path(data_cfg["splits"]["paths"][name]))
        for name in splits.keys()
    }

    loaders = {
        name: build_loader(
            data_cfg,
            name,
            indices[name],
            train_flag,
            batch_size,
            args.fastdev,
        )
        for name, (_, train_flag) in splits.items()
    }

    # Determine aggregation method: CLI > config (config must specify aggregate)
    if args.aggregate is not None:
        agg_method = args.aggregate
        LOGGER.info("Using aggregation method from CLI: %s", agg_method)
    else:
        agg_method = attack_cfg["t_error"].get("aggregate")
        if agg_method is None:
            raise ValueError(
                "Aggregation method must be specified in config (t_error.aggregate) or via --aggregate. "
                "Please set aggregate in configs/attack_qr.yaml::t_error.aggregate"
            )
        LOGGER.info("Using aggregation method from config: %s", agg_method)

    metadata_base = {
        "aggregate": agg_method,
        "timesteps": list(timesteps),
        "ckpt_path": str(checkpoint_path),
        "ckpt_sha256": file_sha256(checkpoint_path) if checkpoint_path.is_file() else None,
    }

    for name, loader in loaders.items():
        filename = f"{args.tag}_{name}.pt" if args.tag else f"{name}.pt"
        out_path = cache_dir / filename
        metadata = {
            **metadata_base,
            "split": name,
            "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if out_path.exists() and not args.force:
            LOGGER.info("Found existing %s; refreshing per-sample vector.", out_path)
            refresh_existing_file(out_path)
            continue
        
        # compute_split_scores now returns a dict with 'scores' and optionally 'stats'
        result = compute_split_scores(
            name,
            loader,
            model,
            timesteps,
            alphas_bar,
            device,
            agg=agg_method,
        )
        per_sample = result["scores"]
        stats = result.get("stats")  # Will be None if not q25 aggregation
        
        save_scores(out_path, per_sample, metadata, agg=agg_method, stats=stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        sys.exit(1)
