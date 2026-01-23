"""DDIM training entrypoint with multi-dataset support.

Supports:
- CIFAR-10 (32x32)
- CIFAR-100 (32x32)
- STL-10 (96x96)
- CelebA (64x64)

Includes watermark exposure logging for MIA experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # Go up to project root (from src/ddpm_ddim/train_ddim.py)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim import select_checkpoints
from src.ddpm_ddim.models.unet import UNetModel, build_unet
from src.ddpm_ddim.schedulers.betas import build_cosine_schedule


LOGGER = get_winston_logger(__name__)


class MultiDatasetSubset(Dataset):
    """Generic dataset subset wrapper with watermark tracking.
    
    Supports CIFAR-10, CIFAR-100, STL-10, CelebA with deterministic transforms.
    Tracks which indices are accessed for watermark exposure logging.
    """

    def __init__(
        self,
        dataset_name: str,
        root: pathlib.Path,
        indices: torch.Tensor,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        image_size: int = 32,
        download: bool = False,
        watermark_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            dataset_name: One of 'cifar10', 'cifar100', 'stl10', 'celeba'
            root: Dataset root directory
            indices: Indices to include in subset
            mean: Normalization mean
            std: Normalization std
            image_size: Target image size
            download: Whether to download dataset
            watermark_indices: Optional list of watermark indices for exposure tracking
        """
        self.dataset_name = dataset_name.lower()
        self.image_size = image_size
        self.watermark_set = set(watermark_indices) if watermark_indices else set()
        self.exposure_counter = Counter()  # Track watermark sample exposure
        
        # Build transforms - resize if needed, then normalize
        transform_list = []
        if self.dataset_name == "stl10" and image_size != 96:
            transform_list.append(transforms.Resize(image_size))
        elif self.dataset_name == "celeba":
            # CelebA needs cropping and resizing to square
            transform_list.extend([
                transforms.CenterCrop(178),  # Crop to face region
                transforms.Resize(image_size),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform = transforms.Compose(transform_list)
        
        # Load base dataset
        if self.dataset_name == "cifar10":
            base_dataset = datasets.CIFAR10(root=str(root), train=True, download=download, transform=transform)
        elif self.dataset_name == "cifar100":
            base_dataset = datasets.CIFAR100(root=str(root), train=True, download=download, transform=transform)
        elif self.dataset_name == "stl10":
            base_dataset = datasets.STL10(root=str(root), split="train", download=download, transform=transform)
        elif self.dataset_name == "celeba":
            base_dataset = datasets.CelebA(root=str(root), split="train", download=download, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.indices = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        self.subset = Subset(base_dataset, self.indices)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        # Return data along with original index for exposure tracking in main process
        # NOTE: Exposure tracking in __getitem__ doesn't work with num_workers > 0
        # because DataLoader workers have separate copies of the dataset.
        # Tracking must be done in the main process using the returned indices.
        original_idx = self.indices[idx]
        data = self.subset[idx]
        return data, original_idx
    
    def track_exposure(self, indices: torch.Tensor) -> None:
        """Track watermark exposure from batch indices (call from main process).
        
        Args:
            indices: Tensor of original indices from a batch
        """
        for idx in indices.tolist():
            if idx in self.watermark_set:
                self.exposure_counter[idx] += 1
    
    def get_watermark_exposure(self) -> Dict[int, int]:
        """Get watermark sample exposure counts."""
        return dict(self.exposure_counter)
    
    def reset_exposure_counter(self) -> None:
        """Reset exposure counter."""
        self.exposure_counter.clear()


# Keep CIFAR10Subset for backward compatibility
class CIFAR10Subset(MultiDatasetSubset):
    """CIFAR-10 subset wrapper - backward compatibility alias."""

    def __init__(
        self,
        root: pathlib.Path,
        indices: torch.Tensor,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        download: bool = False,
    ) -> None:
        super().__init__(
            dataset_name="cifar10",
            root=root,
            indices=indices,
            mean=mean,
            std=std,
            image_size=32,
            download=download,
        )


class EMA:
    """Exponential moving average helper for model weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.ema_model = build_unet(asdict(model.config))  # type: ignore[arg-type]
        self.ema_model.load_state_dict(model.state_dict())
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def to(self, device: torch.device) -> None:
        self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_environment(enable_amp: bool) -> Dict[str, object]:
    state: Dict[str, object] = {}

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        state["cudnn_allow_tf32"] = False
        if hasattr(torch.backends.cudnn, "fp32_math_mode") and hasattr(torch.backends.cudnn, "FP32MathMode"):
            try:
                torch.backends.cudnn.fp32_math_mode = torch.backends.cudnn.FP32MathMode.F32
            except AttributeError:
                pass
            state["cudnn_fp32_math_mode"] = "F32"
        else:
            state["cudnn_fp32_math_mode"] = None
        state["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        state["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    else:
        state["cudnn_benchmark"] = None
        state["cudnn_deterministic"] = None
        state["cudnn_allow_tf32"] = None
        state["cudnn_fp32_math_mode"] = None

    if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul_backend = torch.backends.cuda.matmul
        if hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False
            state["cuda_matmul_allow_tf32"] = False
        else:
            state["cuda_matmul_allow_tf32"] = None
        if hasattr(matmul_backend, "fp32_precision"):
            try:
                matmul_backend.fp32_precision = "ieee"
            except AttributeError:
                pass
            state["cuda_matmul_fp32_precision"] = "ieee"
        else:
            state["cuda_matmul_fp32_precision"] = None
    else:
        state["cuda_matmul_allow_tf32"] = None
        state["cuda_matmul_fp32_precision"] = None

    torch.set_float32_matmul_precision("high")
    state["float32_matmul_precision"] = torch.get_float32_matmul_precision()

    amp_enabled = enable_amp and torch.cuda.is_available()
    state["amp_enabled"] = amp_enabled
    state["amp_mode"] = "torch.amp.autocast" if amp_enabled else "disabled"
    state["amp_device"] = "cuda" if amp_enabled else "cpu"
    return state


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def diffusion_training_step(
    model: UNetModel,
    batch: torch.Tensor,
    alphas_bar: torch.Tensor,
    T: int,
    scaler: torch.amp.GradScaler | None,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    device = batch.device
    bsz = batch.size(0)
    timesteps = torch.randint(0, T, (bsz,), device=device, dtype=torch.long)
    noise = torch.randn_like(batch)
    alpha_bar_t = alphas_bar[timesteps]
    sqrt_alpha = alpha_bar_t.sqrt()[:, None, None, None]
    sqrt_one_minus = (1 - alpha_bar_t).sqrt()[:, None, None, None]
    xt = sqrt_alpha * batch + sqrt_one_minus * noise

    device_type = "cuda" if batch.is_cuda else "cpu"
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, enabled=scaler is not None and scaler.is_enabled())
        if device_type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        pred_noise = model(xt, timesteps)
        loss = F.mse_loss(pred_noise, noise)

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def save_checkpoint(
    model: UNetModel,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_dir: pathlib.Path,
) -> None:
    ckpt_dir = run_dir / f"ckpt_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "state_dict": model.state_dict()}, ckpt_dir / "model.ckpt")
    torch.save({"step": step, "state_dict": ema.ema_model.state_dict()}, ckpt_dir / "ema.ckpt")
    torch.save({"step": step, "state_dict": optimizer.state_dict()}, ckpt_dir / "optim.ckpt")
    LOGGER.info("Checkpoint saved at step %d -> %s", step, ckpt_dir)


def find_latest_checkpoint(run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Find the latest valid checkpoint directory in run_dir.
    
    A valid checkpoint must contain model.ckpt, ema.ckpt, and optim.ckpt files.
    Returns None if no valid checkpoint is found.
    """
    if not run_dir.exists():
        return None
    
    # Find all ckpt_XXXXXX directories
    ckpt_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("ckpt_")],
        key=lambda x: int(x.name.split("_")[1]),
        reverse=True,  # Latest first
    )
    
    # Return the first valid checkpoint (latest step with all required files)
    for ckpt_dir in ckpt_dirs:
        model_ckpt = ckpt_dir / "model.ckpt"
        ema_ckpt = ckpt_dir / "ema.ckpt"
        optim_ckpt = ckpt_dir / "optim.ckpt"
        if model_ckpt.exists() and ema_ckpt.exists() and optim_ckpt.exists():
            return ckpt_dir
    
    return None


def load_checkpoint(
    ckpt_dir: pathlib.Path,
    model: UNetModel,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load checkpoint and return the step number.
    
    Args:
        ckpt_dir: Directory containing model.ckpt, ema.ckpt, optim.ckpt
        model: Model to load state into
        ema: EMA wrapper to load state into
        optimizer: Optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        The training step at which the checkpoint was saved.
    """
    model_ckpt = torch.load(ckpt_dir / "model.ckpt", map_location=device)
    ema_ckpt = torch.load(ckpt_dir / "ema.ckpt", map_location=device)
    optim_ckpt = torch.load(ckpt_dir / "optim.ckpt", map_location=device)
    
    model.load_state_dict(model_ckpt["state_dict"])
    ema.ema_model.load_state_dict(ema_ckpt["state_dict"])
    optimizer.load_state_dict(optim_ckpt["state_dict"])
    
    step = model_ckpt["step"]
    LOGGER.info("Resumed from checkpoint at step %d <- %s", step, ckpt_dir)
    return step


def write_run_metadata(
    run_dir: pathlib.Path,
    model_cfg: Dict,
    data_cfg: Dict,
    seed: int,
    mode: str,
    determinism_state: Dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run_dir)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        LOGGER.warning("Unable to resolve git commit hash")

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": git_hash,
        "seed": seed,
        "mode": mode,
        "model_config": model_cfg,
        "data_config": data_cfg,
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_arch": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "determinism": determinism_state,
        },
    }
    with (run_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    LOGGER.info("Wrote run metadata to %s", run_dir / "run.json")


def load_indices(path: pathlib.Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return torch.tensor(data, dtype=torch.long)


def prepare_fixed_pool(
    dataset: Dataset,
    pool_size: int,
    device: torch.device,
    num_workers: int = 0,
    batch_size: int = 256,
) -> torch.Tensor:
    """Materialize a fixed pool of normalized images for EMA evaluation.

    Returns:
        Tensor `[pool_size, 3, 32, 32]` placed on `device`. If the dataset is
        smaller than `pool_size`, the entire dataset is returned.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    batches = []
    total = 0
    for (images, _), _ in loader:
        # Unpack ((images, labels), indices) format from MultiDatasetSubset
        batches.append(images.to(device))
        total += images.size(0)
        if total >= pool_size:
            break
    if not batches:
        raise RuntimeError("Unable to build fixed pool: dataset is empty")
    pool = torch.cat(batches, dim=0)
    return pool[: min(pool_size, pool.size(0))]


def evaluate_ema_noise_mse(
    ema_model: UNetModel,
    pool: torch.Tensor,
    alphas_bar: torch.Tensor,
    T: int,
    batch_size: int = 256,
) -> float:
    """Compute the diffusion noise-prediction MSE on a fixed pool."""

    ema_model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, pool.size(0), batch_size):
            stop = min(start + batch_size, pool.size(0))
            batch = pool[start:stop]
            bsz = batch.size(0)
            if bsz == 0:
                continue
            timesteps = torch.randint(0, T, (bsz,), device=batch.device, dtype=torch.long)
            noise = torch.randn_like(batch)
            alpha_bar_t = alphas_bar[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha = alpha_bar_t.sqrt()
            sqrt_one_minus = (1 - alpha_bar_t).sqrt()
            xt = sqrt_alpha * batch + sqrt_one_minus * noise
            pred_noise = ema_model(xt, timesteps)
            loss = F.mse_loss(pred_noise, noise, reduction="mean")
            total_loss += loss.item() * bsz
            total += bsz
    return total_loss / max(total, 1)


def _denormalize(
    images: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std_tensor + mean_tensor


def sample_ddim_grid(
    ema_model: UNetModel,
    alphas_bar: torch.Tensor,
    T: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    device: torch.device,
    num_samples: int = 25,
    nrow: int = 5,
    image_size: int = 32,
) -> torch.Tensor:
    """Generate a DDIM sample grid (5×5 by default) for TensorBoard logging."""

    ema_model.eval()
    with torch.no_grad():
        samples = torch.randn(num_samples, 3, image_size, image_size, device=device)
        for step in reversed(range(T)):
            t_batch = torch.full((num_samples,), step, device=device, dtype=torch.long)
            eps = ema_model(samples, t_batch)
            alpha_bar_t = alphas_bar[step]
            sqrt_alpha = alpha_bar_t.sqrt()
            sqrt_one_minus = (1 - alpha_bar_t).sqrt()
            x0_hat = (samples - sqrt_one_minus * eps) / sqrt_alpha
            if step == 0:
                samples = x0_hat
            else:
                alpha_bar_prev = alphas_bar[step - 1]
                samples = alpha_bar_prev.sqrt() * x0_hat + (1 - alpha_bar_prev).sqrt() * eps
        samples = _denormalize(samples, mean, std).clamp(0.0, 1.0)
        grid = make_grid(samples.cpu(), nrow=nrow)
    return grid


def save_watermark_exposure(
    dataset: MultiDatasetSubset,
    run_dir: pathlib.Path,
    checkpoint_step: int,
) -> None:
    """Save watermark exposure statistics."""
    exposure = dataset.get_watermark_exposure()
    if not exposure:
        return
    
    exposure_path = run_dir / "watermark_exposure.json"
    
    # Load existing data if present
    if exposure_path.exists():
        with exposure_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"checkpoints": {}}
    
    # Add current checkpoint's exposure
    data["checkpoints"][str(checkpoint_step)] = {
        "total_samples_seen": sum(exposure.values()),
        "unique_watermarks_seen": len(exposure),
        "min_exposure": min(exposure.values()) if exposure else 0,
        "max_exposure": max(exposure.values()) if exposure else 0,
        "mean_exposure": sum(exposure.values()) / len(exposure) if exposure else 0,
    }
    
    with exposure_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    LOGGER.info(
        "Watermark exposure at step %d: %d unique samples, mean %.1f exposures",
        checkpoint_step,
        len(exposure),
        data["checkpoints"][str(checkpoint_step)]["mean_exposure"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDIM on multiple datasets")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/model_ddim.yaml"))
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--mode", choices=["main", "fastdev"], default="main")
    parser.add_argument("--fastdev", action="store_true", help="Alias for --mode fastdev")
    parser.add_argument("--select-best", action="store_true", help="Select top checkpoints after training")
    args = parser.parse_args()

    if args.fastdev:
        args.mode = "fastdev"

    model_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)
    logging_cfg = model_cfg.get("logging", {})

    seed = model_cfg.get("seed", 0)
    set_global_seeds(seed)
    determinism_state = configure_environment(model_cfg["training"].get("amp", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training indices
    train_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"].get("member_train", "data/splits/member_train.json")))
    if args.mode == "fastdev":
        train_indices = train_indices[:1024]

    # Load watermark indices for exposure tracking (optional)
    watermark_indices = None
    watermark_path = data_cfg["splits"]["paths"].get("watermark_private")
    if watermark_path and pathlib.Path(watermark_path).exists():
        watermark_indices = load_indices(pathlib.Path(watermark_path)).tolist()
        LOGGER.info(f"Loaded {len(watermark_indices)} watermark indices for exposure tracking")

    # Dataset configuration
    dataset_name = data_cfg["dataset"].get("name", "cifar10")
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])  # type: ignore[assignment]
    std = tuple(data_cfg["dataset"]["normalization"]["std"])  # type: ignore[assignment]
    image_shape = data_cfg["dataset"].get("image_shape", [3, 32, 32])
    image_size = image_shape[-1]  # Assume square images

    LOGGER.info(f"Training on {dataset_name} dataset, resolution {image_size}x{image_size}")

    root.mkdir(parents=True, exist_ok=True)
    dataset = MultiDatasetSubset(
        dataset_name=dataset_name,
        root=root,
        indices=train_indices,
        mean=mean,
        std=std,
        image_size=image_size,
        download=data_cfg["dataset"].get("download", False),
        watermark_indices=watermark_indices,
    )
    num_workers = data_cfg["dataset"].get("num_workers", 8)
    batch_size = model_cfg["training"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ema_pool_size = min(logging_cfg.get("ema_pool_size", 4096), len(dataset))
    if args.mode == "fastdev":
        ema_pool_size = min(ema_pool_size, logging_cfg.get("fastdev_ema_pool_size", 1024))
    ema_eval_batch = logging_cfg.get("ema_eval_batch", 256)
    if ema_pool_size <= 0:
        raise RuntimeError("EMA evaluation pool size must be positive")
    LOGGER.info("Preparing EMA evaluation pool of size %d", ema_pool_size)
    ema_pool = prepare_fixed_pool(
        dataset,
        pool_size=ema_pool_size,
        device=device,
        num_workers=num_workers,
        batch_size=ema_eval_batch,
    )

    model = build_unet(model_cfg["model"])
    model.to(device)
    ema = EMA(model, decay=model_cfg["training"]["ema_decay"])
    ema.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["training"]["lr"],
        betas=tuple(model_cfg["training"]["betas"]),
        weight_decay=model_cfg["training"].get("weight_decay", 0.0),
    )

    use_cuda_amp = torch.cuda.is_available() and model_cfg["training"].get("amp", True)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if use_cuda_amp
        else None
    )

    T = model_cfg["diffusion"]["timesteps"]
    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    iterations = model_cfg["training"]["iterations"][args.mode]
    if args.mode == "fastdev":
        fastdev_limit = model_cfg["training"].get("fastdev_limit")
        if fastdev_limit is not None:
            iterations = min(iterations, fastdev_limit)
    checkpoint_interval = model_cfg["training"]["checkpoint_interval"]

    run_dir = pathlib.Path(model_cfg["experiment"]["output_dir"]) / args.mode
    write_run_metadata(run_dir, model_cfg, data_cfg, seed, args.mode, determinism_state)

    tb_dir = run_dir / "tb"
    # #region agent log
    import io; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:675','message':'tb_dir path before SummaryWriter','data':{'tb_dir':str(tb_dir),'exists':tb_dir.exists(),'parent_exists':tb_dir.parent.exists()},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D'})+'\n')
    # #endregion
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    # #region agent log
    import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:676','message':'SummaryWriter created','data':{'tb_dir':str(tb_dir),'exists':tb_dir.exists(),'files':os.listdir(tb_dir) if tb_dir.exists() else []},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D'})+'\n')
    # #endregion
    csv_path = run_dir / "train_log.csv"
    csv_exists = csv_path.exists()
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if (not csv_exists) or csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "lr", "train_loss", "ema_mse"])
        csv_file.flush()

    log_interval = model_cfg["training"]["log_interval"]
    sample_interval = logging_cfg.get("sample_interval", checkpoint_interval)
    if sample_interval <= 0:
        sample_interval = checkpoint_interval
    sample_grid_size = logging_cfg.get("sample_grid_size", 25)
    sample_grid_cols = logging_cfg.get("sample_grid_cols", 5)

    step = 0
    
    # Auto-resume from latest checkpoint if available
    latest_ckpt = find_latest_checkpoint(run_dir)
    if latest_ckpt is not None:
        step = load_checkpoint(latest_ckpt, model, ema, optimizer, device)
        LOGGER.info("Resuming training from step %d (target: %d iterations)", step, iterations)
    else:
        LOGGER.info("Starting training from scratch (target: %d iterations)", iterations)
    
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < iterations:
            for (batch, _), indices in dataloader:
                # Track watermark exposure in main process (works with num_workers > 0)
                dataset.track_exposure(indices)
                batch = batch.to(device)
                loss = diffusion_training_step(
                    model,
                    batch,
                    alphas_bar,
                    T,
                    scaler,
                    optimizer,
                    model_cfg["training"]["grad_clip"],
                )
                ema.update(model)
                step += 1
                if step % log_interval == 0:
                    ema_mse = evaluate_ema_noise_mse(
                        ema.ema_model,
                        ema_pool,
                        alphas_bar,
                        T,
                        batch_size=ema_eval_batch,
                    )
                    current_lr = optimizer.param_groups[0]["lr"]
                    LOGGER.info("step=%d loss=%.4f ema_mse=%.5f", step, loss, ema_mse)
                    tb_writer.add_scalar("train/loss", loss, step)
                    tb_writer.add_scalar("ema/fixed4096_noise_mse", ema_mse, step)
                    csv_writer.writerow([step, current_lr, loss, ema_mse])
                    csv_file.flush()
                if sample_grid_size > 0 and step % sample_interval == 0:
                    # #region agent log
                    import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:734','message':'Before sample_ddim_grid','data':{'step':step,'tb_dir_exists':tb_dir.exists(),'tb_files':os.listdir(tb_dir) if tb_dir.exists() else []},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D,E'})+'\n')
                    # #endregion
                    grid = sample_ddim_grid(
                        ema.ema_model,
                        alphas_bar,
                        T,
                        mean,
                        std,
                        device,
                        num_samples=sample_grid_size,
                        nrow=sample_grid_cols,
                        image_size=image_size,
                    )
                    # #region agent log
                    import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:746','message':'Before tb_writer.add_image','data':{'step':step,'tb_dir_exists':tb_dir.exists(),'tb_files':os.listdir(tb_dir) if tb_dir.exists() else [],'grid_shape':list(grid.shape)},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D,E'})+'\n')
                    # #endregion
                    try:
                        tb_writer.add_image("samples/ddim", grid, step)
                        tb_writer.flush()
                        # #region agent log
                        import io; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:747','message':'tb_writer.add_image SUCCESS','data':{'step':step},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D,E'})+'\n')
                        # #endregion
                    except Exception as e:
                        # #region agent log
                        import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:747','message':'tb_writer.add_image FAILED','data':{'step':step,'error':str(e),'error_type':type(e).__name__,'tb_dir_exists':tb_dir.exists(),'tb_files':os.listdir(tb_dir) if tb_dir.exists() else []},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,B,D,E'})+'\n')
                        # #endregion
                        raise
                if step % checkpoint_interval == 0:
                    # #region agent log
                    import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:748','message':'Before save_checkpoint','data':{'step':step,'tb_dir_exists':tb_dir.exists(),'run_dir_exists':run_dir.exists()},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,C,D'})+'\n')
                    # #endregion
                    save_checkpoint(model, ema, optimizer, step, run_dir)
                    # #region agent log
                    import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:749','message':'After save_checkpoint','data':{'step':step,'tb_dir_exists':tb_dir.exists()},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,C,D'})+'\n')
                    # #endregion
                    # Save watermark exposure stats at each checkpoint
                    if hasattr(dataset, 'get_watermark_exposure'):
                        save_watermark_exposure(dataset, run_dir, step)
                if step >= iterations:
                    break

        save_checkpoint(model, ema, optimizer, step, run_dir)
        # Save final watermark exposure stats (important for short runs that don't hit checkpoint_interval)
        if hasattr(dataset, 'get_watermark_exposure'):
            save_watermark_exposure(dataset, run_dir, step)
        LOGGER.info("Training finished at step %d", step)
    finally:
        # #region agent log
        import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:762','message':'In finally block before cleanup','data':{'tb_dir_exists':tb_dir.exists(),'tb_files':os.listdir(tb_dir) if tb_dir.exists() else []},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D'})+'\n')
        # #endregion
        csv_file.flush()
        csv_file.close()
        try:
            tb_writer.flush()
            tb_writer.close()
            # #region agent log
            import io; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:764','message':'tb_writer cleanup SUCCESS','data':{},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,D,E'})+'\n')
            # #endregion
        except Exception as e:
            # #region agent log
            import io, os; io.open('/home/fjiang4/mia_ddpm_qr copy/.cursor/debug.log', 'a').write(__import__('json').dumps({'location':'train_ddim.py:764','message':'tb_writer cleanup FAILED','data':{'error':str(e),'error_type':type(e).__name__,'tb_dir_exists':tb_dir.exists()},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','hypothesisId':'A,B,D,E'})+'\n')
            # #endregion
            raise

    if args.select_best:
        selection_path = select_checkpoints.run_selection(
            run_dir=run_dir,
            model_config=args.config,
            data_config=args.data,
            device=device,
        )
        LOGGER.info("Checkpoint selection summary written to %s", selection_path)


if __name__ == "__main__":
    main()
