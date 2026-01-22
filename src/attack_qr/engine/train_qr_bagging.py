from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from src.attack_qr.models.qr_resnet18 import ResNet18QR, ResNet18GaussianQR
from src.attack_qr.utils.losses import pinball_loss
from src.attack_qr.utils.seeding import seed_everything
from src.ddpm.data.loader import IndexedDataset, get_dataset, get_transforms


@dataclass
class QuantileTrainingConfig:
    """Configuration for quantile regression bagging training.
    
    Attributes:
        lr: Learning rate for optimizer
        epochs: Number of training epochs per model
        batch_size: Batch size for training
        alpha_list: List of quantile values (tau) to train for
        bootstrap: Whether to use bootstrap sampling
        M: Number of models in the ensemble
        seed: Random seed for reproducibility
        use_log1p: Whether to apply log1p transformation to targets
    """
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    alpha_list: Sequence[float] = (0.01, 0.001)
    bootstrap: bool = True
    M: int = 16
    seed: int = 0
    use_log1p: bool = True
    # Extended fields for Gaussian head + bagging workflow
    weight_decay: float = 0.0
    cosine_anneal: bool = True
    early_stop_patience: int = 10
    val_ratio: float = 0.1
    bootstrap_ratio: float = 0.8
    B: int = 50
    num_workers: int = 0
    device: str = "cuda"


class QuantileScoresDataset(Dataset):
    """Dataset that loads images paired with q25 scores and stats from scores files.
    
    This dataset is designed for the q25 aggregation workflow where:
    - scores: q25 aggregated t-error values (attack scores) [N]
    - stats: Summary statistics from t-error sequence [N, 3]
             Contains [mean_error, std_error, l2_error] per sample
             Note: stats do NOT include q25 itself to avoid information leakage
    
    The dataset returns (image, stats, target_raw, target_log) tuples where:
    - image: Normalized CIFAR-10 image [3, 32, 32]
    - stats: Summary stats from t-error sequence [3]
    - target_raw: Raw q25 score (scalar)
    - target_log: log1p(q25) score for training (scalar)
    
    ALIGNMENT GUARANTEE:
    -------------------
    This dataset assumes that:
    1. The `indices` parameter is loaded from the SAME JSON file used by
       compute_scores.py when generating the scores file.
    2. scores[i] and stats[i, :] correspond to indices[i] (the i-th CIFAR index
       in the JSON file's order).
    3. Both compute_scores.py and this dataset iterate through indices in the
       SAME order (no shuffling during score computation).
    
    The __getitem__ method:
    - Uses `indices[idx]` to fetch the correct CIFAR image
    - Uses `scores[idx]` and `stats[idx]` for the corresponding labels
    
    This ensures image ↔ score ↔ stats are always correctly aligned.
    """
    
    # Default stats dimension (mean, std, L2)
    STATS_DIM = 3
    
    def __init__(
        self,
        data_root: str | Path,
        indices: Sequence[int],
        scores_path: str | Path,
        mean: tuple = (0.5, 0.5, 0.5),
        std: tuple = (0.5, 0.5, 0.5),
        train: bool = False,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            data_root: Root directory for CIFAR-10 data
            indices: List of CIFAR-10 indices to use. MUST be from the SAME JSON
                     file used when computing scores (e.g., data/splits/aux.json).
            scores_path: Path to scores .pt file (e.g., scores/q25_aux.pt)
            mean: Normalization mean for images
            std: Normalization std for images
            train: Whether this is for training (uses train split of CIFAR-10)
        
        Important:
            The indices MUST match the order used in compute_scores.py.
            scores[i] corresponds to indices[i], NOT to CIFAR index i.
        """
        self.indices = list(indices)
        
        # Set up image transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Load CIFAR-10 base dataset
        # Allow automatic download if dataset not found
        base = CIFAR10(root=str(data_root), train=train, download=True, transform=transform)
        self.base_dataset = base
        
        # Load scores and stats from scores file
        scores_path = Path(scores_path)
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores file not found: {scores_path}")
        
        scores_data = torch.load(scores_path, map_location="cpu")
        
        # Extract scores (q25 aggregated values)
        self.scores = scores_data["scores"].float()
        
        # Extract stats (mean, std, L2 of t-error sequence)
        if "stats" not in scores_data:
            raise ValueError(
                f"Scores file {scores_path} does not contain 'stats' field. "
                "Please regenerate scores with the updated compute_scores.py"
            )
        self.stats = scores_data["stats"].float()
        
        # Compute log1p transformed targets for training
        self.log_scores = torch.log1p(self.scores.clamp_min(0))
        
        # Validate dimensions
        if len(self.indices) != len(self.scores):
            raise ValueError(
                f"Mismatch between indices ({len(self.indices)}) and scores ({len(self.scores)})"
            )
        if len(self.scores) != self.stats.shape[0]:
            raise ValueError(
                f"Mismatch between scores ({len(self.scores)}) and stats ({self.stats.shape[0]})"
            )
        
        self.stats_dim = self.stats.shape[1]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        """Return (image, stats, target_raw, target_log) for the given index."""
        cifar_idx = self.indices[idx]
        image, _ = self.base_dataset[cifar_idx]
        
        stats = self.stats[idx]
        target_raw = self.scores[idx]
        target_log = self.log_scores[idx]
        
        return image, stats, target_raw, target_log


class QuantilePairsDataset(Dataset):
    """Legacy dataset using NPZ pairs files (kept for backwards compatibility)."""
    
    def __init__(self, indexed_dataset: IndexedDataset, pairs_by_image: dict[int, dict[str, np.ndarray]]):
        self.dataset = indexed_dataset
        self.pairs_by_image = pairs_by_image
        self.image_ids = sorted(pairs_by_image.keys())
        self.idx_to_pos = {idx: pos for pos, idx in enumerate(self.dataset.indices)}
        if not self.image_ids:
            raise ValueError("No public samples available for quantile regression.")
        example = pairs_by_image[self.image_ids[0]]["stats"]
        self.stats_dim = example.shape[1]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, item: int):
        img_id = int(self.image_ids[item])
        pos = self.idx_to_pos[img_id]
        img, _, _ = self.dataset[pos]
        pair_info = self.pairs_by_image[img_id]
        choice = np.random.randint(0, pair_info["t_error"].shape[0])
        stats = torch.tensor(pair_info["stats"][choice], dtype=torch.float32)
        target = torch.tensor(pair_info["t_error"][choice], dtype=torch.float32)
        return img, stats, target


def load_pairs(npz_path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Load legacy NPZ pairs file (kept for backwards compatibility)."""
    with np.load(npz_path) as data:
        image_ids = data["image_id"].astype(np.int64)
        t_error = data["t_error"].astype(np.float32)
        t_frac = data["t_frac"].astype(np.float32)
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        norm2 = data["norm2"].astype(np.float32)

    pairs: dict[int, dict[str, np.ndarray]] = {}
    unique_ids = np.unique(image_ids)
    for img_id in unique_ids:
        mask = image_ids == img_id
        stats = np.stack([t_frac[mask], mean[mask], std[mask], norm2[mask]], axis=1)
        pairs[int(img_id)] = {
            "t_error": t_error[mask],
            "stats": stats,
        }
    return pairs


def load_scores_data(scores_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load scores and stats from a scores .pt file.
    
    Args:
        scores_path: Path to scores file (e.g., scores/q25_aux.pt)
    
    Returns:
        Dictionary containing 'scores' [N] and 'stats' [N, D] tensors
    """
    scores_path = Path(scores_path)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")
    
    data = torch.load(scores_path, map_location="cpu")
    
    result = {
        "scores": data["scores"].float(),
    }
    
    if "stats" in data:
        result["stats"] = data["stats"].float()
    
    return result


def prepare_dataset(
    dataset_name: str,
    root: str,
    public_indices: Sequence[int],
    img_size: int,
) -> IndexedDataset:
    """Prepare indexed dataset (legacy helper, kept for compatibility)."""
    base_dataset = get_dataset(dataset_name, root=root, download=True)
    transform = get_transforms(img_size, augment=False)
    return IndexedDataset(base_dataset, indices=public_indices, transform=transform)


def train_val_split(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    """Deterministic train/val split helper for bagging workflows."""
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(1, len(dataset) - val_size)
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gaussian negative log-likelihood for scalar targets in log-space."""
    assert mu.shape == log_sigma.shape == target.shape, "Shape mismatch in gaussian_nll_loss"
    sigma = torch.exp(log_sigma).clamp(min=eps)
    z = (target - mu) / sigma
    return (0.5 * z.pow(2) + log_sigma + 0.5 * math.log(2.0 * math.pi)).mean()


def bootstrap_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate bootstrap sample indices."""
    return rng.integers(low=0, high=size, size=size)


def train_single_model_scores(
    dataset: QuantileScoresDataset,
    config: QuantileTrainingConfig,
    model_seed: int,
    device: torch.device,
) -> tuple[ResNet18QR, List[float]]:
    """Train a single quantile regression model using scores-based dataset.
    
    This function trains on the QuantileScoresDataset which provides:
    - (image, stats, target_raw, target_log) tuples
    - stats are the t-error summary statistics (mean, std, L2)
    - target is the q25 score (or log1p(q25) if use_log1p=True)
    
    Args:
        dataset: QuantileScoresDataset instance
        config: Training configuration
        model_seed: Random seed for model initialization
        device: Device to train on
    
    Returns:
        Tuple of (trained model, list of per-epoch losses)
    """
    from mia_logging import get_winston_logger
    logger = get_winston_logger(__name__)
    
    seed_everything(model_seed)
    
    # Get stats dimension from dataset
    stats_dim = getattr(dataset, "stats_dim", QuantileScoresDataset.STATS_DIM)
    
    # Create model with explicit stats_dim
    model = ResNet18QR(num_outputs=len(config.alpha_list), stats_dim=stats_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    alpha_tensor = torch.tensor(config.alpha_list, dtype=torch.float32, device=device)
    
    # Validate first batch to ensure stats_dim alignment
    first_batch_validated = False

    epoch_losses: List[float] = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for images, stats, target_raw, target_log in loader:
            # Validate first batch stats dimension matches model expectation
            if not first_batch_validated:
                actual_stats_dim = stats.shape[1] if stats.dim() > 1 else 1
                if actual_stats_dim != stats_dim:
                    raise ValueError(
                        f"Stats dimension mismatch! Model expects stats_dim={stats_dim}, "
                        f"but first batch has stats.shape={stats.shape} (dim={actual_stats_dim})"
                    )
                logger.info(
                    "First batch validation: images=%s, stats=%s (dim=%d), "
                    "target_raw=%s, model.stats_dim=%d",
                    tuple(images.shape), tuple(stats.shape), actual_stats_dim,
                    tuple(target_raw.shape), model.stats_dim
                )
                first_batch_validated = True
            images = images.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            
            # Select target based on config: log1p(q25) or raw q25
            if config.use_log1p:
                targets = target_log.to(device, non_blocking=True)
            else:
                targets = target_raw.to(device, non_blocking=True)
            
            # Expand targets for multi-quantile prediction [B] -> [B, num_quantiles]
            targets = targets.unsqueeze(1).expand(-1, len(config.alpha_list))
            
            preds = model(images, stats)
            loss = pinball_loss(preds, targets, alpha_tensor, reduction="mean")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        epoch_loss = float(sum(losses) / max(1, len(losses)))
        epoch_losses.append(epoch_loss)
    
    return model, epoch_losses


def train_single_model_gaussian_scores(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: QuantileTrainingConfig,
    logger,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Train a Gaussian head model on log1p targets using NLL."""
    device = torch.device(getattr(cfg, "device", "cuda"))
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    cosine_anneal = getattr(cfg, "cosine_anneal", False)
    early_stop_patience = getattr(cfg, "early_stop_patience", 10)
    epochs = getattr(cfg, "epochs", 30)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine_anneal else None
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_train = 0
        for images, stats, target_raw, target_log in train_loader:
            images = images.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            targets = target_log.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            mu, log_sigma = model(images, stats)
            loss = gaussian_nll_loss(mu, log_sigma, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            num_train += batch_size

        train_loss /= max(1, num_train)

        model.eval()
        val_loss = 0.0
        num_val = 0
        with torch.no_grad():
            for images, stats, target_raw, target_log in val_loader:
                images = images.to(device, non_blocking=True)
                stats = stats.to(device, non_blocking=True)
                targets = target_log.to(device, non_blocking=True)
                mu, log_sigma = model(images, stats)
                loss = gaussian_nll_loss(mu, log_sigma, targets)
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                num_val += batch_size

        val_loss /= max(1, num_val)

        logger.info(f"[Gaussian] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                logger.info("Early stopping (Gaussian head) triggered")
                break

        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state, best_val


def train_single_model(
    dataset: QuantilePairsDataset,
    config: QuantileTrainingConfig,
    model_seed: int,
    device: torch.device,
) -> tuple[ResNet18QR, List[float]]:
    """Train a single quantile regression model using legacy pairs dataset.
    
    This is kept for backwards compatibility with the NPZ pairs workflow.
    For the new q25 workflow, use train_single_model_scores instead.
    """
    seed_everything(model_seed)
    stats_dim = getattr(dataset, "stats_dim", None)
    if stats_dim is None and hasattr(dataset, "dataset"):
        stats_dim = getattr(dataset.dataset, "stats_dim", None)
    if stats_dim is None:
        raise AttributeError("Quantile dataset must expose stats_dim")
    model = ResNet18QR(num_outputs=len(config.alpha_list), stats_dim=stats_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    alpha_tensor = torch.tensor(config.alpha_list, dtype=torch.float32, device=device)

    epoch_losses: List[float] = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for images, stats, targets in loader:
            images = images.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1).expand(-1, len(config.alpha_list))
            preds = model(images, stats)
            loss = pinball_loss(preds, targets, alpha_tensor, reduction="mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = float(sum(losses) / max(1, len(losses)))
        epoch_losses.append(epoch_loss)
    return model, epoch_losses


def train_bagging_ensemble(
    npz_path: str | Path,
    dataset_name: str,
    public_indices: Sequence[int],
    config: QuantileTrainingConfig,
    out_dir: str | Path,
    img_size: int,
    data_root: str,
    device: str | torch.device = "cuda",
    skip_existing: bool = False,
) -> None:
    seed_everything(config.seed)
    device = torch.device(device)
    pairs_by_image = load_pairs(npz_path)
    dataset = prepare_dataset(dataset_name, root=data_root, public_indices=public_indices, img_size=img_size)

    data = QuantilePairsDataset(dataset, pairs_by_image)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    base_manifest = {
        "dataset": dataset_name,
        "alpha_list": list(config.alpha_list),
        "bootstrap": config.bootstrap,
        "M": config.M,
        "seed": config.seed,
        "public_indices": [int(i) for i in public_indices],
        "stats_dim": data.stats_dim,
        "mode": "quantile",
        "target_space": "log1p" if getattr(config, "use_log1p", False) else "raw",
        "models": [],
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in base_manifest.items():
            if key == "models":
                continue
            if manifest.get(key) != value:
                raise ValueError(
                    f"Existing manifest at {manifest_path} has {key}={manifest.get(key)} but expected {value}."
                )
    else:
        manifest = base_manifest

    manifest.setdefault("models", [])
    manifest_lookup = {entry["path"]: entry for entry in manifest["models"]}

    def save_manifest() -> None:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if skip_existing and not manifest_path.exists():
        existing_ckpts = sorted(out_dir.glob("model_*.pt"))
        for ckpt_path in existing_ckpts:
            rel_path = ckpt_path.name
            if rel_path in manifest_lookup:
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            entry = {
                "path": rel_path,
                "seed": ckpt.get("seed"),
                "loss": (ckpt.get("losses") or [None])[-1],
            }
            if ckpt.get("bootstrap_indices") is not None:
                entry["bootstrap_indices"] = ckpt["bootstrap_indices"]
            if ckpt.get("bootstrap_image_ids") is not None:
                entry["bootstrap_image_ids"] = ckpt["bootstrap_image_ids"]
            manifest["models"].append(entry)
            manifest_lookup[rel_path] = entry
        if existing_ckpts:
            save_manifest()

    indices = np.arange(len(data))
    rng_master = np.random.default_rng(config.seed)

    for m in tqdm(range(config.M), desc="Bagging"):
        master_seed = int(rng_master.integers(0, 2**63 - 1))
        rng = np.random.default_rng(master_seed)
        ckpt_path = out_dir / f"model_{m:03d}.pt"
        rel_path = str(ckpt_path.relative_to(out_dir))
        existing_entry = manifest_lookup.get(rel_path)

        if skip_existing and ckpt_path.exists() and existing_entry is not None:
            continue
        if ckpt_path.exists() and not skip_existing:
            raise FileExistsError(f"Checkpoint {ckpt_path} already exists. Remove it or use --skip-existing.")

        if config.bootstrap:
            sampled_indices = bootstrap_indices(len(indices), rng)
            subset = Subset(data, sampled_indices.tolist())
            setattr(subset, "stats_dim", data.stats_dim)
            sampled_list = sampled_indices.tolist()
            sampled_image_ids = [int(data.image_ids[i]) for i in sampled_list]
        else:
            subset = data
            sampled_list = None
            sampled_image_ids = None

        model_seed = int(rng.integers(0, 2**31 - 1))
        seed_everything(model_seed)
        model, losses = train_single_model(subset, config, model_seed, device)
        torch.save(
            {
                "model": model.state_dict(),
                "alpha_list": list(config.alpha_list),
                "seed": model_seed,
                "losses": losses,
                "bootstrap_indices": sampled_list,
                "bootstrap_image_ids": sampled_image_ids,
            },
            ckpt_path,
        )
        entry = {
            "path": rel_path,
            "seed": model_seed,
            "loss": losses[-1] if losses else None,
        }
        if config.bootstrap:
            entry["bootstrap_indices"] = sampled_list
            entry["bootstrap_image_ids"] = sampled_image_ids

        manifest_lookup[rel_path] = entry
        manifest["models"] = [e for e in manifest["models"] if e["path"] != rel_path]
        manifest["models"].append(entry)
        save_manifest()


def train_bagging_ensemble_scores(
    scores_path: str | Path,
    indices_path: str | Path,
    data_root: str | Path,
    config: QuantileTrainingConfig,
    out_dir: str | Path,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
    train_cifar: bool = False,
    device: str | torch.device = "cuda",
    skip_existing: bool = False,
) -> None:
    """Train bagging ensemble using scores files directly (q25 workflow).
    
    This function trains the QR ensemble using:
    - Images from CIFAR-10 (indexed by indices_path)
    - q25 scores and stats from scores_path
    
    The stats (mean_error, std_error, l2_error of t-error sequence) are used
    as additional input features to the model, while the q25 score is used
    as the training target.
    
    Args:
        scores_path: Path to scores file (e.g., scores/q25_aux.pt)
        indices_path: Path to JSON file with CIFAR-10 indices (e.g., data/splits/aux.json)
        data_root: Root directory for CIFAR-10 data
        config: Training configuration
        out_dir: Output directory for model checkpoints
        mean: Normalization mean for images
        std: Normalization std for images
        train_cifar: Whether to use CIFAR-10 train split (False = test split)
        device: Device to train on
        skip_existing: Skip training for models that already exist
    """
    seed_everything(config.seed)
    device = torch.device(device)
    
    # Load indices from JSON file
    indices_path = Path(indices_path)
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_path}")
    with indices_path.open("r", encoding="utf-8") as f:
        indices = json.load(f)
    
    # Create dataset using scores file
    dataset = QuantileScoresDataset(
        data_root=data_root,
        indices=indices,
        scores_path=scores_path,
        mean=mean,
        std=std,
        train=train_cifar,
    )
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    
    base_manifest = {
        "scores_path": str(scores_path),
        "indices_path": str(indices_path),
        "alpha_list": list(config.alpha_list),
        "bootstrap": config.bootstrap,
        "M": config.M,
        "seed": config.seed,
        "use_log1p": config.use_log1p,
        "stats_dim": dataset.stats_dim,
        "num_samples": len(dataset),
        "mode": "quantile",
        "target_space": "log1p" if config.use_log1p else "raw",
        "models": [],
    }
    
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Validate key parameters match
        for key in ["alpha_list", "M", "seed", "stats_dim"]:
            if manifest.get(key) != base_manifest.get(key):
                raise ValueError(
                    f"Existing manifest at {manifest_path} has {key}={manifest.get(key)} "
                    f"but expected {base_manifest.get(key)}."
                )
    else:
        manifest = base_manifest
    
    manifest.setdefault("models", [])
    manifest_lookup = {entry["path"]: entry for entry in manifest["models"]}
    
    def save_manifest() -> None:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    
    # Recover existing checkpoints if manifest doesn't exist
    if skip_existing and not manifest_path.exists():
        existing_ckpts = sorted(out_dir.glob("model_*.pt"))
        for ckpt_path in existing_ckpts:
            rel_path = ckpt_path.name
            if rel_path in manifest_lookup:
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            entry = {
                "path": rel_path,
                "seed": ckpt.get("seed"),
                "loss": (ckpt.get("losses") or [None])[-1],
            }
            if ckpt.get("bootstrap_indices") is not None:
                entry["bootstrap_indices"] = ckpt["bootstrap_indices"]
            manifest["models"].append(entry)
            manifest_lookup[rel_path] = entry
        if existing_ckpts:
            save_manifest()
    
    indices_array = np.arange(len(dataset))
    rng_master = np.random.default_rng(config.seed)
    
    for m in tqdm(range(config.M), desc="Bagging (scores)"):
        master_seed = int(rng_master.integers(0, 2**63 - 1))
        rng = np.random.default_rng(master_seed)
        ckpt_path = out_dir / f"model_{m:03d}.pt"
        rel_path = str(ckpt_path.relative_to(out_dir))
        existing_entry = manifest_lookup.get(rel_path)
        
        if skip_existing and ckpt_path.exists() and existing_entry is not None:
            continue
        if ckpt_path.exists() and not skip_existing:
            raise FileExistsError(
                f"Checkpoint {ckpt_path} already exists. Remove it or use --skip-existing."
            )
        
        # Bootstrap sampling
        if config.bootstrap:
            sampled_indices = bootstrap_indices(len(indices_array), rng)
            subset = Subset(dataset, sampled_indices.tolist())
            # Propagate stats_dim to subset for model creation
            setattr(subset, "stats_dim", dataset.stats_dim)
            sampled_list = sampled_indices.tolist()
        else:
            subset = dataset
            sampled_list = None
        
        model_seed = int(rng.integers(0, 2**31 - 1))
        seed_everything(model_seed)
        
        # Create temporary dataset wrapper for subset that returns 4-tuples
        class SubsetWrapper(Dataset):
            def __init__(self, subset, parent_dataset):
                self.subset = subset
                self.stats_dim = parent_dataset.stats_dim
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                return self.subset[idx]
        
        wrapped_subset = SubsetWrapper(subset, dataset)
        model, losses = train_single_model_scores(wrapped_subset, config, model_seed, device)
        
        torch.save(
            {
                "model": model.state_dict(),
                "alpha_list": list(config.alpha_list),
                "stats_dim": dataset.stats_dim,
                "seed": model_seed,
                "losses": losses,
                "use_log1p": config.use_log1p,
                "bootstrap_indices": sampled_list,
            },
            ckpt_path,
        )
        
        entry = {
            "path": rel_path,
            "seed": model_seed,
            "loss": losses[-1] if losses else None,
        }
        if config.bootstrap:
            entry["bootstrap_indices"] = sampled_list
        
        manifest_lookup[rel_path] = entry
        manifest["models"] = [e for e in manifest["models"] if e["path"] != rel_path]
        manifest["models"].append(entry)
        save_manifest()


def train_bagging_ensemble_gaussian_scores(
    dataset: QuantileScoresDataset,
    cfg: QuantileTrainingConfig,
    logger,
) -> Dict[str, object]:
    """Train a bagging ensemble of Gaussian QR models on aux non-member scores."""
    B = getattr(cfg, "B", getattr(cfg, "M", 1))
    bootstrap_ratio = getattr(cfg, "bootstrap_ratio", 0.8)
    val_ratio = getattr(cfg, "val_ratio", 0.1)
    batch_size = getattr(cfg, "batch_size", 256)
    num_workers = getattr(cfg, "num_workers", 0)
    seed = getattr(cfg, "seed", 0)

    ensembles: List[Dict[str, object]] = []
    generator = torch.Generator().manual_seed(seed)

    for b in range(B):
        logger.info(f"[Gaussian bagging] training model b={b}/{B-1}")

        bootstrap_size = max(1, int(len(dataset) * bootstrap_ratio))
        bootstrap_indices = torch.randint(
            low=0,
            high=len(dataset),
            size=(bootstrap_size,),
            generator=generator,
        ).tolist()

        bootstrap_subset = torch.utils.data.Subset(dataset, bootstrap_indices)
        train_subset, val_subset = train_val_split(bootstrap_subset, val_ratio=val_ratio, seed=seed + b)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        model_seed = seed + b
        seed_everything(model_seed)

        model = ResNet18GaussianQR(stats_dim=dataset.stats_dim)
        best_state, best_val = train_single_model_gaussian_scores(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            logger=logger,
        )

        ensembles.append(
            {
                "state_dict": best_state,
                "val_loss": best_val,
                "bootstrap_indices": bootstrap_indices,
                "seed": model_seed,
            }
        )

    manifest = {
        "mode": "gaussian",
        "B": B,
        "bootstrap_ratio": bootstrap_ratio,
        "stats_dim": dataset.stats_dim,
        "target_space": "log1p",
        "tau_values": getattr(cfg, "alpha_list", ()),
        "seed": seed,
    }

    return {"manifest": manifest, "models": ensembles}
