"""QR-MIA Attack Evaluation Module.

This module implements the evaluation logic for the Quantile Regression-based
Membership Inference Attack (QR-MIA) on diffusion models.

CANONICAL ENSEMBLE AGGREGATION (Step 4):
-----------------------------------------
The ensemble aggregation follows a unified rule:

1. Each model b outputs a predicted non-member τ-quantile: q̂_τ^(b)(x)
2. Aggregate predictions in quantile space (mean aggregation):
   q̂_τ^ens(x) = (1/B) Σ_b q̂_τ^(b)(x)
3. Define the ensemble margin:
   m(x) = q̂_τ^ens(x) - s(x)
   where s(x) is the observed q25 t-error score (lower is more member-like)
4. Use m(x) as the ONLY attack score for all evaluation metrics
   (TPR@FPR, ROC-AUC, etc.)

Interpretation:
- Positive margin: actual score is LOWER than predicted non-member quantile
  => sample is more likely a MEMBER (lower reconstruction error)
- Negative margin: actual score is at or above predicted non-member level
  => sample is more likely a NON-MEMBER

NOTE: Previous "per-model decision + majority vote" logic has been DEPRECATED
in favor of this unified quantile-space aggregation approach.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from src.attacks.eval.metrics import roc_auc, tpr_precision_at_fpr
from src.attack_qr.features.t_error import compute_t_error
from src.attack_qr.models.qr_resnet18 import ResNet18QR, ResNet18GaussianQR
from src.attack_qr.utils.metrics import bootstrap_metrics, compute_roc, interpolate_tpr
from src.attack_qr.utils.seeding import seed_everything, timesteps_seed
from src.ddpm.data.loader import IndexedDataset, get_dataset, get_transforms
from src.ddpm.schedules.noise import DiffusionSchedule


@dataclass
class EvalConfig:
    """Configuration for attack evaluation.
    
    Attributes:
        alpha: Quantile value (tau) for evaluation
        mode: t-error mode ('x0' or 'eps')
        K: Number of timesteps to sample per image
        batch_size: Batch size for evaluation
        bootstrap: Number of bootstrap iterations for confidence intervals
        seed: Random seed
        use_log1p: Whether the model was trained with log1p targets
    """
    alpha: float
    mode: str = "x0"
    K: int = 4
    batch_size: int = 128
    bootstrap: int = 200
    seed: int = 0
    use_log1p: bool = True


# Standard normal for Gaussian quantile computation
_normal = Normal(loc=0.0, scale=1.0)


def gaussian_quantile_from_params(mu: torch.Tensor, log_sigma: torch.Tensor, tau: float) -> torch.Tensor:
    """Compute q_tau(x) in log-space for y ~ N(mu(x), sigma^2(x)).
    
    Gaussian head semantics:
    - alpha is the target FPR over non-member scores (e.g., 1e-3)
    - tau = 1 - alpha is the upper-tail quantile level passed to Normal.icdf
    - This function expects tau to already be the upper-tail quantile level
    """
    z = _normal.icdf(torch.tensor([tau], device=mu.device, dtype=mu.dtype))[0]
    sigma = torch.exp(log_sigma)
    return mu + sigma * z


class EvalScoresDataset(Dataset):
    """Dataset for evaluation that loads images with scores and stats from scores files.
    
    This dataset is used for evaluating the QR-based attack on eval_in/eval_out splits.
    It provides (image, stats, score_raw, score_log) tuples where:
    - image: Normalized CIFAR-10 image [3, 32, 32]
    - stats: Summary stats from t-error sequence [3] (mean_error, std_error, l2_error)
    - score_raw: Raw q25 score (the attack score)
    - score_log: log1p(q25) score
    
    Note: stats do NOT include q25 itself to avoid information leakage.
    
    ALIGNMENT GUARANTEE:
    -------------------
    This dataset assumes that:
    1. The `indices` parameter is loaded from the SAME JSON file used by
       compute_scores.py when generating the scores file.
    2. scores[i] and stats[i, :] correspond to indices[i] (the i-th CIFAR index
       in the JSON file's order).
    3. Both compute_scores.py and this dataset iterate through indices in the
       SAME order (no shuffling during score computation).
    """
    
    STATS_DIM = 3
    
    def __init__(
        self,
        data_root: str | Path,
        indices: Sequence[int],
        scores_path: str | Path,
        mean: tuple = (0.5, 0.5, 0.5),
        std: tuple = (0.5, 0.5, 0.5),
        train: bool = True,
    ) -> None:
        """Initialize the evaluation dataset.
        
        Args:
            data_root: Root directory for CIFAR-10 data
            indices: List of CIFAR-10 indices for this split. MUST be from the SAME
                     JSON file used when computing scores (e.g., data/splits/eval_in.json).
            scores_path: Path to scores .pt file (e.g., scores/q25_eval_in.pt)
            mean: Normalization mean for images
            std: Normalization std for images
            train: Whether to use CIFAR-10 train split (True for eval_in, True for eval_out)
        
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
        self.base_dataset = CIFAR10(
            root=str(data_root),
            train=train,
            download=False,
            transform=transform,
        )
        
        # Load scores and stats from scores file
        scores_path = Path(scores_path)
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores file not found: {scores_path}")
        
        scores_data = torch.load(scores_path, map_location="cpu")
        
        # Extract scores (q25 aggregated values) - these are the attack scores
        self.scores = scores_data["scores"].float()
        
        # Extract stats (mean, std, L2 of t-error sequence) - model input features
        if "stats" not in scores_data:
            raise ValueError(
                f"Scores file {scores_path} does not contain 'stats' field. "
                "Please regenerate scores with the updated compute_scores.py"
            )
        self.stats = scores_data["stats"].float()
        
        # Compute log1p transformed scores
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
        """Return (image, stats, score_raw, score_log) for the given index."""
        cifar_idx = self.indices[idx]
        image, _ = self.base_dataset[cifar_idx]
        
        stats = self.stats[idx]
        score_raw = self.scores[idx]
        score_log = self.log_scores[idx]
        
        return image, stats, score_raw, score_log


def load_quantile_ensemble(
    models_dir: str | Path,
    device: torch.device,
) -> tuple[list[torch.nn.Module], List[float], Dict]:
    """Load QR ensemble (quantile or gaussian) from checkpoint directory."""
    models_dir = Path(models_dir)
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    mode = manifest.get("mode", "quantile")
    alpha_list = manifest.get("alpha_list") or manifest.get("tau_values") or []
    stats_dim = manifest.get("stats_dim", 3)
    use_log1p = manifest.get("use_log1p", True)
    
    ensemble = []
    for entry in manifest["models"]:
        ckpt_path = models_dir / entry["path"]
        ckpt = torch.load(ckpt_path, map_location=device)
        model_stats_dim = ckpt.get("stats_dim", stats_dim)
        if mode == "gaussian":
            model = ResNet18GaussianQR(stats_dim=model_stats_dim).to(device)
        else:
            model = ResNet18QR(num_outputs=len(alpha_list), stats_dim=model_stats_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        ensemble.append(model)
    
    return ensemble, alpha_list, manifest


def prepare_eval_dataloaders(
    dataset_name: str,
    data_root: str,
    member_indices: Sequence[int],
    nonmember_indices: Sequence[int],
    img_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    base = get_dataset(dataset_name, root=data_root, download=True)
    transform = get_transforms(img_size, augment=False)
    member_dataset = IndexedDataset(base, indices=member_indices, transform=transform)
    nonmember_dataset = IndexedDataset(base, indices=nonmember_indices, transform=transform)

    def _loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return _loader(member_dataset), _loader(nonmember_dataset)


def _collect_sample_info(
    ddpm_model: torch.nn.Module,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    indices: Sequence[int],
    dataset_name: str,
    global_seed: int,
    K: int,
    mode: str,
) -> List[dict]:
    device = images.device
    denom = max(1, schedule.T - 1)
    outputs: List[dict] = []
    for img, idx in zip(images, indices):
        idx_int = int(idx)
        rng = np.random.default_rng(timesteps_seed(dataset_name, idx_int, global_seed))
        timesteps = rng.integers(low=0, high=schedule.T, size=K, endpoint=False)
        t_tensor = torch.as_tensor(timesteps, device=device, dtype=torch.long)
        x_batch = img.unsqueeze(0).repeat(K, 1, 1, 1)
        errors = compute_t_error(
            model=ddpm_model,
            schedule=schedule,
            x0=x_batch,
            timesteps=t_tensor,
            dataset_name=dataset_name,
            sample_indices=[idx_int] * K,
            global_seed=global_seed,
            mode=mode,
        )
        mean_val = float(img.mean().item())
        std_val = float(img.std(unbiased=False).item())
        norm2_val = float(torch.linalg.norm(img.float()).item())
        t_frac = t_tensor.float() / denom
        stats = torch.stack(
            [
                t_frac,
                torch.full_like(t_frac, mean_val),
                torch.full_like(t_frac, std_val),
                torch.full_like(t_frac, norm2_val),
            ],
            dim=1,
        )
        outputs.append(
            {
                "image": img.unsqueeze(0),
                "stats": stats,
                "mean_error": float(errors.mean().item()),
                "index": idx_int,
            }
        )
    return outputs


def evaluate_attack(
    ddpm_model: torch.nn.Module,
    schedule: DiffusionSchedule,
    ensemble: list[ResNet18QR],
    alpha_list: Sequence[float],
    config: EvalConfig,
    dataset_name: str,
    data_root: str,
    member_indices: Sequence[int],
    nonmember_indices: Sequence[int],
    img_size: int,
    global_seed: int,
    device: str | torch.device = "cuda",
    out_dir: str | Path = "runs/eval",
) -> Dict:
    device = torch.device(device)
    seed_everything(config.seed)
    ddpm_model.to(device).eval()
    schedule = schedule.to(device)
    for model in ensemble:
        model.to(device).eval()

    try:
        alpha_idx = alpha_list.index(config.alpha)
    except ValueError as exc:
        raise ValueError(f"Alpha {config.alpha} not in ensemble outputs {alpha_list}") from exc

    members_loader, nonmembers_loader = prepare_eval_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        img_size=img_size,
        batch_size=config.batch_size,
    )

    records = []

    def _process(loader, label):
        """Process a split (members or non-members) and compute canonical margin.
        
        CANONICAL AGGREGATION:
        1. Each model predicts τ-quantile: q̂_τ^(b)(x)
        2. Ensemble prediction: q̂_τ^ens(x) = mean(q̂_τ^(b)(x))
        3. Margin: m(x) = q̂_τ^ens(x) - s(x)
        
        The margin m(x) is stored as 'score' in records and used for all metrics.
        """
        for images, _, idxs in tqdm(loader, desc=f"Eval label={label}"):
            images = images.to(device, non_blocking=True)
            sample_infos = _collect_sample_info(
                ddpm_model=ddpm_model,
                schedule=schedule,
                images=images,
                indices=idxs,
                dataset_name=dataset_name,
                global_seed=global_seed,
                K=config.K,
                mode=config.mode,
            )
            # s(x): observed scores (mean t-error)
            mean_errors = torch.tensor([info["mean_error"] for info in sample_infos], device=device)

            # Collect predictions from all models: q̂_τ^(b)(x)
            model_preds = []
            for model in ensemble:
                preds_per_sample = []
                for info in sample_infos:
                    stats = info["stats"]
                    img_rep = info["image"].repeat(stats.size(0), 1, 1, 1)
                    # Get model prediction for target quantile
                    preds = model(img_rep, stats).detach()[:, alpha_idx].mean()
                    preds_per_sample.append(preds)
                model_preds.append(torch.stack(preds_per_sample))

            # Stack predictions: [M, N] where M is ensemble size, N is batch size
            preds_stack = torch.stack(model_preds)
            
            # CANONICAL AGGREGATION: mean in quantile space
            # q̂_τ^ens(x) = (1/B) Σ_b q̂_τ^(b)(x)
            q_ens = preds_stack.mean(dim=0)  # [N]
            
            # CANONICAL MARGIN: m(x) = q̂_τ^ens(x) - s(x)
            # Positive margin => score lower than predicted => more likely MEMBER
            margin = q_ens - mean_errors

            for info, margin_val, t_err in zip(
                sample_infos,
                margin.cpu().tolist(),
                mean_errors.cpu().tolist(),
            ):
                records.append(
                    {
                        "image_id": int(info["index"]),
                        "label": int(label),
                        "score": float(margin_val),  # This IS the canonical margin m(x)
                        "t_error": float(t_err),
                    }
                )

    _process(members_loader, label=1)
    _process(nonmembers_loader, label=0)

    labels = np.array([r["label"] for r in records], dtype=np.int32)
    scores = np.array([r["score"] for r in records], dtype=np.float32)

    roc = compute_roc(labels, scores)
    target_fprs = [0.01, 0.001]
    interpolated = {f: float(interpolate_tpr(roc.fprs, roc.tprs, f)) for f in target_fprs}

    negatives = scores[labels == 0]
    positives = scores[labels == 1]

    def threshold_at_fpr(target: float) -> tuple[float, float, float]:
        if target <= 0:
            return float("inf"), 0.0, 0.0
        sorted_scores = np.sort(negatives)[::-1]
        k = max(1, int(math.ceil(target * len(sorted_scores))))
        threshold = sorted_scores[k - 1]
        actual_fpr = float((negatives > threshold).mean())
        tpr = float((positives > threshold).mean())
        return threshold, actual_fpr, tpr

    calibrated = {}
    for fpr in target_fprs:
        thr, act_fpr, tpr = threshold_at_fpr(fpr)
        calibrated[fpr] = {
            "threshold": float(thr),
            "fpr": float(act_fpr),
            "tpr": float(tpr),
        }

    boot = bootstrap_metrics(labels, scores, target_fprs=target_fprs, n_bootstrap=config.bootstrap, seed=config.seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_scores.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(records, f)
    with (out_dir / "raw_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label", "score", "t_error"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    report = {
        "dataset": dataset_name,
        "alpha": config.alpha,
        "mode": config.mode,
        "K": config.K,
        "M": len(ensemble),
        "metrics": {
            "auc": float(roc.auc),
            "tpr_at": interpolated,
            "calibrated": calibrated,
            "bootstrap": boot,
        },
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_row = {
        "dataset": dataset_name,
        "alpha": config.alpha,
        "mode": config.mode,
        "M": len(ensemble),
        "K": config.K,
        "AUC": roc.auc,
        "TPR@1%": interpolated[0.01],
        "TPR@0.1%": interpolated[0.001],
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    md = (
        "| dataset | alpha | mode | M | K | AUC | TPR@1% | TPR@0.1% |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| {dataset_name} | {config.alpha} | {config.mode} | {len(ensemble)} | {config.K} | {roc.auc:.4f} | "
        f"{interpolated[0.01]:.4f} | {interpolated[0.001]:.4f} |\n"
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    return report


def evaluate_attack_scores(
    ensemble: list[ResNet18QR],
    alpha_list: Sequence[float],
    config: EvalConfig,
    member_scores_path: str | Path,
    nonmember_scores_path: str | Path,
    member_indices_path: str | Path,
    nonmember_indices_path: str | Path,
    data_root: str | Path,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
    train_cifar_member: bool = True,
    train_cifar_nonmember: bool = True,
    device: str | torch.device = "cuda",
    out_dir: str | Path = "runs/eval",
) -> Dict:
    """Evaluate QR-based attack using pre-computed scores files (q25 workflow).
    
    CANONICAL ENSEMBLE AGGREGATION:
    --------------------------------
    The attack uses a unified aggregation rule in quantile space:
    
    1. Each model b predicts non-member τ-quantile: q̂_τ^(b)(x)
    2. Ensemble prediction (mean aggregation):
       q̂_τ^ens(x) = (1/B) Σ_b q̂_τ^(b)(x)
    3. Ensemble margin:
       m(x) = q̂_τ^ens(x) - s(x)
       where s(x) is the observed q25 t-error score
    4. m(x) is the ONLY attack score used for all metrics
    
    Interpretation:
    - Positive margin: actual score LOWER than predicted non-member quantile
      => sample is more likely a MEMBER (lower reconstruction error)
    - Negative margin: actual score at or above predicted level
      => sample is more likely a NON-MEMBER
    
    Attack Flow:
    1. Load pre-computed q25 scores and stats from scores files
    2. For each sample, each model predicts τ-quantile: q̂_τ^(b)(x)
    3. Aggregate in quantile space: q̂_τ^ens(x) = mean(q̂_τ^(b))
    4. Compute margin: m(x) = q̂_τ^ens(x) - s(x)
    5. Use margin m(x) as the attack score for ROC-AUC, TPR@FPR, etc.
    
    NOTE: Previous "per-model decision + majority vote" has been DEPRECATED.
    All metrics are computed using the margin m(x) as the attack score.
    
    Args:
        ensemble: List of trained QR models
        alpha_list: List of quantile values the models predict
        config: Evaluation configuration
        member_scores_path: Path to member scores file (e.g., scores/q25_eval_in.pt)
        nonmember_scores_path: Path to non-member scores file (e.g., scores/q25_eval_out.pt)
        member_indices_path: Path to member indices JSON (e.g., data/splits/eval_in.json)
        nonmember_indices_path: Path to non-member indices JSON (e.g., data/splits/eval_out.json)
        data_root: Root directory for CIFAR-10 data
        mean: Normalization mean for images
        std: Normalization std for images
        train_cifar_member: Whether to use CIFAR-10 train split for members
        train_cifar_nonmember: Whether to use CIFAR-10 train split for non-members
        device: Device for inference
        out_dir: Output directory for reports
    
    Returns:
        Dictionary containing evaluation metrics and report
    """
    device = torch.device(device)
    seed_everything(config.seed)
    
    for model in ensemble:
        model.to(device).eval()
    
    # Find the index for the target alpha
    try:
        alpha_idx = alpha_list.index(config.alpha)
    except ValueError as exc:
        raise ValueError(f"Alpha {config.alpha} not in ensemble outputs {alpha_list}") from exc
    
    # Load indices from JSON files
    with Path(member_indices_path).open("r", encoding="utf-8") as f:
        member_indices = json.load(f)
    with Path(nonmember_indices_path).open("r", encoding="utf-8") as f:
        nonmember_indices = json.load(f)
    
    # Create datasets from scores files
    member_dataset = EvalScoresDataset(
        data_root=data_root,
        indices=member_indices,
        scores_path=member_scores_path,
        mean=mean,
        std=std,
        train=train_cifar_member,
    )
    
    nonmember_dataset = EvalScoresDataset(
        data_root=data_root,
        indices=nonmember_indices,
        scores_path=nonmember_scores_path,
        mean=mean,
        std=std,
        train=train_cifar_nonmember,
    )
    
    # Create data loaders
    member_loader = DataLoader(
        member_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    nonmember_loader = DataLoader(
        nonmember_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    records = []
    
    def _process(loader, label: int, indices_list: List[int]):
        """Process a single split (members or non-members).
        
        CANONICAL AGGREGATION:
        1. Each model predicts τ-quantile: q̂_τ^(b)(x)
        2. Ensemble prediction: q̂_τ^ens(x) = (1/B) Σ_b q̂_τ^(b)(x)
        3. Margin: m(x) = q̂_τ^ens(x) - s(x)
        
        The margin m(x) is stored as 'score' and used for all metrics.
        """
        sample_idx = 0
        for images, stats, score_raw, score_log in tqdm(loader, desc=f"Eval label={label}"):
            images = images.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            
            # s(x): observed score (q25 t-error)
            # Use log-transformed score if model was trained with log1p
            if config.use_log1p:
                s_x = score_log.to(device, non_blocking=True)
            else:
                s_x = score_raw.to(device, non_blocking=True)
            
            # Collect predictions from all ensemble models: q̂_τ^(b)(x)
            model_preds = []
            for model in ensemble:
                with torch.no_grad():
                    preds = model(images, stats)  # [B, num_quantiles]
                    model_preds.append(preds[:, alpha_idx])  # [B]
            
            # Stack predictions: [M, B] where M is ensemble size
            preds_stack = torch.stack(model_preds)
            
            # CANONICAL AGGREGATION: mean in quantile space
            # q̂_τ^ens(x) = (1/B) Σ_b q̂_τ^(b)(x)
            q_ens = preds_stack.mean(dim=0)  # [B]
            
            # CANONICAL MARGIN: m(x) = q̂_τ^ens(x) - s(x)
            # Positive margin => score lower than predicted => more likely MEMBER
            margin = q_ens - s_x  # [B]
            
            # Record results
            batch_size = images.size(0)
            for i in range(batch_size):
                global_idx = sample_idx + i
                records.append({
                    "image_id": int(indices_list[global_idx]),
                    "label": int(label),
                    "score": float(margin[i].cpu().item()),  # This IS the canonical margin m(x)
                    "q25_score": float(score_raw[i].cpu().item()),
                })
            sample_idx += batch_size
    
    # Process members (label=1) and non-members (label=0)
    _process(member_loader, label=1, indices_list=member_indices)
    _process(nonmember_loader, label=0, indices_list=nonmember_indices)
    
    # Compute metrics
    labels = np.array([r["label"] for r in records], dtype=np.int32)
    scores = np.array([r["score"] for r in records], dtype=np.float32)
    
    roc = compute_roc(labels, scores)
    target_fprs = [0.01, 0.001]
    interpolated = {f: float(interpolate_tpr(roc.fprs, roc.tprs, f)) for f in target_fprs}
    
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    
    def threshold_at_fpr(target: float) -> tuple[float, float, float]:
        if target <= 0:
            return float("inf"), 0.0, 0.0
        sorted_scores = np.sort(negatives)[::-1]
        k = max(1, int(math.ceil(target * len(sorted_scores))))
        threshold = sorted_scores[k - 1]
        actual_fpr = float((negatives > threshold).mean())
        tpr = float((positives > threshold).mean())
        return threshold, actual_fpr, tpr
    
    calibrated = {}
    for fpr in target_fprs:
        thr, act_fpr, tpr = threshold_at_fpr(fpr)
        calibrated[fpr] = {
            "threshold": float(thr),
            "fpr": float(act_fpr),
            "tpr": float(tpr),
        }
    
    boot = bootstrap_metrics(
        labels, scores,
        target_fprs=target_fprs,
        n_bootstrap=config.bootstrap,
        seed=config.seed,
    )
    
    # Save results
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = out_dir / "raw_scores.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(records, f)
    
    with (out_dir / "raw_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "label", "score", "q25_score"]
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    
    report = {
        "alpha": config.alpha,
        "M": len(ensemble),
        "use_log1p": config.use_log1p,
        "num_members": int((labels == 1).sum()),
        "num_nonmembers": int((labels == 0).sum()),
        "metrics": {
            "auc": float(roc.auc),
            "tpr_at": interpolated,
            "calibrated": calibrated,
            "bootstrap": boot,
        },
    }
    
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    summary_row = {
        "alpha": config.alpha,
        "M": len(ensemble),
        "AUC": roc.auc,
        "TPR@1%": interpolated[0.01],
        "TPR@0.1%": interpolated[0.001],
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    
    md = (
        "| alpha | M | AUC | TPR@1% | TPR@0.1% |\n"
        "|---|---|---|---|---|\n"
        f"| {config.alpha} | {len(ensemble)} | {roc.auc:.4f} | "
        f"{interpolated[0.01]:.4f} | {interpolated[0.001]:.4f} |\n"
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    
    return report


def compute_gaussian_ensemble_margin_for_split(
    models: List[torch.nn.Module],
    dataloader: DataLoader,
    alpha: float,
    tau: float,
    device: torch.device,
    logger,
    sanity_state: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-space ensemble margins for one split using Gaussian heads.
    
    Gaussian head semantics:
    - alpha is the target FPR over non-member scores
    - tau = 1 - alpha is the upper-tail quantile level for Normal.icdf
    - Margin is q_log_ens - y_log; larger => more member-like
    """
    all_margins: List[torch.Tensor] = []
    all_score_logs: List[torch.Tensor] = []
    if sanity_state is None:
        sanity_state = {}
    sanity_logged = bool(sanity_state.get("logged", False))  # emit sanity log only once per overall eval run

    for images, stats, score_raw, score_log in tqdm(dataloader, desc=f"Gaussian eval tau={tau}"):
        images = images.to(device, non_blocking=True)
        stats = stats.to(device, non_blocking=True)
        y_log = score_log.to(device, non_blocking=True)

        per_model_quantiles: List[torch.Tensor] = []
        first_mu: Optional[torch.Tensor] = None
        first_log_sigma: Optional[torch.Tensor] = None
        for model in models:
            model.eval()
            with torch.no_grad():
                mu, log_sigma = model(images, stats)
                if not sanity_logged and first_mu is None:
                    first_mu = mu
                    first_log_sigma = log_sigma
                q_log = gaussian_quantile_from_params(mu, log_sigma, tau)
                per_model_quantiles.append(q_log)

        if not sanity_logged and first_mu is not None and first_log_sigma is not None:
            z = _normal.icdf(torch.tensor([tau], device=first_mu.device, dtype=first_mu.dtype))[0]
            logger.info(
                "Gaussian head sanity: alpha=%.1e, tau=%.6f, z=%.4f, mean(mu)=%.4f, mean(log_sigma)=%.4f",
                alpha,
                tau,
                float(z.item()),
                float(first_mu.mean().item()),
                float(first_log_sigma.mean().item()),
            )
            sanity_logged = True
            sanity_state["logged"] = True

        q_log_stack = torch.stack(per_model_quantiles, dim=1)  # [B, num_models]
        q_log_ens = q_log_stack.mean(dim=1)  # [B]

        margins = q_log_ens - y_log  # log-space margin
        all_margins.append(margins.cpu())
        all_score_logs.append(y_log.cpu())

    margins_all = torch.cat(all_margins, dim=0)
    scores_log_all = torch.cat(all_score_logs, dim=0)

    logger.info(
        "[Gaussian ensemble] alpha=%.1e tau=%.6f mean_margin=%.6f std_margin=%.6f "
        "mean_score_log=%.6f std_score_log=%.6f",
        alpha,
        tau,
        float(margins_all.mean().item()),
        float(margins_all.std(unbiased=False).item()),
        float(scores_log_all.mean().item()),
        float(scores_log_all.std(unbiased=False).item()),
    )

    return margins_all, scores_log_all


def evaluate_attack_scores_gaussian(
    ensemble: list[torch.nn.Module],
    config: EvalConfig,
    member_scores_path: str | Path,
    nonmember_scores_path: str | Path,
    member_indices_path: str | Path,
    nonmember_indices_path: str | Path,
    data_root: str | Path,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
    train_cifar_member: bool = True,
    train_cifar_nonmember: bool = True,
    device: str | torch.device = "cuda",
    out_dir: str | Path = "runs/eval",
    logger=None,
) -> Dict:
    """Evaluate Gaussian-head ensemble using log-space margins.
    
    Gaussian head semantics:
    - alpha in config is the target FPR (right-tail over non-member scores)
    - tau = 1 - alpha is the upper-tail quantile level passed to Normal.icdf
    - A one-time sanity log (alpha, tau, z, mean mu/log_sigma) is emitted on first batch
    """
    # Gaussian QR evaluation semantics:
    #   - alpha (config) = target FPR for non-member scores
    #   - tau = 1 - alpha is the upper-tail quantile level
    #   - margin = q_tau^{log,ens}(x) - log1p(score(x)); larger => more member-like
    device = torch.device(device)
    seed_everything(config.seed)
    for model in ensemble:
        model.to(device).eval()

    if logger is None:
        from mia_logging import get_winston_logger
        logger = get_winston_logger(__name__)

    with Path(member_indices_path).open("r", encoding="utf-8") as f:
        member_indices = json.load(f)
    with Path(nonmember_indices_path).open("r", encoding="utf-8") as f:
        nonmember_indices = json.load(f)

    member_dataset = EvalScoresDataset(
        data_root=data_root,
        indices=member_indices,
        scores_path=member_scores_path,
        mean=mean,
        std=std,
        train=train_cifar_member,
    )
    nonmember_dataset = EvalScoresDataset(
        data_root=data_root,
        indices=nonmember_indices,
        scores_path=nonmember_scores_path,
        mean=mean,
        std=std,
        train=train_cifar_nonmember,
    )

    member_loader = DataLoader(
        member_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    nonmember_loader = DataLoader(
        nonmember_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # alpha is target FPR; tau is derived upper-tail quantile level for Normal.icdf
    tau = 1.0 - float(config.alpha)

    sanity_state = {"logged": False}

    margins_in, _ = compute_gaussian_ensemble_margin_for_split(
        models=ensemble,
        dataloader=member_loader,
        alpha=float(config.alpha),
        tau=tau,
        device=device,
        logger=logger,
        sanity_state=sanity_state,
    )
    margins_out, _ = compute_gaussian_ensemble_margin_for_split(
        models=ensemble,
        dataloader=nonmember_loader,
        alpha=float(config.alpha),
        tau=tau,
        device=device,
        logger=logger,
        sanity_state=sanity_state,
    )

    target_fprs = [0.01, 0.001]
    # Metrics are computed exclusively from ensemble margins (larger = member-like)
    auc = roc_auc(margins_in, margins_out)
    tpr_results = {
        fpr: tpr_precision_at_fpr(
            margins_in,
            margins_out,
            target_fpr=fpr,
            num_bootstrap=config.bootstrap,
            seed=config.seed,
        )
        for fpr in target_fprs
    }

    logger.info(
        "[Gaussian QR] Metrics based on margins: AUC=%.6f, target_fprs=%s, tpr_at=%s",
        float(auc),
        target_fprs,
        [float(tpr_results[f]["tpr"]) for f in target_fprs],
    )

    calibrated = {
        fpr: {
            "threshold": float(res["threshold"]),
            "fpr": float(res["achieved_fpr"]),
            "tpr": float(res["tpr"]),
            "precision": float(res["precision"]),
            "fpr_error": float(res["fpr_error"]),
            "counts": res["counts"],
            "bootstrap": res.get("bootstrap"),
        }
        for fpr, res in tpr_results.items()
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_scores.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {"label": 1, "score": float(s)} for s in margins_in
            ] + [
                {"label": 0, "score": float(s)} for s in margins_out
            ],
            f,
        )

    report = {
        "alpha": config.alpha,
        "M": len(ensemble),
        "use_log1p": True,
        "num_members": int(margins_in.numel()),
        "num_nonmembers": int(margins_out.numel()),
        "metrics": {
            "auc": float(auc),
            "tpr_at": {f: float(tpr_results[f]["tpr"]) for f in target_fprs},
            "precision_at": {f: float(tpr_results[f]["precision"]) for f in target_fprs},
            "calibrated": calibrated,
        },
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_row = {
        "alpha": config.alpha,
        "M": len(ensemble),
        "AUC": auc,
        "TPR@1%": tpr_results[0.01]["tpr"],
        "TPR@0.1%": tpr_results[0.001]["tpr"],
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    md = (
        "| alpha | M | AUC | TPR@1% | TPR@0.1% |\n"
        "|---|---|---|---|---|\n"
        f"| {config.alpha} | {len(ensemble)} | {auc:.4f} | "
        f"{tpr_results[0.01]['tpr']:.4f} | {tpr_results[0.001]['tpr']:.4f} |\n"
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    return report
