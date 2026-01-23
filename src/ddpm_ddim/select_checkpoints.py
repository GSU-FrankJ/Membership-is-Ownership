"""Quick checkpoint selection for DDIM training runs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # Go up to project root (from src/ddpm_ddim/select_checkpoints.py)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from mia_logging import get_winston_logger
from src.attacks.scores import t_error_aggregate, uniform_timesteps
from src.attacks.scores.compute_scores import SplitDataset, load_indices
from src.attacks.eval.metrics import roc_auc
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule


LOGGER = get_winston_logger(__name__)


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_dataloader(
    data_cfg: Dict,
    split: str,
    indices: List[int],
    batch_size: int,
    limit: int | None,
) -> DataLoader:
    train_flag = split != "aux"
    dataset_name = data_cfg["dataset"].get("name", "cifar10")
    dataset = SplitDataset(
        pathlib.Path(data_cfg["dataset"]["root"]),
        indices[: limit] if limit is not None else indices,
        train=train_flag,
        mean=tuple(data_cfg["dataset"]["normalization"]["mean"]),
        std=tuple(data_cfg["dataset"]["normalization"]["std"]),
        dataset_name=dataset_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["dataset"].get("num_workers", 8),
        pin_memory=True,
    )


def _compute_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    timesteps: List[int],
    alphas_bar: torch.Tensor,
    device: torch.device,
    agg: str = "q25",
) -> torch.Tensor:
    """Compute t-error scores with specified aggregation method.
    
    Args:
        model: Denoising model
        loader: DataLoader for the split
        timesteps: Timesteps to sample
        alphas_bar: Diffusion schedule
        device: Compute device
        agg: Aggregation method (q25 default for best separability)
    
    Returns:
        Tensor of aggregated t-error scores
    """
    scores = []
    for images, _ in loader:
        images = images.to(device)
        batch_scores = t_error_aggregate(images, timesteps, model, alphas_bar, agg=agg)
        scores.append(batch_scores.cpu())
    return torch.cat(scores) if scores else torch.empty(0)


def evaluate_checkpoint(
    ckpt_path: pathlib.Path,
    model_cfg: Dict,
    data_cfg: Dict,
    timesteps: List[int],
    sample_limit: int,
    device: torch.device,
    agg: str = "q25",
) -> Dict:
    """Evaluate checkpoint using t-error scores.
    
    Args:
        ckpt_path: Path to checkpoint file
        model_cfg: Model configuration dictionary
        data_cfg: Data configuration dictionary
        timesteps: List of timesteps to evaluate
        sample_limit: Maximum number of samples to evaluate
        device: Compute device
        agg: Aggregation method (q25 default for best separability)
    
    Returns:
        Dictionary with evaluation metrics
    """
    state = torch.load(ckpt_path, map_location=device)
    model = build_unet(model_cfg["model"])
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()

    LOGGER.info("Evaluating checkpoint %s (agg=%s)", ckpt_path, agg)

    aux_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["aux"]))
    eval_out_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["eval_out"]))

    batch_size = model_cfg["training"].get("selection_batch_size", 256)
    # load_indices returns a list directly, no need for .tolist()
    aux_list = aux_indices if isinstance(aux_indices, list) else aux_indices.tolist()
    out_list = eval_out_indices if isinstance(eval_out_indices, list) else eval_out_indices.tolist()
    loader_aux = _build_dataloader(data_cfg, "aux", aux_list, batch_size, sample_limit)
    loader_out = _build_dataloader(data_cfg, "eval_out", out_list, batch_size, sample_limit)

    _betas, alphas_bar = build_cosine_schedule(model_cfg["diffusion"]["timesteps"])
    alphas_bar = alphas_bar.to(device)

    scores_aux = _compute_scores(model, loader_aux, timesteps, alphas_bar, device, agg=agg)
    scores_out = _compute_scores(model, loader_out, timesteps, alphas_bar, device, agg=agg)

    metric = {
        "roc_auc": roc_auc(scores_aux, scores_out),
        "mean_aux": scores_aux.mean().item() if scores_aux.numel() else 0.0,
        "mean_out": scores_out.mean().item() if scores_out.numel() else 0.0,
        "delta_mean": (scores_aux.mean() - scores_out.mean()).item()
        if scores_aux.numel() and scores_out.numel()
        else 0.0,
        "num_aux": int(scores_aux.numel()),
        "num_out": int(scores_out.numel()),
    }
    return metric


def run_selection(
    run_dir: pathlib.Path,
    model_config: pathlib.Path,
    data_config: pathlib.Path,
    device: torch.device | None = None,
    top_k: int = 3,
    timesteps: int = 10,
    sample_limit: int = 2048,
    agg: str = "q25",
) -> pathlib.Path:
    """Evaluate EMA checkpoints under `run_dir` and persist selection results.
    
    Args:
        run_dir: Directory containing checkpoint subdirectories
        model_config: Path to model configuration YAML
        data_config: Path to data configuration YAML
        device: Compute device (defaults to cuda if available)
        top_k: Number of top checkpoints to select
        timesteps: Number of timesteps to evaluate
        sample_limit: Maximum number of samples to evaluate per checkpoint
        agg: Aggregation method (q25 default for best separability)
    
    Returns:
        Path to the selection results JSON file
    """
    model_cfg = load_yaml(model_config)
    data_cfg = load_yaml(data_config)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dirs = sorted(run_dir.glob("ckpt_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    schedule = uniform_timesteps(model_cfg["diffusion"]["timesteps"], timesteps)
    results = []

    for ckpt_dir in ckpt_dirs:
        step = int(ckpt_dir.name.split("_")[-1])
        ckpt_path = ckpt_dir / "ema.ckpt"
        if not ckpt_path.exists():
            LOGGER.warning("Skipping %s (missing ema.ckpt)", ckpt_dir)
            continue
        metrics = evaluate_checkpoint(
            ckpt_path,
            model_cfg,
            data_cfg,
            schedule,
            sample_limit,
            device,
            agg=agg,
        )
        metrics["step"] = step
        metrics["ckpt_path"] = str(ckpt_path)
        results.append(metrics)

    if not results:
        raise RuntimeError("No checkpoints evaluated; aborting selection.")

    results.sort(
        key=lambda m: (-m["roc_auc"], -m["delta_mean"], -m["mean_aux"]),
    )
    selected = results[:top_k]

    payload = {
        "parameters": {
            "timesteps": timesteps,
            "sample_limit": sample_limit,
            "top_k": top_k,
        },
        "evaluated": results,
        "selected": selected,
    }

    out_path = run_dir / "model_selection.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # Create best_for_mia.ckpt symlink pointing to the best checkpoint
    if selected:
        best_ckpt = pathlib.Path(selected[0]["ckpt_path"])
        best_link = run_dir / "best_for_mia.ckpt"
        # Remove existing symlink if present
        if best_link.is_symlink() or best_link.exists():
            best_link.unlink()
        # Create relative symlink to the best EMA checkpoint
        relative_path = best_ckpt.relative_to(run_dir)
        best_link.symlink_to(relative_path)
        LOGGER.info("Created best_for_mia.ckpt -> %s", relative_path)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select top EMA checkpoints via quick separability scoring.")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-config", type=pathlib.Path, required=True)
    parser.add_argument("--data-config", type=pathlib.Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--sample-limit", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None, help="torch device override (e.g., cuda:0)")
    parser.add_argument("--aggregate", type=str, default="q25", help="Aggregation method (q25 default for best separability, or mean, q10, q20, median)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else None
    out_path = run_selection(
        run_dir=args.run_dir,
        model_config=args.model_config,
        data_config=args.data_config,
        device=device,
        top_k=args.top_k,
        timesteps=args.timesteps,
        sample_limit=args.sample_limit,
        agg=args.aggregate,
    )
    LOGGER.info("Selection results written to %s", out_path)


if __name__ == "__main__":
    main()
