#!/usr/bin/env python3
"""CLI entry point for training QR bagging ensemble using attack_qr/engine (ResNet18QR + image + stats).

This CLI supports two training workflows:
1. Scores-based workflow (--use-scores): Uses pre-computed q25 scores and stats from scores files
2. Pairs-based workflow (default): Uses NPZ pairs files with per-timestep t-error data

For the q25 workflow, use --use-scores to train with:
- Images from CIFAR-10 (aux split)
- q25 scores as training targets
- Stats (mean_error, std_error, l2_error) as model input features
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attack_qr.engine.build_pairs import build_t_error_pairs
from src.attack_qr.engine.train_qr_bagging import (
    QuantileScoresDataset,
    QuantileTrainingConfig,
    train_bagging_ensemble,
    train_bagging_ensemble_gaussian_scores,
    train_bagging_ensemble_scores,
)
from src.ddpm.data.loader import IndexedDataset, get_dataset, get_transforms
from src.attacks.scores.compute_scores import load_indices
from src.ddpm.engine.checkpoint_utils import load_ddpm_model
from mia_logging import get_winston_logger
from torch.utils.data import DataLoader

LOGGER = get_winston_logger(__name__)


def main() -> None:
    """Main CLI entry point for training QR bagging ensemble.
    
    This is the unified training pipeline using:
    - attack_qr/engine/train_qr_bagging.py
    - ResNet18QR model (image + stats inputs)
    - Config-driven parameters
    
    Supports two workflows:
    1. --use-scores: Train using pre-computed q25 scores and stats from scores files
    2. Default: Train using NPZ pairs files with per-timestep t-error data
    """
    parser = argparse.ArgumentParser(
        description="Train quantile regression bagging ensemble (attack_qr/engine + ResNet18QR)"
    )
    parser.add_argument("--config", type=Path, required=True, help="Attack YAML config (configs/attack_qr.yaml)")
    parser.add_argument("--data-config", type=Path, default=None, help="Data config (default: configs/data_cifar10.yaml)")
    parser.add_argument("--pairs", type=Path, default=None, help="NPZ file with t-error pairs (auto-build if not provided)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for models (default: runs/attack_qr/ensembles)")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training for existing checkpoints")
    parser.add_argument("--force-rebuild-pairs", action="store_true", help="Force rebuild pairs NPZ even if exists")
    # New arguments for scores-based workflow
    parser.add_argument("--use-scores", action="store_true", 
                        help="Use scores-based workflow (load q25 scores and stats from scores files)")
    parser.add_argument("--scores-path", type=Path, default=None,
                        help="Path to scores file (e.g., scores/q25_aux.pt). Auto-detected if not provided.")
    parser.add_argument("--scores-tag", type=str, default=None,
                        help="Tag for scores files. If not provided, uses t_error.aggregate from config (typically q25).")
    args = parser.parse_args()

    # Load configs
    attack_cfg = yaml.safe_load(args.config.read_text())
    if args.data_config is None:
        data_config_path = PROJECT_ROOT / "configs" / "data_cifar10.yaml"
    else:
        data_config_path = args.data_config
    data_cfg = yaml.safe_load(data_config_path.read_text())

    # Resolve scores-tag from config if not provided
    if args.scores_tag is None:
        args.scores_tag = attack_cfg.get("t_error", {}).get("aggregate", "q25")
        LOGGER.info("Using scores-tag from config t_error.aggregate: %s", args.scores_tag)

    # Extract parameters from config
    dataset_name = attack_cfg.get("dataset", attack_cfg.get("dataset_name", "cifar10"))
    img_size = attack_cfg.get("img_size", 32)
    seed = attack_cfg.get("seed", 20251030)
    
    bag_cfg = attack_cfg.get("bagging", {})
    train_cfg = attack_cfg.get("train", {})
    qr_cfg = attack_cfg.get("qr", {})
    qr_mode = qr_cfg.get("mode", "quantile")
    
    # Build QuantileTrainingConfig
    # Note: attack_qr/engine uses alpha_list (quantiles), not tau_values
    # Map tau_values to alpha_list for compatibility
    tau_values = bag_cfg.get("tau_values", [0.001, 0.0001])
    alpha_list = tuple(tau_values)  # Use tau_values as alpha_list
    
    qt_config = QuantileTrainingConfig(
        lr=train_cfg.get("lr", 1e-3),
        epochs=train_cfg.get("epochs", 50),
        batch_size=train_cfg.get("batch_size", 256),
        alpha_list=alpha_list,
        bootstrap=bag_cfg.get("bootstrap", True),
        M=bag_cfg.get("B", bag_cfg.get("M", 50)),  # Use B from config, fallback to M
        B=bag_cfg.get("B", bag_cfg.get("M", 50)),
        seed=seed,
        use_log1p=train_cfg.get("log1p", True),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        cosine_anneal=train_cfg.get("cosine_anneal", True),
        val_ratio=train_cfg.get("val_ratio", 0.1),
        bootstrap_ratio=bag_cfg.get("bootstrap_ratio", 0.8),
        num_workers=train_cfg.get("num_workers", 0),
        early_stop_patience=train_cfg.get("early_stop_patience", 10),
        device=args.device,
    )

    # Determine output directory
    if args.out is None:
        output_root = Path(attack_cfg.get("logging", {}).get("output_dir", "runs/attack_qr"))
        out_dir = output_root / "ensembles"
    else:
        out_dir = args.out
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get normalization parameters from data config
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])

    # ========== SCORES-BASED WORKFLOW (q25) ==========
    if args.use_scores:
        LOGGER.info("Using scores-based workflow (q25)")
        
        # Determine scores path
        cache_dir = Path(attack_cfg["t_error"]["cache_dir"])
        if args.scores_path is not None:
            scores_path = args.scores_path
        else:
            scores_path = cache_dir / f"{args.scores_tag}_aux.pt"
        
        if not scores_path.exists():
            raise FileNotFoundError(
                f"Scores file not found: {scores_path}\n"
                "Please run 'python tools/compute_scores.py --tag q25' first."
            )
        
        # Get aux indices path
        aux_indices_path = Path(data_cfg["splits"]["paths"]["aux"])
        if not aux_indices_path.exists():
            raise FileNotFoundError(f"Aux indices file not found: {aux_indices_path}")
        
        LOGGER.info("Training with scores: %s", scores_path)
        LOGGER.info("Using aux indices: %s", aux_indices_path)
        LOGGER.info("Config: M=%d, alpha_list=%s, use_log1p=%s", 
                   qt_config.M, qt_config.alpha_list, qt_config.use_log1p)
        manifest_path = out_dir / "manifest.json"

        if qr_mode == "gaussian":
            aux_indices = load_indices(aux_indices_path)
            # Defensive guard: Gaussian training must use aux/public (non-member) only
            scores_stem = scores_path.stem
            if "eval" in scores_stem:
                raise ValueError(f"Gaussian training must use aux scores only, but got scores file: {scores_path}")
            if "aux" not in scores_stem:
                LOGGER.warning(
                    "Gaussian training expected aux scores; scores file name lacks 'aux': %s", scores_path
                )
            if "eval" in str(aux_indices_path):
                raise ValueError(f"Gaussian training must not load eval indices: {aux_indices_path}")

            dataset = QuantileScoresDataset(
                data_root=args.data_root,
                indices=aux_indices,
                scores_path=scores_path,
                mean=mean,
                std=std,
                train=False,
            )

            result = train_bagging_ensemble_gaussian_scores(
                dataset=dataset,
                cfg=qt_config,
                logger=LOGGER,
            )

            manifest = result["manifest"]
            manifest["mode"] = "gaussian"
            manifest["alpha_list"] = list(alpha_list)
            manifest["tau_values"] = list(alpha_list)
            manifest["use_log1p"] = True
            manifest["target_space"] = qr_cfg.get("target_space", "log1p")
            manifest["models"] = []

            for b, model_entry in enumerate(result["models"]):
                ckpt_path = out_dir / f"model_{b:03d}.pt"
                torch.save(
                    {
                        "model": model_entry["state_dict"],
                        "stats_dim": manifest["stats_dim"],
                        "seed": model_entry.get("seed"),
                        "loss": model_entry.get("val_loss"),
                        "bootstrap_indices": model_entry.get("bootstrap_indices"),
                    },
                    ckpt_path,
                )
                manifest["models"].append(
                    {
                        "path": ckpt_path.name,
                        "seed": model_entry.get("seed"),
                        "loss": model_entry.get("val_loss"),
                        "bootstrap_indices": model_entry.get("bootstrap_indices"),
                    }
                )

            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            LOGGER.info("Training complete (gaussian scores workflow). Models saved to: %s", out_dir)
            return

        # Train using quantile scores-based workflow
        train_bagging_ensemble_scores(
            scores_path=scores_path,
            indices_path=aux_indices_path,
            data_root=args.data_root,
            config=qt_config,
            out_dir=out_dir,
            mean=mean,
            std=std,
            train_cifar=False,  # aux uses test split
            device=args.device,
            skip_existing=args.skip_existing,
        )
        LOGGER.info("Training complete (scores workflow). Models saved to: %s", out_dir)
        return

    # ========== PAIRS-BASED WORKFLOW (legacy) ==========
    LOGGER.info("Using pairs-based workflow (legacy)")
    
    # Determine pairs path
    if args.pairs is None:
        pairs_dir = out_dir.parent / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        pairs_path = pairs_dir / f"pairs_{dataset_name}_seed{seed}.npz"
    else:
        pairs_path = Path(args.pairs)

    # Build pairs if needed
    if not pairs_path.exists() or args.force_rebuild_pairs:
        LOGGER.info("Building t-error pairs: %s", pairs_path)
        
        # Load DDPM model
        model_cfg_path = Path(attack_cfg["model"]["config"])
        checkpoint_root = Path(attack_cfg["model"]["checkpoint_root"])
        prefer_ema = attack_cfg["model"].get("prefer_ema", True)
        
        # Find latest checkpoint
        ckpt_dirs = sorted(checkpoint_root.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")
        latest_ckpt_dir = ckpt_dirs[-1]
        ckpt_path = latest_ckpt_dir / ("ema.ckpt" if prefer_ema else "model.ckpt")
        
        device = args.device if isinstance(args.device, str) else str(args.device)
        ddpm_model, schedule, _ = load_ddpm_model(ckpt_path, device=device)
        
        # Load public indices
        split_json_path = Path(data_cfg["splits"]["paths"].get("aux", data_cfg["splits"]["paths"].get("public")))
        if split_json_path.exists():
            public_indices = load_indices(split_json_path)
        else:
            # Fallback: use all indices from pairs if available
            LOGGER.warning("Split JSON not found, will infer public indices from pairs")
            public_indices = None
        
        # Prepare dataset
        base_dataset = get_dataset(dataset_name, root=args.data_root, download=True)
        transform = get_transforms(img_size, augment=False)
        if public_indices is not None:
            indexed_dataset = IndexedDataset(base_dataset, indices=public_indices, transform=transform)
        else:
            # Use all training indices as fallback
            train_indices = list(range(len(base_dataset)))
            indexed_dataset = IndexedDataset(base_dataset, indices=train_indices, transform=transform)
        
        dataloader = DataLoader(indexed_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        
        # Build pairs
        public_cfg = attack_cfg.get("public", {})
        K = public_cfg.get("K", 12)
        mode = public_cfg.get("mode", "x0")
        
        build_t_error_pairs(
            model=ddpm_model,
            schedule=schedule,
            dataloader=dataloader,
            dataset_name=dataset_name,
            global_seed=seed,
            K=K,
            mode=mode,
            out_path=pairs_path,
            device=device,
        )
        LOGGER.info("Pairs built: %s", pairs_path)
    else:
        LOGGER.info("Using existing pairs: %s", pairs_path)

    # Load public indices for training
    split_json_path = Path(data_cfg["splits"]["paths"].get("aux", data_cfg["splits"]["paths"].get("public")))
    if split_json_path.exists():
        public_indices = load_indices(split_json_path)
    else:
        # Infer from pairs
        with np.load(pairs_path) as data:
            public_indices = np.unique(data["image_id"]).tolist()
        LOGGER.info("Inferred %d public indices from pairs", len(public_indices))

    # Train ensemble
    LOGGER.info("Training bagging ensemble: M=%d, alpha_list=%s", qt_config.M, qt_config.alpha_list)
    train_bagging_ensemble(
        npz_path=pairs_path,
        dataset_name=dataset_name,
        public_indices=public_indices,
        config=qt_config,
        out_dir=out_dir,
        img_size=img_size,
        data_root=args.data_root,
        device=args.device,
        skip_existing=args.skip_existing,
    )
    LOGGER.info("Training complete (pairs workflow). Models saved to: %s", out_dir)


if __name__ == "__main__":
    main()

