#!/usr/bin/env python3
"""CLI entry point for evaluating QR-MIA attack using attack_qr/engine (ResNet18QR + image + stats).

This CLI supports two evaluation workflows:
1. Scores-based workflow (--use-scores): Uses pre-computed q25 scores and stats from scores files
2. Legacy workflow (default): Computes t-errors on-the-fly during evaluation

For the q25 workflow, use --use-scores to evaluate with:
- Images from CIFAR-10 (eval_in/eval_out splits)
- q25 scores as attack decision basis
- Stats (mean_error, std_error, l2_error) as model input features
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attack_qr.engine.eval_attack import (
    EvalConfig,
    evaluate_attack,
    evaluate_attack_scores,
    evaluate_attack_scores_gaussian,
    load_quantile_ensemble,
)
from src.ddpm.data.loader import load_split_indices
from src.ddpm.engine.checkpoint_utils import load_ddpm_model
from src.attacks.scores.compute_scores import load_indices as load_single_indices
from mia_logging import get_winston_logger

LOGGER = get_winston_logger(__name__)


def main() -> None:
    """Main CLI entry point for evaluating QR-MIA attack.
    
    This is the unified evaluation pipeline using:
    - attack_qr/engine/eval_attack.py
    - ResNet18QR model (image + stats inputs)
    - Config-driven parameters
    
    Supports two workflows:
    1. --use-scores: Evaluate using pre-computed q25 scores and stats from scores files
    2. Default: Compute t-errors on-the-fly during evaluation
    """
    parser = argparse.ArgumentParser(
        description="Evaluate QR-MIA attack (attack_qr/engine + ResNet18QR)"
    )
    parser.add_argument("--config", type=Path, required=True, help="Attack YAML config (configs/attack_qr.yaml)")
    parser.add_argument("--data-config", type=Path, default=None, help="Data config (default: configs/data_cifar10.yaml)")
    parser.add_argument("--ensemble", type=Path, default=None, help="QR ensemble directory (default: runs/attack_qr/ensembles)")
    parser.add_argument("--ddpm-ckpt", type=Path, default=None, help="DDPM checkpoint (auto-detect from config if not provided)")
    parser.add_argument("--report-dir", type=Path, default=None, help="Report output directory (default: runs/attack_qr/reports/<timestamp>)")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha for evaluation")
    parser.add_argument("--mode", type=str, default=None, help="Override t-error mode (default: x0)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    # New arguments for scores-based workflow
    parser.add_argument("--use-scores", action="store_true",
                        help="Use scores-based workflow (load q25 scores and stats from scores files)")
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
    seed = args.seed if args.seed is not None else attack_cfg.get("seed", 20251030)
    eval_cfg = attack_cfg.get("eval", {})
    train_cfg = attack_cfg.get("train", {})
    qr_cfg = attack_cfg.get("qr", {})
    qr_mode_cfg = qr_cfg.get("mode", "quantile")
    
    # Build EvalConfig
    alpha = args.alpha if args.alpha is not None else eval_cfg.get("alpha", 0.01)
    mode = args.mode if args.mode is not None else eval_cfg.get("mode", "x0")
    use_log1p = train_cfg.get("log1p", True)
    
    eval_config = EvalConfig(
        alpha=alpha,
        mode=mode,
        K=eval_cfg.get("K", 4),
        batch_size=eval_cfg.get("batch_size", 128),
        bootstrap=eval_cfg.get("bootstrap", 200),
        seed=seed,
        use_log1p=use_log1p,
    )

    # Determine ensemble directory
    if args.ensemble is None:
        output_root = Path(attack_cfg.get("logging", {}).get("output_dir", "runs/attack_qr"))
        ensemble_dir = output_root / "ensembles"
    else:
        ensemble_dir = Path(args.ensemble)
    
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

    device = torch.device(args.device)

    # Load QR ensemble
    ensemble, alpha_list, manifest = load_quantile_ensemble(ensemble_dir, device=device)
    manifest_mode = manifest.get("mode", "quantile")
    effective_mode = manifest_mode or qr_mode_cfg or "quantile"
    LOGGER.info("Loaded QR ensemble: %d models, mode=%s, alpha_list=%s", len(ensemble), effective_mode, alpha_list)
    
    # Validate alpha for quantile mode; for gaussian any alpha is allowed
    if effective_mode != "gaussian":
        if eval_config.alpha not in alpha_list:
            if args.alpha is None:
                eval_config.alpha = alpha_list[0]
                LOGGER.warning(
                    "Alpha %.4f from config not in ensemble alpha_list %s. "
                    "Auto-selecting first alpha from ensemble: %.4f",
                    eval_cfg.get("alpha", 0.01), alpha_list, eval_config.alpha
                )
            else:
                raise ValueError(
                    f"Specified alpha {eval_config.alpha} not in ensemble alpha_list {alpha_list}. "
                    f"Please use one of: {alpha_list}"
                )
    else:
        if args.alpha is None:
            tau_values = manifest.get("tau_values") or alpha_list
            if tau_values:
                eval_config.alpha = tau_values[0]
                LOGGER.info("Gaussian mode: alpha not provided, defaulting to first tau: %.4f", eval_config.alpha)
    
    # Override use_log1p from manifest if available
    if "use_log1p" in manifest:
        eval_config.use_log1p = manifest["use_log1p"]
        LOGGER.info("Using use_log1p=%s from ensemble manifest", eval_config.use_log1p)

    # Get normalization parameters from data config
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])

    # Determine report directory
    if args.report_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(attack_cfg.get("logging", {}).get("output_dir", "runs/attack_qr"))
        report_dir = output_root / "reports" / timestamp
    else:
        report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ========== SCORES-BASED WORKFLOW (q25 / gaussian) ==========
    if args.use_scores:
        LOGGER.info("Using scores-based evaluation workflow (q25)")
        
        # Determine scores paths
        cache_dir = Path(attack_cfg["t_error"]["cache_dir"])
        member_scores_path = cache_dir / f"{args.scores_tag}_eval_in.pt"
        nonmember_scores_path = cache_dir / f"{args.scores_tag}_eval_out.pt"
        
        if not member_scores_path.exists():
            raise FileNotFoundError(
                f"Member scores file not found: {member_scores_path}\n"
                "Please run 'python tools/compute_scores.py --tag q25' first."
            )
        if not nonmember_scores_path.exists():
            raise FileNotFoundError(
                f"Non-member scores file not found: {nonmember_scores_path}\n"
                "Please run 'python tools/compute_scores.py --tag q25' first."
            )
        
        # Get indices paths
        member_indices_path = Path(data_cfg["splits"]["paths"]["eval_in"])
        nonmember_indices_path = Path(data_cfg["splits"]["paths"]["eval_out"])
        
        LOGGER.info("Evaluating with scores:")
        LOGGER.info("  Members: %s", member_scores_path)
        LOGGER.info("  Non-members: %s", nonmember_scores_path)
        LOGGER.info("  Config: alpha=%.4f, use_log1p=%s", eval_config.alpha, eval_config.use_log1p)
        
        # Run scores-based evaluation
        if effective_mode == "gaussian":
            report = evaluate_attack_scores_gaussian(
                ensemble=ensemble,
                config=eval_config,
                member_scores_path=member_scores_path,
                nonmember_scores_path=nonmember_scores_path,
                member_indices_path=member_indices_path,
                nonmember_indices_path=nonmember_indices_path,
                data_root=args.data_root,
                mean=mean,
                std=std,
                train_cifar_member=True,  # eval_in uses train split
                train_cifar_nonmember=True,  # eval_out uses train split
                device=device,
                out_dir=report_dir,
                logger=LOGGER,
            )
        else:
            report = evaluate_attack_scores(
                ensemble=ensemble,
                alpha_list=alpha_list,
                config=eval_config,
                member_scores_path=member_scores_path,
                nonmember_scores_path=nonmember_scores_path,
                member_indices_path=member_indices_path,
                nonmember_indices_path=nonmember_indices_path,
                data_root=args.data_root,
                mean=mean,
                std=std,
                train_cifar_member=True,  # eval_in uses train split
                train_cifar_nonmember=True,  # eval_out uses train split
                device=device,
                out_dir=report_dir,
            )
        
        LOGGER.info("Evaluation complete (scores workflow). Report saved to: %s", report_dir)
        LOGGER.info("AUC: %.4f, TPR@1%%: %.4f, TPR@0.1%%: %.4f", 
                    report["metrics"]["auc"], 
                    report["metrics"]["tpr_at"][0.01],
                    report["metrics"]["tpr_at"][0.001])
        return

    # ========== LEGACY WORKFLOW (compute t-errors on-the-fly) ==========
    if effective_mode == "gaussian":
        raise ValueError("Gaussian mode requires --use-scores evaluation workflow.")
    LOGGER.info("Using legacy evaluation workflow (compute t-errors on-the-fly)")
    
    # Load DDPM model (only needed for legacy workflow)
    if args.ddpm_ckpt is None:
        model_cfg_path = Path(attack_cfg["model"]["config"])
        checkpoint_root = Path(attack_cfg["model"]["checkpoint_root"])
        prefer_ema = attack_cfg["model"].get("prefer_ema", True)
        
        # Find latest checkpoint
        ckpt_dirs = sorted(checkpoint_root.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")
        latest_ckpt_dir = ckpt_dirs[-1]
        ddpm_ckpt_path = latest_ckpt_dir / ("ema.ckpt" if prefer_ema else "model.ckpt")
    else:
        ddpm_ckpt_path = Path(args.ddpm_ckpt)
    
    ddpm_model, schedule, ddpm_meta = load_ddpm_model(ddpm_ckpt_path, device=device)
    dataset_name = ddpm_meta.get("dataset", "cifar10")
    img_size = ddpm_meta.get("img_size", 32)
    
    LOGGER.info("Loaded DDPM model from: %s", ddpm_ckpt_path)
    LOGGER.info("Dataset: %s, Image size: %d", dataset_name, img_size)

    # Load split indices
    # Try to find splits.json first (full split file)
    split_json_path = Path(data_cfg["splits"]["paths"]["eval_in"]).parent / "splits.json"
    if split_json_path.exists():
        split_info = load_split_indices(split_json_path)
        z_indices = split_info.get("z", [])
        holdout_indices = split_info.get("holdout", [])
    else:
        # Fallback: use eval_in as members, eval_out as non-members
        LOGGER.warning("splits.json not found, using eval_in/eval_out as member/non-member splits")
        eval_in_path = Path(data_cfg["splits"]["paths"]["eval_in"])
        eval_out_path = Path(data_cfg["splits"]["paths"]["eval_out"])
        if not eval_in_path.exists() or not eval_out_path.exists():
            raise FileNotFoundError(f"Split files not found: {eval_in_path}, {eval_out_path}")
        z_indices = load_single_indices(eval_in_path)
        holdout_indices = load_single_indices(eval_out_path)
    
    # Sample member indices (same size as holdout)
    rng = np.random.default_rng(seed)
    if len(z_indices) < len(holdout_indices):
        raise ValueError(f"Not enough Z indices ({len(z_indices)}) for holdout size ({len(holdout_indices)})")
    member_indices = rng.choice(z_indices, size=len(holdout_indices), replace=False).tolist()
    
    LOGGER.info("Member indices: %d, Non-member indices: %d", len(member_indices), len(holdout_indices))

    # Run evaluation
    LOGGER.info("Running evaluation: alpha=%.4f, mode=%s, K=%d", eval_config.alpha, eval_config.mode, eval_config.K)
    report = evaluate_attack(
        ddpm_model=ddpm_model,
        schedule=schedule,
        ensemble=ensemble,
        alpha_list=alpha_list,
        config=eval_config,
        dataset_name=dataset_name,
        data_root=args.data_root,
        member_indices=member_indices,
        nonmember_indices=holdout_indices,
        img_size=img_size,
        global_seed=seed,
        device=device,
        out_dir=report_dir,
    )

    LOGGER.info("Evaluation complete (legacy workflow). Report saved to: %s", report_dir)
    LOGGER.info("AUC: %.4f, TPR@1%%: %.4f, TPR@0.1%%: %.4f", 
                report["metrics"]["auc"], 
                report["metrics"]["tpr_at"][0.01],
                report["metrics"]["tpr_at"][0.001])


if __name__ == "__main__":
    main()

