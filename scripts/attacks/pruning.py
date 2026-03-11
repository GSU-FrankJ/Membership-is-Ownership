"""
Structured L1 channel pruning for diffusion model checkpoints.

Supports DDIM (MiO), WDM, and EDM (Zhao) model types.

Usage:
    python scripts/attacks/pruning.py \
        --checkpoint /path/to/ckpt \
        --rate 0.3 \
        --output-dir /path/to/output \
        --model-type ddim|wdm|edm
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import torch
import torch.nn.utils.prune as prune

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WDM_REPO = PROJECT_ROOT / "experiments" / "baseline_comparison" / "wdm_repo"
if str(WDM_REPO) not in sys.path:
    sys.path.insert(0, str(WDM_REPO))

EDM_DIR = PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo" / "edm"
if str(EDM_DIR) not in sys.path:
    sys.path.insert(0, str(EDM_DIR))

from src.ddpm_ddim.models.unet import build_unet


def load_ddim_model(checkpoint_path: pathlib.Path, model_cfg_path: pathlib.Path | None, device: str):
    """Load a DDIM model from checkpoint."""
    import yaml

    if model_cfg_path is not None:
        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f)
    else:
        model_cfg = {}

    model = build_unet(model_cfg.get("model", {}))

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_wdm_model(checkpoint_path: pathlib.Path, device: str):
    """Load a WDM model from checkpoint."""
    from wdm.script_util import create_model_and_diffusion
    from scripts.baselines.wdm_adapter import WDM_CIFAR10_CONFIG

    model, _ = create_model_and_diffusion(**WDM_CIFAR10_CONFIG)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_edm_model(checkpoint_path: pathlib.Path, device: str):
    """Load an EDM model from pickle checkpoint."""
    # EDM pickles need torch_utils/dnnlib from the EDM codebase
    import importlib
    edm_path = str(EDM_DIR)
    if edm_path not in sys.path:
        sys.path.insert(0, edm_path)
    # Force reimport to pick up EDM's torch_utils
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("torch_utils") or mod_name.startswith("dnnlib"):
            del sys.modules[mod_name]

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(device)
    net.eval()
    return net


MIN_CHANNELS = 4


def _is_conv2d(module: torch.nn.Module) -> bool:
    """Check if module is Conv2d, including EDM's persistence-wrapped Conv2d."""
    if isinstance(module, torch.nn.Conv2d):
        return True
    # EDM uses torch_utils.persistence which wraps classes in Decorator;
    # isinstance fails but the MRO class names still contain 'Conv2d'
    return type(module).__name__ == "Conv2d" and hasattr(module, "weight") and module.weight.dim() == 4


def apply_pruning(model: torch.nn.Module, rate: float) -> dict:
    """Apply structured L1 channel pruning to all Conv2d layers.

    Skips layers with <= MIN_CHANNELS output channels to protect the
    final RGB projection.

    Returns summary dict with counts.
    """
    pruned_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        if _is_conv2d(module):
            if module.out_channels <= MIN_CHANNELS:
                skipped_count += 1
                continue
            prune.ln_structured(module, name="weight", amount=rate, n=1, dim=0)
            prune.remove(module, "weight")
            pruned_count += 1

    return {"pruned": pruned_count, "skipped": skipped_count}


def main():
    parser = argparse.ArgumentParser(description="Structured L1 channel pruning")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--rate", type=float, default=0.3, help="Pruning rate (default: 0.3)")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-type", type=str, default="ddim", choices=["ddim", "wdm", "edm"])
    parser.add_argument("--model-config", type=pathlib.Path, default=None,
                        help="Model YAML config (DDIM only, auto-detected if not provided)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.model_type == "wdm":
        print(f"Loading WDM model from {args.checkpoint}")
        model = load_wdm_model(args.checkpoint, args.device)
    elif args.model_type == "edm":
        print(f"Loading EDM model from {args.checkpoint}")
        model = load_edm_model(args.checkpoint, args.device)
    else:
        # Auto-detect model config for DDIM
        model_cfg_path = args.model_config
        if model_cfg_path is None:
            ckpt_str = str(args.checkpoint).lower()
            for ds in ["cifar10", "cifar100", "stl10", "celeba"]:
                if ds in ckpt_str:
                    model_cfg_path = PROJECT_ROOT / f"configs/model_ddim_{ds}.yaml"
                    break
            if model_cfg_path is None:
                model_cfg_path = PROJECT_ROOT / "configs/model_ddim_cifar10.yaml"
                print(f"WARNING: Could not detect dataset, defaulting to {model_cfg_path}")
        print(f"Loading DDIM model from {args.checkpoint}")
        model = load_ddim_model(args.checkpoint, model_cfg_path, args.device)

    print(f"Applying {args.rate*100:.0f}% L1 structured pruning...")
    summary = apply_pruning(model, args.rate)
    print(f"  Pruned: {summary['pruned']} layers")
    print(f"  Skipped: {summary['skipped']} layers (<= {MIN_CHANNELS} output channels)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "edm":
        # Save EDM as pickle to preserve the EDMPrecond wrapper
        out_path = args.output_dir / f"pruned_{args.rate:.0%}.pkl".replace("%", "pct")
        with open(out_path, "wb") as f:
            pickle.dump({"ema": model}, f)
    elif args.model_type == "wdm":
        out_path = args.output_dir / f"pruned_{args.rate:.0%}.pt".replace("%", "pct")
        torch.save(model.state_dict(), out_path)
    else:
        out_path = args.output_dir / f"pruned_{args.rate:.0%}.pt".replace("%", "pct")
        torch.save({"model": model.state_dict()}, out_path)

    print(f"Saved pruned checkpoint: {out_path}")


if __name__ == "__main__":
    main()
