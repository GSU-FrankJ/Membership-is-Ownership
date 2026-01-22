#!/usr/bin/env python3

"""Sampling with the 10-step deterministic DDIM finetuned model (EMA by default)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import subprocess
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from src.ddpm_ddim.models import build_unet
from src.ddpm_ddim.schedulers import build_cosine_schedule
from src.ddpm_ddim.samplers.ddim10 import build_linear_timesteps, ddim_sample_differentiable


LOGGER = get_winston_logger(__name__)


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _denormalize(images: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    imgs = images * std_t + mean_t
    return imgs.clamp(0.0, 1.0)


def _setup_determinism(seed: int, device: torch.device) -> torch.Generator:
    """Configure deterministic behavior and return a seeded generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - best-effort guard
        LOGGER.warning("Deterministic algorithms are not fully available on this platform")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator(device=device).manual_seed(seed)


def _compute_sha256(path: pathlib.Path) -> str:
    """Return SHA256 for a file using a streaming read to stay memory-friendly."""

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sha256_file(paths: List[pathlib.Path], sha_path: pathlib.Path) -> None:
    """Write deterministic SHA256 sums for provided paths."""

    lines = []
    for p in sorted(paths):
        rel = p.relative_to(sha_path.parent)
        lines.append(f"{_compute_sha256(p)}  {rel}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote SHA256 manifest with %d entries to %s", len(paths), sha_path)


def _git_commit() -> str | None:
    """Return current git commit hash if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        LOGGER.warning("Git commit unavailable; proceeding without commit hash")
        return None


def _ensure_checkpoint_exists(checkpoint: pathlib.Path) -> None:
    """Fail fast if checkpoint is missing and list nearby EMA candidates."""

    if checkpoint.exists():
        return
    candidates = sorted(checkpoint.parent.glob("*_ema.pt"))
    formatted = "\n".join([f"- {c.name}" for c in candidates]) if candidates else "None found"
    raise FileNotFoundError(
        f"Checkpoint not found at {checkpoint}. Nearby EMA candidates:\n{formatted}"
    )


def _prepare_out_dir(out_dir: pathlib.Path, overwrite: bool) -> None:
    """Create output directory or fail if it already exists and overwrite is False."""

    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists. Use --overwrite to replace it.")
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _save_grid(images: torch.Tensor, out_dir: pathlib.Path) -> pathlib.Path:
    """Save a square grid image (max 16x16) and return its path."""

    grid_count = min(len(images), 256)
    grid_rows = min(int(grid_count ** 0.5), 16)
    grid = make_grid(images[: grid_rows * grid_rows], nrow=grid_rows)
    grid_path = out_dir / f"grid_{grid_rows}x{grid_rows}.png"
    save_image(grid, grid_path)
    LOGGER.info("Saved sample grid to %s", grid_path)
    return grid_path


def _save_individual_images(images: torch.Tensor, images_dir: pathlib.Path) -> List[pathlib.Path]:
    """Save individual samples as PNGs under images_dir and return their paths."""

    images_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, img in enumerate(images):
        path = images_dir / f"{idx:06d}.png"
        save_image(img, path)
        saved.append(path)
    LOGGER.info("Saved %d individual samples to %s", len(saved), images_dir)
    return saved


def _write_manifest(
    manifest_path: pathlib.Path,
    args: argparse.Namespace,
    checkpoint: pathlib.Path,
    git_commit: str | None,
    grid_path: pathlib.Path | None,
    image_paths: List[pathlib.Path] | None,
    sha_path: pathlib.Path | None,
) -> None:
    """Persist run metadata so downstream checks can reproduce the invocation."""

    payload = {
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "checkpoint": str(checkpoint.resolve()),
        "git_commit": git_commit,
        "args": {
            "model_config": str(args.model_config),
            "data_config": str(args.data_config),
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "device": args.device,
            "grad_checkpoint": args.grad_checkpoint,
            "seed": args.seed,
            "t_start": args.t_start,
            "ddim_fp32": args.ddim_fp32,
            "save_grid": args.save_grid,
            "save_individual": args.save_individual,
        },
        "artifacts": {
            "grid": str(grid_path) if grid_path else None,
            "images_dir": str(pathlib.Path(image_paths[0]).parent) if image_paths else None,
            "sha256": str(sha_path) if sha_path else None,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote manifest to %s", manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples with 10-step DDIM (EMA finetuned checkpoint).")
    parser.add_argument("--model-config", type=pathlib.Path, default=pathlib.Path("configs/model_ddim.yaml"))
    parser.add_argument("--data-config", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to EMA finetuned checkpoint")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("samples_ddim10"))
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (off for sampling by default)")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for deterministic sampling")
    parser.add_argument("--t-start", type=int, default=900, help="Starting timestep for DDIM10 (default: 900)")
    parser.add_argument("--ddim-fp32", action=argparse.BooleanOptionalAction, default=True, help="Force DDIM algebra to run in float32")
    parser.add_argument("--save-grid", action=argparse.BooleanOptionalAction, default=True, help="Save grid PNG (grid_16x16.png)")
    parser.add_argument("--save-individual", action=argparse.BooleanOptionalAction, default=True, help="Save individual PNGs under images/")
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True, help="Write manifest.json with run metadata")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an existing out-dir by overwriting its contents")
    parser.add_argument("--save-sha", action=argparse.BooleanOptionalAction, default=True, help="Write sha256.txt over saved PNGs")
    return parser.parse_args()


def run_sampling(args: argparse.Namespace) -> Dict[str, object]:
    """Run deterministic sampling and persist artifacts."""

    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_checkpoint_exists(args.checkpoint)
    _prepare_out_dir(args.out_dir, args.overwrite)

    generator = _setup_determinism(args.seed, device)
    model = build_unet(model_cfg["model"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    T = model_cfg["diffusion"]["timesteps"]
    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)
    t_start = args.t_start if args.t_start is not None else T - 1
    timesteps = build_linear_timesteps(T, 10, start=t_start)
    LOGGER.info("Using DDIM timesteps (start=%d): %s", t_start, timesteps)

    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])  # type: ignore[assignment]
    std = tuple(data_cfg["dataset"]["normalization"]["std"])  # type: ignore[assignment]

    generated = []
    remaining = args.num_samples
    bsz = args.batch_size

    while remaining > 0:
        curr = min(remaining, bsz)
        noise = torch.randn((curr, 3, 32, 32), device=device, generator=generator)
        with torch.no_grad():
            samples = ddim_sample_differentiable(
                model=model,
                alphas_bar=alphas_bar,
                shape=noise.shape,
                timesteps=timesteps,
                device=device,
                use_checkpoint=args.grad_checkpoint,
                noise=noise,
                ddim_fp32=args.ddim_fp32,
            )
        samples = _denormalize(samples, mean, std).cpu()
        generated.append(samples)
        remaining -= curr

    images = torch.cat(generated, dim=0)

    grid_path = _save_grid(images, args.out_dir) if args.save_grid else None
    image_paths = _save_individual_images(images, args.out_dir / "images") if args.save_individual else None

    sha_path: pathlib.Path | None = args.out_dir / "sha256.txt" if args.save_sha else None
    if args.save_sha:
        sha_targets: List[pathlib.Path] = []
        if grid_path:
            sha_targets.append(grid_path)
        if image_paths:
            sha_targets.extend(image_paths)
        if sha_targets:
            _write_sha256_file(sha_targets, sha_path)  # type: ignore[arg-type]
        else:
            LOGGER.info("No artifacts to hash; skipping sha256 manifest")
            sha_path = None

    manifest_path = args.out_dir / "manifest.json"
    git_commit = _git_commit()
    if args.write_manifest:
        _write_manifest(
            manifest_path=manifest_path,
            args=args,
            checkpoint=args.checkpoint,
            git_commit=git_commit,
            grid_path=grid_path,
            image_paths=image_paths,
            sha_path=sha_path,
        )

    LOGGER.info("Completed sampling run with %d images", len(images))
    return {
        "out_dir": args.out_dir,
        "grid": grid_path,
        "images": image_paths,
        "manifest": manifest_path if args.write_manifest else None,
        "sha256": sha_path if args.save_sha else None,
        "git_commit": git_commit,
        "timesteps": timesteps,
    }


def main() -> None:
    args = parse_args()
    run_sampling(args)


if __name__ == "__main__":
    main()
