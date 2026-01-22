#!/usr/bin/env python3

"""Run two DDIM10 sampling passes and assert bitwise determinism via SHA256."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

from mia_logging import get_winston_logger
from scripts.sample_ddim10 import (
    _compute_sha256,
    _prepare_out_dir,
    run_sampling,
)


LOGGER = get_winston_logger(__name__)


def _collect_pngs(out_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return all PNG artifacts (grid + individual images)."""

    paths = list(out_dir.glob("grid_*.png"))
    paths.extend(sorted((out_dir / "images").glob("*.png")))
    return sorted(paths)


def _hash_bundle(out_dir: pathlib.Path) -> Dict[str, str]:
    """Return relative path -> sha256 for PNG bundle."""

    hashes: Dict[str, str] = {}
    pngs = _collect_pngs(out_dir)
    if not pngs:
        raise RuntimeError(f"No PNG artifacts found under {out_dir}")
    for path in pngs:
        rel = path.relative_to(out_dir)
        hashes[str(rel)] = _compute_sha256(path)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Determinism checker for DDIM10 sampling.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to EMA finetuned checkpoint")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for deterministic sampling")
    parser.add_argument("--t-start", type=int, default=900, help="Starting timestep for DDIM10")
    parser.add_argument("--ddim-fp32", action=argparse.BooleanOptionalAction, default=True, help="Force DDIM algebra to run in float32")
    parser.add_argument("--out-root", type=pathlib.Path, required=True, help="Parent directory that will contain run1/ and run2/")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing out-root by overwriting its contents")
    return parser.parse_args()


def _build_run_args(base_args: argparse.Namespace, out_dir: pathlib.Path) -> argparse.Namespace:
    """Create a shallow copy of args with a new out_dir."""

    return argparse.Namespace(
        model_config=pathlib.Path("configs/model_ddim.yaml"),
        data_config=pathlib.Path("configs/data_cifar10.yaml"),
        checkpoint=base_args.checkpoint,
        num_samples=base_args.num_samples,
        batch_size=base_args.batch_size,
        device=base_args.device,
        out_dir=out_dir,
        grad_checkpoint=base_args.grad_checkpoint,
        seed=base_args.seed,
        t_start=base_args.t_start,
        ddim_fp32=base_args.ddim_fp32,
        save_grid=True,
        save_individual=True,
        write_manifest=True,
        overwrite=base_args.overwrite,
        save_sha=True,
    )


def main() -> None:
    args = parse_args()
    _prepare_out_dir(args.out_root, args.overwrite)

    run1_dir = args.out_root / "run1"
    run2_dir = args.out_root / "run2"

    LOGGER.info("Starting determinism check with out dirs %s and %s", run1_dir, run2_dir)

    run_sampling(_build_run_args(args, run1_dir))
    run_sampling(_build_run_args(args, run2_dir))

    hashes1 = _hash_bundle(run1_dir)
    hashes2 = _hash_bundle(run2_dir)

    if hashes1 == hashes2:
        LOGGER.info("Determinism check PASS: all SHA256 hashes match")
        sys.exit(0)

    LOGGER.error("Determinism check FAIL: mismatched hashes detected")
    mismatch_keys = sorted(set(hashes1.keys()) ^ set(hashes2.keys()))
    shared = sorted(set(hashes1.keys()) & set(hashes2.keys()))

    if mismatch_keys:
        LOGGER.error("Files present only in one run: %s", mismatch_keys)
    for key in shared:
        if hashes1[key] != hashes2[key]:
            LOGGER.error("Mismatch: %s | run1=%s run2=%s", key, hashes1[key], hashes2[key])

    sys.exit(1)


if __name__ == "__main__":
    main()
