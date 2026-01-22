#!/usr/bin/env python3
"""
Generate deterministic membership splits with watermark manifests for MIA experiments.

This script creates:
- watermark_private.json: K indices for private member set W_D
- eval_nonmember.json: K indices disjoint from W_D for evaluation
- member_train.json: Full training set including W_D
- manifest.json: SHA256 hashes + metadata for reproducibility

Usage:
    python scripts/generate_splits.py --dataset cifar100
    python scripts/generate_splits.py --dataset stl10
    python scripts/generate_splits.py --dataset celeba
    python scripts/generate_splits.py --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torchvision import datasets, transforms
from tqdm import tqdm

from mia_logging import get_winston_logger

LOGGER = get_winston_logger(__name__)


# Dataset configurations with split parameters
DATASET_PARAMS = {
    "cifar10": {
        "total_train": 50000,
        "total_test": 10000,
        "watermark_size": 5000,
        "eval_size": 5000,
        "config_path": "configs/data_cifar10.yaml",
    },
    "cifar100": {
        "total_train": 50000,
        "total_test": 10000,
        "watermark_size": 5000,
        "eval_size": 5000,
        "config_path": "configs/data_cifar100.yaml",
    },
    "stl10": {
        "total_train": 5000,  # Labeled train split
        "total_test": 8000,
        "watermark_size": 1000,
        "eval_size": 1000,
        "config_path": "configs/data_stl10.yaml",
    },
    "celeba": {
        "total_train": 162770,
        "total_test": 19962,
        "watermark_size": 5000,
        "eval_size": 5000,
        "config_path": "configs/data_celeba.yaml",
    },
}


def load_yaml(path: pathlib.Path) -> Dict:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sample_indices(generator: torch.Generator, population: int, k: int) -> List[int]:
    """Sample k indices from population without replacement."""
    perm = torch.randperm(population, generator=generator)
    return perm[:k].tolist()


def compute_image_hash(image: torch.Tensor) -> str:
    """Compute SHA256 hash of an image tensor."""
    # Convert to bytes and hash
    img_bytes = image.numpy().tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


def get_dataset_loader(dataset_name: str, root: pathlib.Path, train: bool = True):
    """Get dataset for hash computation (no transforms, raw images)."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=str(root), train=train, download=True)
    elif dataset_name == "cifar100":
        return datasets.CIFAR100(root=str(root), train=train, download=True)
    elif dataset_name == "stl10":
        split = "train" if train else "test"
        return datasets.STL10(root=str(root), split=split, download=True)
    elif dataset_name == "celeba":
        split = "train" if train else "test"
        # CelebA needs special handling - download may require manual steps
        try:
            return datasets.CelebA(root=str(root), split=split, download=True)
        except Exception as e:
            LOGGER.warning(f"CelebA download failed: {e}. Please download manually.")
            return None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_splits(
    dataset_name: str,
    seed: int,
    watermark_size: Optional[int] = None,
    eval_size: Optional[int] = None,
) -> Dict[str, List[int]]:
    """
    Build deterministic membership splits for a dataset.
    
    Creates:
    - watermark_private: K indices for private watermark set W_D
    - eval_nonmember: K indices from train set, disjoint from W_D
    - member_train: Full training indices (watermark samples included)
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed for reproducibility
        watermark_size: Override watermark size K
        eval_size: Override eval size
    
    Returns:
        Dictionary with splits
    """
    params = DATASET_PARAMS[dataset_name]
    generator = torch.Generator().manual_seed(seed)
    
    total_train = params["total_train"]
    K = watermark_size or params["watermark_size"]
    eval_K = eval_size or params["eval_size"]
    
    # Ensure we don't exceed dataset size
    K = min(K, total_train // 4)
    eval_K = min(eval_K, total_train // 4)
    
    # Full training indices
    all_train_indices = list(range(total_train))
    
    # Sample watermark indices (private member set W_D)
    watermark_private = sample_indices(generator, total_train, K)
    watermark_set = set(watermark_private)
    
    # Sample eval_nonmember from remaining indices (disjoint from W_D)
    non_watermark_pool = [idx for idx in all_train_indices if idx not in watermark_set]
    non_watermark_tensor = torch.tensor(non_watermark_pool)
    eval_perm = torch.randperm(len(non_watermark_pool), generator=generator)
    eval_nonmember = non_watermark_tensor[eval_perm[:eval_K]].tolist()
    
    # member_train includes all training data (W_D is a subset)
    member_train = all_train_indices
    
    LOGGER.info(
        "[%s] Splits: watermark_private=%d, eval_nonmember=%d, member_train=%d",
        dataset_name,
        len(watermark_private),
        len(eval_nonmember),
        len(member_train),
    )
    
    return {
        "watermark_private": watermark_private,
        "eval_nonmember": eval_nonmember,
        "member_train": member_train,
    }


def compute_watermark_hashes(
    dataset_name: str,
    root: pathlib.Path,
    watermark_indices: List[int],
    max_samples: Optional[int] = None,
) -> Dict[int, str]:
    """
    Compute SHA256 hashes for watermark samples.
    
    Args:
        dataset_name: Name of the dataset
        root: Dataset root directory
        watermark_indices: Indices of watermark samples
        max_samples: Max samples to hash (for testing)
    
    Returns:
        Dict mapping index -> SHA256 hash
    """
    LOGGER.info("Computing SHA256 hashes for watermark samples...")
    
    dataset = get_dataset_loader(dataset_name, root, train=True)
    if dataset is None:
        LOGGER.warning("Could not load dataset for hashing. Skipping hash computation.")
        return {}
    
    hashes = {}
    indices_to_hash = watermark_indices[:max_samples] if max_samples else watermark_indices
    
    for idx in tqdm(indices_to_hash, desc="Hashing watermark samples"):
        try:
            # Get raw image (PIL Image)
            if dataset_name == "celeba":
                img, _ = dataset[idx]
            else:
                img, _ = dataset[idx]
            
            # Convert PIL to tensor for consistent hashing
            if hasattr(img, 'tobytes'):
                # PIL Image
                img_bytes = img.tobytes()
            else:
                # Already tensor
                img_bytes = img.numpy().tobytes()
            
            hashes[idx] = hashlib.sha256(img_bytes).hexdigest()
        except Exception as e:
            LOGGER.warning(f"Failed to hash sample {idx}: {e}")
    
    LOGGER.info(f"Computed {len(hashes)} hashes")
    return hashes


def generate_manifest(
    dataset_name: str,
    seed: int,
    splits: Dict[str, List[int]],
    hashes: Dict[int, str],
    output_dir: pathlib.Path,
) -> Dict:
    """
    Generate manifest with metadata for reproducibility.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed used
        splits: Generated splits
        hashes: SHA256 hashes of watermark samples
        output_dir: Output directory
    
    Returns:
        Manifest dictionary
    """
    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "splits": {
            "watermark_private": {
                "count": len(splits["watermark_private"]),
                "indices_file": "watermark_private.json",
            },
            "eval_nonmember": {
                "count": len(splits["eval_nonmember"]),
                "indices_file": "eval_nonmember.json",
            },
            "member_train": {
                "count": len(splits["member_train"]),
                "indices_file": "member_train.json",
            },
        },
        "watermark_hashes": {
            "count": len(hashes),
            "hashes_file": "sample_hashes.json",
        },
        "verification": {
            "watermark_disjoint_from_eval": len(
                set(splits["watermark_private"]) & set(splits["eval_nonmember"])
            ) == 0,
            "watermark_subset_of_train": set(splits["watermark_private"]).issubset(
                set(splits["member_train"])
            ),
        },
    }
    
    return manifest


def save_splits(
    splits: Dict[str, List[int]],
    hashes: Dict[int, str],
    manifest: Dict,
    output_dir: pathlib.Path,
) -> None:
    """Save all split files and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual split files
    for name, indices in splits.items():
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(indices, f)
        LOGGER.info(f"Wrote {path} ({len(indices)} indices)")
    
    # Save hashes
    if hashes:
        hashes_path = output_dir / "sample_hashes.json"
        # Convert int keys to strings for JSON
        hashes_str = {str(k): v for k, v in hashes.items()}
        with hashes_path.open("w", encoding="utf-8") as f:
            json.dump(hashes_str, f, indent=2)
        LOGGER.info(f"Wrote {hashes_path} ({len(hashes)} hashes)")
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    LOGGER.info(f"Wrote {manifest_path}")


def download_dataset(dataset_name: str, root: pathlib.Path) -> bool:
    """Download dataset if not present. Returns True if successful."""
    root.mkdir(parents=True, exist_ok=True)
    
    try:
        if dataset_name == "cifar10":
            datasets.CIFAR10(root=str(root), train=True, download=True)
            datasets.CIFAR10(root=str(root), train=False, download=True)
        elif dataset_name == "cifar100":
            datasets.CIFAR100(root=str(root), train=True, download=True)
            datasets.CIFAR100(root=str(root), train=False, download=True)
        elif dataset_name == "stl10":
            datasets.STL10(root=str(root), split="train", download=True)
            datasets.STL10(root=str(root), split="test", download=True)
        elif dataset_name == "celeba":
            # CelebA often requires manual download
            try:
                datasets.CelebA(root=str(root), split="train", download=True)
                datasets.CelebA(root=str(root), split="test", download=True)
            except Exception as e:
                LOGGER.warning(
                    f"CelebA auto-download failed: {e}\n"
                    "Please download manually from: "
                    "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8"
                )
                return False
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        LOGGER.info(f"Dataset {dataset_name} downloaded successfully")
        return True
        
    except Exception as e:
        LOGGER.error(f"Failed to download {dataset_name}: {e}")
        return False


def process_dataset(
    dataset_name: str,
    seed: int,
    output_base: pathlib.Path,
    skip_download: bool = False,
    skip_hashes: bool = False,
    max_hash_samples: Optional[int] = None,
) -> None:
    """
    Process a single dataset: download, build splits, compute hashes, save.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed
        output_base: Base output directory (splits will be in output_base/{dataset_name}/)
        skip_download: Skip downloading dataset
        skip_hashes: Skip computing SHA256 hashes
        max_hash_samples: Max samples to hash (for testing)
    """
    LOGGER.info(f"Processing dataset: {dataset_name}")
    
    params = DATASET_PARAMS[dataset_name]
    config_path = PROJECT_ROOT / params["config_path"]
    
    # Load config if exists
    if config_path.exists():
        data_cfg = load_yaml(config_path)
        root = pathlib.Path(data_cfg["dataset"]["root"])
        seed = data_cfg.get("splits", {}).get("seed", seed)
        watermark_size = data_cfg.get("splits", {}).get("watermark_size")
    else:
        root = PROJECT_ROOT / "data" / dataset_name
        watermark_size = None
    
    set_seeds(seed)
    
    # Download dataset
    if not skip_download:
        if not download_dataset(dataset_name, root):
            LOGGER.warning(f"Skipping {dataset_name} due to download failure")
            return
    
    # Build splits
    splits = build_splits(dataset_name, seed, watermark_size=watermark_size)
    
    # Compute hashes for watermark samples
    if skip_hashes:
        hashes = {}
    else:
        hashes = compute_watermark_hashes(
            dataset_name, root, splits["watermark_private"], max_hash_samples
        )
    
    # Generate manifest
    output_dir = output_base / dataset_name
    manifest = generate_manifest(dataset_name, seed, splits, hashes, output_dir)
    
    # Save everything
    save_splits(splits, hashes, manifest, output_dir)
    
    LOGGER.info(f"[{dataset_name}] Split generation complete -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate membership splits with watermark manifests"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cifar10", "cifar100", "stl10", "celeba", "all"],
        help="Dataset to generate splits for (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Base output directory for splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20251030,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets (assume already present)",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip computing SHA256 hashes (faster but less verifiable)",
    )
    parser.add_argument(
        "--max-hash-samples",
        type=int,
        default=None,
        help="Max samples to hash (for testing)",
    )
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.dataset == "all":
        dataset_list = ["cifar10", "cifar100", "stl10", "celeba"]
    else:
        dataset_list = [args.dataset]
    
    # Process each dataset
    for dataset_name in dataset_list:
        try:
            process_dataset(
                dataset_name,
                args.seed,
                args.output_dir,
                args.skip_download,
                args.skip_hashes,
                args.max_hash_samples,
            )
        except Exception as e:
            LOGGER.error(f"Failed to process {dataset_name}: {e}")
            if args.dataset != "all":
                raise
    
    LOGGER.info(f"Split generation complete for: {', '.join(dataset_list)}")


if __name__ == "__main__":
    main()
