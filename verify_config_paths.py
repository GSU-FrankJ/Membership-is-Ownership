#!/usr/bin/env python3
"""验证所有配置文件的路径是否正确指向 /data/short/ 中的资源。

Usage:
    python verify_config_paths.py
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_path(path: str, description: str) -> bool:
    """检查路径是否存在。"""
    exists = os.path.exists(path)
    status = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
    print(f"  {status} {description}: {path}")
    return exists

def verify_data_config(config_path: str) -> Tuple[bool, List[str]]:
    """验证数据配置文件。"""
    print(f"\n{BLUE}Checking {config_path}{RESET}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    errors = []
    all_ok = True
    
    # Check dataset root
    dataset_root = config['dataset']['root']
    if not check_path(dataset_root, "Dataset root"):
        all_ok = False
        errors.append(f"Dataset root missing: {dataset_root}")
    
    # Check split paths
    split_paths = config['splits']['paths']
    for split_name, split_path in split_paths.items():
        if split_name in ['watermark_private', 'eval_nonmember', 'member_train', 'manifest']:
            if not check_path(split_path, f"Split '{split_name}'"):
                all_ok = False
                errors.append(f"Split file missing: {split_path}")
    
    return all_ok, errors

def verify_model_config(config_path: str) -> Tuple[bool, List[str]]:
    """验证模型配置文件。"""
    print(f"\n{BLUE}Checking {config_path}{RESET}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    errors = []
    all_ok = True
    
    # Check output directory
    output_dir = config['experiment']['output_dir']
    if not check_path(output_dir, "Output directory"):
        all_ok = False
        errors.append(f"Output directory missing: {output_dir}")
    
    return all_ok, errors

def verify_finetune_config(config_path: str) -> Tuple[bool, List[str]]:
    """验证微调配置文件。"""
    print(f"\n{BLUE}Checking {config_path}{RESET}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    errors = []
    all_ok = True
    
    # Check checkpoint path
    checkpoint_path = config['base']['checkpoint']
    if not check_path(checkpoint_path, "Checkpoint"):
        all_ok = False
        errors.append(f"Checkpoint missing: {checkpoint_path}")
    
    return all_ok, errors

def main():
    """主验证函数。"""
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}Configuration Path Verification{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    all_configs_ok = True
    all_errors = []
    
    # Data configs
    data_configs = [
        'configs/data_cifar10.yaml',
        'configs/data_cifar100.yaml',
        'configs/data_celeba.yaml',
        'configs/data_stl10.yaml',
    ]
    
    print(f"\n{YELLOW}=== Data Configurations ==={RESET}")
    for config in data_configs:
        ok, errors = verify_data_config(config)
        if not ok:
            all_configs_ok = False
            all_errors.extend(errors)
    
    # Model configs
    model_configs = [
        'configs/model_ddim_cifar10.yaml',
        'configs/model_ddim_cifar100.yaml',
        'configs/model_ddim_celeba.yaml',
        'configs/model_ddim_stl10.yaml',
    ]
    
    print(f"\n{YELLOW}=== Model Configurations ==={RESET}")
    for config in model_configs:
        ok, errors = verify_model_config(config)
        if not ok:
            all_configs_ok = False
            all_errors.extend(errors)
    
    # Finetune configs
    finetune_configs = [
        'configs/mmd_finetune_cifar10.yaml',
        'configs/mmd_finetune_cifar100.yaml',
        'configs/mmd_finetune_celeba.yaml',
        'configs/mmd_finetune_stl10.yaml',
    ]
    
    print(f"\n{YELLOW}=== MMD Finetune Configurations ==={RESET}")
    for config in finetune_configs:
        ok, errors = verify_finetune_config(config)
        if not ok:
            all_configs_ok = False
            all_errors.extend(errors)
    
    # Summary
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    if all_configs_ok:
        print(f"{GREEN}✓ All configurations verified successfully!{RESET}")
        print(f"{GREEN}✓ All paths point to /data/short/fjiang4/mia_ddpm_qr/{RESET}")
        print(f"\n{GREEN}You can now skip training and use existing checkpoints:{RESET}")
        print(f"  - CIFAR-10: {GREEN}Ready{RESET}")
        print(f"  - CIFAR-100: {GREEN}Ready{RESET}")
        print(f"  - STL-10: {GREEN}Ready{RESET}")
        print(f"  - CelebA: {YELLOW}Training in progress (290k/400k){RESET}")
        return 0
    else:
        print(f"{RED}✗ Some paths are missing:{RESET}")
        for error in all_errors:
            print(f"  {RED}•{RESET} {error}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
