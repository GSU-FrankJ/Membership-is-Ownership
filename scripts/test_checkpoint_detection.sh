#!/usr/bin/env bash
# Quick test script to verify checkpoint detection

set -euo pipefail

RUNS_DIR="/data/short/fjiang4/mia_ddpm_qr/runs"
DATASETS=("cifar10" "cifar100" "stl10")

echo "=========================================="
echo "Testing Checkpoint Detection"
echo "=========================================="

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Testing: $ds ---"
    
    OUTPUT_DIR="${RUNS_DIR}/ddim_${ds}/main"
    OUTPUT_CKPT="${OUTPUT_DIR}/best_for_mia.ckpt"
    
    echo "Checking: $OUTPUT_CKPT"
    
    if [[ -f "$OUTPUT_CKPT" ]]; then
        echo "✓ SKIP: $OUTPUT_CKPT already exists"
        ls -lh "$OUTPUT_CKPT"
    else
        echo "✗ NOT FOUND: Would start training"
    fi
done

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
