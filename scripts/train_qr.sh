#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SOURCE_DIR}/.." && pwd)"

ATTACK_CONFIG="${PROJECT_ROOT}/configs/attack_qr.yaml"
DATA_CONFIG="${PROJECT_ROOT}/configs/data_cifar10.yaml"

echo "[train_qr] Attack config: ${ATTACK_CONFIG}"
echo "[train_qr] Using MAIN PATH: attack_qr/engine + ResNet18QR (image + stats)"

python "${PROJECT_ROOT}/attack_qr/engine/cli_train.py" \
  --config "${ATTACK_CONFIG}" \
  --data-config "${DATA_CONFIG}" "$@"


