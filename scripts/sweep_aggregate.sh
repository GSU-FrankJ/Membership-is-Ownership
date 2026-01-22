#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SOURCE_DIR}/.." && pwd)"

TARGET_STEP="${TARGET_STEP:-300000}"
CKPT="${ROOT_DIR}/runs/ddim_cifar10/main/ckpt_${TARGET_STEP}/ema.ckpt"
CFG_ATTACK="${ROOT_DIR}/configs/attack_qr.yaml"
AGGREGATES=(mean min q10)

for AGG in "${AGGREGATES[@]}"; do
  TAG="step_${TARGET_STEP}__agg_${AGG}"

  bash "${ROOT_DIR}/scripts/compute_scores.sh" --config "${CFG_ATTACK}" \
    --ckpt "${CKPT}" --tag "${TAG}" --aggregate "${AGG}"

  bash "${ROOT_DIR}/scripts/train_qr.sh" --config "${CFG_ATTACK}" \
    --out "${ROOT_DIR}/runs/attack_qr/${TAG}" \
    --scores-tag "${TAG}"

  bash "${ROOT_DIR}/scripts/eval_qr.sh" --config "${CFG_ATTACK}" \
    --scores-tag "${TAG}" \
    --ensemble "${ROOT_DIR}/runs/attack_qr/${TAG}/ensembles/bagging.pt"
done
