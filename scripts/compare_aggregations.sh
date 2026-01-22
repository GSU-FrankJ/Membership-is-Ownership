#!/bin/bash
# Compare multiple t-error aggregation strategies on the same checkpoint.

set -euo pipefail

# Required configuration (can be overridden via env vars)
CKPT="${CKPT:-runs/ddim_cifar10/main/ckpt_400000/ema.ckpt}"
CONFIG="${CONFIG:-configs/attack_qr.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_cifar10.yaml}"
DEVICE="${DEVICE:-cuda}"
AGGREGATES_RAW="${AGGREGATES:-"mean q10"}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PNG="${OUTPUT_PNG:-aggregation_comparison_${TIMESTAMP}.png}"

# Normalize aggregation list (space-delimited)
AGGREGATES_LIST="$(echo "${AGGREGATES_RAW}" | xargs || true)"
if [[ -z "${AGGREGATES_LIST}" ]]; then
  echo "❌ No aggregation methods provided. Set \$AGGREGATES (e.g., 'mean q05 q10 median')."
  exit 1
fi

IFS=' ' read -r -a AGG_METHODS <<< "${AGGREGATES_LIST}"
AGG_COUNT="${#AGG_METHODS[@]}"

echo "=========================================="
echo "Aggregation Comparison Experiment"
echo "=========================================="
echo "Checkpoint : ${CKPT}"
echo "Config     : ${CONFIG}"
echo "Aggregates : ${AGGREGATES_LIST}"
echo "Timestamp  : ${TIMESTAMP}"
echo "=========================================="
echo ""

# Run score computation for each aggregation method
for agg in "${AGG_METHODS[@]}"; do
  echo "------------------------------------------"
  echo "Computing scores with aggregation: ${agg}"
  echo "------------------------------------------"
  python tools/compute_scores.py \
    --config "${CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --aggregate "${agg}" \
    --tag "compare_${agg}_${TIMESTAMP}" \
    --force
  echo ""
done

echo "✅ Score generation finished for ${AGG_COUNT} aggregation methods."
echo ""

# Build comma-separated list for downstream scripts
AGGREGATES_CSV="$(printf "%s," "${AGG_METHODS[@]}")"
AGGREGATES_CSV="${AGGREGATES_CSV%,}"

# Generate markdown summary
python scripts/analyze_aggregation_comparison.py \
  --timestamp "${TIMESTAMP}" \
  --ckpt "${CKPT}" \
  --aggregates "${AGGREGATES_CSV}"

# Generate visualization
python scripts/visualize_aggregation_comparison.py \
  --timestamp "${TIMESTAMP}" \
  --aggregates "${AGGREGATES_CSV}" \
  --output "${OUTPUT_PNG}"

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo "Summary report : comparison_report_${TIMESTAMP}.md"
echo "Visualization  : ${OUTPUT_PNG}"
echo "Score tensors  : scores/compare_<agg>_${TIMESTAMP}_eval_{in,out}.pt"
echo ""
