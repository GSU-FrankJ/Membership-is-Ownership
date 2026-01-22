#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/fitcheck.sh [options]

Options:
  --ckpt <path>           Checkpoint to evaluate (default: newest runs/ddim_*/main/ema_*.ckpt)
  --config_model <file>   Model config (default: configs/model_ddim.yaml)
  --config_attack <file>  Attack config (default: configs/attack_qr.yaml)
  --run_qr                Also run QR train/eval before aggregation
  --tag <name>            Optional tag appended to the result directory
  -h, --help              Print this message

Note:
  - Diagnostic sampling has been removed (no --n_samples option)
  - Aggregation method follows configs/attack_qr.yaml (default: q25)
EOF
}

find_latest_ckpt() {
  python - <<'PY'
import pathlib, time
patterns = [
    "runs/ddim_*/main/ema_*.ckpt",
    "runs/ddim_*/main/ckpt_*/ema*.ckpt",
    "runs/*/main/ema_*.ckpt",
    "runs/**/*ema*.ckpt",
]
paths = []
root = pathlib.Path(".")
for pattern in patterns:
    paths.extend(root.glob(pattern))
if not paths:
    raise SystemExit
latest = max(paths, key=lambda p: p.stat().st_mtime)
print(latest.as_posix())
PY
}

detect_model_name() {
  local ckpt_path="$1"
  local guess="unknown"
  if [[ "${ckpt_path}" =~ runs/([^/]+)/ ]]; then
    guess="${BASH_REMATCH[1]}"
  elif [[ "${ckpt_path}" =~ (cifar10|celeba|imagenet|mnist) ]]; then
    guess="${BASH_REMATCH[1]}"
  fi
  echo "${guess}"
}

detect_step_label() {
  local ckpt_path="$1"
  if [[ "${ckpt_path}" =~ ema_([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "${ckpt_path}" =~ ckpt_([0-9]+)/ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "${ckpt_path}" =~ step_([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo "nostep"
}

CKPT=""
CONFIG_MODEL="configs/model_ddim.yaml"
CONFIG_ATTACK="configs/attack_qr.yaml"
RUN_QR=0
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --config_model)
      CONFIG_MODEL="$2"
      shift 2
      ;;
    --config_attack)
      CONFIG_ATTACK="$2"
      shift 2
      ;;
    --run_qr)
      RUN_QR=1
      shift 1
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[fitcheck] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CKPT}" ]]; then
  CKPT="$(find_latest_ckpt || true)"
fi

if [[ -z "${CKPT}" ]]; then
  echo "[fitcheck] ERROR: unable to locate a checkpoint automatically. Use --ckpt." >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[fitcheck] ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

MODEL_NAME="$(detect_model_name "${CKPT}")"
STEP_LABEL="$(detect_step_label "${CKPT}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUTDIR="results/fitcheck/${MODEL_NAME}/${STEP_LABEL}_${TIMESTAMP}"

if [[ -n "${TAG}" ]]; then
  OUTDIR="${BASE_OUTDIR}/${TAG}"
else
  OUTDIR="${BASE_OUTDIR}"
fi

ARTIFACTS_DIR="${OUTDIR}/artifacts"
PLOTS_DIR="${OUTDIR}/plots"
DIAG_DIR="${OUTDIR}/diagnostics"

mkdir -p "${ARTIFACTS_DIR}" "${PLOTS_DIR}" "${DIAG_DIR}"

echo "[fitcheck] Checkpoint: ${CKPT}"
echo "[fitcheck] Model tag : ${MODEL_NAME}"
echo "[fitcheck] Step label: ${STEP_LABEL}"
echo "[fitcheck] Output dir: ${OUTDIR}"

# Extract aggregation method from config (required for consistency)
AGGREGATE_METHOD=$(python - <<PY
import yaml
import pathlib
config_path = pathlib.Path("${CONFIG_ATTACK}")
with config_path.open("r") as f:
    cfg = yaml.safe_load(f)
aggregate = cfg.get("t_error", {}).get("aggregate")
if aggregate is None:
    raise ValueError("t_error.aggregate must be set in ${CONFIG_ATTACK}")
print(aggregate)
PY
)

# If no tag provided, use aggregation method as tag to ensure consistency
if [[ -z "${TAG}" ]]; then
  TAG="${AGGREGATE_METHOD}"
  echo "[fitcheck] No tag provided; using aggregation method as tag: ${TAG}"
fi

echo "[fitcheck] Aggregation method: ${AGGREGATE_METHOD}"
echo "[fitcheck] Score tag: ${TAG}"

# Step 1: reconstruction scores (aggregation method follows configs/attack_qr.yaml)
echo "[fitcheck] Computing reconstruction scores..."
if [[ -f "scripts/compute_scores.sh" ]]; then
  if [[ -n "${TAG}" ]]; then
    bash "scripts/compute_scores.sh" \
      --config "${CONFIG_ATTACK}" \
      --ckpt "${CKPT}" \
      --tag "${TAG}" \
      --force
  else
    bash "scripts/compute_scores.sh" \
      --config "${CONFIG_ATTACK}" \
      --ckpt "${CKPT}" \
      --force
  fi
elif [[ -f "tools/compute_scores.py" ]]; then
  if [[ -n "${TAG}" ]]; then
    python "tools/compute_scores.py" \
      --config "${CONFIG_ATTACK}" \
      --ckpt "${CKPT}" \
      --tag "${TAG}" \
      --force
  else
    python "tools/compute_scores.py" \
      --config "${CONFIG_ATTACK}" \
      --ckpt "${CKPT}" \
      --force
  fi
else
  echo "[fitcheck] WARNING: no scoring entry point found; skipping score computation." >&2
fi

if [[ -d "scores" ]]; then
  echo "[fitcheck] Copying score artifacts..."
  mapfile -t SCORE_FILES < <(find "scores" -type f \( \
    -name '*eval_in.pt' -o -name '*eval_out.pt' -o \
    -name '*per_sample_scores.pt' -o -name 'hist_*.png' -o \
    -name 'summary*.json' -o -name '*.md' \
  \) 2>/dev/null || true)
  if [[ ${#SCORE_FILES[@]} -gt 0 ]]; then
    IFS=$'\n' SCORE_FILES=($(
      printf "%s\n" "${SCORE_FILES[@]}" | while IFS= read -r item; do
        if [[ -f "${item}" ]]; then
          printf "%s %s\n" "$(stat -c "%Y" "${item}")" "${item}"
        fi
      done | sort -nr | cut -d' ' -f2-
    ))
    for file in "${SCORE_FILES[@]}"; do
      cp --parents "${file}" "${ARTIFACTS_DIR}"
    done
    echo "[fitcheck] Copied files:"
    printf '  - %s\n' "${SCORE_FILES[@]}"
  else
    echo "[fitcheck] WARNING: no matching score artifacts found under scores/."
  fi
else
  echo "[fitcheck] WARNING: scores/ directory not found; skipping artifact copy."
fi

# Step 2: optional QR attack
if [[ "${RUN_QR}" -eq 1 ]]; then
  echo "[fitcheck] Running QR training/evaluation with scores-tag=${TAG}..."
  if [[ -f "scripts/train_qr.sh" ]]; then
    if ! bash "scripts/train_qr.sh" --config "${CONFIG_ATTACK}" --scores-tag "${TAG}"; then
      echo "[fitcheck] WARNING: train_qr.sh failed (continuing)." >&2
    fi
  else
    echo "[fitcheck] WARNING: scripts/train_qr.sh missing; skipping QR training."
  fi
  if [[ -f "scripts/eval_qr.sh" ]]; then
    if ! bash "scripts/eval_qr.sh" --config "${CONFIG_ATTACK}" --scores-tag "${TAG}"; then
      echo "[fitcheck] WARNING: eval_qr.sh failed (continuing)." >&2
    fi
  else
    echo "[fitcheck] WARNING: scripts/eval_qr.sh missing; skipping QR evaluation."
  fi

  if [[ -d "runs/attack_qr/reports" ]]; then
    if compgen -G "runs/attack_qr/reports/*" > /dev/null; then
      LATEST_REPORT="$(ls -1dt runs/attack_qr/reports/* | head -n1)"
    else
      LATEST_REPORT=""
    fi
    if [[ -n "${LATEST_REPORT}" && -d "${LATEST_REPORT}" ]]; then
      echo "[fitcheck] Copying QR report artifacts from ${LATEST_REPORT}..."
      for pattern in "summary.json" "*.md" "*.png"; do
        shopt -s nullglob
        for report_file in "${LATEST_REPORT}"/${pattern}; do
          cp --parents "${report_file}" "${ARTIFACTS_DIR}"
        done
        shopt -u nullglob
      done
    else
      echo "[fitcheck] WARNING: no QR reports found under runs/attack_qr/reports/."
    fi
  else
    echo "[fitcheck] WARNING: runs/attack_qr/reports/ missing; QR artifacts unavailable."
  fi
fi

# Step 3: aggregation/report
if [[ ! -f "tools/fitcheck.py" ]]; then
  echo "[fitcheck] ERROR: tools/fitcheck.py is missing; cannot aggregate." >&2
  exit 1
fi

echo "[fitcheck] Aggregating metrics..."
python "tools/fitcheck.py" --ckpt "${CKPT}" --outdir "${OUTDIR}"

OUTDIR_ABS="$(cd "${OUTDIR}" && pwd)"

echo "SUMMARY: ${OUTDIR_ABS}/summary.json"
echo "REPORT : ${OUTDIR_ABS}/REPORT.md"
echo "PLOTS  : ${OUTDIR_ABS}/plots/"
echo "DIAG   : ${OUTDIR_ABS}/diagnostics/"
