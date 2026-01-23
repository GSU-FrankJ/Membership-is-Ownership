#!/usr/bin/env bash
#
# run_all.sh - 多数据集 MIA 水印实验全流程 Pipeline
#
# 使用方法:
#   bash run_all.sh 2>&1 | tee run_all_$(date +%Y%m%d_%H%M%S).log
#
# 支持: 失败即停 (set -e), 日志重定向, 产物保护
#

set -euo pipefail  # 失败即停, 未定义变量报错

###############################################################################
# 配置区
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH，确保 Python 能找到项目根目录的模块 (如 mia_logging)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 数据集列表 
DATASETS=("cifar10" "cifar100" "stl10" "celeba")

# 公共参数
SEED=20251030
DEVICE="cuda"
K_TIMESTEPS=50
AGG="q25"

# Smoke test 模式: 设为非空值启用 (如 "100"), 空值表示完整运行
# 建议先设 MAX_SAMPLES_SMOKE="100" 跑通 pipeline，再设为空跑完整实验
MAX_SAMPLES_SMOKE="100"

# 输出目录
SPLITS_DIR="data/splits"
# Use absolute path to match the updated configs that point to /data/short/
RUNS_DIR="/data/short/fjiang4/mia_ddpm_qr/runs"
REPORTS_DIR="${RUNS_DIR}/attack_qr/reports"

# 日志目录
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

###############################################################################
# 工具函数
###############################################################################

log_step() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP $1: $2"
    echo "========================================================================"
}

check_file() {
    if [[ -f "$1" ]]; then
        echo "  ✓ EXISTS: $1"
        return 0
    else
        echo "  ✗ MISSING: $1"
        return 1
    fi
}

# Check if dataset splits are ready (all 4 files exist)
check_dataset_ready() {
    local ds="$1"
    local all_exist=true
    for f in watermark_private.json eval_nonmember.json member_train.json manifest.json; do
        if [[ ! -f "$SPLITS_DIR/$ds/$f" ]]; then
            all_exist=false
            break
        fi
    done
    if $all_exist; then
        return 0
    else
        return 1
    fi
}

###############################################################################
# STEP 1: 生成 Splits / Manifest
###############################################################################

log_step 1 "Generate Splits & Manifests (all datasets)"

# 命令: 生成所有数据集的 splits
# --skip-download: 假设数据集已下载 (首次运行请去掉此参数)
# --skip-hashes: 跳过 SHA256 计算 (加速，验证时可去掉)
python scripts/generate_splits.py \
    --dataset all \
    --output-dir "$SPLITS_DIR" \
    --seed "$SEED" \
    --skip-hashes \
    2>&1 | tee "$LOG_DIR/step1_generate_splits.log"

# Expected Output Files:
echo ""
echo "Checking Step 1 outputs..."
READY_DATASETS=()
SKIPPED_DATASETS=()
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    check_file "$SPLITS_DIR/$ds/watermark_private.json" || true
    check_file "$SPLITS_DIR/$ds/eval_nonmember.json" || true
    check_file "$SPLITS_DIR/$ds/member_train.json" || true
    check_file "$SPLITS_DIR/$ds/manifest.json" || true
    
    # Track which datasets are ready
    if check_dataset_ready "$ds"; then
        READY_DATASETS+=("$ds")
    else
        SKIPPED_DATASETS+=("$ds")
        echo "  ⚠️  SKIPPING $ds (splits incomplete - manual download may be required)"
    fi
done

echo ""
echo "Ready datasets: ${READY_DATASETS[*]}"
if [[ ${#SKIPPED_DATASETS[@]} -gt 0 ]]; then
    echo "Skipped datasets: ${SKIPPED_DATASETS[*]}"
fi

# Update DATASETS to only include ready ones
DATASETS=("${READY_DATASETS[@]}")

###############################################################################
# STEP 2: 训练 Model A (每个数据集)
###############################################################################

log_step 2 "Train Model A (all datasets)"

# 注意: 这是长时间任务，建议在 tmux/screen 中运行
# CIFAR-100: ~24h, STL-10: ~12h, CelebA: ~36h (1x A100)
# 如果 best_for_mia.ckpt 已存在，跳过训练 (支持断点续跑)

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Training Model A for: $ds ---"
    
    MODEL_CONFIG="configs/model_ddim_${ds}.yaml"
    DATA_CONFIG="configs/data_${ds}.yaml"
    OUTPUT_DIR="${RUNS_DIR}/ddim_${ds}/main"
    OUTPUT_CKPT="${OUTPUT_DIR}/best_for_mia.ckpt"
    
    if [[ -f "$OUTPUT_CKPT" ]]; then
        echo "  SKIP: $OUTPUT_CKPT already exists"
        continue
    fi
    
    # 训练命令
    # --mode main: 完整训练 (400k iters for CIFAR-100/CelebA, 200k for STL-10)
    # --select-best: 训练完成后自动选择最优 checkpoint
    python src/ddpm_ddim/train_ddim.py \
        --config "$MODEL_CONFIG" \
        --data "$DATA_CONFIG" \
        --mode main \
        --select-best \
        2>&1 | tee "$LOG_DIR/step2_train_model_a_${ds}.log"
    
    # Expected Output Files:
    echo "Checking Model A outputs for $ds..."
    check_file "$OUTPUT_CKPT"
    check_file "${OUTPUT_DIR}/watermark_exposure.json"
    check_file "${OUTPUT_DIR}/run.json"
done

###############################################################################
# STEP 3: MMD Finetune -> Model B (每个数据集)
###############################################################################

log_step 3 "MMD Finetune to create Model B (all datasets)"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Creating Model B for: $ds ---"
    
    MMD_CONFIG="configs/mmd_finetune_${ds}.yaml"
    OUTPUT_DIR="${RUNS_DIR}/mmd_finetune/${ds}/model_b"
    OUTPUT_CKPT="${OUTPUT_DIR}/ckpt_0500_ema.pt"
    
    if [[ -f "$OUTPUT_CKPT" ]]; then
        echo "  SKIP: $OUTPUT_CKPT already exists"
        continue
    fi
    
    # Finetune 命令
    # --iters 500: 500 步 MMD finetune
    # --seed: 保证可复现
    python scripts/finetune_mmd_ddm.py \
        --config "$MMD_CONFIG" \
        --out "$OUTPUT_DIR" \
        --seed "$SEED" \
        --iters 500 \
        2>&1 | tee "$LOG_DIR/step3_finetune_model_b_${ds}.log"
    
    # Expected Output Files:
    echo "Checking Model B outputs for $ds..."
    check_file "$OUTPUT_CKPT"
    check_file "${OUTPUT_DIR}/configs/mmd_finetune.yaml"
done

###############################################################################
# STEP 4: Ownership Evaluation (每个数据集, 两个 split)
###############################################################################

log_step 4 "Ownership Evaluation (all datasets, watermark + nonmember)"

SPLITS=("watermark_private" "eval_nonmember")

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Evaluating ownership for: $ds ---"
    
    MODEL_A="${RUNS_DIR}/ddim_${ds}/main/best_for_mia.ckpt"
    MODEL_B="${RUNS_DIR}/mmd_finetune/${ds}/model_b/ckpt_0500_ema.pt"
    OUTPUT_DIR="${REPORTS_DIR}/${ds}"
    
    mkdir -p "$OUTPUT_DIR"
    
    for split in "${SPLITS[@]}"; do
        echo "  Evaluating split: $split"
        
        OUTPUT_JSON="${OUTPUT_DIR}/baseline_comparison_${ds}_${split}.json"
        
        # 如果已存在，跳过
        if [[ -f "$OUTPUT_JSON" ]]; then
            echo "    SKIP: $OUTPUT_JSON already exists"
            continue
        fi
        
        # Evaluation 命令
        # --split: 指定评估哪个 split
        # --k-timesteps: t-error 采样的 timestep 数量
        # --agg: 聚合方式 (q25 = 25th percentile)
        # --save-pdf: 生成 PDF 可视化报告
        # --max-samples: smoke test 时限制样本数
        python scripts/eval_ownership.py \
            --dataset "$ds" \
            --split "$split" \
            --model-a "$MODEL_A" \
            --model-b "$MODEL_B" \
            --baselines-config configs/baselines_by_dataset.yaml \
            --output "$OUTPUT_DIR" \
            --device "$DEVICE" \
            --k-timesteps "$K_TIMESTEPS" \
            --agg "$AGG" \
            --batch-size 256 \
            --save-pdf \
            ${MAX_SAMPLES_SMOKE:+--max-samples $MAX_SAMPLES_SMOKE} \
            2>&1 | tee "$LOG_DIR/step4_eval_${ds}_${split}.log"
        
        # Expected Output Files:
        check_file "$OUTPUT_JSON"
        check_file "${OUTPUT_DIR}/t_error_distributions_${split}.npz"
        check_file "${OUTPUT_DIR}/report_${ds}_${split}.pdf"
    done
done

###############################################################################
# STEP 5: Cross-Dataset Summary
###############################################################################

log_step 5 "Generate Cross-Dataset Summary"

# 汇总所有数据集、所有 split 的结果
python scripts/generate_cross_dataset_summary.py \
    --reports-dir "$REPORTS_DIR" \
    --output "${REPORTS_DIR}/summary_all_datasets.csv" \
    --datasets "${DATASETS[@]}" \
    --splits watermark_private eval_nonmember \
    2>&1 | tee "$LOG_DIR/step5_cross_dataset_summary.log"

# Expected Output Files:
echo ""
echo "Checking Step 5 outputs..."
check_file "${REPORTS_DIR}/summary_all_datasets.csv"
check_file "${REPORTS_DIR}/summary_all_datasets.json"
check_file "${REPORTS_DIR}/summary_report.md"

###############################################################################
# 完成
###############################################################################

log_step "DONE" "All experiments completed!"

echo ""
echo "========== FINAL SUMMARY =========="
echo "Logs:    $LOG_DIR/"
echo "Splits:  $SPLITS_DIR/"
echo "Models:  $RUNS_DIR/"
echo "Reports: $REPORTS_DIR/"
echo ""
echo "Key output files:"
for ds in "${DATASETS[@]}"; do
    echo "  [$ds]"
    echo "    - Model A: ${RUNS_DIR}/ddim_${ds}/main/best_for_mia.ckpt"
    echo "    - Model B: ${RUNS_DIR}/mmd_finetune/${ds}/model_b/ckpt_0500_ema.pt"
    echo "    - Report (watermark): ${REPORTS_DIR}/${ds}/baseline_comparison_${ds}_watermark_private.json"
    echo "    - Report (nonmember): ${REPORTS_DIR}/${ds}/baseline_comparison_${ds}_eval_nonmember.json"
done
echo ""
echo "Cross-dataset summary:"
echo "  - CSV:      ${REPORTS_DIR}/summary_all_datasets.csv"
echo "  - JSON:     ${REPORTS_DIR}/summary_all_datasets.json"
echo "  - Markdown: ${REPORTS_DIR}/summary_report.md"
echo "==================================="
