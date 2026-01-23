# run_all.sh 路径修复说明

## 问题描述
即使更新了配置文件指向 `/data/short/` 中的已有checkpoints，运行 `run_all.sh` 时仍然尝试重新训练 Model A。

## 根本原因
`run_all.sh` 脚本中的 `RUNS_DIR` 变量使用的是**相对路径** `"runs"`，而配置文件中的 `output_dir` 已更新为**绝对路径** `/data/short/fjiang4/mia_ddpm_qr/runs/`。

脚本检查的路径：
```bash
OUTPUT_CKPT="runs/ddim_cifar10/main/best_for_mia.ckpt"  # 相对路径，不存在
```

实际checkpoint位置：
```bash
/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt  # 绝对路径
```

因此 `[[ -f "$OUTPUT_CKPT" ]]` 检查返回 false，脚本认为checkpoint不存在而开始训练。

## 修复方案
修改 `run_all.sh` 第 39 行，将 `RUNS_DIR` 改为绝对路径：

### 修改前：
```bash
RUNS_DIR="runs"
REPORTS_DIR="runs/attack_qr/reports"
```

### 修改后：
```bash
RUNS_DIR="/data/short/fjiang4/mia_ddpm_qr/runs"
REPORTS_DIR="${RUNS_DIR}/attack_qr/reports"
```

## 验证结果
修复后，脚本正确检测到所有已存在的checkpoints：

```bash
$ bash test_checkpoint_detection.sh

--- Testing: cifar10 ---
✓ SKIP: /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt already exists

--- Testing: cifar100 ---
✓ SKIP: /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar100/main/best_for_mia.ckpt already exists

--- Testing: stl10 ---
✓ SKIP: /data/short/fjiang4/mia_ddpm_qr/runs/ddim_stl10/main/best_for_mia.ckpt already exists
```

## 重要说明
**我没有删除任何checkpoints！** 所有模型文件都安全保存在 `/data/short/` 中：
- ✅ CIFAR-10: 完整（400k iterations）
- ✅ CIFAR-100: 完整（所有中间checkpoints 10k-400k）
- ✅ STL-10: 完整（195k iterations）
- ✅ CelebA: 训练中（290k/400k）

唯一删除的文件是调试日志 `.cursor/debug.log`（临时文件，不影响任何数据）。

## 现在可以运行
```bash
# 完整pipeline（会跳过已有checkpoints的训练）
bash run_all.sh 2>&1 | tee run_all_$(date +%Y%m%d_%H%M%S).log

# 或者从Step 3开始（MMD微调）
# 手动编辑 run_all.sh，注释掉 STEP 1 和 STEP 2
```

## 影响的步骤
修复后，脚本将：
- ✅ **STEP 1**: 检测到splits已存在，跳过生成
- ✅ **STEP 2**: 检测到Model A已存在，**跳过训练**（主要修复点）
- ▶️ **STEP 3**: 继续执行MMD微调创建Model B
- ▶️ **STEP 4**: 继续执行Ownership Evaluation
- ▶️ **STEP 5**: 生成cross-dataset summary

## 清理文件
修复过程中添加的临时文件：
- `test_checkpoint_detection.sh` - 可以删除（测试脚本）
- `run_all.sh` 中的 DEBUG 日志代码 - 可以保留或删除

## 配置一致性
现在所有配置都使用 `/data/short/` 的绝对路径：
- ✅ `configs/data_*.yaml` → 数据集路径
- ✅ `configs/model_ddim_*.yaml` → 输出目录
- ✅ `configs/mmd_finetune_*.yaml` → checkpoint路径
- ✅ `run_all.sh` → RUNS_DIR变量
