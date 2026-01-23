# 使用已有Checkpoints和数据集的指南

## 概述
本项目已配置为使用 `/data/short/fjiang4/mia_ddpm_qr/` 中已经训练好的checkpoints和数据集，**无需重新训练**。

## ✅ 验证配置
首先运行验证脚本确认所有路径正确：
```bash
python3 verify_config_paths.py
```

## 📊 当前状态

### 可立即使用的数据集
| 数据集 | 状态 | Checkpoint | Iterations |
|--------|------|------------|-----------|
| CIFAR-10 | ✅ Ready | `best_for_mia.ckpt` | 400,000 |
| CIFAR-100 | ✅ Ready | `best_for_mia.ckpt` | 400,000 |
| STL-10 | ✅ Ready | `best_for_mia.ckpt` | 195,000 |
| CelebA | ✅ Ready | `ckpt_400000/ema.ckpt` | 400,000 ✅ |

## 🚀 快速开始

### 1. 跳过训练阶段（推荐）
由于checkpoints已存在，可以直接跳到evaluation步骤：

```bash
# 构建 t-error pairs (需要从已有checkpoints生成)
python scripts/2_build_t_error_pairs.py --dataset cifar10

# 训练 Quantile Regression 攻击模型
python scripts/3_train_qr_bagging.py --dataset cifar10

# 评估攻击性能
python scripts/4_eval_attack.py --dataset cifar10
```

### 2. MMD微调（Theft Simulation）
使用已有的Model A创建Model B：

```bash
# CIFAR-10
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar10.yaml

# CIFAR-100
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar100.yaml

# STL-10
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_stl10.yaml
```

### 3. Checkpoint选择（可选）
如果需要重新选择最佳checkpoint：

```bash
python src/ddpm_ddim/select_checkpoints.py \
    --model-config configs/model_ddim_cifar10.yaml \
    --data-config configs/data_cifar10.yaml \
    --run-dir /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main
```

## 📁 路径结构

### 数据集位置
```
/data/short/fjiang4/mia_ddpm_qr/data/
├── cifar10/          # CIFAR-10 数据集
├── cifar100/         # CIFAR-100 数据集
├── celeba/           # CelebA 数据集
├── stl10/            # STL-10 数据集
└── splits/           # 数据集划分文件
    ├── cifar10/
    │   ├── watermark_private.json
    │   ├── eval_nonmember.json
    │   ├── member_train.json
    │   └── manifest.json
    ├── cifar100/
    ├── celeba/
    └── stl10/
```

### Checkpoint位置
```
/data/short/fjiang4/mia_ddpm_qr/runs/
├── ddim_cifar10/main/
│   └── best_for_mia.ckpt → experiments/production/model_a/best_for_mia.ckpt
├── ddim_cifar100/main/
│   └── best_for_mia.ckpt → ckpt_400000/ema.ckpt
├── ddim_celeba/main/
│   ├── ckpt_400000/ema.ckpt  ✅ 训练完成！
│   └── best_for_mia.ckpt (待选择后创建软链接)
└── ddim_stl10/main/
    └── best_for_mia.ckpt → ckpt_195000/ema.ckpt
```

## ⚙️ 配置文件说明

### 已更新的配置
所有配置文件已更新为绝对路径：

**数据配置** (`configs/data_*.yaml`):
- `dataset.root`: 指向 `/data/short/.../data/{dataset}/`
- `splits.paths.*`: 指向 `/data/short/.../data/splits/{dataset}/`

**模型配置** (`configs/model_ddim_*.yaml`):
- `experiment.output_dir`: 指向 `/data/short/.../runs/ddim_{dataset}/`

**微调配置** (`configs/mmd_finetune_*.yaml`):
- `base.checkpoint`: 指向 `/data/short/.../runs/ddim_{dataset}/main/best_for_mia.ckpt`

## 🔍 验证数据加载
测试数据集是否能正确加载：

```bash
python3 -c "
import yaml
import torch
from pathlib import Path

# 读取配置
with open('configs/data_cifar10.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

print('Dataset root:', cfg['dataset']['root'])
print('Root exists:', Path(cfg['dataset']['root']).exists())

# 尝试加载数据集
from torchvision import datasets, transforms
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(
    root=cfg['dataset']['root'],
    train=True,
    download=False,
    transform=transform
)
print(f'Dataset loaded: {len(dataset)} samples')
print('✅ CIFAR-10 数据集验证成功！')
"
```

## 🎯 工作流程示例

### 完整的MIA实验流程（CIFAR-10）
```bash
# Step 1: 验证配置
python3 verify_config_paths.py

# Step 2: 构建 t-error pairs（使用已有checkpoint）
python scripts/2_build_t_error_pairs.py \
    --dataset cifar10 \
    --checkpoint /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt

# Step 3: 训练 QR 攻击模型
python scripts/3_train_qr_bagging.py --dataset cifar10

# Step 4: 评估攻击
python scripts/4_eval_attack.py --dataset cifar10

# Step 5: 分析结果
python scripts/analyze_aggregation_comparison.py --dataset cifar10
```

### Model Theft Simulation
```bash
# 使用 Model A 创建 Model B
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar10.yaml

# 比较 Model A 和 Model B 的 t-error
python tools/compute_scores.py \
    --model-a /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --model-b runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt
```

## ⚠️ 注意事项

### CelebA Checkpoint选择
CelebA 模型已完成训练（400k iterations）。下一步：
1. **运行checkpoint选择脚本**（如果尚未完成）：
   ```bash
   python src/ddpm_ddim/select_checkpoints.py \
       --model-config configs/model_ddim_celeba.yaml \
       --data-config configs/data_celeba.yaml \
       --run-dir /data/short/fjiang4/mia_ddpm_qr/runs/ddim_celeba/main
   ```
2. 脚本会自动创建 `best_for_mia.ckpt` 软链接
3. 然后可以运行所有CelebA的实验

### 训练新模型（不推荐）
如果确实需要重新训练（不推荐），可以：
1. 修改配置中的 `output_dir` 为新路径（避免覆盖现有checkpoints）
2. 运行训练脚本：
```bash
python src/ddpm_ddim/train_ddim.py \
    --config configs/model_ddim_cifar10.yaml \
    --mode main
```

但这会花费大量时间（CIFAR-10约30小时）。

## 📖 相关文档
- [完整更新说明](CONFIGURATION_UPDATE_SUMMARY.md) - 详细的配置更改记录
- [项目文档](docs/README.md) - 完整的项目文档
- [方法论](docs/methodology/) - 研究方法说明

## 🐛 故障排除

### 问题：找不到数据集
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'data/cifar10'
```
**解决**：运行 `python3 verify_config_paths.py` 确认配置已更新

### 问题：找不到checkpoint
```bash
FileNotFoundError: runs/ddim_cifar10/main/best_for_mia.ckpt
```
**解决**：确认配置使用绝对路径 `/data/short/fjiang4/mia_ddpm_qr/...`

### 问题：CUDA内存不足
```bash
RuntimeError: CUDA out of memory
```
**解决**：减小配置中的 `batch_size`

## ✨ 优势总结
- ✅ **节省时间**：跳过30小时+的训练过程
- ✅ **节省资源**：不占用GPU进行重复训练
- ✅ **一致性**：所有实验使用相同的训练checkpoints
- ✅ **可复现**：使用固定的400k iteration checkpoints
- ✅ **即时开始**：立即进入attack evaluation阶段
