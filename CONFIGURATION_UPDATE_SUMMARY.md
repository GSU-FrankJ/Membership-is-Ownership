# Configuration Update Summary

## 问题描述
项目配置原先使用相对路径指向本地 `data/` 目录，但实际的数据集和checkpoints已经存在于 `/data/short/fjiang4/mia_ddpm_qr/` 中，因此不需要重新训练。

## 解决方案
更新所有配置文件，使其指向 `/data/short/` 中已有的数据集和checkpoints。

## 已更新的配置文件

### 数据集配置 (Data Configs)
所有数据集配置已更新为使用绝对路径：

1. **configs/data_cifar10.yaml**
   - `dataset.root`: `data/cifar10` → `/data/short/fjiang4/mia_ddpm_qr/data/cifar10`
   - `splits.paths.*`: 所有路径更新为 `/data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/*`
   - `download`: 保持 `false`

2. **configs/data_cifar100.yaml**
   - `dataset.root`: `data/cifar100` → `/data/short/fjiang4/mia_ddpm_qr/data/cifar100`
   - `splits.paths.*`: 所有路径更新为 `/data/short/fjiang4/mia_ddpm_qr/data/splits/cifar100/*`
   - `download`: 保持 `false`

3. **configs/data_celeba.yaml**
   - `dataset.root`: `data/celeba` → `/data/short/fjiang4/mia_ddpm_qr/data/celeba`
   - `splits.paths.*`: 所有路径更新为 `/data/short/fjiang4/mia_ddpm_qr/data/splits/celeba/*`
   - `download`: 保持 `false`

4. **configs/data_stl10.yaml**
   - `dataset.root`: `data/stl10` → `/data/short/fjiang4/mia_ddpm_qr/data/stl10`
   - `splits.paths.*`: 所有路径更新为 `/data/short/fjiang4/mia_ddpm_qr/data/splits/stl10/*`
   - `download`: 保持 `false`

### 模型配置 (Model Configs)
模型输出目录更新为使用已有的checkpoints位置：

5. **configs/model_ddim_cifar10.yaml**
   - `experiment.output_dir`: `runs/ddim_cifar10` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10`

6. **configs/model_ddim_cifar100.yaml**
   - `experiment.output_dir`: `runs/ddim_cifar100` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar100`

7. **configs/model_ddim_celeba.yaml**
   - `experiment.output_dir`: `runs/ddim_celeba` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_celeba`

8. **configs/model_ddim_stl10.yaml**
   - `experiment.output_dir`: `runs/ddim_stl10` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_stl10`

### MMD微调配置 (MMD Finetune Configs)
微调配置的checkpoint路径更新为已有的best checkpoints：

9. **configs/mmd_finetune_cifar10.yaml**
   - `base.checkpoint`: `runs/ddim_cifar10/main/best_for_mia.ckpt` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt`

10. **configs/mmd_finetune_cifar100.yaml**
    - `base.checkpoint`: `runs/ddim_cifar100/main/best_for_mia.ckpt` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar100/main/best_for_mia.ckpt`

11. **configs/mmd_finetune_celeba.yaml**
    - `base.checkpoint`: `runs/ddim_celeba/main/best_for_mia.ckpt` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_celeba/main/best_for_mia.ckpt`

12. **configs/mmd_finetune_stl10.yaml**
    - `base.checkpoint`: `runs/ddim_stl10/main/best_for_mia.ckpt` → `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_stl10/main/best_for_mia.ckpt`

## 验证状态

### 已验证存在的Checkpoints ✅
- `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt` (软链接到experiments)
- `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar100/main/best_for_mia.ckpt` → `ckpt_400000/ema.ckpt`
- `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_stl10/main/best_for_mia.ckpt` → `ckpt_195000/ema.ckpt`
- `/data/short/fjiang4/mia_ddpm_qr/runs/ddim_celeba/main/ckpt_400000/ema.ckpt` ✅ **训练完成！** (400k iterations)

### 数据集状态 ✅
所有数据集已存在于 `/data/short/fjiang4/mia_ddpm_qr/data/`:
- cifar10 ✅
- cifar100 ✅
- celeba ✅
- stl10 ✅
- splits/ (所有split文件) ✅

## 后续步骤

### ✅ 所有数据集就绪 (CIFAR-10, CIFAR-100, STL-10, CelebA)
**所有四个数据集**现在都可以直接开始以下工作：
```bash
# 1. 为CelebA选择最佳checkpoint（如果尚未完成）
python src/ddpm_ddim/select_checkpoints.py \
    --model-config configs/model_ddim_celeba.yaml \
    --data-config configs/data_celeba.yaml \
    --run-dir /data/short/fjiang4/mia_ddpm_qr/runs/ddim_celeba/main

# 2. 运行完整的evaluation pipeline（所有数据集）
bash run_all.sh 2>&1 | tee run_all_$(date +%Y%m%d_%H%M%S).log

# 或单独运行每个数据集
python scripts/eval_ownership.py --dataset cifar10
python scripts/eval_ownership.py --dataset cifar100
python scripts/eval_ownership.py --dataset stl10
python scripts/eval_ownership.py --dataset celeba

# 3. MMD微调（theft simulation）- 所有数据集
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar10.yaml
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_cifar100.yaml
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_stl10.yaml
python scripts/finetune_mmd_ddm.py --config configs/mmd_finetune_celeba.yaml  # 现在可用！
```

## 配置文件对比

### 之前（相对路径）
```yaml
dataset:
  root: data/cifar10
splits:
  paths:
    watermark_private: data/splits/cifar10/watermark_private.json
```

### 现在（绝对路径）
```yaml
dataset:
  root: /data/short/fjiang4/mia_ddpm_qr/data/cifar10
splits:
  paths:
    watermark_private: /data/short/fjiang4/mia_ddpm_qr/data/splits/cifar10/watermark_private.json
```

## 优势
1. ✅ **无需重新训练** - 直接使用已有的400k iterations checkpoints
2. ✅ **无需重新下载数据集** - 所有数据集已存在
3. ✅ **无需重新生成splits** - split文件已存在
4. ✅ **节省时间和计算资源** - CIFAR-10训练需要~30小时，现在跳过
5. ✅ **保持一致性** - 使用相同的训练checkpoints进行所有实验

## 注意事项
- MMD微调的 `output_dir` 仍使用相对路径 `runs/mmd_finetune/`，因为这是新生成的模型
- `configs/mmd_finetune_cifar10_ddim10.yaml` 使用不同的路径结构（experiments/production/），暂未修改
- `configs/model_ddim.yaml` 保持不变（用于生产环境配置）
