# Phase 12 汇报：Latent vs Pixel Space + Verification Protocol 修正

> 日期：2026-03-24
> 分支：fix/pixel-space-caption
> 模型：SD v1.4 + LoRA (A6, 1000 members, 80 epochs)

---

## 1. 背景与动机

Phase 11 完成了 3-point ownership verification，但遗留两个问题：

- **Issue A**：代码在 latent space (4×64×64) 计算 t-error，论文 Eq.6 定义的是 pixel space (3×512×512) 下的 ||x - x̂||²/(H×W×C)
- **Issue B**：推理时用空 prompt ("")，但 LoRA 训练时用的是 per-image COCO caption

Phase 12 的目标：系统比较 latent vs pixel 空间 + caption 条件对 membership detection 和 verification 的影响。

---

## 2. 实验设计

### 2.1 变量

| 因素 | 选项 |
|------|------|
| Error Space | Latent (4×64×64, sum) vs Pixel (3×512×512, mean) |
| Text Conditioning | Empty prompt ("") vs Per-image COCO caption |

### 2.2 实验矩阵

| 变体 | Space | Caption | 数据来源 |
|------|-------|---------|---------|
| V0 | Latent | Empty | 复用 Phase 11 CSVs |
| V1 | Latent | Per-image | 新跑 |
| V2 | Pixel | Per-image | 新跑 |

- 未跑 Pixel+Empty 组合（之前 quick eval 已证明是最差的，AUC=0.916）
- 每个变体跑 3 个模型：A6 (owner), B1 (domain-shift adversary), B2 (task-shift adversary)

### 2.3 评估数据

- 1000 members + 200 non-members（phase11_w_only.json）
- 串行执行，单 GPU (V100 32GB)
- V1 latent: ~12 min/model, V2 pixel: ~50 min/model（pixel 因 VAE decode 慢 4x）

---

## 3. Scoring 结果

### 3.1 Quick Eval（100 members + 100 non-members, A6 模型）

这是之前在 fix/pixel-space-caption 分支做的 2×2 完整消融：

| 变体 | Space | Caption | AUC | TPR@1% | Cohen's d | Time |
|------|-------|---------|-----|--------|-----------|------|
| V1 | Latent | Empty | 0.9811 | 0.4900 | 2.80 | 122s |
| **V2** | **Latent** | **Caption** | **0.9988** | **0.9100** | **3.21** | **124s** |
| V3 | Pixel | Empty | 0.9163 | 0.2700 | 1.71 | 485s |
| V4 | Pixel | Caption | 0.9696 | 0.5900 | 2.04 | 493s |

**结论**：Caption 是更大的提升因素（AUC +0.018），Pixel space 反而损害性能（AUC -0.065）。

### 3.2 Full-Scale（1000 + 200, 3 个模型）

| 变体 | 模型 | AUC | TPR@1% | Cohen's d | Mem Mean | Non-Mem Mean |
|------|------|-----|--------|-----------|----------|-------------|
| V0 (lat+empty) | A6 (owner) | 0.9871 | 0.7760 | 2.80 | -20.35 | +2.56 |
| V0 (lat+empty) | B1 (domain) | 0.9722 | 0.4920 | 2.49 | -17.79 | -0.90 |
| V0 (lat+empty) | B2 (task) | 0.9078 | 0.3730 | 1.79 | -2.00 | +13.64 |
| **V1 (lat+cap)** | **A6 (owner)** | **0.9956** | **0.8090** | **3.24** | **-29.19** | **+3.77** |
| V1 (lat+cap) | B1 (domain) | 0.9882 | 0.6140 | 2.85 | -23.75 | -0.93 |
| V1 (lat+cap) | B2 (task) | 0.9780 | 0.7560 | 2.62 | -13.82 | +10.68 |
| V2 (pix+cap) | A6 (owner) | 0.9590 | 0.3920 | 2.00 | -0.0003 | +0.0001 |
| V2 (pix+cap) | B1 (domain) | 0.9306 | 0.1900 | 1.72 | -0.0002 | +0.0001 |
| V2 (pix+cap) | B2 (task) | 0.8717 | 0.2240 | 1.44 | -0.0002 | +0.0002 |

### 3.3 Scoring 关键发现

**Caption 的效果**（V0 → V1，同为 latent）：
- AUC: 0.987 → 0.996 (+0.009)
- TPR@1%: 0.776 → 0.809 (+0.033)
- Cohen's d: 2.80 → 3.24 (+0.44)

**Pixel 的影响**（V1 → V2，同有 caption）：
- AUC: 0.996 → 0.959 (-0.037)
- TPR@1%: 0.809 → 0.392 (-0.417)
- Cohen's d: 3.24 → 2.00 (-1.24)

**对 adversary 的鲁棒性**（B2 task-shift 的 AUC 下降）：
- V0 (lat+empty): -0.079
- V1 (lat+cap): **-0.018**（最小下降）
- V2 (pix+cap): -0.087

---

## 4. Verification Protocol 分析（重要发现）

### 4.1 Algorithm 2 在 LoRA 模型上的适用性

严格按照论文 Algorithm 2 的定义实现 verification protocol：

| Criterion | 定义 | 通过条件 |
|-----------|------|---------|
| C1 Consistency | t-test(S_A, S_B) on W | p > 0.05 |
| C2 Separation | t-test(S_A, S_ref) on W + Cohen's d | p < 1e-6 AND \|d\| > 2.0 |
| C3 Ratio | mean(S_ref) / mean(S_A) | ratio > 5.0 |

其中 S_A = owner 的 raw t-error, S_B = suspect 的 raw t-error, S_ref = baseline (SD v1.4) 的 raw t-error。

### 4.2 Verification 结果

| 变体 | Adversary | C1 (p>0.05) | C2 (\|d\|>2.0) | C3 (ratio>5.0) | Verdict |
|------|-----------|-------------|----------------|----------------|---------|
| V0 lat+empty | B1 | **PASS** (p=0.874) | FAIL (0.057) | FAIL (1.011x) | REJECTED |
| V0 lat+empty | B2 | **PASS** (p=0.254) | FAIL (0.057) | FAIL (1.011x) | REJECTED |
| V1 lat+cap | B1 | **PASS** (p=0.734) | FAIL (0.082) | FAIL (1.016x) | REJECTED |
| V1 lat+cap | B2 | **PASS** (p=0.337) | FAIL (0.082) | FAIL (1.016x) | REJECTED |
| V2 pix+cap | B1 | **PASS** (p=0.926) | FAIL (0.018) | FAIL (1.012x) | REJECTED |
| V2 pix+cap | B2 | **PASS** (p=0.853) | FAIL (0.018) | FAIL (1.012x) | REJECTED |

### 4.3 根因分析

LoRA fine-tuning 只修改了 UNet cross-attention 的 q/k/v/out 权重（rank=64, ~33.5M params / 860M total = 3.9%），对 base model 的重建误差只有 **~1% 的扰动**：

```
Raw t-error (latent+caption, A6 on W):
  S_A  (owner):    mean = 1864.95
  S_ref (baseline): mean = 1894.15
  ratio = 1894.15 / 1864.95 = 1.016x
```

论文的 Algorithm 2 是为 **from-scratch 训练的 DDIM** 设计的，那种情况下：
- Owner model 完全在 W 上训练 → raw t-error 极低
- Baseline 从未见过 W → raw t-error 很高
- Ratio 轻松达到 19x+

但 LoRA derivative model 的 raw t-error 被 base model 的巨大 variance (std~358) 淹没，membership signal 只存在于 delta (score_tgt - score_ref) 里。

---

## 5. 总结与下一步

### 5.1 确认的结论

| 结论 | 证据 |
|------|------|
| **Latent+Caption 是最优 scoring 配置** | AUC=0.996, d=3.24, TPR@1%=0.809 |
| **Caption 比 space 选择更重要** | Caption: +0.009 AUC. Pixel: -0.037 AUC |
| **Pixel space 有害** | VAE decode 引入有损噪声，稀释 membership signal |
| **Caption 提升 adversary 鲁棒性** | B2 降幅: -0.018 (cap) vs -0.079 (empty) |

### 5.2 待解决的问题

| 问题 | 说明 |
|------|------|
| **Algorithm 2 不适用于 derivative models** | Raw score ratio ~1.01x，远达不到 5.0 阈值。需要为 LoRA/adapter 模型设计新的 verification protocol |
| **Latent 的 sum vs mean 不一致** | 代码用 sum(4×64×64)，论文定义 mean/(H×W×C)。对 AUC 无影响（只是常数缩放），但导致 latent 和 pixel 的 score 数值不可直接比较 |
| **Paper 需要新增 discussion** | 讨论为什么 Algorithm 2 对 derivative model 失效，以及 delta-based verification 的合理性和重新校准方案 |

### 5.3 建议的方向

1. **设计 Delta-based Verification Protocol**：用 delta = score_tgt - score_ref 替代 raw score，重新校准 C2/C3 阈值
2. **归一化修正**：统一 latent 和 pixel 的归一化方式（都用 mean/(H×W×C)），确保跨空间可比
3. **在论文中明确区分**：from-scratch model（原 Algorithm 2 适用）vs derivative model（需要 adapted protocol）
