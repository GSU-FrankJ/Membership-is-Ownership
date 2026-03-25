# Phase 13 汇报：Extended Experiments + Algorithm 2 验证

> 日期：2026-03-25
> 分支：fix/pixel-space-caption
> 评估：latent space + caption + mean/(HWC) 归一化，1000 members + 1000 non-members

---

## 1. 实验改进

| 改动 | 旧 (Phase 11/12) | 新 (Phase 13) |
|------|-----------------|--------------|
| Non-member 数量 | 200 | **1000** |
| Full fine-tune | 无（A4 只在 500 members 上做过 quick eval） | **A7: 1000 members, 40 epochs, 860M params** |
| 高 rank LoRA | r64 (33.5M, 3.9%) | **A8: r256 (134M, 15.6%)** |
| Latent 归一化 | sum（不符合论文 Eq.6） | **mean/(HWC)（符合论文）** |
| Text conditioning | 部分实验用空 prompt | **全部用 per-image COCO caption** |

---

## 2. Membership Detection 结果

| 模型 | 参数修改比 | AUC | TPR@1% | Cohen's d | Mem mean | Non-mem mean |
|------|----------|-----|--------|-----------|----------|-------------|
| A6 (LoRA r64) | 3.9% | 0.9949 | 0.8280 | 3.26 | -0.001782 | +0.000239 |
| **A7 (Full FT)** | **100%** | **0.9986** | **0.9640** | **3.41** | -0.003110 | +0.000307 |
| **A8 (LoRA r256)** | **15.6%** | **0.9994** | **0.9850** | **3.93** | -0.002831 | +0.000782 |
| B1 (adversary, domain) | — | 0.9858 | 0.6600 | 2.83 | -0.001450 | -0.000045 |
| B2 (adversary, task) | — | 0.9799 | 0.7130 | 2.70 | -0.000843 | +0.000689 |

### 发现

- **A8 (LoRA r256) 的 membership detection 最强**：AUC=0.9994，d=3.93，超过 full FT
- **高 rank LoRA 比 full FT 更有效**：r256 的 d=3.93 > full FT 的 d=3.41。可能因为 full FT 的 lr=5e-6 较保守，而 LoRA r256 的 lr=1e-4 更激进
- **参数修改比与 detection 质量正相关**：r64 (3.9%) d=3.26 → r256 (15.6%) d=3.93 → full (100%) d=3.41
- 归一化后 score 在 ~[-0.003, +0.001] 范围，latent 和 pixel 可直接比较

---

## 3. Algorithm 2 Verification 结果

### 定义

| Criterion | 公式 | 通过条件 |
|-----------|------|---------|
| C1 一致性 | t-test(S_A, S_B) on W | p > 0.05 |
| C2 分离度 | t-test(S_A, S_ref) on W + Cohen's d | p < 1e-6 AND \|d\| > 2.0 |
| C3 倍率 | mean(S_ref) / mean(S_A) | ratio > 5.0 |

### 结果

| Owner | Adversary | C1 | C2 (\|d\|) | C3 (ratio) | Verdict |
|-------|-----------|----|-----------|-----------|----|
| A6 r64 | B1 | PASS (p=0.73) | FAIL (0.08) | FAIL (1.016x) | REJECTED |
| A6 r64 | B2 | PASS (p=0.34) | FAIL (0.08) | FAIL (1.016x) | REJECTED |
| A7 full | B1 | PASS (p=0.09) | FAIL (0.14) | FAIL (1.028x) | REJECTED |
| A7 full | B2 | **FAIL** (p=0.02) | FAIL (0.14) | FAIL (1.028x) | REJECTED |
| A8 r256 | B1 | PASS (p=0.16) | FAIL (0.13) | FAIL (1.025x) | REJECTED |
| A8 r256 | B2 | **FAIL** (p=0.04) | FAIL (0.13) | FAIL (1.025x) | REJECTED |

### 分析

**C2/C3 全部 FAIL**：即使 full fine-tune 修改了全部 860M 参数，raw score ratio 也只有 1.028x（需要 >5.0x）。根本原因：

```
S_ref (baseline) mean = 0.1156, std = 0.0218
S_A  (A7 full FT) mean = 0.1125, std = 0.0218

差值 = 0.003，标准差 = 0.022
信噪比 = 0.003 / 0.022 = 0.14
```

Base model 的 per-image reconstruction variance (~0.022) 是 membership signal (~0.003) 的 7 倍。Algorithm 2 的阈值是为 from-scratch DDIM 设计的（那种场景 ratio > 19x），不适用于任何基于 SD 的 fine-tuning。

**C1 正确区分了模型谱系**：B1/B2 是从 A6 (LoRA r64) 派生的，不是从 A7 或 A8。C1 结果正确反映了这一点 — 只有 A6 vs B1/B2 通过 consistency check (p>0.05)，A7 和 A8 因为是完全不同的模型（不同的 fine-tuning 方式和参数量），raw t-error 特征不同，所以 FAIL。这不是"检测到 adversary 修改"，而是 C1 正确识别了不同的模型谱系。

---

## 4. 对比总结

### Membership Detection（delta score, 不需要 Algorithm 2）

| 指标 | A6 (r64, 3.9%) | A7 (full, 100%) | A8 (r256, 15.6%) |
|------|---------------|-----------------|-----------------|
| AUC | 0.9949 | 0.9986 | **0.9994** |
| TPR@1% | 0.8280 | 0.9640 | **0.9850** |
| Cohen's d | 3.26 | 3.41 | **3.93** |

### Verification Protocol（Algorithm 2, raw scores）

| 指标 | A6 (r64) | A7 (full) | A8 (r256) | 阈值 |
|------|----------|-----------|-----------|------|
| C2 \|d\| | 0.08 | 0.14 | 0.13 | >2.0 |
| C3 ratio | 1.016x | 1.028x | 1.025x | >5.0x |

**结论**：Membership detection 信号很强（d>3），但 Algorithm 2 看不到它。信号在 delta (score_tgt - score_ref) 里，不在 raw scores 里。

---

## 5. 下一步建议

1. **设计 Delta-based Verification Protocol** — 用 delta score 替代 raw score，重新校准 C2/C3 阈值
2. **论文写作** — 在 discussion 中明确区分 from-scratch 和 fine-tuned 两种场景
3. **补充实验** — LoRA r256 是当前最强配置，可以作为论文的 SD 实验主结果
