# Phase 07: Broader Post-Theft Robustness Evaluation

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Address the reviewer's concern that (1) MiO is an extension of existing paradigms [5,6], and (2) the robustness evaluation against post-theft attacks is too narrow (only MMD fine-tuning).

---

## 1. Reviewer Question

> The concept of leveraging membership signals from a subset of training data for model ownership protection has been previously explored [5,6]. This paper can be viewed as an extension of this paradigm to the domain of diffusion models, with specialized designs based on diffusion-specific membership inference attacks. While the feasibility of this approach is demonstrated, the empirical evaluation remains insufficient to fully establish its effectiveness and superiority. Specifically, the assessment of robustness against post-theft is too narrow, as it only considers MMD fine-tuning. A more comprehensive evaluation involving a broader range of model modification attacks is necessary.

## 2. Current Paper Position

### What the paper already tests (reviewer may have missed pruning)

| Attack | Tested? | Result | Location |
|--------|---------|--------|----------|
| MMD fine-tuning (500 iter) | Yes | PASS (d=24.18, ratio=24.7x) | Table 6, Section 5.2 |
| 30% structured L1 pruning | Yes | PASS (FID 10.7, criteria hold) | Table 6, Section 5.2 |
| Retroactive claim fabrication | Yes | REJECTED | Appendix retroactive |
| Domain-shift FT (SD v1.4 LoRA) | Yes (unpublished) | PASS (AUC=0.972, d=2.21) | SD experiment Phase 11 |
| Task-shift FT (SD v1.4 LoRA) | Yes (unpublished) | PARTIAL (AUC=0.908, d=1.73) | SD experiment Phase 11 |

**Note**: The reviewer says "only considers MMD fine-tuning" — the paper ALSO tests pruning (Table 6). The reviewer may have overlooked this or considers pruning insufficient.

### What is NOT tested

| Attack | Practical relevance | Difficulty to add |
|--------|-------------------|------------------|
| Standard SGD fine-tuning (vanilla, no MMD) | Very high — most common attack | Low (~2h GPU) |
| LoRA / PEFT fine-tuning | Very high — dominant method in SD community | Low (infra exists from SD experiment) |
| Quantization (INT8, INT4) | High — standard deployment optimization | Very low (minutes, no training) |
| Weight perturbation (Gaussian noise) | Medium — theoretical attack | Very low (minutes, no training) |
| Higher pruning rates (50%, 70%) | Medium — extends existing test | Very low (minutes) |
| Model merging / weight averaging | Medium — popular in SD community | Low (~1h) |
| Distillation | High — but principled boundary (Phase 01) | High (~16h GPU) |
| Targeted memorization erasure | Theoretical — adversary specifically removes MiO signal | Out of scope (line 886) |

## 3. Analysis

### The novelty concern ([5,6])

The reviewer cites [5,6] as prior work using membership signals for ownership. These likely refer to works applying MIA for IP verification on classifiers (similar to Dataset Inference discussed in Phase 03). Our response (consistent with Phase 03):

- **Acknowledge the paradigm**: MiO belongs to the population-level MIA-for-ownership family
- **Emphasize non-trivial contributions**: t-error scoring with Q25 multi-timestep aggregation, Gaussian QR for arbitrary FPR, three-point verification protocol — none of these exist in [5,6] and all are necessitated by the diffusion model setting
- Cross-reference Phase 03 rebuttal for full argument

### The robustness concern — honest assessment

**The reviewer has a valid point.** The paper tests two attacks (MMD-FT + pruning) on the DDIM experiments. This is narrow. However:

1. **Quick wins exist**: Quantization, weight perturbation, and higher pruning rates require zero training and can be tested in minutes. Standard SGD fine-tuning is also straightforward.

2. **SD experiments already extend coverage**: Phase 11 tests domain-shift and task-shift fine-tuning on SD v1.4 — these results should be integrated into the paper.

3. **Distillation is a principled boundary**: Phase 01 argues this approximates full retraining and is shared with ALL weight-based methods.

### Expected outcomes for untested attacks

| Attack | Expected result | Reasoning |
|--------|----------------|-----------|
| **SGD fine-tuning** (vanilla) | PASS | Weaker than MMD-FT (no explicit distribution matching); if MMD-FT passes, SGD should too |
| **LoRA fine-tuning** | PASS | SD Phase 11 shows domain-shift LoRA passes (d=2.21); modifies fewer parameters than full FT |
| **INT8 quantization** | PASS | Minimal weight change (~0.1% relative error); t-error is continuous, not sensitive to small perturbations |
| **INT4 quantization** | Likely PASS | Larger perturbation but model still generates reasonable images; memorization likely preserved |
| **Gaussian noise (σ=0.01)** | PASS | Small perturbation to weights; analogous to quantization |
| **Gaussian noise (σ=0.1)** | Degraded | Model quality degrades significantly; may still pass with reduced d |
| **50% pruning** | PASS | 30% passes; 50% likely still passes with reduced d |
| **70%+ pruning** | Likely FAIL | Severe capacity loss; model quality degrades substantially |
| **Model merging** | PASS | Averaging with another model dilutes memorization but doesn't erase it |
| **Distillation** | Likely FAIL | Student never sees W directly (Phase 01 analysis) |

---

## 4. Rebuttal Draft

---

**Response to Reviewer — Novelty and Robustness Evaluation**

**On novelty.** We acknowledge and appreciate the connection to prior work using membership signals for ownership verification in discriminative models [5,6]. As discussed in our updated Related Work (see our response on Dataset Inference), MiO belongs to this broader paradigm. However, extending it to diffusion models requires non-trivial, domain-specific designs: (1) a new membership signal based on single-step reconstruction error aggregated across timesteps (Q25 t-error), as diffusion models lack the classification features used in [5,6]; (2) Gaussian Quantile Regression providing closed-form FPR thresholds without retraining; and (3) a three-point verification protocol (consistency, separation, ratio) that provides interpretable, auditable evidence suitable for IP disputes.

**On robustness evaluation.** We note that the paper evaluates two attacks, not one: MMD fine-tuning (Table 6) AND 30% structured L1 pruning (Table 6). Both pass all verification criteria. We acknowledge, however, that a broader attack suite would strengthen the evaluation.

We have extended the robustness evaluation to include the following additional attacks on CIFAR-10:

| Attack | d (clean: 24.1) | Ratio (clean: 24.5x) | Verified? |
|--------|-----|-------|-----------|
| MMD fine-tuning (500 iter) | 24.2 | 24.7x | PASS |
| Structured L1 pruning 30% | [TBD] | [TBD] | PASS |
| Structured L1 pruning 50% | [TBD] | [TBD] | [TBD] |
| Standard SGD fine-tuning (500 iter) | [TBD] | [TBD] | [TBD] |
| INT8 quantization | [TBD] | [TBD] | [TBD] |
| Gaussian weight noise (σ=0.01) | [TBD] | [TBD] | [TBD] |
| LoRA fine-tuning (rank 4, 500 iter) | [TBD] | [TBD] | [TBD] |

[Numbers to be filled after experiments run.]

Additionally, our Stable Diffusion experiments (new Appendix section) evaluate domain-shift fine-tuning (LoRA, 2000 steps on disjoint data: d=2.21, VERIFIED) and task-shift fine-tuning (LoRA, 2000 steps on synthetic data: d=1.73, partial signal retention). We discuss distillation as a principled boundary shared with all weight-based methods [see our response on distillation attacks].

---

## 5. Proposed Experiments (require professor approval)

### Experiment Set A: Zero-training attacks (minutes each, CIFAR-10)

```bash
# All use existing Model A checkpoint, just modify weights and re-eval
conda activate mio

# A1: INT8 quantization
python scripts/eval_ownership.py --dataset cifar10 \
  --model-a <path> --quantize int8

# A2: INT4 quantization
python scripts/eval_ownership.py --dataset cifar10 \
  --model-a <path> --quantize int4

# A3: Gaussian weight noise (multiple sigma levels)
python scripts/eval_ownership.py --dataset cifar10 \
  --model-a <path> --noise-sigma 0.001 0.005 0.01 0.05 0.1

# A4: Higher pruning rates
python scripts/eval_ownership.py --dataset cifar10 \
  --model-a <path> --prune 0.3 0.5 0.7

# Total: ~30 min on 1 GPU (just inference, no training)
```

**Note**: `eval_ownership.py` may need small modifications to support `--quantize` and `--noise-sigma` flags. These are straightforward (~50 lines each).

### Experiment Set B: Fine-tuning variants (~2-4h each, CIFAR-10)

```bash
# B1: Standard SGD fine-tuning (vanilla, no MMD)
python scripts/finetune_vanilla.py --config configs/vanilla_ft_cifar10.yaml \
  --iterations 500 --lr 5e-6

# B2: LoRA fine-tuning on CIFAR-10
# (needs LoRA adapter implementation for DDIM UNet — moderate effort)

# B3: Fine-tuning with different iteration counts (100, 500, 1000, 2000)
# Shows degradation curve

# Total: ~8h on 1 GPU
```

### Experiment Set C: Model merging (~1h, CIFAR-10)

```bash
# C1: Average Model A with public baseline (50/50 weight average)
python scripts/merge_models.py --model-a <path> --model-b <baseline> --alpha 0.5
python scripts/eval_ownership.py --dataset cifar10 --model-a <merged>
```

### Total compute estimate

| Set | GPU time | Scripts needed | Priority |
|-----|---------|----------------|----------|
| A (quantization, noise, pruning) | ~30 min | Minor eval_ownership.py mods | **HIGH** — max impact, min effort |
| B (SGD, LoRA fine-tuning) | ~8h | New finetune_vanilla.py + LoRA adapter | MEDIUM |
| C (model merging) | ~1h | New merge_models.py | LOW |
| **Total** | **~10h** | | |

---

## 6. Proposed Paper Changes

### Change 1: Expanded robustness table (replace Table 6)

```latex
\begin{table}[t]
\caption{Robustness under post-theft model modifications (CIFAR-10).
$|d|$ and ratio are computed against public baselines.
\cmark: all three verification criteria pass.
\xmark: at least one criterion fails.}
\label{tab:robustness_extended}
\centering\small
\begin{tabular}{@{}lccc@{}}
\toprule
Attack & FID & $|d|$ / Ratio & Verified? \\
\midrule
Clean (no attack) & 6.21 & 24.1 / 24.5x & \cmark \\
\addlinespace
\multicolumn{4}{@{}l}{\textit{Fine-tuning attacks}} \\
MMD-FT (500 iter) & 6.58 & 24.2 / 24.7x & \cmark \\
SGD-FT (500 iter) & [TBD] & [TBD] & [TBD] \\
\addlinespace
\multicolumn{4}{@{}l}{\textit{Compression attacks}} \\
Pruning 30\% & 10.7 & [TBD] & \cmark \\
Pruning 50\% & [TBD] & [TBD] & [TBD] \\
INT8 quantization & [TBD] & [TBD] & [TBD] \\
\addlinespace
\multicolumn{4}{@{}l}{\textit{Perturbation attacks}} \\
Gaussian noise $\sigma{=}0.01$ & [TBD] & [TBD] & [TBD] \\
\bottomrule
\end{tabular}
\end{table}
```

### Change 2: Add one sentence to Section 5.2 acknowledging paradigm lineage

```latex
Our approach shares the population-level verification philosophy
with prior work on membership-based ownership for
classifiers~\citep{[5],[6],maini2021dataset}; the key technical
challenges specific to diffusion models---multi-timestep scoring,
noise-schedule-aware aggregation, and closed-form FPR
calibration---are detailed in
Sections~\ref{sec:t_error}--\ref{sec:criteria}.
```

> **NOTE**: Need to identify what [5] and [6] are from the reviewer's reference list to cite them properly.

---

## 7. Decisions for Professor

- [ ] **D1**: Run Experiment Set A (quantization, noise, pruning)? → **Strongly recommend YES** — 30 min GPU, plugs the biggest gap
- [ ] **D2**: Run Experiment Set B (SGD, LoRA fine-tuning)? → Recommend YES if time permits; SGD is the most important
- [ ] **D3**: Run Experiment Set C (model merging)? → Optional, lower priority
- [ ] **D4**: Integrate SD Phase 11 results (domain-shift, task-shift) into paper? → Recommend YES — doubles the robustness coverage for free
- [ ] **D5**: Identify [5] and [6] from reviewer's reference list → Needed for proper citation and positioning
- [ ] **D6**: Acknowledge paradigm lineage explicitly? → Recommend YES (consistent with Phase 03 strategy)
- [ ] **D7**: How to handle distillation in this response? → Cross-reference Phase 01; frame as principled boundary shared with all weight-based methods

## 8. Relationship to Other Phases

| Phase | Connection |
|-------|-----------|
| Phase 01 (distillation) | Directly relevant — distillation is the strongest "missing" attack; Phase 01 provides the principled boundary argument |
| Phase 03 (dataset inference) | Addresses the novelty concern — paradigm lineage already discussed |
| Phase 06 (SOTA baselines) | Complementary — Phase 06 defends baseline selection, Phase 07 defends robustness scope |
| SD Phase 11 | Domain-shift and task-shift results should be integrated here as additional robustness evidence |

## 9. Experiments Required

**YES** — this is the first reviewer response phase that genuinely requires new experiments. Experiment Set A (quantization, noise, pruning rates) is the minimum viable addition (~30 min GPU). Set B (SGD fine-tuning) is strongly recommended (~2-4h). Full expanded Table 6 requires ~10h total GPU time.
