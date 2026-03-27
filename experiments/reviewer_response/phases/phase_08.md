# Phase 08: Reference Model Selection Ablation

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Address the reviewer's request for ablation studies on reference model selection: how does the choice of reference model affect verification results? Is architecture/data matching required?

---

## 1. Reviewer Question

> The proposed method incorporates a reference model as a 'baseline' to calibrate sample difficulty, thereby mitigating the impact of varying sample memorability on membership inference. However, the paper lacks a detailed discussion and ablation studies regarding the selection of this reference model. It remains unclear whether (and to what extent) the choice of different reference models affects the final verification results, and whether there is a strict requirement for the reference model to share the same architecture or training data. Such dependencies could potentially limit the practical applicability of the proposed framework.

## 2. Current Paper Position

- **Line 551**: Lists ONE reference model per dataset — gives no sense of sensitivity
- **Discussion line 882**: "for novel domains, general-purpose reference models provide conservative bounds" — unsupported assertion
- **No ablation** on reference model selection anywhere in the paper
- **Code already supports multi-reference-model evaluation**: `eval_ownership.py` loads all baselines from `configs/baselines_by_dataset.yaml`, computes per-baseline d/ratio/p, and applies conservative criteria (`min(all_d) > 2.0`, `min(all_ratios) > 5.0`)
- **Phase 09 of baseline_comparison** was designed to run this but **never executed**

## 3. Analysis: What We Can Already Infer (Cross-Dataset Evidence)

The paper implicitly tests different reference model types across datasets, but never frames it as a reference model sensitivity analysis:

| Dataset | Reference Model | Architecture match? | Data distribution match? | Resolution match? | d | Ratio |
|---------|----------------|-------------------|------------------------|------------------|---|-------|
| CIFAR-10 | ddpm-cifar10-32 | Different UNet config | Same data (100% overlap) | Yes (32x32) | 23.9 | 24.4x |
| CIFAR-100 | ddpm-cifar10-32 | Different UNet config | Different dataset | Yes (32x32) | 18.5 | 19.2x |
| STL-10 | ddpm-ema-bedroom-256 | Different UNet config | Different domain entirely | No (256→96 resize) | 33.4 | 101x |
| CelebA | ddpm-celebahq-256 | Different UNet config | Related (CelebA-HQ ⊂ CelebA) | No (256→64 resize) | 26.1 | 26.6x |

**Key observations:**
1. **Architecture never matches** — our DDIM UNet (128ch, [1,2,2,2], cosine schedule) differs from all Google/CompVis reference models. Yet all pass with d > 18.
2. **Data match is irrelevant** — CIFAR-10 has 100% overlap (d=23.9); STL-10 has 0% overlap with a bedroom model (d=33.4). Both pass.
3. **Resolution mismatch is handled** — STL-10 and CelebA reference models are resized from 256 down. Still works.
4. **Mismatched reference models give STRONGER separation** (Phase 05 argument)

### What's missing for a proper ablation

We need **multiple reference models evaluated on the SAME dataset** to show sensitivity. The config already registers these:

**CIFAR-10 (3 reference models):**
- `ddpm-cifar10` (matched) — d=23.9 (already computed)
- `ddpm-bedroom` (mismatched) — **NOT computed yet**
- `random-32` (untrained lower bound) — **NOT computed yet**

**CelebA (4 reference models):**
- `ddpm-celebahq` (matched) — d=26.1 (already computed)
- `ldm-celebahq` (matched, different architecture: LDM vs DDPM) — **NOT computed yet**
- `ddpm-bedroom` (mismatched) — **NOT computed yet**
- `random-64` (untrained lower bound) — **NOT computed yet**

Running these would produce a proper ablation table showing d and ratio for each reference model type on the same dataset.

---

## 4. Proposed Experiment: Reference Model Ablation

### What to run

```bash
conda activate mio

# CIFAR-10: eval all 3 registered reference models
python scripts/eval_ownership.py \
  --dataset cifar10 \
  --model-a /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
  --model-b /data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt \
  --baselines-config configs/baselines_by_dataset.yaml \
  --output /data/short/fjiang4/experiments/baseline_comparison/results/multi_baseline/cifar10/

# CelebA: eval all 4 registered reference models
python scripts/eval_ownership.py \
  --dataset celeba \
  --model-a <celeba_model_a_path> \
  --model-b <celeba_model_b_path> \
  --baselines-config configs/baselines_by_dataset.yaml \
  --output /data/short/fjiang4/experiments/baseline_comparison/results/multi_baseline/celeba/
```

### Compute estimate

| Dataset | Reference models to compute | Time per reference model | Total |
|---------|---------------------------|------------------------|-------|
| CIFAR-10 | 2 new (bedroom, random) | ~20 min each | ~40 min |
| CelebA | 3 new (ldm-celebahq, bedroom, random) | ~30 min each | ~90 min |
| **Total** | | | **~2h GPU** |

All inference-only — no training required.

### Expected output: per-reference-model ablation table

| Dataset | Reference Model | Role | Architecture | Training Data | d | Ratio | Verified? |
|---------|----------------|------|-------------|--------------|---|-------|-----------|
| CIFAR-10 | ddpm-cifar10 | Matched | Google DDPM | CIFAR-10 (same) | 23.9 | 24.4x | PASS |
| CIFAR-10 | ddpm-bedroom | Mismatched | Google DDPM 256 | LSUN Bedroom | [TBD, expect >30] | [TBD, expect >50x] | PASS |
| CIFAR-10 | random-32 | Lower bound | Untrained UNet | None | [TBD, expect >40] | [TBD, expect >100x] | PASS |
| CelebA | ddpm-celebahq | Matched | Google DDPM 256 | CelebA-HQ | 26.1 | 26.6x | PASS |
| CelebA | ldm-celebahq | Matched (diff arch) | CompVis LDM | CelebA-HQ | [TBD] | [TBD] | PASS |
| CelebA | ddpm-bedroom | Mismatched | Google DDPM 256 | LSUN Bedroom | [TBD, expect >30] | [TBD, expect >50x] | PASS |
| CelebA | random-64 | Lower bound | Untrained UNet | None | [TBD, expect >40] | [TBD, expect >100x] | PASS |

**Expected finding**: All reference models pass verification. Mismatched/random reference models give STRONGER separation. The conservative criterion (min d across all reference models) is dominated by the matched reference model, which is the hardest case.

---

## 5. Rebuttal Draft

---

**Response to Reviewer — Reference Model Selection**

We thank the reviewer for this suggestion. We have conducted an ablation study evaluating verification performance across reference models of varying domain relevance for the same dataset.

**Setup.** For each dataset, we evaluate three classes of reference models: (1) *matched* — trained on the same or related domain; (2) *mismatched* — trained on an unrelated domain; and (3) *lower bound* — a randomly initialized (untrained) model. All reference models use different architectures from our owner model (DDIM UNet, 128ch, cosine schedule).

**Results on CIFAR-10:**

| Reference Model | Role | d | Ratio | Verified? |
|----------------|------|---|-------|-----------|
| google/ddpm-cifar10-32 | Matched | 23.9 | 24.4x | PASS |
| google/ddpm-ema-bedroom-256 | Mismatched | [TBD] | [TBD] | [TBD] |
| Random UNet (untrained) | Lower bound | [TBD] | [TBD] | [TBD] |

[Numbers to be filled after experiments.]

**Key findings:**
1. **Architecture matching is NOT required.** None of our reference models share the same architecture as the owner model, yet all achieve d >> 2.0. The verification signal depends on the memorization fingerprint (unique to the owner's training trajectory), not on architectural similarity.
2. **Training data matching is NOT required.** Domain-mismatched reference models yield *stronger* separation because they cannot reconstruct the watermark set at all. An untrained random model provides the ultimate lower bound.
3. **The conservative criterion uses min(d) across ALL reference models**, so the verification threshold is set by the *hardest* (matched) reference model. Adding more reference models can only tighten, never loosen, the verification.

These results demonstrate that MiO imposes no strict requirement on reference model architecture or training data, and that practically any independently trained model serves as a valid reference.

---

## 6. Proposed Paper Changes

### Change 1: New ablation table (appendix)

```latex
\begin{table}[h]
\caption{Reference model sensitivity ablation on CIFAR-10.
All reference models use different architectures from the owner model.
The conservative verification criterion reports $\min |d|$
across all reference models.}
\label{tab:baseline_ablation}
\centering\small
\begin{tabular}{@{}llllcc@{}}
\toprule
Reference Model & Role & Arch. & Training Data & $|d|$ & Ratio \\
\midrule
ddpm-cifar10-32 & Matched & Google DDPM & CIFAR-10 & 23.9 & 24.4$\times$ \\
ddpm-bedroom-256 & Mismatched & Google DDPM & LSUN Bed. & [TBD] & [TBD] \\
Random UNet & Lower bound & Same & None & [TBD] & [TBD] \\
\midrule
\multicolumn{4}{@{}l}{Conservative (min across all)} & [TBD] & [TBD] \\
\bottomrule
\end{tabular}
\end{table}
```

### Change 2: Add 1-2 sentences to Discussion (after reference model availability paragraph)

```latex
An ablation across matched, mismatched, and untrained reference models
(Appendix~\ref{app:baseline_ablation}) confirms that verification
does not require architectural or distributional similarity between
the reference and owner models; mismatched reference models yield strictly
stronger separation, and the conservative criterion reports the
minimum $|d|$ across all reference models.
```

### Change 3: Expand Section 5.1 setup (line 551)

Replace single-reference-model description with multi-reference-model framing:
```latex
For each dataset we evaluate against multiple public reference models
spanning three roles: \emph{matched} (same or related domain),
\emph{mismatched} (unrelated domain), and \emph{lower bound}
(randomly initialized, untrained).
Table~\ref{tab:main_results} reports the conservative (minimum)
$|d|$ across all reference models; per-reference-model breakdowns
appear in Appendix~\ref{app:baseline_ablation}.
```

---

## 7. Decisions for Professor

- [ ] **D1**: Run the reference model ablation experiment? → **Strongly recommend YES** — ~2h GPU (inference only), directly answers the reviewer's primary ask
- [ ] **D2**: Run for CIFAR-10 only or all 4 datasets? → Recommend CIFAR-10 + CelebA (covers matched/mismatched/random + LDM vs DDPM architecture comparison)
- [ ] **D3**: Place ablation table in appendix or main text? → Recommend appendix (saves main text space) with 1-2 sentence summary in Discussion
- [ ] **D4**: Update Table 1 to report conservative min(d) instead of single-reference-model d? → Recommend YES — makes the multi-reference-model protocol explicit
- [ ] **D5**: Merge this with Phase 09 (multi-reference-model expansion) of baseline_comparison? → Recommend YES — same experiment, same infrastructure

## 8. Relationship to Other Phases

| Phase | Connection |
|-------|-----------|
| Phase 04 (baseline access) | Phase 04 argues reference models are available; Phase 08 shows the CHOICE doesn't matter |
| Phase 05 (mismatched baseline) | Phase 05 provides the theoretical argument; Phase 08 provides the empirical ablation |
| Phase 09 (baseline_comparison) | Directly overlaps — same multi-reference-model eval infrastructure |
| Phase 07 (robustness) | Independent but complementary — Phase 07 varies the attack, Phase 08 varies the reference |

## 9. Experiments Required — YES (inference only)

~2h GPU total. No training. Uses existing `eval_ownership.py` with `--baselines-config configs/baselines_by_dataset.yaml`. The per-reference-model breakdown is already computed by the script (lines 557-573). Just needs to be run and results collected.
