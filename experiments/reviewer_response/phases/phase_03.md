# Phase 03: Dataset Inference & CDI Comparison

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Address the reviewer question comparing MiO with Dataset Inference (DI) and Centralized Dataset Inference (CDI). Prepare a rebuttal draft and proposed paper changes.

---

## 1. Reviewer Question

> How does the proposed method compare with prior work on dataset inference and CDI? How is the general idea related to the core idea of dataset inference?

## 2. Current Paper Position

- **Related Work** (Section 2, line 254): Covers model watermarking and MIA only. **No mention of dataset inference or CDI.**
- **Problem Setup** (Section 3, line 316): "From Membership Inference to Ownership Verification" paragraph articulates the shift from per-sample to population-level testing — but does not cite DI/CDI as prior art for this idea.
- **Bibliography**: No entries for Maini et al. (2021) or Maini et al. (2024).

## 3. Key Prior Work

### Dataset Inference (DI) — Maini, Yaghini & Papernot, ICLR 2021
- **Problem**: Given a suspect classifier, determine if it was trained on a private dataset
- **Method**: Extract features from the suspect model's penultimate layer for private data vs reference data. Compute pairwise feature-space distances. Train a meta-classifier on these distance statistics to distinguish "trained on D" from "not trained on D"
- **Key insight**: Population-level test over a dataset, not per-sample decisions
- **Requirements**: Shadow models trained on disjoint data as reference

### LLM Dataset Inference — Maini, Wyllie, Sablayrolles, Balle & Papernot, NeurIPS 2024
- **Extension**: Adapts DI paradigm to large language models
- **Improvement**: Centralized one-sided test that eliminates the shadow model requirement
- **Significance**: Shows the paradigm generalizes beyond classifiers

### Paradigm Evolution
```
Classifiers (DI, 2021) --> LLMs (LLM-DI, 2024) --> Diffusion Models (MiO, ours)
```

---

## 4. Analysis: Shared Foundation and Key Differences

### Shared philosophical foundation

Both MiO and DI/CDI recognize that **per-sample membership inference is insufficient for IP claims** and propose **population-level statistical hypothesis testing** on a designated evidence set against reference models.

### Key differences

| Dimension | Dataset Inference (DI/CDI) | MiO (ours) |
|---|---|---|
| **Model domain** | Discriminative (classifiers) / LLMs | Generative (diffusion models) |
| **Membership signal** | Feature-space distances: classifier's learned representations position training data differently from unseen data | Reconstruction error (t-error): single-step denoising fidelity on training vs unseen samples |
| **Signal source** | Penultimate-layer features (requires meaningful representations) | Diffusion model's denoising function (exploits memorization gap) |
| **Statistical test** | Meta-classifier on distance statistics (DI); centralized one-sided test (CDI) | Three-point conjunction: consistency (p>0.05), separation (p<10^-6, \|d\|>2.0), ratio (>5x) |
| **Baseline/reference** | Shadow models on disjoint data (DI); centralized reference (CDI) | Public HuggingFace checkpoints + random reference models; no shadow training |
| **FPR control** | Meta-classifier threshold (DI); centralized p-value (CDI) | Gaussian QR: closed-form quantile at arbitrary FPR |
| **Proactive modification** | None (post-hoc) | None (post-hoc) |
| **Verification output** | Binary (DI) or p-value (CDI) | Three interpretable criteria with human-readable thresholds |

### Why MiO is NOT a trivial port of DI to diffusion models

1. **Signal fundamentally different**: DI relies on classifiers producing separable features for seen vs unseen data. Diffusion models have no classification head — membership signal lives in reconstruction fidelity across a noise schedule. Designing t-error scoring (multi-timestep, Q25 aggregation) is a non-trivial, architecture-specific contribution.

2. **Multi-timestep challenge**: Classifiers have a single forward pass. Diffusion models have T timesteps, not all carrying equal membership signal. Q25 aggregation and timestep selection (ablation in Sec 5.2) address a challenge absent in the classifier setting.

3. **Gaussian QR for FPR control**: DI trains a binary meta-classifier; CDI uses a centralized test. MiO introduces Gaussian QR with bagging ensemble, providing closed-form quantile thresholds at arbitrary FPR without retraining.

4. **Three-point verification**: DI produces a single accept/reject. MiO's conjunction of consistency + separation + ratio is stricter and more auditable, designed for adversarial IP disputes.

---

## 5. Rebuttal Draft

---

**Response to Reviewer — Comparison with Dataset Inference and CDI**

We thank the reviewer for this important connection. Dataset Inference (DI; Maini et al., ICLR 2021) and its extension to LLMs (Maini et al., NeurIPS 2024) indeed share the same philosophical foundation as our work: both recognize that per-sample membership inference is insufficient for IP claims and propose population-level statistical hypothesis testing against reference models. This paradigm has evolved from classifiers (DI) to large language models (LLM Dataset Inference) — our work extends it to generative diffusion models, completing coverage of the three major model families. We view MiO as belonging to this broader paradigm of dataset-level ownership verification via membership signals.

However, the technical realization differs substantially across model domains:

**(1) Membership signal.** DI/CDI operate on discriminative models and detect ownership via feature-space distances — the classifier's learned representations position training data differently from unseen data. In diffusion models, there is no classification head or meaningful feature bottleneck; the membership signal instead resides in *reconstruction fidelity* across the noise schedule. Our t-error scoring with Q25 multi-timestep aggregation (Section 4.1) addresses the unique challenge of extracting a robust membership signal from a T-step denoising process — a problem absent in the classifier setting.

**(2) Statistical framework.** DI trains a binary meta-classifier on distance statistics and thresholds its output; CDI improves this with a centralized one-sided test that eliminates shadow models. MiO takes a different approach: Gaussian Quantile Regression provides closed-form quantile thresholds at arbitrary FPR without retraining, and the three-point verification protocol (consistency, separation, ratio) provides a conjunction of interpretable, independently auditable criteria rather than a single accept/reject decision.

**(3) Reference model requirements.** DI originally requires training shadow models on disjoint data; CDI removes this. MiO uses publicly available pretrained checkpoints (e.g., HuggingFace DDPM models) as reference models, requiring no additional training.

We will add a discussion of DI/CDI to the Related Work section to properly contextualize MiO within this lineage.

---

## 6. Proposed Paper Changes

### Change 1: New paragraph in Related Work (after MIA paragraph, ~line 272 in ICML2026/main.tex)

```latex
\paragraph{Dataset Inference.}
Dataset inference~\citep{maini2021dataset} recasts membership
inference as a population-level hypothesis test for dataset
ownership verification: given a suspect classifier, it tests
whether the model's feature representations on a private dataset
are statistically distinguishable from those of reference models
trained on disjoint data.
This paradigm has since been extended to large language
models~\citep{maini2024llmdi}.
\textit{Our work brings the population-level verification
philosophy to generative diffusion models, where the membership
signal resides in reconstruction fidelity rather than feature-space
distances, requiring domain-specific scoring (multi-timestep
t-error with Q25 aggregation) and a distinct statistical framework
(Gaussian quantile regression with three-point verification).}
```

### Change 2: New bib entries (both ICML2026/references.bib and ACM/references.bib)

```bibtex
@inproceedings{maini2021dataset,
  title={Dataset Inference: Ownership Resolution in Machine Learning},
  author={Maini, Pratyush and Yaghini, Mohammad and Papernot, Nicolas},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{maini2024llmdi,
  title={{LLM} Dataset Inference: Did you train on my dataset?},
  author={Maini, Pratyush and Wyllie, Hengrui Jia and Sablayrolles, Alexandre and Balle, Borja and Papernot, Nicolas},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

### Change 3 (optional): Cross-reference in Problem Setup (after line 354)

```latex
This population-level perspective parallels dataset inference
approaches for classifiers~\citep{maini2021dataset,maini2024llmdi},
though the membership signal and statistical machinery differ
substantially between discriminative and generative settings
(see Section~\ref{sec:related}).
```

---

## 7. Decisions for Professor

- [ ] **D1**: Include DI/CDI in Related Work? → Recommend: YES (new paragraph, reviewer explicitly asked)
- [ ] **D2**: Add cross-reference in Problem Setup (line 354)? → Recommend: YES (connects our formulation to DI lineage)
- [ ] **D3**: Citation confirmed — cite both Maini et al. ICLR 2021 + NeurIPS 2024
- [ ] **D4**: Tone — acknowledge shared philosophy openly, then stress non-trivial technical differences
- [ ] **D5**: Experimental comparison needed? → Recommend: NO (DI/CDI target classifiers, cannot be applied to diffusion models)

---

## 8. No Experiments Required

DI/CDI are designed for discriminative models (classifiers) and LLMs respectively. The membership signal (feature-space distances) is fundamentally incompatible with diffusion model architecture. A direct experimental comparison is infeasible and unnecessary — the response is purely textual.
