# STATE — Reviewer Response

> Last updated: 2026-03-27
> Current phase: **Q2q9 subfolder created (7 files)**
> Overall progress: ██░░░░░░░░░ 1/8

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-25 | Created reviewer_response experiment | Centralize all reviewer rebuttals with analysis + experiment designs |
| 2026-03-25 | Added Phase 02: reference model overlap sensitivity | Reviewer asks about practical reference model selection and confidence degradation under overlap |
| 2026-03-25 | Added Phase 03: dataset inference & CDI comparison | Reviewer asks how MiO relates to DI/CDI paradigm; text-only response, no experiments |
| 2026-03-25 | Added Phase 04: reference model access realism at scale | Reviewer questions reference model assumption; key defense = 100% overlap still works + open-weight ecosystem |
| 2026-03-25 | Added Phase 05: no same-distribution reference model | STL-10 already uses mismatched reference model (bedroom→STL-10, d=33.4); mismatched = stronger separation |
| 2026-03-25 | Added Phase 06: SOTA baseline coverage | Reviewer conflates output watermarking with model ownership; taxonomy argument + qualitative table |
| 2026-03-25 | Added Phase 07: broader post-theft robustness | Valid concern; propose quantization/noise/pruning (30min) + SGD-FT + integrate SD Phase 11 results |
| 2026-03-25 | Added Phase 08: reference model ablation | Reviewer wants ablation on reference model choice; ~2h inference-only experiment; merges with Phase 09 baseline_comparison |
| 2026-03-26 | Added Phase 09: STL-10 domain mismatch + multi-reference-model | Reviewer correctly identifies bedroom reference model confound; expanded to 3-4 reference models/dataset with conservative ALL-must-pass criteria |
| 2026-03-26 | CIFAR-10 multi-reference-model complete | 3 reference models PASS, min d=24.14, separation_range=(24.14, 63.89) |
| 2026-03-26 | STL-10 multi-reference-model complete | 4 reference models PASS, min d=29.30, separation_range=(29.30, 116.65); bedroom d=33.48 confirmed as inflated vs church d=29.30 |
| 2026-03-27 | Restructured by reviewer: Q2q9 subfolder | 7 files (W1-W4, Q1-Q3) with per-item analysis + rebuttal drafts; phases/ kept as shared analysis backend |
| 2026-03-27 | Restructured by reviewer: kdBn subfolder | 8 files (W1-W5, Q1-Q3); 3 new concerns (W3: TPR@FPR, W4: naming/H0, W5: writing) |
| 2026-03-27 | Restructured by reviewer: pFey subfolder | 3 files (W1-W3); all map to existing phases (06, 03+07, 08+09) |

## Phase Tracker

| Phase | Reviewer Concern | Status | Notes |
|-------|-----------------|--------|-------|
| 01 | Distillation robustness | 📝 DRAFT | Analysis + rebuttal draft complete, pending professor discussion |
| 02 | Reference model overlap sensitivity & selection guidance | 📝 PLAN | Detailed plan with 4 sub-experiments (E1–E4), pending professor discussion on D1–D4 |
| 03 | Dataset inference & CDI comparison | 📝 DRAFT | Rebuttal draft + related work paragraph ready; no experiments needed; pending professor discussion |
| 04 | Reference model access realism at scale | 📝 DRAFT | Rebuttal draft ready; **CRITICAL**: line 551 "disjoint" claim is wrong, must fix; ties to Phase 02 |
| 05 | No same-distribution reference model available | 📝 DRAFT | Existing STL-10 result (d=33.4, bedroom reference model) already answers this; no new experiments needed |
| 06 | SOTA baseline coverage (Stable Sig, Tree-Ring, etc.) | 📝 DRAFT | Taxonomy argument: output WM ≠ model ownership; WDM/Zhao are correct comparisons; qualitative table proposed |
| 07 | Broader post-theft robustness + novelty concern | 📝 DRAFT | **NEEDS EXPERIMENTS**: Set A (quant/noise, 20min) + Set B (SGD-FT, 2-4h); **Table 6 pruning PASS is wrong (actual FAIL)**; [5][6] identified as DI lineage |
| 08 | Reference model selection ablation | 📝 DRAFT | **NEEDS EXPERIMENTS**: multi-reference-model eval on CIFAR-10+CelebA (~2h inference); merges with baseline_comparison Phase 09 |
| 09 | STL-10 domain-mismatched reference model & multi-reference-model robustness | 🔬 RUNNING | CIFAR-10 + STL-10 complete (PASS); CIFAR-100 + CelebA running; branch `feat/multi-baseline` |

---

## Reviewer 1 (Q2q9) — Per-Item Tracker

| File | Concern | Status | Experiment? | Related Phases |
|------|---------|--------|-------------|----------------|
| W1 | Single reference model + STL-10 domain mismatch | 🔬 RUNNING | Yes (Phase 09 in progress) | 08, 09 |
| W2 | Reference model selection guidance + overlap degradation | 📝 DRAFT | Optional (Phase 02 overlap curve) | 02, 04 |
| W3 | White-box access limits applicability | 📝 DRAFT | No | NEW |
| W4 | Distillation attack not addressed | 📝 DRAFT | Optional (~12-16h GPU) | 01 |
| Q1 | Domain-relevant STL-10 reference model + multiple reference models | 🔬 RUNNING | Yes (Phase 09 in progress) | 08, 09 |
| Q2 | Practical reference model selection methodology | 📝 DRAFT | Optional (Phase 02 overlap curve) | 02, 04 |
| Q3 | Distillation effectiveness | 📝 DRAFT | Optional (~12-16h GPU) | 01 |

## Reviewer 3 (pFey) — Per-Item Tracker

| File | Concern | Status | Experiment? | Related Phases |
|------|---------|--------|-------------|----------------|
| W1 | SOTA baseline coverage (Stable Sig, Tree-Ring, etc.) | 📝 DRAFT | No (taxonomy argument) | 06 |
| W2 | Novelty ([5,6] extension) + narrow robustness | 📝 DRAFT | Yes (Set A: 20min, Set B: 2-4h) | 03, 07 |
| W3 | Reference model selection ablation | 🔬 RUNNING | Yes (Phase 09 in progress) | 08, 09 |

---

## Reviewer 2 (kdBn) — Per-Item Tracker

| File | Concern | Status | Experiment? | Related Phases |
|------|---------|--------|-------------|----------------|
| W1 | Missing DI/CDI related work | 📝 DRAFT | No | 03 |
| W2 | Reference model access unrealistic | 📝 DRAFT | No | 04 |
| W3 | TPR@0.1%FPR reporting | 📝 DRAFT | Yes (~30min recompute) | NEW + 07 |
| W4 | Model naming confusion + null hypothesis | 📝 DRAFT | No | NEW |
| W5 | Writing: hyperparams to appendix | 📝 DRAFT | No | NEW |
| Q1 | Compare with DI/CDI | 📝 DRAFT | No | 03 |
| Q2 | Reference model access realism | 📝 DRAFT | No | 04 |
| Q3 | No same-distribution reference models | 📝 DRAFT | No (existing data) | 05, 09 |

---

## Phase 02 — Key Decisions Pending

| ID | Question | Options | Recommendation |
|----|----------|---------|----------------|
| D1 | DDIM training iterations | 400k (24-30h) / 100k fastdev (6-12h) / 200k (12-18h) | Pilot with 100k on 3 levels first |
| D2 | Number of overlap levels | 3 (0%,50%,100%) / 5 / 6 | Start with 3, add more if curve is interesting |
| D3 | SD experiment scope | E4a (CLIP check) / E4b (contaminated LoRA) / both | E4b more valuable |
| D4 | Paper placement | Appendix / Discussion expansion / new Experiments subsection | To discuss |

> **Terminology note**: In this experiment, "baseline" refers exclusively to watermark methods (WDM, Zhao, SleeperMark). Non-watermark models (HuggingFace pretrained, SD v1.4 base, random init) are called "reference models."

## Next Actions
- [ ] Professor review of Phase 01 (distillation) rebuttal draft
- [ ] Professor review of Phase 02 (reference model overlap) plan — decide D1–D4
- [ ] After D1–D4 decisions: execute Phase 02 experiments
- [ ] Professor review of Phase 03 (dataset inference/CDI) rebuttal draft — decide D1–D5
- [ ] After Phase 03 approval: add DI/CDI paragraph to Related Work + bib entries
- [ ] Professor review of Phase 04 (reference model realism) rebuttal draft — decide D1–D5
- [ ] After Phase 04 approval: fix line 551 + expand Discussion paragraph + Threat Model footnote
- [ ] Professor review of Phase 05 (no same-distribution reference model) rebuttal draft
- [ ] After Phase 05 approval: annotate Table 1 + add Discussion sentence
- [ ] Professor review of Phase 06 (SOTA baselines) — decide D1–D6, especially D1 (taxonomy only vs experiment)
- [ ] Verify Shallow Diffuse exact citation from reviewer's reference list
- [ ] After Phase 06 approval: add taxonomy paragraph + qualitative table + new citations
- [ ] Professor review of Phase 07 (robustness + novelty) — decide D1–D7
- [x] ~~Identify reviewer's [5] and [6]~~ → [5]=Maini ICLR 2021 (DI), [6]=NeurIPS 2022 (DI for self-supervised)
- [ ] After Phase 07 approval: run experiments Set A (30min) + Set B (2-4h), expand Table 6
- [ ] Professor review of Phase 08 (reference model ablation) — decide D1–D5
- [ ] After Phase 08 approval: run multi-baseline eval (~2h inference), produce ablation table
- [ ] Phase 09: wait for CIFAR-100 + CelebA eval to finish (~3h remaining)
- [ ] Phase 09: once all 4 datasets complete, fill TBD values in phase_09.md rebuttal and paper changes
- [ ] Professor review of Phase 09 — decide D1-D5 (conservative table, appendix, setup text, discussion, line 882 fix)
