# STATE — Reviewer Response

> Last updated: 2026-03-25
> Current phase: **Phase 08 drafted**
> Overall progress: ██░░░░░░░░░ 1/8

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-25 | Created reviewer_response experiment | Centralize all reviewer rebuttals with analysis + experiment designs |
| 2026-03-25 | Added Phase 02: baseline overlap sensitivity | Reviewer asks about practical baseline selection and confidence degradation under overlap |
| 2026-03-25 | Added Phase 03: dataset inference & CDI comparison | Reviewer asks how MiO relates to DI/CDI paradigm; text-only response, no experiments |
| 2026-03-25 | Added Phase 04: baseline access realism at scale | Reviewer questions baseline assumption; key defense = 100% overlap still works + open-weight ecosystem |
| 2026-03-25 | Added Phase 05: no same-distribution baseline | STL-10 already uses mismatched baseline (bedroom→STL-10, d=33.4); mismatched = stronger separation |
| 2026-03-25 | Added Phase 06: SOTA baseline coverage | Reviewer conflates output watermarking with model ownership; taxonomy argument + qualitative table |
| 2026-03-25 | Added Phase 07: broader post-theft robustness | Valid concern; propose quantization/noise/pruning (30min) + SGD-FT + integrate SD Phase 11 results |
| 2026-03-25 | Added Phase 08: reference model ablation | Reviewer wants ablation on baseline choice; ~2h inference-only experiment; merges with Phase 09 baseline_comparison |

## Phase Tracker

| Phase | Reviewer Concern | Status | Notes |
|-------|-----------------|--------|-------|
| 01 | Distillation robustness | 📝 DRAFT | Analysis + rebuttal draft complete, pending professor discussion |
| 02 | Baseline overlap sensitivity & selection guidance | 📝 PLAN | Detailed plan with 4 sub-experiments (E1–E4), pending professor discussion on D1–D4 |
| 03 | Dataset inference & CDI comparison | 📝 DRAFT | Rebuttal draft + related work paragraph ready; no experiments needed; pending professor discussion |
| 04 | Baseline access realism at scale | 📝 DRAFT | Rebuttal draft ready; **CRITICAL**: line 551 "disjoint" claim is wrong, must fix; ties to Phase 02 |
| 05 | No same-distribution baseline available | 📝 DRAFT | Existing STL-10 result (d=33.4, bedroom baseline) already answers this; no new experiments needed |
| 06 | SOTA baseline coverage (Stable Sig, Tree-Ring, etc.) | 📝 DRAFT | Taxonomy argument: output WM ≠ model ownership; WDM/Zhao are correct comparisons; qualitative table proposed |
| 07 | Broader post-theft robustness + novelty concern | 📝 DRAFT | **NEEDS EXPERIMENTS**: Set A (quant/noise/prune, 30min) + Set B (SGD-FT, 2-4h); integrate SD Phase 11 |
| 08 | Reference model selection ablation | 📝 DRAFT | **NEEDS EXPERIMENTS**: multi-baseline eval on CIFAR-10+CelebA (~2h inference); merges with baseline_comparison Phase 09 |

---

## Phase 02 — Key Decisions Pending

| ID | Question | Options | Recommendation |
|----|----------|---------|----------------|
| D1 | DDIM training iterations | 400k (24-30h) / 100k fastdev (6-12h) / 200k (12-18h) | Pilot with 100k on 3 levels first |
| D2 | Number of overlap levels | 3 (0%,50%,100%) / 5 / 6 | Start with 3, add more if curve is interesting |
| D3 | SD experiment scope | E4a (CLIP check) / E4b (contaminated LoRA) / both | E4b more valuable |
| D4 | Paper placement | Appendix / Discussion expansion / new Experiments subsection | To discuss |

## Next Actions
- [ ] Professor review of Phase 01 (distillation) rebuttal draft
- [ ] Professor review of Phase 02 (baseline overlap) plan — decide D1–D4
- [ ] After D1–D4 decisions: execute Phase 02 experiments
- [ ] Professor review of Phase 03 (dataset inference/CDI) rebuttal draft — decide D1–D5
- [ ] After Phase 03 approval: add DI/CDI paragraph to Related Work + bib entries
- [ ] Professor review of Phase 04 (baseline realism) rebuttal draft — decide D1–D5
- [ ] After Phase 04 approval: fix line 551 + expand Discussion paragraph + Threat Model footnote
- [ ] Professor review of Phase 05 (no same-distribution baseline) rebuttal draft
- [ ] After Phase 05 approval: annotate Table 1 + add Discussion sentence
- [ ] Professor review of Phase 06 (SOTA baselines) — decide D1–D6, especially D1 (taxonomy only vs experiment)
- [ ] Verify Shallow Diffuse exact citation from reviewer's reference list
- [ ] After Phase 06 approval: add taxonomy paragraph + qualitative table + new citations
- [ ] Professor review of Phase 07 (robustness + novelty) — decide D1–D7
- [ ] Identify reviewer's [5] and [6] references for proper citation
- [ ] After Phase 07 approval: run experiments Set A (30min) + Set B (2-4h), expand Table 6
- [ ] Professor review of Phase 08 (reference model ablation) — decide D1–D5
- [ ] After Phase 08 approval: run multi-baseline eval (~2h inference), produce ablation table
