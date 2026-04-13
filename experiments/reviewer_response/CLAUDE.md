# Reviewer Response — Claude Code Instructions

> This file applies only to the `experiments/reviewer_response/` experiment.
> The root `.claude/CLAUDE.md` contains global instructions; this file adds experiment-specific rules.

## Experiment Goal
Address reviewer comments for the MiO paper (ICML 2026 / ACM submission).
Each phase corresponds to one reviewer question or concern, containing analysis, rebuttal draft, and (where applicable) proposed experiments.

## Scope
- Rebuttal drafts ready for professor review
- Theoretical analysis of reviewer concerns
- Experiment designs (run only after professor approval)
- Paper revision suggestions (apply only after professor approval)

## Workflow Rules
1. **Start every session** by reading `experiments/reviewer_response/STATE.md`.
2. **Load only the current phase**: `phases/phase_XX.md`.
3. **After completing work**: update STATE.md with results and next steps.
4. **No paper changes without approval**: drafts go in phase files, not directly into `ICML2026/` or `ACM/`.
5. **No experiments without approval**: phase files include experiment designs, but GPU work requires professor sign-off.

## Phase Naming
Each phase = one reviewer concern. Name format:
```
phase_01.md  — Distillation robustness
phase_02.md  — (next reviewer comment)
...
```

## Key Paper Files
```
ICML2026/main.tex          # ICML submission
ACM/main.tex               # ACM submission (kept in sync)
```

## Related Experiments
Results from these experiments may be referenced in rebuttals:
- `experiments/baseline_comparison/` — DDIM pipeline, 4-dataset results
- `experiments/sd_watermark_comp/` — SD v1.4 LoRA + SleeperMark
