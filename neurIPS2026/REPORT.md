# NeurIPS 2026 Port — Diff Report

## 1. Command / Macro Substitutions (ICML → NeurIPS)

| ICML | NeurIPS | Notes |
|---|---|---|
| `\usepackage{icml2026}` | `\usepackage{neurips_2026}` | Default = anonymous double-blind |
| `\icmltitlerunning{...}` | Dropped | NeurIPS has no running header |
| `\icmltitle{...}` | `\title{...}` | Standard LaTeX |
| `\begin{icmlauthorlist}...\end{icmlauthorlist}` | Dropped | Replaced by `\author{Anonymous Author(s)}` |
| `\icmlauthor{...}{...}` | Dropped | |
| `\icmlaffiliation{...}{...}` | Dropped | |
| `\icmlcorrespondingauthor{...}{...}` | Dropped | |
| `\printAffiliationsAndNotice{}` | Dropped | NeurIPS uses `\maketitle` |
| `\twocolumn[...]` | Dropped | NeurIPS is single-column |
| `\onecolumn` (appendix) | Dropped | Already single-column |
| `\vskip 0.3in` (post-author) | Dropped | |
| `\vskip 0.1in` / `\vskip -0.1in` (tables) | Removed | NeurIPS single-col doesn't need ICML table vskips |
| `\begin{figure*}` | `\begin{figure}` | Single-column; no spanning needed |
| `\bibliographystyle{icml2026}` | `\bibliographystyle{plainnat}` | NeurIPS loads natbib; plainnat is compatible |
| `\section*{Impact Statement}` | `\begin{ack}...\end{ack}` | ack environment auto-hides in anonymized builds |
| `\begin{center}...\end{center}` (tables) | `\centering` | Cleaner for NeurIPS single-col |

## 2. Packages Added / Removed

| Package | Action | Reason |
|---|---|---|
| `icml2026` | Removed | Replaced by `neurips_2026` |
| `algorithm` | Added explicitly | Was loaded by icml2026.sty; NeurIPS sty does not load it |
| `algorithmic` | Added explicitly | Same as above |
| `algorithm.sty`, `algorithmic.sty` | Copied to neurIPS2026/ | Local copies from ICML dir for portability |
| `inputenc`, `fontenc`, `hyperref`, `url`, `nicefrac`, `microtype` | Kept from NeurIPS template | Not duplicated |
| `amsmath`, `amssymb`, `amsthm` | Kept from ICML | Not loaded by neurips_2026.sty |
| `amsfonts` | Kept from template | Technically redundant with amssymb but harmless |
| `booktabs`, `xcolor` | Kept from template | Also needed by paper content |

## 3. Anonymization Fixes

| Item | Status |
|---|---|
| Author names (Feng Jiang, Zuobin Xiong, An Huang, Zhipeng Cai, Yingshu Li) | Replaced with `Anonymous Author(s)` |
| Email (fjiang4@student.gsu.edu) | Not present — dropped with `\icmlcorrespondingauthor` |
| Affiliation (Georgia State / GSU) | Not present — dropped with `\icmlaffiliation` |
| `\thanks{}` footnotes | None in NeurIPS version |
| GitHub URLs | None found in any file |
| Self-citations (first-person) | None found |
| Bib file "Xiong" match | Different person (Jie Xiong in Peng et al. 2023) — not a co-author |

**Verdict: CLEAN** — no anonymization leaks detected.

## 4. Page Count Delta

| Metric | ICML | NeurIPS | Delta |
|---|---|---|---|
| Format | Two-column | Single-column | — |
| Body page limit | 8 pages | 9 pages | +1 page headroom |
| Actual body pages | ~8 (two-col) | **~10.5 (single-col)** | **~1.5 pages over limit** |
| Total pages (with appendix + checklist) | ~13 | 22 | +9 (checklist adds 6pp) |

**The body exceeds the 9-page NeurIPS limit by approximately 1.5 pages.** No content was removed or shrunk to fit. Candidate cuts (to be discussed with Frank):
1. **Move Table 7 (aggregation ablation) to appendix** (~0.3pp savings) — ablation text can reference appendix
2. **Condense Controlled Baseline Comparison (Sec 5.4)** — currently ~1.5pp of prose + 2 tables; could be tightened
3. **Move Table 4 (threshold sensitivity) to appendix** (~0.2pp savings) — it's confirmatory, not primary evidence
4. **Tighten the TikZ figure** — Figure 1 currently occupies most of a page in single-column; could be scaled down
5. **Condense Problem Setup / Threat Model** — currently ~2 pages; threat model could be shortened

## 5. Checklist Questions Needing Frank's Input

| Q# | Question | Current Answer | Issue |
|---|---|---|---|
| Q5 | Open access to data and code | `\answerTODO{}` | Will an anonymized code repo be provided with the NeurIPS submission? |

All other 14 questions are answered with grounded justifications.

## 6. Best-Guess Resolutions

| Item | Decision | Rationale |
|---|---|---|
| Impact statement placement | Moved to `\begin{ack}...\end{ack}` | NeurIPS convention; auto-hidden in anonymous mode |
| Bibliography style | `plainnat` | Standard NeurIPS-compatible style; all \citep/\citet resolve |
| `Eq.` cross-reference in ablation text | Removed dangling reference (ICML had "Eq.") | Sentence reworded to remove specific equation pointer |
| `\ref{sec:cost}` → `\ref{app:cost}` | Fixed | Cost section is in appendix, not main body |

## 7. Files in neurIPS2026/

```
neurIPS2026/
├── main.tex          # NeurIPS-formatted paper
├── main.pdf          # Compiled PDF (22 pages)
├── references.bib    # Bibliography (copied from ICML2026/)
├── checklist.tex     # NeurIPS mandatory checklist (pre-filled)
├── neurips_2026.sty  # NeurIPS style file (UNMODIFIED)
├── algorithm.sty     # Algorithm float (copied from ICML2026/)
├── algorithmic.sty   # Algorithmic pseudocode (copied from ICML2026/)
├── STATE.md          # Phase tracking document
├── REPORT.md         # This file
└── main.{aux,bbl,blg,log,out}  # Build artifacts
```

---

## Follow-up Pass 1

### Item 1: Checklist Q5 (Open access to data and code)
- **Action**: Changed `\answerTODO{}` → `\answerYes{}` in `checklist.tex`.
- **Justification text** (verbatim per Frank's instruction): "An anonymized code repository containing the MiO verification protocol, training scripts, and evaluation pipeline will be provided as supplementary material with this submission. See Section~5.1 for setup instructions and Appendix~A for hyperparameters."
- **Section placeholders resolved**: `[Section]` → `Section~\ref{sec:setup}` (Sec 5.1), `[Appendix]` → `Appendix~\ref{app:details}` (App A).
- **FLAG**: Frank must prepare the anonymized supplementary zip before submission deadline.
- **Issues surfaced**: None.

### Item 2: Bibliography style verification
- **Action**: Full recompile with `plainnat`. Inspected `.bbl` output for 5 specific citations.
- **(a) Tree-Ring** (`wen2023tree`): Renders as "Wen et al. (2023)" with full author list. arXiv venue correct.
- **(b) WDM** (`peng2023watermark`): Renders as "Peng et al. (2023)". arXiv venue correct.
- **(c) Zhao et al.** (`zhao2023recipe`): Renders as "Zhao et al. (2023)" with full 6-author list. arXiv venue correct.
- **(d) \citet calls**: None present in paper — only `\cite` and `\citep` used. Both render correctly with plainnat.
- **(e) Missing fields**: 0 bibtex warnings. All 25 bbl entries complete.
- **Issues surfaced**: None.

### Item 3: Figure caption audit (figure* → figure)
- **Action**: Grepped all 17 `\caption` entries for spatial language ("left panel", "right panel", "top row", "bottom row", "(a)...(b)" references).
- **Result**: Zero hits. The paper has one TikZ figure (no subfigures) and 15 standalone tables. No captions reference two-column-specific spatial arrangement.
- **Captions edited**: None required.
- **Issues surfaced**: None.

### Item 4: ack environment audit (Impact Statement misplacement)
- **Found**: `\begin{ack}...\end{ack}` contained an ICML-style Impact Statement paragraph (not acknowledgments/funding).
- **Action per decision tree (b)**:
  1. Moved the substantive content (positive impact + risks + mitigations) into the checklist's Broader Impacts question (Q10) as the justification text, with concrete section references (Section 4.4 for criteria, Section 6 for discussion).
  2. Left `\begin{ack}...\end{ack}` **empty** (with a comment: "Leave empty for anonymous submission. Add funding/thanks for camera-ready.")
  3. Updated the previous checklist Q10 justification which incorrectly referenced "the acknowledgments section."
- **Issues surfaced**: None. The impact content is now in the right place for reviewers to see it.

### Item 5: Float specifier warnings
- **Found**: 8x `\begin{table}[h]` and 1x `\begin{algorithm}[h]` in the appendix.
- **Action**: Changed all to `[!htbp]` (9 substitutions total).
- **Result**: Float warnings dropped from 7 to **0** after recompile.
- **Issues surfaced**: None. No layout change observed — appendix floats still appear in the same positions.

### Gate Re-run Results (after all 5 items)
| Gate | Result | Notes |
|---|---|---|
| G1 (zero errors) | **PASS** | 0 errors |
| G3 (all refs/cites resolve) | **PASS** | 0 undefined references, 0 undefined citations |
| G4 (no Type 3 fonts) | **PASS** | 26 Type 1 embedded, 0 Type 3 |
| Float warnings | **PASS** | 0 (was 7) |
| G2 (body ≤ 9 pages) | **Not re-run** | Intentionally over; Frank will handle page cuts |
