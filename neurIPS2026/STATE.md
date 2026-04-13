# NeurIPS 2026 Port — STATE.md

## (a) ICML → NeurIPS Command Mapping Decisions

| ICML Command | NeurIPS Action | Status |
|---|---|---|
| `\usepackage{icml2026}` | → `\usepackage{neurips_2026}` | DONE |
| `\icmltitlerunning{...}` | DROP (NeurIPS has no running title) | DONE |
| `\icmltitle{...}` | → `\title{...}` | DONE |
| `\begin{icmlauthorlist}...\end{icmlauthorlist}` | DROP entirely | DONE |
| `\icmlauthor{...}{...}` | DROP (use `\author{}` with anonymous placeholder) | DONE |
| `\icmlaffiliation{...}{...}` | DROP | DONE |
| `\icmlcorrespondingauthor{...}{...}` | DROP | DONE |
| `\printAffiliationsAndNotice{}` | DROP (NeurIPS uses `\maketitle`) | DONE |
| `\twocolumn[...]` | DROP (NeurIPS is single-column) | DONE |
| `\bibliographystyle{icml2026}` | → `\bibliographystyle{plainnat}` | DONE |
| `\onecolumn` (appendix) | DROP (already single-column) | DONE |
| `\vskip 0.3in` (after author block) | DROP | DONE |
| `\section*{Impact Statement}` | → Content moved to checklist Broader Impacts justification; `\begin{ack}` left empty | DONE |
| `\vskip 0.1in` / `\vskip -0.1in` (table spacing) | REMOVED (NeurIPS single-col doesn't need ICML vskips) | DONE |
| `figure*` (two-column spanning) | → `figure` (NeurIPS is single-column) | DONE |

### Packages: Keep / Drop / Add

| Package | Decision | Reason |
|---|---|---|
| `icml2026` | DROP | Replaced by `neurips_2026` |
| `amsmath,amssymb,amsthm` | KEEP | Not in NeurIPS sty (only amsfonts in template) |
| `graphicx` | KEEP | Not loaded by NeurIPS sty |
| `booktabs` | DROP | Template .tex loads it (but sty doesn't — KEEP to be safe) |
| `subcaption` | KEEP | Needed for subfigures |
| `multirow` | KEEP | Used in tables |
| `xspace` | KEEP | Custom macros use it |
| `pifont` | KEEP | \cmark/\xmark |
| `tikz` + libs | KEEP | Figure 1 uses TikZ |
| `algorithm` | ADD | Was provided by icml2026.sty |
| `algorithmic` | ADD | Was provided by icml2026.sty |
| `inputenc` (utf8) | KEEP from template | Already in template |
| `fontenc` (T1) | KEEP from template | Already in template |
| `hyperref` | KEEP from template | Already in template |
| `url` | KEEP from template | Already in template |
| `amsfonts` | DROP | Loaded via amssymb |
| `nicefrac` | KEEP from template | May be useful |
| `microtype` | KEEP from template | Already in template |
| `xcolor` | DROP from explicit load | NeurIPS sty doesn't load it, but template .tex does — KEEP |

## (b) Anonymization Items Removed

| Item | File:Line | Action | Status |
|---|---|---|---|
| Author names (Feng Jiang et al.) | main.tex:57 | Replaced with `Anonymous Author(s)` | CLEAN |
| Email (fjiang4@student.gsu.edu) | N/A | Not present in NeurIPS version | CLEAN |
| Affiliation (Georgia State / GSU) | N/A | Not present in NeurIPS version | CLEAN |
| `\icmlcorrespondingauthor` | N/A | Dropped entirely | CLEAN |
| `\thanks{}` footnotes | main.tex | None present | CLEAN |
| GitHub URLs (GSU-FrankJ) | N/A | None found | CLEAN |
| Self-citations (first-person) | main.tex | None found | CLEAN |
| Ack environment content | main.tex:838 | Emptied; impact statement moved to checklist Q10 justification | CLEAN |
| `references.bib` "Xiong" hit | references.bib:288 | Different person (Jie Xiong in Peng et al.) — not co-author | OK |
| STATE.md "Frank" mention | STATE.md:68 | Internal tracking file, not submitted | N/A |

## (c) Page Count After Each Major Phase

| Phase | Body Pages | Total Pages | Notes |
|---|---|---|---|
| ICML source | ~8 (two-column) | ~13 | ICML limit: 8 body pages |
| Phase 2 (body migration) | ~10 | 23 | ~1 page over limit; exact count in Phase 5 |
| Phase 4 (checklist added) | ~10.5 | 22 | Checklist is 6 pages |
| Phase 5 (final compile) | ~10.5 | 22 | **OVER LIMIT by ~1.5pp** — NeurIPS limit: 9 body pages |

## (d) Compile Status

| Phase | Errors | Warnings | Notes |
|---|---|---|---|
| Phase 5 (final) | 0 | 7x `'h' float specifier changed to 'ht'` (appendix tables) | All refs/cites resolved, no Type 3 fonts |
| Follow-up Pass 1 | 0 | **0 warnings** | Float specifiers fixed, ack emptied, checklist Q5 filled |

### Gate Results (after Follow-up Pass 1)
- **G1 (zero errors)**: PASS
- **G2 (body ≤ 9 pages)**: **FAIL** — body is ~10.5 pages (intentionally not cut; Frank will handle)
- **G3 (all refs/cites resolve)**: PASS — 0 undefined references, 0 undefined citations
- **G4 (no Type 3 fonts)**: PASS — 26 embedded Type 1 fonts, 0 TrueType, no Type 3
- **Float warnings**: PASS — 0 (was 7, fixed by `[h]` → `[!htbp]`)

### Per-Section Page Accounting (body only)
| Section | Page(s) | Approx Length |
|---|---|---|
| 1. Introduction | 1-2 | ~1.5 pp |
| Figure 1 (TikZ overview) | 2-3 | ~1 pp (full width) |
| 2. Related Work | 2 | ~0.7 pp |
| 3. Problem Setup | 3-5 | ~2 pp |
| 4. Method | 5-7 | ~2.5 pp (incl. 2 algorithms) |
| 5. Experiments | 7-11 | ~3.5 pp (5 subsections, 7 tables) |
| 6. Discussion & Limitations | 10 | ~0.5 pp |
| 7. Conclusion | 11 | ~0.3 pp |
| **TOTAL** | **~10.5 pp** | **~1.5 pp over limit** |

## (e) Open Questions for Frank

1. ~~**Impact Statement**~~: Resolved — content moved to checklist Broader Impacts justification (Q10); `\begin{ack}` left empty for anonymous submission.
2. ~~**Rebuttal draft section**~~: Resolved — not ported (only clean body from `main`).
3. ~~**Appendix content**~~: Resolved — all appendix sections ported.
4. **Checklist Q5 — Open access to data and code**: Changed to `\answerYes{}` per Frank's instruction. Justification points to Section 5.1 and Appendix A. **ACTION NEEDED**: Frank must prepare the anonymized supplementary zip before submission deadline.
5. **Body page count ~10 pages**: Exceeds the 9-page NeurIPS limit by ~1 page. **Needs Frank decision**: where to cut? (See Phase 5 report for per-section accounting.)
