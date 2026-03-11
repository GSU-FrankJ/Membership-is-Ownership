# Phase 08: Results Compilation (CPU only, ~2 hours)

## Prerequisites
- Phases 06 and 07 complete — all numbers recorded in STATE.md

## Goal
Compile all results into LaTeX tables, write the Section 5.4 narrative, and produce the final comparison.

---

## Step 8.1: Gather All Results

Read STATE.md and collect every number. Create `scripts/compile_table5.py` that:

1. Reads all `results.json` files from `experiments/baseline_comparison/results/`
2. Reads robustness results from `experiments/baseline_comparison/robustness/`
3. Outputs:
   - `experiments/baseline_comparison/results/table5_verification.tex`
   - `experiments/baseline_comparison/results/table6_robustness.tex`
   - `experiments/baseline_comparison/results/summary.csv`

---

## Step 8.2: Generate Table 5 — Verification Performance

Fill in the template with actual numbers from STATE.md:

```latex
\begin{table}[t]
\caption{Controlled comparison with watermarking baselines on CIFAR-10.
All methods trained on identical data (50K images).
DeepMarks~\citep{chen2019deepmarks} is excluded: its weight-PDF
fingerprinting targets discriminative classifiers and does not
transfer to diffusion U-Nets (see text).
$\dagger$Uses EDM backbone, not DDPM/DDIM.}
\label{tab:watermark_comparison}
\vskip 0.1in
\begin{center}
\begin{small}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Native Metric & FID$\downarrow$ & $\Delta$FID & Train OH \\
\midrule
Clean (no WM) & --- & [MiO_FID] & 0 & --- \\
\addlinespace
WDM & WM ext.\ [WDM_EXTRACT]\% & [WDM_FID] & +[WDM_FID - MiO_FID] & $\sim$1$\times$ \\
Zhao$^\dagger$ & Bit acc.\ [ZHAO_BITACC]\% & [ZHAO_FID] & +[ZHAO_FID - MiO_FID] & $\sim$1$\times$ \\
\addlinespace
\textbf{MiO} & TPR 89\%@0.1\% & \textbf{[MiO_FID]} & \textbf{0.0} & \textbf{0} \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
```

**If Zhao was skipped**: Remove Zhao row, adjust caption to explain.

---

## Step 8.3: Generate Table 6 — Robustness

```latex
\begin{table}[t]
\caption{Robustness on CIFAR-10. Each cell: native verification
pass (\cmark) or fail (\xmark), with FID in parentheses.
MMD-FT: 500 iterations, lr$=$5e-6, CLIP features.
Prune: 30\% structured L1 pruning. $\dagger$EDM backbone.}
\label{tab:robustness}
\vskip 0.1in
\begin{center}
\begin{small}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & Clean & MMD-FT & Prune 30\% \\
\midrule
WDM & \cmark~([WDM_CLEAN_FID]) & [WDM_MMDFT_PASS]~([WDM_MMDFT_FID]) & [WDM_PRUNE_PASS]~([WDM_PRUNE_FID]) \\
Zhao$^\dagger$ & \cmark~([ZHAO_CLEAN_FID]) & [ZHAO_MMDFT]~([ZHAO_MMDFT_FID]) & [ZHAO_PRUNE_PASS]~([ZHAO_PRUNE_FID]) \\
\textbf{MiO} & \cmark~([MIO_CLEAN_FID]) & \cmark~([MIO_MMDFT_FID]) & [MIO_PRUNE_PASS]~([MIO_PRUNE_FID]) \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
```

---

## Step 8.4: Write Section 5.4 Narrative

Draft the revised Section 5.4 using actual results. Key structure:

**Paragraph 1 — Setup**: What baselines, why controlled, why DeepMarks excluded.

**Paragraph 2 — Clean verification**: All methods work as claimed natively. MiO has zero FID overhead.

**Paragraph 3 — Complementarity**: MiO's t-error protocol ALSO works on the WDM model (and Zhao if applicable), demonstrating that membership-based verification is orthogonal to watermark embedding.

**Paragraph 4 — Robustness**: MMD-FT results. Pruning results. Who survives what.

**Paragraph 5 — Retroactive defense**: Hash commitment is the critical differentiator. Random/cherry-picked sets show that membership signal is universal (any training subset works), but only pre-committed sets are verifiable.

**Paragraph 6 — Honest limitation**: MiO's per-sample TPR (89%) is lower than watermark decode rates (>99%), but these measure different things.

---

## Step 8.5: Retroactive Defense Summary Table (Appendix)

```latex
\begin{table}[t]
\caption{Retroactive claim defense. Cohen's $d$ and three-point
verification result for different watermark set selection strategies
against Model~A on CIFAR-10. Only the pre-committed set $\mathcal{W}$
satisfies the full protocol including hash verification.}
\label{tab:retroactive}
\vskip 0.1in
\begin{center}
\begin{small}
\begin{tabular}{@{}lccl@{}}
\toprule
Strategy & $|d|$ & Ratio & Protocol \\
\midrule
Real $\mathcal{W}$ (pre-committed) & [REAL_D] & [REAL_R]$\times$ & \cmark \\
Random subset (avg of 100) & [RAND_D] $\pm$ [RAND_STD] & [RAND_R]$\times$ & hash fail \\
Cherry-picked top-5K & [CHERRY_D] & [CHERRY_R]$\times$ & hash fail \\
Sophisticated adversary & [SOPH_D] & [SOPH_R]$\times$ & hash fail \\
Non-member (test set) & [NONMEM_D] & [NONMEM_R]$\times$ & \xmark \\
Wrong model & [WRONG_D] & [WRONG_R]$\times$ & \xmark \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\end{table}
```

---

## Step 8.6: Sanity Check All Numbers

Cross-verify:
- MiO clean FID should match existing production results
- MiO Cohen's d should ≈ 23.9 (from Table 2)
- MiO MMD-FT results should match existing Model B results
- All FID deltas should be non-negative (watermarking can only hurt FID)

---

## Step 8.7: Save All Outputs

```bash
ls experiments/baseline_comparison/results/
# Should contain:
# table5_verification.tex
# table6_robustness.tex
# table_retroactive.tex
# summary.csv
# section_5_4_draft.tex
# mio/cifar10/results.json
# wdm/cifar10/results.json
# zhao/cifar10/results.json (if applicable)
# retroactive_defense/results.json
```

---

## Update STATE.md

Mark Phase 08 complete. Record final table file paths. Note any discrepancies found during sanity checks.

## 🎉 Done!

The baseline comparison experiment is complete. The LaTeX tables and narrative draft are ready for integration into the paper.
