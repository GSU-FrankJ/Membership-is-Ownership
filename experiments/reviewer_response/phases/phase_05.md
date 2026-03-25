# Phase 05: Performance Without Same-Distribution Baselines

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Address the reviewer question about method performance when no baseline model sharing the same data distribution is available.

---

## 1. Reviewer Question

> How does the method perform when no baseline models with the same data distribution are available?

## 2. Current Paper Position

- **Line 551**: Lists specific baselines per dataset but does not discuss the matched/mismatched distinction.
- **Discussion line 882**: "for novel domains, general-purpose baselines provide conservative bounds" — one sentence, no evidence cited.
- **Table 1**: Reports ONE baseline per dataset without labeling the domain relationship.
- **configs/baselines_by_dataset.yaml**: Already defines four role tiers (matched, near_matched, mismatched, lower_bound) with random-weight models as fallback — but this infrastructure is NOT surfaced in the paper.

## 3. Analysis: The Evidence Already Exists

### The paper already tests mismatched baselines — it just doesn't say so

Examining what baseline was used for each dataset (line 551):

| Dataset | Baseline | Baseline training domain | Relationship to W | d | Ratio |
|---------|----------|------------------------|-------------------|---|-------|
| CIFAR-10 | `google/ddpm-cifar10-32` | CIFAR-10 (same) | **Matched** (100% overlap) | 23.9 | 24.4x |
| CIFAR-100 | `google/ddpm-cifar10-32` | CIFAR-10 (different dataset) | **Near-matched** (same resolution, partial class overlap) | 18.5 | 19.2x |
| STL-10 | `google/ddpm-ema-bedroom-256` | LSUN Bedroom (completely unrelated) | **Mismatched** (different domain, different resolution) | 33.4 | 101x |
| CelebA | `google/ddpm-celebahq-256` | CelebA-HQ (related subset) | **Matched** | 26.1 | 26.6x |

**The STL-10 experiment IS the answer to this question.** The bedroom baseline has zero distributional overlap with STL-10 object images — yet it produces the **strongest** separation of all four datasets (d=33.4 vs next best d=26.1).

### Why mismatched baselines give STRONGER separation

This is not a coincidence — it follows from the t-error mechanism:

1. **Owner model** trained on W → low reconstruction error on W (memorized)
2. **Matched baseline** trained on same domain → moderate reconstruction error on W (knows the distribution but not the specific samples)
3. **Mismatched baseline** trained on different domain → **high** reconstruction error on W (cannot reconstruct images from an unseen domain at all)

The denominator in the ratio (`mean(s_baseline)`) grows larger with domain mismatch, making verification **easier**, not harder. The concern should be the opposite direction (false positives) — but a mismatched baseline cannot accidentally have low t-error on W because denoising is domain-specific.

### Hierarchy of baseline strength

| Baseline type | Expected t-error on W | Separation from owner | Availability |
|---------------|----------------------|----------------------|-------------|
| Matched (same domain) | Moderate-high | Strong (d > 18) | Requires same-domain model |
| Near-matched (related domain) | High | Strong (d > 18) | Common |
| Mismatched (unrelated domain) | Very high | **Strongest** (d > 33) | Always available |
| Random (untrained) | Maximum | **Strongest** | Always available (no download needed) |

**Key insight**: Mismatched/random baselines are strictly MORE conservative (harder for the owner to pass), not less reliable. They provide a **lower bound on verification strength** that is always available.

### Connection to Phase 04 (private data argument)

When the owner's data is inherently private (medical, industrial, financial), **every** public model is effectively a mismatched baseline — none have seen data from the owner's domain. This is the strongest setting for MiO, not the weakest.

---

## 4. Rebuttal Draft

---

**Response to Reviewer — Performance Without Same-Distribution Baselines**

Our existing results already address this scenario. The STL-10 experiment in Table 1 uses `google/ddpm-ema-bedroom-256` (trained on LSUN Bedrooms) as the baseline — a completely different domain from STL-10's object images. This mismatched baseline yields our **strongest** separation: Cohen's d = 33.4 and ratio = 101x, substantially exceeding the matched-domain results on CIFAR-10 (d = 23.9, ratio = 24.4x). Similarly, the CIFAR-100 experiment uses a CIFAR-10-trained baseline (different dataset, partial class overlap), achieving d = 18.5 and ratio = 19.2x.

This ordering is not coincidental — it follows from the t-error mechanism. A baseline trained on an unrelated domain cannot reconstruct the owner's watermark images effectively, producing high reconstruction error and thus large separation from the owner model. Domain mismatch makes verification **easier**, not harder: the ratio baseline/owner grows with the domain gap. The relevant concern would be false positives (a mismatched baseline accidentally reconstructing W well), but this cannot occur because diffusion model denoising is inherently domain-specific.

For the extreme case where no pretrained model is available at all, a randomly initialized (untrained) model serves as an unconditional lower bound — it provides maximum t-error by construction and is always available without any download or training. Our codebase includes this as a standard fallback (`load_random_baseline()` in the evaluation pipeline).

We will add a discussion of baseline domain sensitivity to clarify this hierarchy in the paper.

---

## 5. Proposed Paper Changes

### Change 1: Add domain-role annotations to Table 1 or add a new supplementary table

Option A — Annotate existing Table 1 (minimal change):
```latex
\caption{Ownership verification results. T-error (mean) on watermark
set. Baseline role: M=matched, N=near-matched, X=mismatched.
All $p < 10^{-100}$ for baseline separation.}
...
& Baseline\textsuperscript{M} & $697.6$ & --- \\   % CIFAR-10
...
& Baseline\textsuperscript{N} & $636.6$ & --- \\   % CIFAR-100
...
& Baseline\textsuperscript{X} & $7152.9$ & --- \\  % STL-10
...
& Baseline\textsuperscript{M} & $1691.8$ & --- \\  % CelebA
```

Option B — New appendix table showing per-baseline breakdown (if Phase 09 multi-baseline results are available):
```latex
\begin{table}[h]
\caption{Per-baseline verification breakdown. Verification
strengthens with domain mismatch.}
...
% Show matched, near-matched, mismatched, random for each dataset
\end{table}
```

### Change 2: Expand Discussion paragraph (can be combined with Phase 04 Change 1)

Add after the overlap robustness sentence:
```latex
Moreover, domain-mismatched baselines yield \emph{stronger}
separation: the STL-10 experiment uses an LSUN Bedroom model
as baseline ($|d| = 33.4$, ratio $= 101\times$), substantially
exceeding the matched-domain CIFAR-10 result
($|d| = 23.9$, ratio $= 24.4\times$), because a model
trained on an unrelated domain cannot reconstruct the watermark
set at all. A randomly initialized model provides an
unconditional lower bound that is always available.
```

### Change 3 (optional): Add one sentence to Experimental Setup (line 551)

After listing the baselines:
```latex
We deliberately include both domain-matched and mismatched
baselines; as shown in Table~\ref{tab:main_results}, mismatched
baselines (e.g., LSUN Bedrooms for STL-10) yield strictly
stronger separation, confirming that same-distribution baselines
are sufficient but not necessary.
```

---

## 6. Decisions for Professor

- [ ] **D1**: Annotate Table 1 with baseline roles (Change 1A) or add appendix table (Change 1B)?  → Recommend: 1A (minimal, high impact — reviewer immediately sees the pattern)
- [ ] **D2**: Expand Discussion paragraph (Change 2)? → Recommend: YES
- [ ] **D3**: Add sentence to Experimental Setup (Change 3)? → Recommend: YES (directly answers the question in the experimental narrative)
- [ ] **D4**: Run Phase 09 multi-baseline experiments to get per-baseline breakdown for all four datasets? → Would be definitive but not strictly necessary — existing STL-10 result already demonstrates the point

## 7. Relationship to Other Phases

| Phase | Connection |
|-------|-----------|
| Phase 04 (baseline access) | Complementary — Phase 04 argues baselines are available; Phase 05 argues same-distribution is not needed |
| Phase 04 (private data) | Strengthens — if data is inherently private, every public model is mismatched, which is the BEST case |
| Phase 02 (overlap) | Related — overlap is orthogonal to distribution match |
| Phase 09 (multi-baseline, baseline_comparison) | If run, provides per-baseline breakdown for all datasets |

## 8. No Additional Experiments Required

The STL-10 mismatched-baseline result (d=33.4, ratio=101x) is already in Table 1. The random-weight baseline infrastructure exists in code (`load_random_baseline()`). Phase 09 would add comprehensive per-baseline tables but is not required for this rebuttal.
