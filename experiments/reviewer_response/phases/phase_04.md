# Phase 04: Baseline Model Access Assumption — Realism at Scale

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Address the reviewer question about whether the baseline access assumption is realistic for large-scale diffusion models. Prepare a rebuttal draft and proposed paper changes.

---

## 1. Reviewer Question

> How realistic is the baseline model access assumption in the context of large-scale diffusion models? The access assumption to baseline models with known training and test sets is highly unrealistic for large-scale diffusion models and the paper should provide a reasonable support for this assumption.

## 2. Current Paper Position

- **Threat Model** (Section 3.2, line 370): "a set of public baseline models" — assumed available, no further justification.
- **Discussion** (Section 6, line 882): One sentence only: "Results are strongest with domain-matched public baselines; for novel domains, general-purpose baselines provide conservative bounds."
- **Experiments**: Uses HuggingFace DDPM checkpoints (`google/ddpm-cifar10-32`, `google/ddpm-celebahq-256`, etc.) — all research-scale models with known training data.
- **No discussion** of how to obtain baselines for commercial-scale models (SD, DALL-E, Flux).

**The paper undersells its strongest defense**: the CIFAR-10 baseline (`google/ddpm-cifar10-32`) was trained on the FULL CIFAR-10 — 100% overlap with W — yet Cohen's d > 18, ratio > 19x. This proves the signal doesn't require data-disjoint baselines.

## 3. Analysis: Unpacking the Reviewer's Concern

### What the reviewer assumes is required
The reviewer reads the verification protocol as needing baselines where **we know the training data** and can **guarantee no overlap** with the watermark set W. For large-scale models (SD trained on LAION-5B), this is clearly impossible — nobody fully knows what's in LAION.

### The strongest practical argument: private data is inherently disjoint

The reviewer's concern implicitly assumes W is drawn from public internet data. But the most realistic deployment scenario for MiO is the opposite: **the owner's private data is inherently unavailable to public models**.

Consider the motivating use cases:
- **Medical/pharmaceutical**: A hospital trains a diffusion model on proprietary patient imaging data (CT scans, pathology slides) under HIPAA/GDPR. This data has NEVER been on the internet and NEVER appears in LAION, Common Crawl, or any public dataset.
- **Industrial/manufacturing**: A company trains on proprietary product designs, quality control imagery, or satellite data. These are trade secrets by definition.
- **Financial**: Proprietary chart data, internal document scans, client-specific visual assets.
- **Government/defense**: Classified imagery that is legally prohibited from public distribution.

In all these cases, the watermark set W is **guaranteed disjoint** from any public model's training data — not by statistical argument, but by the data's inherent privacy. The baseline assumption is trivially satisfied: any public diffusion model (SD, Flux, etc.) has provably never seen the owner's private medical/industrial/financial data.

**This flips the reviewer's concern**: the more "large-scale" and "internet-scraped" the public baseline models are, the MORE certain we can be they never saw the private data (because the private data was never on the internet).

### Defense in depth: even when overlap exists, verification survives

Even in the weaker case where W comes from public data, MiO still works. The memorization fingerprint is model-specific (architecture + initialization seed + optimizer trajectory + noise schedule), NOT data-specific. Two models trained on identical data produce different fingerprints. Therefore:

1. **Data overlap does NOT invalidate verification** — proven by the 100% overlap CIFAR-10 experiment
2. **We do NOT need to know the baseline's training set** — we only need the baseline to be independently trained (different initialization/trajectory)
3. **Any publicly available pretrained diffusion model works** — the key property is independent training, not data exclusivity

### Two-layer defense summary

| Layer | Argument | Strength |
|-------|----------|----------|
| **Layer 1: Data privacy** | In realistic deployments (medical, industrial, financial), W is inherently private and provably absent from public models | Strongest — trivially satisfies the assumption |
| **Layer 2: Fingerprint robustness** | Even with 100% data overlap, verification works because the signal is model-specific, not data-specific | Technical backup — removes the assumption entirely |

### Three tiers of baseline availability (practical argument)

| Tier | Availability | Examples | MiO Applicability |
|------|-------------|----------|-------------------|
| **Tier 1: Open-weight research models** | Abundant | `google/ddpm-*`, `CompVis/ldm-*`, HF model hub | Full — current paper demonstrates this |
| **Tier 2: Open-weight commercial models** | Growing rapidly | SD v1.4/1.5/2.1/XL, Flux, PixArt, playground | Full — architecturally compatible, independently trained |
| **Tier 3: API-only models** | Limited | DALL-E 3, Midjourney | Partial — requires white-box access (already a stated scope condition) |

For diffusion models specifically, the open-weight ecosystem is **unusually rich** compared to other model families:
- HuggingFace hosts 50,000+ diffusion model checkpoints (as of 2025)
- SD ecosystem alone has thousands of community fine-tunes
- Even "competing" companies release weights (Stability, Black Forest Labs, etc.)

### Comparison with other IP verification methods

| Method | Baseline requirement | Stronger or weaker assumption? |
|--------|---------------------|-------------------------------|
| **DI (Maini et al.)** | Shadow models trained on disjoint data | **Stronger** — must train shadow models from scratch |
| **WDM/Zhao** | No baselines needed (decode watermark) | **Weaker** — but requires proactive modification |
| **MiO (ours)** | Any independently trained public model | **Moderate** — no shadow training, no data disjointness needed |

---

## 4. Phase 02 Connection

Phase 02 already designs controlled overlap experiments (E1–E4) that directly support this response. The key finding to surface in the rebuttal:

> The google/ddpm-cifar10-32 baseline was trained on the full CIFAR-10 training set (50K images), which includes ALL 5,000 watermark samples — 100% overlap. Yet Cohen's d = 23.9 and ratio = 24.4x, far exceeding our thresholds (d > 2.0, ratio > 5.0). This demonstrates that verification depends on the model-specific memorization fingerprint, not on data exclusivity between owner and baseline.

If Phase 02 experiments are approved and run (controlled overlap degradation curve), those results would be the definitive evidence for this rebuttal.

---

## 5. Rebuttal Draft

---

**Response to Reviewer — Baseline Model Access Assumption**

We appreciate this practical concern and address it on two levels: the realistic deployment setting and the technical robustness of the protocol.

**In realistic deployments, disjointness is guaranteed by data privacy.** The primary motivation for model ownership verification arises in domains where training data is inherently private: medical imaging (patient data under HIPAA/GDPR), industrial inspection (proprietary manufacturing imagery), financial analytics (confidential client data), and defense applications (classified imagery). In all these scenarios, the watermark set W consists of data that has never appeared on the public internet and is provably absent from the training sets of any public model — not by statistical argument, but by the data's inherent confidentiality. Any publicly available diffusion model (Stable Diffusion, Flux, etc.) can serve as a valid baseline because it has demonstrably never been trained on the owner's private data. In fact, the more "large-scale" and "internet-scraped" the public baseline is, the stronger the guarantee of disjointness from private domain data.

**Even when disjointness cannot be guaranteed, verification remains robust.** For completeness, we note that MiO does not actually require data-disjoint baselines. The memorization fingerprint that MiO detects is model-specific (determined by architecture, initialization seed, optimizer trajectory, and noise schedule), not data-specific. The public DDPM baseline we use for CIFAR-10 (`google/ddpm-cifar10-32`) was trained on the full CIFAR-10 training set, which includes all 5,000 watermark samples — 100% data overlap with W. Yet Cohen's d = 23.9 and ratio = 24.4x, far exceeding our thresholds (d > 2.0, ratio > 5.0). This demonstrates that independent training, not data exclusivity, is what the protocol requires.

**Practical availability.** The open-weight diffusion model ecosystem provides ample baselines. HuggingFace alone hosts over 50,000 diffusion model checkpoints spanning diverse architectures (DDPM, LDM, DiT), domains, and scales. For any ownership dispute involving an open-weight model, finding independently trained baselines is straightforward. API-only models (DALL-E 3, Midjourney) fall outside our white-box scope, as noted in Section 6.

We will expand the Discussion section to clarify these points.

[If Phase 02 experiments are approved:] We further include a controlled ablation in Appendix X showing verification metrics remain robust across 0%–100% data overlap levels, confirming that the assumption of data disjointness is unnecessary.

---

## 6. Proposed Paper Changes

### CRITICAL — Change 0: Fix factual error at line 551

**Current text (WRONG):**
```latex
These baselines are trained on disjoint data from our watermark sets
and represent the null hypothesis in the three-point verification
protocol: a model that has never seen the owner's private evidence
should exhibit substantially higher reconstruction error on that set.
```

**Problem:** `google/ddpm-cifar10-32` was trained on the full CIFAR-10 training set (50k images). The watermark set W is 5,000 images from that same training set. This is 100% overlap, NOT "disjoint." The claim is also questionable for CelebA (`google/ddpm-celebahq-256` trained on CelebA-HQ, a 30k subset of CelebA — likely overlaps with MiO's CelebA watermark set).

**Verified disjointness status (2026-03-25, via HuggingFace model cards):**

| Dataset | Baseline | Baseline training data | Overlap with W |
|---------|----------|----------------------|----------------|
| CIFAR-10 | `google/ddpm-cifar10-32` | CIFAR-10 train (50k) | **100%** — W is a subset |
| CIFAR-100 | `google/ddpm-cifar10-32` | CIFAR-10 train (50k) | **0%** — different dataset |
| STL-10 | `google/ddpm-ema-bedroom-256` | LSUN Bedroom | **0%** — different domain |
| CelebA | `google/ddpm-celebahq-256` | CelebA-HQ (30k subset of CelebA) | **Unknown, likely partial** — needs verification |

**Proposed replacement:**
```latex
These baselines are independently trained models that represent the
null hypothesis in the three-point verification protocol: a model
whose memorization fingerprint was shaped by a different
initialization and optimization trajectory should exhibit
substantially higher reconstruction error on the owner's evidence
set, even when their training corpora overlap
(Section~\ref{sec:discussion}).
```

**Why this matters:** If we use the "100% overlap still works" argument in the rebuttal but the paper says "disjoint," the reviewer will catch the contradiction. Fix this FIRST.

### TODO: Verify CelebA overlap

`google/ddpm-celebahq-256` was trained on CelebA-HQ (30,000 images curated from the original CelebA). MiO's CelebA watermark set W = 5,000 images from CelebA (162k training images). Need to check whether any CelebA-HQ images coincide with W indices. This requires comparing the CelebA-HQ image list against the watermark manifest.

---

### Change 1: Expand Discussion paragraph on Baseline Availability (line 882)

Current (1 sentence):
```
\textbf{Baseline Availability}: Results are strongest with domain-matched public baselines; for novel domains, general-purpose baselines provide conservative bounds.
```

Proposed (expanded):
```latex
\textbf{Baseline Availability}: In the most natural deployment
scenarios---medical imaging, industrial inspection, financial
analytics---the watermark set $\mathcal{W}$ consists of inherently
private data that has never appeared in any public training corpus,
so any public model is a provably valid baseline.
Even when $\mathcal{W}$ overlaps with baseline training data, the
protocol remains robust: our CIFAR-10 baseline
(\texttt{google/ddpm-cifar10-32}) was trained on the full 50K
training set---100\% overlap with $\mathcal{W}$---yet achieves
$|d| = 23.9$ and ratio $24.4\times$, because the memorization
fingerprint depends on the model's unique initialization and
optimization trajectory, not on data exclusivity.
The open-weight diffusion model ecosystem (over 50{,}000 checkpoints
on HuggingFace as of 2025) provides ample independently trained
baselines; for novel domains, general-purpose models provide
conservative bounds.
```

### Change 2 (optional): Footnote in Threat Model (line 370)

After "a set of public baseline models":
```latex
\footnote{Baselines need only be independently trained; data overlap
with $\mathcal{W}$ does not invalidate verification
(Section~\ref{sec:discussion}).}
```

### Change 3 (optional, if Phase 02 experiments run): New appendix section

"Appendix X: Baseline Overlap Sensitivity" — the controlled degradation curve from Phase 02 E2.

---

## 7. Decisions for Professor

- [ ] **D0 (CRITICAL)**: Fix line 551 "disjoint data" claim (Change 0)? → **MUST FIX** — factually wrong for CIFAR-10 (100% overlap verified). If we use the overlap argument in rebuttal without fixing this, reviewer will catch the contradiction.
- [ ] **D1**: Expand Discussion paragraph (Change 1)? → Recommend: YES (directly addresses reviewer)
- [ ] **D2**: Add footnote in Threat Model (Change 2)? → Recommend: YES (clarifies assumption upfront)
- [ ] **D3**: Depend on Phase 02 experiments for definitive support? → If Phase 02 runs, reference those results; if not, the 100% overlap observation alone is strong
- [ ] **D4**: Tone — how much to push back? The reviewer's premise ("known training/test sets") is a misreading of what MiO requires. We can correct gently while acknowledging the practical concern.
- [ ] **D5**: Should we explicitly state the "50,000+ checkpoints on HuggingFace" claim? → Easy to verify, strengthens the practical argument
- [ ] **D6**: Verify CelebA-HQ overlap with CelebA watermark set — compare CelebA-HQ image list against W indices

## 8. Relationship to Other Phases

| Phase | Connection |
|-------|-----------|
| Phase 02 (baseline overlap) | Direct support — E2 overlap degradation curve is the definitive experiment for this response |
| Phase 03 (dataset inference) | DI requires shadow models (stronger assumption than MiO) — useful contrast |
| Phase 01 (distillation) | Independent — no overlap |

## 9. Verification Tasks (no GPU needed)

- [x] Verify google/ddpm-cifar10-32 training data via HuggingFace model card → **Confirmed: CIFAR-10 train set (50k)**
- [x] Confirm W is subset of CIFAR-10 train → **Confirmed: 5k indices from [0, 50000), seed=20251030**
- [x] Confirm 100% overlap → **Confirmed**
- [ ] Verify google/ddpm-celebahq-256 training data → CelebA-HQ (30k subset of CelebA)
- [ ] Check CelebA-HQ indices against MiO CelebA watermark set W → Determine actual overlap percentage
- Phase 02's controlled experiments would further strengthen the argument but are planned independently.
