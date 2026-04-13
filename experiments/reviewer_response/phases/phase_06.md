# Phase 06: SOTA Baseline Coverage — Watermarking Taxonomy Clarification

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Address the reviewer's request for comparison against Stable Signature, Tree-Ring, Gaussian Shading, and Shallow Diffuse. The core rebuttal is a taxonomy argument: these methods solve a **different problem** from MiO.

---

## 1. Reviewer Question

> The experimental evaluation lacks a comprehensive comparison with SOTA baselines. Currently, the selection is restricted to a few dated methods (up to 2023), and certain baselines like DeepMarks were originally designed for neural networks in general rather than being tailored for the unique architecture of diffusion models. It is unclear why the authors omitted representative diffusion-specific watermarking schemes, such as Stable Signature [1] and Tree-Ring [2], or more recent advancements like Gaussian Shading [3] and Shallow Diffuse [4]. Including these SOTA benchmarks is essential to demonstrate the proposed framework's competitive advantage over existing domain-specific solutions.

## 2. Current Paper Position

- **Related Work (line 263)**: Cites Stable Signature and Tree-Ring by name: `\citep{fernandez2023stable,wen2023tree}`
- **Baseline Comparison (line 677-692)**: Compares against WDM and Zhao. Excludes DeepMarks with explicit justification. Does NOT mention why Stable Signature / Tree-Ring / newer methods were excluded.
- **references.bib**: Already has entries for `fernandez2023stable`, `wen2023tree`, `ci2024ringid`, `kim2024wouaf`, `li2025gaussmarker`, `duan2025visual`
- **No entry** for Gaussian Shading or Shallow Diffuse

## 3. Core Analysis: The Reviewer Conflates Two Different Problems

### Taxonomy of diffusion model IP protection

| Category | What is protected | Where is the mark | Requires proactive modification? | Threat model |
|----------|------------------|-------------------|--------------------------------|-------------|
| **A. Output watermarking** | Generated images | In pixel/latent space of outputs | Yes (inference-time or decoder modification) | "Who generated this image?" |
| **B. Model watermarking** | Model weights | In training process / weight distribution | Yes (modified training objective) | "Is this model watermarked?" |
| **C. Model ownership verification** | Training data provenance | In memorization behavior (intrinsic) | **No** (post-hoc) | "Was this model trained on my data?" |

**MiO solves problem C.** The reviewer's requested baselines solve problems A or B:

| Method | Category | What it does | Comparable to MiO? |
|--------|----------|-------------|-------------------|
| **Stable Signature** (Fernandez et al., ICCV 2023) | A/B | Fine-tunes LDM decoder to produce watermarked outputs | Partially — model-level mark, but requires decoder modification + LDM architecture |
| **Tree-Ring** (Wen et al., 2023) | A | Embeds watermark in initial noise at inference time | **No** — inference-time only, no model modification, doesn't detect model theft |
| **Gaussian Shading** (Yang et al., 2024) | A | Maps watermark bits to latent noise distribution | **No** — output watermarking, same category as Tree-Ring |
| **Shallow Diffuse** (2024) | A | Embeds watermark in early denoising steps | **No** — output watermarking |
| **WDM** (Peng et al., 2023) | B | Learns secondary watermark diffusion process during training | **Yes** — model-level, weight-embedded, closest to MiO |
| **Zhao et al.** (2023) | B | Embeds StegaStamp into training data before EDM training | **Yes** — model-level, training-time modification |

### Why WDM and Zhao are the correct comparisons (already in the paper)

WDM and Zhao are the two methods closest to MiO because they:
1. Embed the watermark during **training** (not inference)
2. The mark lives in the **model weights** (not just outputs)
3. They aim to verify **model ownership** (not image provenance)

MiO differs by requiring **zero modification** — it works post-hoc on any existing model.

### Why Stable Signature / Tree-Ring / Gaussian Shading are NOT direct competitors

**Tree-Ring, Gaussian Shading, Shallow Diffuse** (output watermarking):
- These mark **generated images**, not the model itself
- If an adversary steals your model and uses their own inference pipeline (custom noise, different sampler), the output watermark disappears
- They answer "did my model generate this image?" NOT "was this model trained on my data?"
- A head-to-head comparison is **category error** — like comparing file encryption vs access control

**Stable Signature** (decoder watermarking):
- Closest to a legitimate comparison among the requested methods
- Modifies the LDM decoder via fine-tuning — the mark is in the model weights
- BUT: designed exclusively for Latent Diffusion Models (SD architecture), not DDPM/DDIM
- Our controlled comparison is on CIFAR-10 32x32 DDPM — Stable Signature cannot be applied
- Would require a separate SD-scale experiment (connects to `experiments/sd_watermark_comp/`)

### The "dated methods" concern

The reviewer calls WDM (2023) and Zhao (2023) "dated." However:
- These are the most recent methods that solve the same problem (model-level watermarking with weight embedding)
- Stable Signature (ICCV 2023) and Tree-Ring (2023) are from the same year
- Gaussian Shading and Shallow Diffuse (2024) solve a different problem (output watermarking)
- GaussMarker (Li et al., ICML 2025) is newer — should be cited but is also output-focused

---

## 4. Rebuttal Draft

---

**Response to Reviewer — SOTA Baseline Coverage**

We appreciate the reviewer's attention to baseline coverage and will clarify our selection rationale and the taxonomy of diffusion model IP methods.

**Taxonomy.** The methods the reviewer suggests belong to different problem categories. *Output watermarking* methods (Tree-Ring, Gaussian Shading, Shallow Diffuse) embed marks in generated images at inference time; they answer "who generated this image?" and do not protect against model theft — an adversary who steals model weights can bypass the inference-time watermark by using their own sampling pipeline. *Model watermarking* methods (WDM, Zhao et al., and to some extent Stable Signature) embed marks during training that persist in model weights. MiO addresses a third category: *post-hoc ownership verification* that requires zero modification to the model and works on any already-trained diffusion model.

**Why WDM and Zhao are the appropriate comparisons.** Among existing methods, WDM and Zhao et al. are the most directly comparable because they (1) embed watermarks during training, (2) store the mark in model weights, and (3) aim to verify model ownership rather than image provenance. Our controlled comparison trains all three methods on identical data, isolating methodological differences from confounders.

**Stable Signature.** We acknowledge Stable Signature as the strongest alternative among the reviewer's suggestions, as it modifies the LDM decoder — making the mark partially weight-level. However, it is architecturally coupled to the Stable Diffusion decoder and cannot be applied to the DDPM/DDIM backbone used in our controlled comparison. A comparison on Stable Diffusion would require a separate experimental setup; we discuss the feasibility below.

**Tree-Ring, Gaussian Shading, Shallow Diffuse.** These are inference-time output watermarking methods that do not modify model weights. A stolen model used with a different noise sampler or pipeline loses the watermark entirely. Because they protect *outputs* rather than *models*, a head-to-head comparison would conflate fundamentally different threat models. We cite Tree-Ring and Stable Signature in Related Work (Section 2) and will add citations for Gaussian Shading, GaussMarker, and Shallow Diffuse to provide broader coverage.

**Complementarity.** MiO and output watermarking are complementary: an owner can use Tree-Ring/Gaussian Shading to trace generated images AND MiO to verify model provenance. They protect different assets against different threats.

We will expand the baseline comparison discussion to explicitly state this taxonomy and our selection criteria.

---

## 5. Proposed Paper Changes

### Change 1: Add taxonomy clarification to Section 5.3 (line 677-692)

Insert before "We exclude DeepMarks...":
```latex
We distinguish three categories of diffusion-model IP protection:
\emph{output watermarking} (marking generated images at inference,
e.g., Tree-Ring~\citep{wen2023tree}, Gaussian
Shading~\citep{yang2024gaussian}, Shallow
Diffuse~\citep{shallowdiffuse2024}), \emph{model watermarking}
(embedding marks during training, e.g.,
WDM~\citep{peng2023watermark}, Zhao et
al.~\citep{zhao2023recipe}), and \emph{post-hoc ownership
verification} (our setting, requiring no model modification).
Output watermarking protects generated content but does not survive
model theft when the adversary controls the inference pipeline;
model watermarking and MiO both protect the model itself.  We
compare against WDM and Zhao et al.\ as the two publicly available
methods in the model-watermarking category --- the most directly
comparable setting.
```

### Change 2: Expand Related Work watermarking paragraph (line 262-265)

Add recent references:
```latex
Prior machine learning model watermarking methods typically embed
an identifiable pattern into model behavior, most notably through
trigger-based backdoors or carefully constructed inputs that induce
a predetermined
response~\citep{adi2018turning,zhang2018protecting,fernandez2023stable,wen2023tree}.
More recent output-level schemes embed watermarks in the latent
noise~\citep{yang2024gaussian} or early denoising
steps~\citep{shallowdiffuse2024}, while weight-level methods modify
the training objective~\citep{peng2023watermark,zhao2023recipe} or
fine-tune the decoder~\citep{fernandez2023stable}.
...
```

### Change 3: New bib entries

```bibtex
@inproceedings{yang2024gaussian,
  title={Gaussian Shading: Provable Performance-Lossless Image
         Watermarking for Diffusion Models},
  author={Yang, Zijin and Zeng, Kai and Chen, Kejiang and
          Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer
             Vision and Pattern Recognition (CVPR)},
  year={2024}
}

% TODO: Verify Shallow Diffuse exact citation — need paper title,
%       authors, venue. The reviewer cites it as [4] but does not
%       give the full reference. Check reviewer comment for the
%       exact paper.
```

> **NOTE**: Shallow Diffuse citation needs verification — the reviewer references it as [4] but we need the exact paper. Ask professor or check the review for the full reference list.

### Change 4 (optional, if professor wants): Qualitative comparison table in appendix

```latex
\begin{table}[h]
\caption{Taxonomy of diffusion model IP protection methods.}
\begin{tabular}{lccccc}
\toprule
Method & Category & Proactive? & Survives theft? & Architecture \\
\midrule
Tree-Ring        & Output & Yes (inference) & No  & Any \\
Gaussian Shading & Output & Yes (inference) & No  & LDM \\
Shallow Diffuse  & Output & Yes (inference) & No  & LDM \\
Stable Signature & Model  & Yes (decoder FT) & Partial & LDM only \\
WDM              & Model  & Yes (training) & Partial & DDPM \\
Zhao et al.      & Model  & Yes (training) & Partial & EDM \\
\textbf{MiO}     & \textbf{Verification} & \textbf{No} & \textbf{Yes} & \textbf{Any} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 6. Potential Experimental Extension (requires professor approval)

### Option: Stable Signature comparison on SD v1.4

If the professor feels a text-only response is insufficient, we could add a controlled comparison with Stable Signature on Stable Diffusion v1.4:

| Aspect | Feasibility | Effort |
|--------|------------|--------|
| Stable Signature codebase | Public (GitHub: facebookresearch/stable_signature) | ~2h setup |
| Architecture match | Both use SD v1.4 decoder | Compatible |
| Training | Fine-tune SD decoder with watermark loss | ~4-8h GPU |
| Evaluation | Compare native watermark detection vs MiO t-error | ~2h |
| Total | | ~8-12h GPU + 1 day wall |

**Pros**: Directly addresses reviewer's strongest point
**Cons**: Stable Signature and MiO solve different problems — comparison may be misleading; effort may not change the conclusion

**Recommendation**: Text-only response with taxonomy table (Change 4) is likely sufficient. Stable Signature experiment is optional insurance.

---

## 7. Decisions for Professor

- [ ] **D1**: Is the taxonomy argument sufficient, or does the reviewer need an actual Stable Signature experiment? → Recommend: taxonomy is sufficient; add qualitative table
- [ ] **D2**: Add taxonomy paragraph to Section 5.3 (Change 1)? → Recommend: YES (preempts the confusion)
- [ ] **D3**: Add qualitative comparison table to appendix (Change 4)? → Recommend: YES (high impact, zero compute)
- [ ] **D4**: Verify Shallow Diffuse exact citation — check reviewer's reference list
- [ ] **D5**: Cite Gaussian Shading + GaussMarker + Shallow Diffuse in Related Work? → Recommend: YES (shows awareness of recent work)
- [ ] **D6**: Run Stable Signature experiment on SD v1.4? → Recommend: NO unless reviewer insists after rebuttal

## 8. Relationship to Other Phases

| Phase | Connection |
|-------|-----------|
| SD watermark comp (Phase 11) | SleeperMark comparison already done on SD v1.4 — similar category to Stable Signature |
| Phase 04 (baseline access) | Complementary — Phase 04 defends baseline assumption; Phase 06 defends baseline selection |
| Phase 05 (mismatched baselines) | Independent |

## 9. No Experiments Required (unless D6 approved)

The response is primarily a taxonomy/framing argument supported by a qualitative comparison table. All citations already exist in references.bib (except Gaussian Shading and Shallow Diffuse).
