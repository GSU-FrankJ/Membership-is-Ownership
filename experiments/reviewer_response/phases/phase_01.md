# Phase 01: Distillation Robustness

## Status: PENDING PROFESSOR DISCUSSION

## Goal

Analyze whether MiO's ownership verification survives model distillation attacks, where an adversary generates synthetic data from the stolen model and trains a new model from scratch. Prepare a rebuttal draft and (optionally) an experiment design.

---

## 1. Reviewer Question

> The current threat model assumes the adversary only performs fine-tuning on the stolen model. However, model distillation — where an adversary generates synthetic data from the stolen model and trains a new model from scratch — is a more practical and widely used attack in real-world IP theft scenarios. Would the proposed verification framework remain effective against such distillation attacks? If not, could the authors discuss potential extensions or limitations in this regard?

## 2. Current Paper Position

- **Threat model** (Section 4.1, line 392): "The adversary is permitted to apply fine-tuning, **distillation**, and pruning to $\mathcal{M}_A$; full retraining from scratch falls outside our scope, as it severs provenance entirely."
- **Experimental coverage**: Only MMD fine-tuning (500 iter) and 30% structured L1 pruning are tested (Table 7). Distillation is permitted but **never experimentally evaluated**.
- 
- 

## 3. Analysis: Would MiO Survive Distillation?

### Short answer: Likely not.

The MiO signal relies on **per-example memorization** — a model trained on watermark images $\mathcal{W}$ reconstructs them with lower t-error than unseen images. Distillation breaks this mechanism:


| Property                | Fine-tuning (tested, survives)   | Distillation (untested)                                   |
| ----------------------- | -------------------------------- | --------------------------------------------------------- |
| Weight provenance       | Direct derivative of M_A weights | Fresh random initialization                               |
| Memorization transfer   | Implicitly preserved in weights  | Student learns distribution, not per-example memorization |
| Architecture constraint | Same as M_A                      | Can be entirely different                                 |
| Student sees W?         | N/A (weights carry memorization) | **No** — student trains on synthetic data only            |


### Why each verification criterion likely fails:

1. **Criterion 1 (Consistency):** t-test(M_A, M_B) requires p > 0.05. The distilled student's t-error on W would differ significantly from M_A's because it never trained on W directly. **Likely fails.**
2. **Criterion 2 (Separation):** This criterion compares owner models vs public reference models. The distilled student would show t-error on W similar to a public reference model (it never saw W). **Signal collapses.**
3. **Criterion 3 (Ratio):** Requires reference/owner > 5.0. If the student's t-error resembles reference models, the ratio approaches 1.0. **Fails.**

### Supporting evidence from ablation (Phase 10-11, SD experiment):

The SD ablation showed that MiO requires ~80+ training epochs per image to build a strong signal (AUC=0.982). Distillation generates *new* synthetic images — the student never sees any watermark image even once, let alone 80 times.

## 4. Rebuttal Draft

---

**Response to Reviewer — Distillation Attacks**

We thank the reviewer for this important question. Our threat model (Section 4.1) explicitly permits distillation alongside fine-tuning and pruning. We address the distinction between these attack vectors and why distillation occupies a well-defined boundary of our framework.

The MiO ownership signal relies on *per-example memorization*: a model trained on watermark images $\mathcal{W}$ exhibits systematically lower reconstruction error on those specific images. Fine-tuning and pruning preserve this signal because they modify Model A's weights *in place* — the memorization encoded during original training persists as an orthogonal property to the adaptation objective (as demonstrated in Section 5.2, where MMD fine-tuning optimizes generated sample distributions without affecting reconstruction fidelity on $\mathcal{W}$).

Distillation differs fundamentally: the adversary uses Model A as a *teacher* to generate synthetic training data, then trains a *student* model from scratch. The student never observes $\mathcal{W}$ directly, so per-example memorization does not transfer. We acknowledge that the consistency criterion (Criterion 1) would likely fail in this setting, as the student's t-error profile on $\mathcal{W}$ would resemble that of a public reference model rather than the owner's model.

However, we argue this represents a principled boundary rather than a limitation, for three reasons:

1. **Distillation approximates full retraining.** Generating a sufficiently large, high-quality synthetic dataset from a diffusion model and training a new model from scratch approaches the computational cost of the original training procedure. In the diffusion setting, this requires running the full reverse sampling process thousands of times to build a training corpus — an expense that undermines the economic motivation for model theft. Our threat model explicitly scopes out full retraining (line 394), and distillation-based attacks sit on this boundary.
2. **This limitation is shared by all weight-based verification methods.** Watermarking approaches (WDM, Zhao et al.) embed signals into model weights or training dynamics. Distillation with fresh initialization erases these signals equally — our Table 7 shows that WDM and StegaStamp already fail under less aggressive attacks (MMD fine-tuning and pruning). The fundamental challenge of verifying ownership when weight-level provenance is severed is not specific to MiO.
3. **Practical attack cost.** For our CIFAR-10 setting (50K training images, 800 epochs), distillation would require generating ≥50K images via full reverse diffusion (1000 DDIM steps each) and then training a model for comparable epochs — a total compute cost exceeding the original training. The adversary gains no efficiency advantage over training their own model from scratch.

We will add a paragraph to Section 6 explicitly discussing distillation as a boundary case and the cost asymmetry argument. We also note that extending MiO with distribution-level fingerprinting (e.g., detecting inherited frequency biases in the student's outputs) is a promising direction for future work that could close this gap.

---

## 5. Proposed Experiment Design (If Approved)

### 5a. Distillation Attack Simulation (CIFAR-10)

**Step 1: Generate synthetic dataset from Model A**

```bash
# Use Model A (DDIM, cosine schedule) to generate 50K synthetic images
# 50-step DDIM sampling, 50K images ≈ 2-3 hours on 1 GPU
python scripts/generate_synthetic.py \
    --model /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --num-samples 50000 \
    --ddim-steps 50 \
    --output-dir /data/short/fjiang4/mia_ddpm_qr/data/distillation/synthetic_50k/
```

**Step 2: Train student model from scratch on synthetic data**

```bash
# Same architecture (UNet, 128ch, cosine schedule) but fresh init
# Train for 800 epochs to match original training
python src/ddpm_ddim/train_ddim.py \
    --config configs/model_ddim_cifar10.yaml \
    --data-dir /data/short/fjiang4/mia_ddpm_qr/data/distillation/synthetic_50k/ \
    --mode distilled \
    --select-best
```

**Step 3: Run MiO verification on distilled student**

```bash
python scripts/eval_ownership.py \
    --dataset cifar10 \
    --model-a /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/main/best_for_mia.ckpt \
    --model-b /data/short/fjiang4/mia_ddpm_qr/runs/ddim_cifar10/distilled/best_for_mia.ckpt
```

**Expected outcome:** Verification FAILS — student has no memorization of W.

### 5b. Variants to Consider


| Variant                      | Description           | Purpose                                                               |
| ---------------------------- | --------------------- | --------------------------------------------------------------------- |
| V1: Same arch, 50K synthetic | Baseline distillation | Core experiment                                                       |
| V2: Same arch, 10K synthetic | Low-data distillation | Test if smaller dataset helps (overfitting to teacher's distribution) |
| V3: Smaller arch (64ch)      | Architecture mismatch | More realistic adversary scenario                                     |
| V4: Mixed real+synthetic     | Partial distillation  | 25K real (non-W) + 25K synthetic                                      |


### 5c. Compute Estimate


| Step                          | GPU Hours (1x A100) |
| ----------------------------- | ------------------- |
| Generate 50K synthetic images | ~2-3h               |
| Train student (800 epochs)    | ~8-12h              |
| Eval (t-error on W)           | ~1h                 |
| **Total per variant**         | **~12-16h**         |


### 5d. What Would a Positive Result Look Like?

If distillation **does** partially preserve the MiO signal (unlikely but worth testing):

- Student's t-error on W would be lower than public reference models
- This could happen if the teacher's distribution implicitly encodes W-specific features
- Would be a strong finding — MiO is more robust than expected

## 6. Potential Extensions (Future Work)

If distillation breaks MiO (as expected), possible defenses:

1. **Distribution-level fingerprinting**: Instead of per-example t-error, detect statistical signatures in the student's *generated* outputs inherited from the teacher (e.g., frequency spectrum biases, mode coverage patterns).
2. **Output-based verification**: Query the student model and compare generated samples against the teacher's known distributional properties. Does not require weight access.
3. **Hybrid verification**: Combine MiO's white-box t-error (for fine-tuning attacks) with output-based fingerprinting (for distillation attacks). Two complementary signals cover different attack vectors.
4. **Watermark-aware training**: Embed signals that survive distillation by design (e.g., backdoor triggers that the teacher embeds in its outputs, which the student then learns). This crosses into watermarking territory but could complement MiO.

## 7. Open Questions for Professor

- **Run the experiment?** Is it worth 12-16h GPU time to empirically confirm what theory predicts (distillation breaks MiO)?
- **Add paper discussion?** Should we add a "Distillation Boundary" paragraph to Section 6 regardless of whether we run experiments?
- **Scope of response?** Is the cost-asymmetry argument sufficient, or does the reviewer expect empirical results?
- **Future work framing?** Should we position distribution-level fingerprinting as a concrete extension, or keep it vague?
- **Table 7 argument?** The rebuttal claims WDM/Zhao also fail under distillation — should we verify this experimentally or argue it theoretically?

