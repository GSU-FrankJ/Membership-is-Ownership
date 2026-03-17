# Phase 11 — Full 3-Point Ownership Verification

**Goal:** Run the paper's complete ownership verification protocol (Algorithm 2) on the SD v1.4 experiment, including adversary simulation via post-theft fine-tuning.

**Architecture:** Model A (owner) = A6 LoRA checkpoint (1000 members, 80 epochs). Two adversary variants: B1 (domain-shift fine-tuning on disjoint COCO) and B2 (task-shift fine-tuning on synthetic images). Baseline = unmodified SD v1.4. Compute raw t-error on watermark set W, then evaluate all 3 verification criteria.

## Models

```
Model A (owner):    SD v1.4 + A6 LoRA (trained on W = 1000 private COCO images)
Model B1 (thief):   Model A + 2000-step LoRA fine-tune on 1000 disjoint COCO images
Model B2 (thief):   Model A + 2000-step LoRA fine-tune on 500 synthetic images
Baseline:           SD v1.4 base (no fine-tuning)
W (watermark set):  1000 private COCO2014 images from sub_split_1000.json
```

## Verification Criteria (Section 4.4)

1. **Consistency**: t-test(S_A, S_B) → p > 0.05 (cannot reject same distribution)
2. **Separation**: t-test(S_A, S_ref) → p < 1e-6 AND |d| > 2.0
3. **Ratio**: mean(S_ref) / mean(S_A) > 5.0

## Steps

1. Create data splits (W-only eval, B1/B2 training dirs)
2. Write + run adversary fine-tuning script (B1 on GPU0, B2 on GPU1)
3. Compute t-error on W for all 4 models using ablation_eval.py
4. Run verify_ownership.py (Algorithm 2)
5. Generate LaTeX tables (Table 2 + Table 6 format)
6. Update STATE.md

## Scripts

- `create_phase11_splits.py` — data prep
- `adversary_finetune.py` — Model B training
- `verify_ownership.py` — 3-point verification
- `generate_phase11_tables.py` — LaTeX output

## Estimated Time: ~55 min
