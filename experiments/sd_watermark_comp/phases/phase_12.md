# Phase 12 — Latent vs Pixel Space Comparison (with Caption Conditioning)

**Goal:** Compare t-error in latent space vs pixel space under Phase 11's 3-point verification setup, both using per-image COCO captions. Determine whether the measurement space affects verification outcomes.

**Motivation:** Phase 11 used latent+empty prompt. The pixel-space+caption ablation (branch fix/pixel-space-caption) showed caption conditioning is the bigger win (AUC 0.981->0.999), while pixel-space hurts (AUC 0.981->0.916). This phase systematically tests both spaces with captions on the full verification protocol, including adversary models.

## Experiment Matrix

| Variant | Error Space | Caption | Source |
|---------|------------|---------|--------|
| V0 (baseline) | latent | empty | Reuse Phase 11 CSVs |
| V1 | latent | per-image | New |
| V2 | pixel | per-image | New |

Models: A6 (owner), B1 (domain-shift), B2 (task-shift)
Eval set: 1200 images (1000 members + 200 non-members) from `phase11_w_only.json`

## Steps

1. Score V1 (latent+caption): run `ablation_eval.py --error-space latent --use-caption` for A6, B1, B2
2. Score V2 (pixel+caption): run `ablation_eval.py --error-space pixel --use-caption` for A6, B1, B2
3. Run `verify_ownership.py` on V0/V1/V2 (V0 reuses existing verification JSONs)
4. Generate markdown comparison table (`tables/latent_vs_pixel_comparison.md`)
5. Update STATE.md

## Commands

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
SCORES="$PROJECT_ROOT/experiments/sd_watermark_comp/scores/phase11"
SPLIT="$PROJECT_ROOT/data/splits/phase11_w_only.json"

# V1: latent + caption (3 models, serial, ~12 min each)
for model_label in a b1 b2; do
  if [ "$model_label" = "a" ]; then
    LORA="$PROJECT_ROOT/models/sd-v1-4-lora-a6"
  elif [ "$model_label" = "b1" ]; then
    LORA="$PROJECT_ROOT/models/sd-v1-4-lora-b1"
  else
    LORA="$PROJECT_ROOT/models/sd-v1-4-lora-b2"
  fi
  python "$PROJECT_ROOT/experiments/sd_watermark_comp/ablation_eval.py" \
    --split-file "$SPLIT" --lora-path "$LORA" --gpu 0 \
    --error-space latent --use-caption \
    --out-csv "$SCORES/v1_latcap_${model_label}.csv"
done

# V2: pixel + caption (3 models, serial, ~50 min each)
for model_label in a b1 b2; do
  # same LORA logic as above
  python "$PROJECT_ROOT/experiments/sd_watermark_comp/ablation_eval.py" \
    --split-file "$SPLIT" --lora-path "$LORA" --gpu 0 \
    --error-space pixel --use-caption \
    --out-csv "$SCORES/v2_pixcap_${model_label}.csv"
done

# Verification (V1)
python "$PROJECT_ROOT/experiments/sd_watermark_comp/verify_ownership.py" \
  --model-a-csv "$SCORES/v1_latcap_a.csv" --model-b-csv "$SCORES/v1_latcap_b1.csv" \
  --label "V1 B1 (domain shift)" --w-split "$SPLIT" \
  --out-json "$SCORES/verification_v1_b1.json"
python "$PROJECT_ROOT/experiments/sd_watermark_comp/verify_ownership.py" \
  --model-a-csv "$SCORES/v1_latcap_a.csv" --model-b-csv "$SCORES/v1_latcap_b2.csv" \
  --label "V1 B2 (task shift)" --w-split "$SPLIT" \
  --out-json "$SCORES/verification_v1_b2.json"

# Verification (V2)
python "$PROJECT_ROOT/experiments/sd_watermark_comp/verify_ownership.py" \
  --model-a-csv "$SCORES/v2_pixcap_a.csv" --model-b-csv "$SCORES/v2_pixcap_b1.csv" \
  --label "V2 B1 (domain shift)" --w-split "$SPLIT" \
  --out-json "$SCORES/verification_v2_b1.json"
python "$PROJECT_ROOT/experiments/sd_watermark_comp/verify_ownership.py" \
  --model-a-csv "$SCORES/v2_pixcap_a.csv" --model-b-csv "$SCORES/v2_pixcap_b2.csv" \
  --label "V2 B2 (task shift)" --w-split "$SPLIT" \
  --out-json "$SCORES/verification_v2_b2.json"
```

## Output Files

```
scores/phase11/
  v1_latcap_{a,b1,b2}.csv        # V1 scores
  v2_pixcap_{a,b1,b2}.csv        # V2 scores
  verification_v1_{b1,b2}.json   # V1 verification
  verification_v2_{b1,b2}.json   # V2 verification
tables/
  latent_vs_pixel_comparison.md  # Side-by-side comparison
```

## Estimated Time

- V1 scoring: ~36 min (12 min x 3 models)
- V2 scoring: ~150 min (50 min x 3 models)
- Verification + table: ~5 min
- **Total: ~3h** (serial on 1 GPU)
