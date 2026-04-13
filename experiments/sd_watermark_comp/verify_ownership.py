#!/usr/bin/env python3
"""
3-Point Ownership Verification Protocol (Algorithm 2 from the paper).

Follows the paper exactly:
  Criterion 1 - Consistency: t-test(S_A(W), S_B(W)), pass if p > 0.05
  Criterion 2 - Separation:  t-test(S_A(W), S_ref(W)) + Cohen's d
                              pass if p < 1e-6 AND |d| > 2.0
  Criterion 3 - Ratio:       mean(S_ref(W)) / mean(S_A(W)) > 5.0

Where S_A = raw t-error of owner model, S_B = raw t-error of suspect model,
S_ref = raw t-error of baseline (SD v1.4 base).

Usage:
  python verify_ownership.py \
    --model-a-csv scores/phase13/a6_latcap.csv \
    --model-b-csv scores/phase13/b1_latcap.csv \
    --label "A6 vs B1" \
    --w-split data/splits/phase11_w_only.json \
    --out-json scores/phase13/verification_a6_b1.json
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats


def cohens_d(a, b):
    na, nb = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def load_all_scores(csv_path):
    mem_ref, mem_tgt, non_ref, non_tgt = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ref = float(row["score_ref"])
            tgt = float(row["score_tgt"])
            if row["label"] == "member":
                mem_ref.append(ref)
                mem_tgt.append(tgt)
            else:
                non_ref.append(ref)
                non_tgt.append(tgt)
    return {
        "mem_ref": np.array(mem_ref),
        "mem_tgt": np.array(mem_tgt),
        "mem_delta": np.array(mem_tgt) - np.array(mem_ref),
        "non_ref": np.array(non_ref),
        "non_tgt": np.array(non_tgt),
        "non_delta": np.array(non_tgt) - np.array(non_ref),
    }


def main():
    parser = argparse.ArgumentParser(description="3-point verification (Algorithm 2)")
    parser.add_argument("--model-a-csv", required=True, help="Owner model scores CSV")
    parser.add_argument("--model-b-csv", required=True, help="Suspect model scores CSV")
    parser.add_argument("--label", default="", help="Label for this verification run")
    parser.add_argument("--w-split", required=True, help="W split JSON (for hash)")
    parser.add_argument("--out-json", required=True, help="Output JSON")
    args = parser.parse_args()

    model_a_csv = args.model_a_csv
    model_b_csv = args.model_b_csv
    label = args.label

    # W hash
    with open(args.w_split) as f:
        w_data = json.load(f)
    w_ids = sorted([m["image_id"] for m in w_data["members"]])
    w_hash = hashlib.sha256(json.dumps(w_ids).encode()).hexdigest()

    A = load_all_scores(model_a_csv)
    B = load_all_scores(model_b_csv)

    S_A_W = A["mem_tgt"]
    S_B_W = B["mem_tgt"]
    S_ref_W = A["mem_ref"]

    n_w = len(S_A_W)

    print(f"\n--- Raw t-error on W ({n_w} images) ---")
    print(f"  S_A  (owner):     mean={np.mean(S_A_W):.6f}, std={np.std(S_A_W):.6f}")
    print(f"  S_B  (suspect):   mean={np.mean(S_B_W):.6f}, std={np.std(S_B_W):.6f}")
    print(f"  S_ref (baseline): mean={np.mean(S_ref_W):.6f}, std={np.std(S_ref_W):.6f}")

    results = {"label": label, "n_members": int(n_w), "w_hash": w_hash}

    # === Criterion 1: Consistency ===
    t1, p1 = stats.ttest_ind(S_A_W, S_B_W, equal_var=False)
    d1 = cohens_d(S_A_W, S_B_W)
    pass1 = bool(p1 > 0.05)
    results["criterion1"] = {
        "name": "Consistency: t-test(S_A, S_B) on W",
        "t_stat": float(t1), "p_value": float(p1), "cohens_d": float(d1),
        "threshold": "p > 0.05", "pass": pass1,
    }
    print(f"\n--- C1: Consistency (S_A vs S_B on W) ---")
    print(f"  t={t1:.4f}, p={p1:.4e}, d={d1:.4f}")
    print(f"  {'PASS' if pass1 else 'FAIL'}")

    # === Criterion 2: Separation ===
    t2, p2 = stats.ttest_ind(S_A_W, S_ref_W, equal_var=False)
    d2 = cohens_d(S_A_W, S_ref_W)
    pass2_p = bool(p2 < 1e-6)
    pass2_d = bool(abs(d2) > 2.0)
    pass2 = pass2_p and pass2_d
    results["criterion2"] = {
        "name": "Separation: t-test(S_A, S_ref) on W",
        "t_stat": float(t2), "p_value": float(p2), "cohens_d": float(d2),
        "threshold": "p < 1e-6 AND |d| > 2.0", "pass": pass2,
    }
    print(f"\n--- C2: Separation (S_A vs S_ref on W) ---")
    print(f"  t={t2:.4f}, p={p2:.4e}, |d|={abs(d2):.4f}")
    print(f"  p<1e-6: {'PASS' if pass2_p else 'FAIL'} | |d|>2.0: {'PASS' if pass2_d else 'FAIL'}")
    print(f"  {'PASS' if pass2 else 'FAIL'}")

    # === Criterion 3: Ratio ===
    mean_ref = np.mean(S_ref_W)
    mean_a = np.mean(S_A_W)
    ratio = mean_ref / mean_a if mean_a > 0 else float('inf')
    pass3 = bool(ratio > 5.0)
    results["criterion3"] = {
        "name": "Ratio: mean(S_ref) / mean(S_A) on W",
        "mean_S_ref": float(mean_ref), "mean_S_A": float(mean_a),
        "ratio": float(ratio), "threshold": "ratio > 5.0", "pass": pass3,
    }
    print(f"\n--- C3: Ratio (mean(S_ref) / mean(S_A) on W) ---")
    print(f"  {mean_ref:.6f} / {mean_a:.6f} = {ratio:.6f}x")
    print(f"  {'PASS' if pass3 else 'FAIL'}")

    # === Verdict ===
    verified = pass1 and pass2 and pass3
    results["verdict"] = "VERIFIED" if verified else "REJECTED"
    results["all_pass"] = [pass1, pass2, pass3]
    results["stats"] = {
        "S_A_mean": float(np.mean(S_A_W)), "S_A_std": float(np.std(S_A_W)),
        "S_B_mean": float(np.mean(S_B_W)), "S_B_std": float(np.std(S_B_W)),
        "S_ref_mean": float(np.mean(S_ref_W)), "S_ref_std": float(np.std(S_ref_W)),
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {results['verdict']}")
    print(f"  C1 Consistency: {'PASS' if pass1 else 'FAIL'} (p={p1:.4e}, d={d1:.4f})")
    print(f"  C2 Separation:  {'PASS' if pass2 else 'FAIL'} (|d|={abs(d2):.4f})")
    print(f"  C3 Ratio:       {'PASS' if pass3 else 'FAIL'} ({ratio:.6f}x)")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
