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
    --model-a-csv scores/phase11/model_a.csv \
    --model-b-csv scores/phase11/model_b1.csv \
    --label "B1 (domain shift)" \
    --w-split data/splits/phase11_w_only.json \
    --out-json scores/phase11/verification_b1.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_all_scores(csv_path):
    """Load score CSV and return separate member/nonmember arrays."""
    mem_tgt, mem_ref, mem_delta = [], [], []
    non_tgt, non_ref, non_delta = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tgt = float(row["score_tgt"])
            ref = float(row["score_ref"])
            delta = float(row["score"])
            if row["label"] == "member":
                mem_tgt.append(tgt)
                mem_ref.append(ref)
                mem_delta.append(delta)
            else:
                non_tgt.append(tgt)
                non_ref.append(ref)
                non_delta.append(delta)
    return {
        "mem_tgt": np.array(mem_tgt), "mem_ref": np.array(mem_ref), "mem_delta": np.array(mem_delta),
        "non_tgt": np.array(non_tgt), "non_ref": np.array(non_ref), "non_delta": np.array(non_delta),
    }


def cohens_d(a, b):
    """Compute Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def verify(model_a_csv, model_b_csv, label, w_split_path=None):
    """Run the 3-point verification protocol."""
    print(f"\n{'='*60}")
    print(f"OWNERSHIP VERIFICATION - {label}")
    print(f"{'='*60}")

    # Step 0: Hash commitment
    w_hash = None
    if w_split_path:
        with open(w_split_path) as f:
            w_split = json.load(f)
        w_ids_sorted = sorted(str(m["image_id"]) for m in w_split["members"])
        w_hash = hashlib.sha256("\n".join(w_ids_sorted).encode()).hexdigest()
        print(f"\nHash commitment SHA-256(sort(W)): {w_hash[:32]}...")
        print(f"W size: {len(w_ids_sorted)} images")

    # Load all scores
    A = load_all_scores(model_a_csv)
    B = load_all_scores(model_b_csv)

    # Raw t-error scores on W (Algorithm 2)
    # S_A = owner model's raw t-error on W
    # S_B = suspect model's raw t-error on W
    # S_ref = baseline's raw t-error on W (= score_ref column, same for all CSVs)
    S_A_W = A["mem_tgt"]       # owner raw t-error on W
    S_B_W = B["mem_tgt"]       # suspect raw t-error on W
    S_ref_W = A["mem_ref"]     # baseline (SD v1.4) raw t-error on W

    n_w = len(S_A_W)

    print(f"\n--- Raw t-error on W ({n_w} images) ---")
    print(f"  S_A  (owner):    mean={np.mean(S_A_W):.4f}, std={np.std(S_A_W):.4f}")
    print(f"  S_B  (suspect):  mean={np.mean(S_B_W):.4f}, std={np.std(S_B_W):.4f}")
    print(f"  S_ref (baseline): mean={np.mean(S_ref_W):.4f}, std={np.std(S_ref_W):.4f}")

    results = {"label": label, "n_members": int(n_w)}
    if w_hash:
        results["w_hash"] = w_hash

    # === Criterion 1: Consistency (Algorithm 2, O1) ===
    # If M_B is derived from M_A, they should have similar t-error on W.
    # t-test(S_A(W), S_B(W)), pass if p > 0.05
    t1, p1 = stats.ttest_ind(S_A_W, S_B_W, equal_var=False)
    d1 = cohens_d(S_A_W, S_B_W)
    pass1 = bool(p1 > 0.05)
    results["criterion1"] = {
        "name": "Consistency: t-test(S_A, S_B) on W",
        "t_stat": float(t1), "p_value": float(p1), "cohens_d": float(d1),
        "threshold": "p > 0.05",
        "pass": pass1,
    }
    print(f"\n--- Criterion 1: Consistency (S_A vs S_B on W) ---")
    print(f"  t-test: t={t1:.4f}, p={p1:.4e}")
    print(f"  Cohen's d: {d1:.4f}")
    print(f"  Threshold: p > 0.05")
    print(f"  Result: {'PASS' if pass1 else 'FAIL'}")

    # === Criterion 2: Separation (Algorithm 2, O2) ===
    # Owner model must reconstruct W better than baseline.
    # t-test(S_A(W), S_ref(W)), pass if p < 1e-6 AND |d| > 2.0
    t2, p2 = stats.ttest_ind(S_A_W, S_ref_W, equal_var=False)
    d2 = cohens_d(S_A_W, S_ref_W)
    pass2_p = bool(p2 < 1e-6)
    pass2_d = bool(abs(d2) > 2.0)
    pass2 = pass2_p and pass2_d
    results["criterion2"] = {
        "name": "Separation: t-test(S_A, S_ref) on W",
        "t_stat": float(t2), "p_value": float(p2), "cohens_d": float(d2),
        "threshold": "p < 1e-6 AND |d| > 2.0",
        "pass": pass2,
    }
    print(f"\n--- Criterion 2: Separation (S_A vs S_ref on W) ---")
    print(f"  t-test: t={t2:.4f}, p={p2:.4e}")
    print(f"  Cohen's d: {d2:.4f} (|d|={abs(d2):.4f})")
    print(f"  Thresholds: p < 1e-6 {'PASS' if pass2_p else 'FAIL'} | "
          f"|d| > 2.0 {'PASS' if pass2_d else 'FAIL'}")
    print(f"  Result: {'PASS' if pass2 else 'FAIL'}")

    # === Criterion 3: Ratio (Algorithm 2, O3) ===
    # mean(S_ref(W)) / mean(S_A(W)) > 5.0
    mean_ref = np.mean(S_ref_W)
    mean_a = np.mean(S_A_W)
    if mean_a > 0:
        ratio = mean_ref / mean_a
    else:
        ratio = float('inf')
    pass3 = bool(ratio > 5.0)
    results["criterion3"] = {
        "name": "Ratio: mean(S_ref) / mean(S_A) on W",
        "mean_S_ref": float(mean_ref),
        "mean_S_A": float(mean_a),
        "ratio": float(ratio),
        "threshold": "ratio > 5.0",
        "pass": pass3,
    }
    print(f"\n--- Criterion 3: Ratio (mean(S_ref) / mean(S_A) on W) ---")
    print(f"  mean(S_ref)={mean_ref:.4f} / mean(S_A)={mean_a:.4f} = {ratio:.4f}x")
    print(f"  Threshold: ratio > 5.0")
    print(f"  Result: {'PASS' if pass3 else 'FAIL'}")

    # === Final verdict ===
    verified = pass1 and pass2 and pass3
    results["verdict"] = "VERIFIED" if verified else "REJECTED"
    results["all_pass"] = [pass1, pass2, pass3]

    # Additional stats for analysis
    results["stats"] = {
        "S_A_mean": float(np.mean(S_A_W)),
        "S_A_std": float(np.std(S_A_W)),
        "S_B_mean": float(np.mean(S_B_W)),
        "S_B_std": float(np.std(S_B_W)),
        "S_ref_mean": float(np.mean(S_ref_W)),
        "S_ref_std": float(np.std(S_ref_W)),
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {'VERIFIED' if verified else 'REJECTED'}")
    print(f"  Criterion 1 (Consistency): {'PASS' if pass1 else 'FAIL'} (p={p1:.4e}, d={d1:.4f})")
    print(f"  Criterion 2 (Separation):  {'PASS' if pass2 else 'FAIL'} (|d|={abs(d2):.4f})")
    print(f"  Criterion 3 (Ratio):       {'PASS' if pass3 else 'FAIL'} ({ratio:.4f}x)")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a-csv", required=True)
    parser.add_argument("--model-b-csv", required=True)
    parser.add_argument("--label", default="Model B")
    parser.add_argument("--w-split", default=None, help="Path to W split JSON for hash commitment")
    parser.add_argument("--out-json", default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = verify(args.model_a_csv, args.model_b_csv, args.label, args.w_split)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
