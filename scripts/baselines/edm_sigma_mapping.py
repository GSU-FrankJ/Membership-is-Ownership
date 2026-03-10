"""
Validates sigma <-> alpha_bar mapping for EDM compatibility.

Run on CPU before committing GPU hours to Zhao et al.
Exit code 0 = all pass (GO), non-zero = fail (NO-GO).

The math:
  DDIM forward: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
  EDM forward:  x_sigma = x0 + sigma * eps
  Mapping:      sigma(t) = sqrt((1 - alpha_bar_t) / alpha_bar_t)
  Inverse:      alpha_bar(sigma) = 1 / (1 + sigma^2)
"""

import math
import sys

import numpy as np
import torch


def alpha_bar_cosine(t, T=1000, s=0.008):
    """Cosine schedule alpha_bar -- matches src/ddpm_ddim/schedulers/betas.py"""
    f_t = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos((s / (1 + s)) * np.pi / 2) ** 2
    return f_t / f_0


def sigma_from_alpha_bar(alpha_bar):
    return np.sqrt((1.0 - alpha_bar) / alpha_bar)


def alpha_bar_from_sigma(sigma):
    return 1.0 / (1.0 + sigma ** 2)


def main():
    passed = 0
    total = 5

    # Test 1: Roundtrip t -> sigma -> alpha_bar ~ alpha_bar_t
    print("=== Test 1: Roundtrip ===")
    for t in [0, 100, 250, 500, 750, 999]:
        ab = alpha_bar_cosine(t)
        sigma = sigma_from_alpha_bar(ab)
        ab_rt = alpha_bar_from_sigma(sigma)
        err = abs(ab - ab_rt)
        status = "OK" if err < 1e-10 else "FAIL"
        print(f"  t={t:4d}  alpha_bar={ab:.6f}  sigma={sigma:.4f}  err={err:.2e} {status}")
        if err >= 1e-10:
            print(f"  FAIL at t={t}")
            break
    else:
        passed += 1
        print("  PASS")

    # Test 2: sigma range covers EDM [0.002, 80]
    print("\n=== Test 2: Sigma Range Coverage ===")
    sigmas = [sigma_from_alpha_bar(alpha_bar_cosine(t)) for t in range(1000)]
    sig_min, sig_max = min(sigmas), max(sigmas)
    print(f"  Our range:  [{sig_min:.4f}, {sig_max:.2f}]")
    print(f"  EDM range:  [0.002, 80.0]")
    ok = sig_min < 0.01 and sig_max > 50
    print(f"  {'PASS' if ok else 'WARNING: incomplete coverage'}")
    if ok:
        passed += 1

    # Test 3: SNR monotonicity (sigma increases with t)
    print("\n=== Test 3: SNR Monotonicity ===")
    monotonic = all(sigmas[i] <= sigmas[i + 1] for i in range(len(sigmas) - 1))
    print(f"  sigma monotonically increasing: {'PASS' if monotonic else 'FAIL'}")
    if monotonic:
        passed += 1

    # Test 4: Boundary conditions
    print("\n=== Test 4: Boundaries ===")
    s0 = sigmas[0]
    s999 = sigmas[999]
    ok4 = s0 < 0.02 and s999 > 10
    print(f"  sigma(0)={s0:.4f} (expect ~0), sigma(999)={s999:.2f} (expect >>1): {'PASS' if ok4 else 'FAIL'}")
    if ok4:
        passed += 1

    # Test 5: Noise equivalence on dummy batch
    print("\n=== Test 5: Noise Equivalence ===")
    x0 = torch.randn(4, 3, 32, 32)
    eps = torch.randn_like(x0)
    t_test = 500
    ab = float(alpha_bar_cosine(t_test))
    sig = float(sigma_from_alpha_bar(ab))
    x_ddim = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
    x_edm = x0 + sig * eps
    diff = (x_ddim - math.sqrt(ab) * x_edm).abs().max().item()
    ok5 = diff < 1e-5
    print(f"  max |x_ddim - sqrt(alpha_bar)*x_edm| = {diff:.2e}: {'PASS' if ok5 else 'FAIL'}")
    if ok5:
        passed += 1

    # Verdict
    print(f"\n{'=' * 40}")
    print(f"RESULT: {passed}/{total} tests passed")
    if passed == total:
        print("GO -- proceed with Zhao et al. EDM pipeline")
        sys.exit(0)
    else:
        print("NO-GO -- Zhao falls back to cited-only comparison")
        sys.exit(1)


if __name__ == "__main__":
    main()
