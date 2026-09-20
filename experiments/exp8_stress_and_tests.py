"""Analytic stress test and independent correctness checks for experiment 8."""
from __future__ import annotations
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from exp8_multimodel_budget import (EVALUATOR, GameSummary, budget_from_summary,
    certify, bernstein_reference, independent_gl, write_csv, write_json)


def brute_force(K, Ut, w):
    d = K.shape[1]
    phi = np.zeros(d)
    U = Ut + K
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for mask in itertools.product((0, 1), repeat=d - 1):
            present = np.zeros(d, dtype=bool)
            present[others] = mask
            s = sum(mask)
            weight = 1 / (d * math.comb(d - 1, s))
            absent = np.where(present, U, Ut)
            before = w @ np.prod(absent, axis=1)
            absent[:, i] = U[:, i]
            after = w @ np.prod(absent, axis=1)
            phi[i] += weight * (after - before)
    return phi


def checks(output):
    rng = np.random.default_rng(20260915)
    rows = []
    for d in (2, 5, 8):
        for p in (1, 3, 8):
            U, Ut = np.exp(rng.normal(0, .5, (2, p, d)))
            # Include opposite-signed mixture weights, but positive factors.
            w = rng.normal(size=p)
            K = U - Ut
            brute = brute_force(K, Ut, w)
            reference = bernstein_reference(K, Ut, w)
            scipy_exact = independent_gl(K, Ut, w, (d + 1) // 2)
            actual = (w[:, None] * EVALUATOR.phi_matrix_prefix_scan(K, (d + 1) // 2, Ut=Ut)).sum(0)
            for name, value in (("Bernstein", reference), ("SciPy", scipy_exact), ("package", actual)):
                np.testing.assert_allclose(value, brute, rtol=2e-12, atol=2e-12)
                rows.append(dict(d=d, n_games=p, implementation=name,
                                 max_error=float(np.max(abs(value - brute)))))
    # Equal factors give a closed-form attribution by symmetry and efficiency.
    for d in (50, 1000):
        K = np.full((1, d), .5)
        Ut = np.full((1, d), .5)
        reference = bernstein_reference(K, Ut, np.ones(1))
        analytic = np.full(d, -np.expm1(d * np.log(.5)) / d)
        np.testing.assert_allclose(reference, analytic, rtol=2e-12, atol=2e-15)
        rows.append(dict(d=d, n_games=1, implementation="Bernstein_vs_analytic",
                         max_error=float(np.max(abs(reference - analytic)))))
    write_json(output / "reference_tests.json", rows)
    print(f"Passed {len(rows)} independent reference comparisons", flush=True)


def stress(output):
    """Bounded-output, coherent positive factors, not fitted synthetic data.

v(S)=r^{|S intersect J|-k}; active fraction 30%. This is the neutral product
game with raw factors r>1 and coefficient r^-k, evaluated in its algebraically
equivalent normalized representation present=1, absent=1/r to avoid underflow.
Every coalition value is <=1. Exact active phi=(1-r^-k)/k. Normalization is
part of the fixed model, not a per-instance or post-error rescaling.
"""
    rows, curves = [], []
    for d in (50, 100, 250, 500, 1000):
        k = round(.3 * d)
        for r in (1.1, 1.5, 3., 10.):
            U, Ut = np.ones((1, d)), np.ones((1, d))
            Ut[:, :k] = 1 / r
            K, w = U - Ut, np.ones(1)
            summary = GameSummary(d=d).update(K, Ut, w)
            reference = np.zeros(d)
            reference[:k] = -np.expm1(-k * np.log(r)) / k
            m_cert = budget_from_summary(summary, 1e-6).m_q
            last = max(m_cert + 5, (k + 1) // 2)
            ms = np.arange(1, last + 1)
            errors = []
            for m in ms:
                phi = EVALUATOR.phi_matrix_prefix_scan(K, int(m), Ut=Ut)[0]
                error = float(np.max(abs(phi - reference)))
                errors.append(error)
                curves.append(dict(d=d, active_features=k, raw_factor=r, m_q=int(m),
                                   observed_error=error, certified_bound=certify(summary, int(m))))
            first = int(ms[np.asarray(errors) <= 1e-6][0])
            rows.append(dict(d=d, active_features=k, raw_factor=r,
                             m_observed=first, m_certified=m_cert, exact_threshold=(d+1)//2,
                             active_exact_threshold=(k+1)//2, lambda_max=summary.lambda_max,
                             A_max=summary.A_max, error_at_certified=errors[m_cert-1]))
    write_csv(output / "controlled_stress.csv", rows)
    write_csv(output / "controlled_stress_curves.csv", curves)
    write_json(output / "controlled_stress.json", rows)
    print("Stress at d=1000:", [r for r in rows if r["d"] == 1000], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    checks(output)
    stress(output)
