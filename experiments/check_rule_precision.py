"""Is the a priori certificate really violated at m_q = 21, or is the measurement node-limited?

``numpy.polynomial.legendre.leggauss`` returns float64 nodes and weights, so a float64 (or even
float128) evaluation of the rule carries an error of order ``|f'| * 1e-16`` that does not decrease
with ``m_q`` -- it is the accuracy of the *rule itself*, not of the quadrature.  This script
recomputes the Gauss--Legendre rule to 50 decimal digits (Newton refinement on the Legendre
recurrence, in ``decimal``) and evaluates the integrand there, so the only thing left in the
difference between ``m_q`` and the exactness threshold is genuine quadrature truncation.
"""
from __future__ import annotations

import sys
from decimal import Decimal as D, getcontext

import numpy as np

sys.path.insert(0, "../src")
getcontext().prec = 50

from common import fit_krr, synthetic_regression                      # noqa: E402
from quadrashap.product_games.budget import ellipse_bound              # noqa: E402


def legendre_p_dp(n: int, x: D):
    """(P_n(x), P_n'(x)) by the three-term recurrence, in decimal."""
    p0, p1 = D(1), x
    for k in range(1, n):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    dp = n * (x * p1 - p0) / (x * x - 1)
    return p1, dp


def gauss_legendre_01_dp(m: int):
    """Nodes and weights of the m-point rule on [0, 1], to the current decimal precision."""
    x0, _ = np.polynomial.legendre.leggauss(m)          # float64 starting points
    xs, ws = [], []
    for guess in x0:
        x = D(float(guess))
        for _ in range(60):                              # Newton; quadratic, converges in a handful
            p, dp = legendre_p_dp(m, x)
            step = p / dp
            x -= step
            if abs(step) < D(10) ** (-45):
                break
        p, dp = legendre_p_dp(m, x)
        xs.append((x + 1) / 2)                           # map [-1,1] -> [0,1]
        ws.append(D(2) / ((1 - x * x) * dp * dp) / 2)
    return xs, ws


def integral_dp(u_dp, i: int, m: int) -> D:
    """int_0^1 prod_{j != i} (1 - t + t u_j) dt by the m-point rule, in decimal."""
    xs, ws = gauss_legendre_01_dp(m)
    total = D(0)
    for t, w in zip(xs, ws):
        prod = D(1)
        one_minus_t = 1 - t
        for j, uj in enumerate(u_dp):
            if j != i:
                prod *= one_minus_t + t * uj
        total += w * prod
    return total


def main() -> None:
    d, gs, n_train, m_test = 60, 30.0, 300, 21
    X, y, _ = synthetic_regression(n_train + 5, d, 10, seed=0, scale=1.0)
    model = fit_krr(X[:n_train], y[:n_train], gamma_scale=gs)
    x = X[n_train]
    U = np.exp(-(gs / d) * (x[None, :] - X[:n_train]) ** 2)
    thr = (d + 1) // 2

    print(f"d={d}, gamma*d={gs}, exactness threshold {thr}, testing m_q={m_test}")
    print(f"{'game':>5} {'i':>3} {'Lambda_i':>9} {'true |I-I_m| (50 digits)':>26} {'certified':>13} {'ratio':>10}")
    worst = 0.0
    for r in (0, 1, 123):
        u_dp = [D(float(v)) for v in U[r]]
        K = np.abs(U[r] - 1.0)
        M = np.maximum(np.abs(U[r]), 1.0)
        for i in (0, 5):
            lam_i = float((K / M).sum() - K[i] / M[i])
            err = abs(integral_dp(u_dp, i, m_test) - integral_dp(u_dp, i, thr))
            Pi = float(np.exp(np.log(M).sum() - np.log(M[i])))
            bound = Pi * ellipse_bound(m_test, lam_i)
            ratio = float(err) / bound
            worst = max(worst, ratio)
            print(f"{r:>5} {i:>3} {lam_i:>9.2f} {float(err):>26.6e} {bound:>13.3e} {ratio:>10.2e}")
    print(f"\nworst true/certified = {worst:.3e}  -> {'VIOLATED' if worst > 1 else 'the certificate holds'}")

    # For contrast: the same differences evaluated with float64 nodes, which is what the experiment does.
    t64, w64 = np.polynomial.legendre.leggauss(m_test)
    t64, w64 = (t64 + 1) / 2, w64 / 2
    te, we = np.polynomial.legendre.leggauss(thr)
    te, we = (te + 1) / 2, we / 2

    def f64(u, i, t, w):
        v = np.delete(u, i)
        return float(sum(wq * np.prod(1 - tq + tq * v) for tq, wq in zip(t, w)))

    e64 = abs(f64(U[0], 0, t64, w64) - f64(U[0], 0, te, we))
    print(f"same quantity with float64 nodes (game 0, i=0): {e64:.3e}")


if __name__ == "__main__":
    main()
