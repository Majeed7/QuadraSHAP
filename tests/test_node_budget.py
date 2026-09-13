"""A priori node budget (Appendix "Choosing the Number of Nodes for a Prescribed Accuracy").

Runs under pytest or directly (``python tests/test_node_budget.py``).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy
from quadrashap.product_games.budget import (
    ellipse_bound, closed_form_bound, closed_form_budget, optimal_ellipse,
    GameSummary, budget_from_summary, node_budget, certify,
)


def test_bound_reference_values():
    """Values from the appendix (computed independently with SciPy's root finder)."""
    for lam, m, ref in [(434, 32, 4.06e-4), (434, 48, 3.07e-9), (94, 16, 7.94e-5), (2124, 96, 2.13e-7)]:
        assert abs(ellipse_bound(m, lam) / ref - 1) < 5e-3, (lam, m)
    # stationarity: (Lambda/2) sinh s = 2m + 2/(e^{2s}-1)
    s = optimal_ellipse(32, 434.0)
    assert abs(0.5 * 434 * math.sinh(s) - 64 - 2 / math.expm1(2 * s)) < 1e-8
    assert ellipse_bound(5, 0.0) == 0.0


def test_bound_monotone_and_closed_form_dominates():
    for lam in (10.0, 94.0, 434.0, 2124.0):
        prev = math.inf
        for m in range(1, 200):
            b = ellipse_bound(m, lam)
            assert b <= prev * (1 + 1e-12); prev = b
            if 2 <= m <= lam / 4:
                assert closed_form_bound(m, lam) >= b * (1 - 1e-12)
        for m in (8, 32, 64):
            assert ellipse_bound(m, lam) <= ellipse_bound(m, 2 * lam) * (1 + 1e-12)


def test_certified_budgets_match_appendix_table():
    for lam, full, cf in [(94, (14, 20, 25), (17, 22, None)), (434, (31, 41, 52), (39, 48, 59)),
                          (680, (39, 52, 65), (50, 61, 74)), (2124, (70, 92, 116), (91, 111, 133))]:
        summ = GameSummary(d=1000); summ.lambda_max = float(lam); summ.A = np.ones(1000)
        assert tuple(budget_from_summary(summ, e).m_q for e in (1e-3, 1e-6, 1e-10)) == full, lam
        assert tuple(closed_form_budget(e, lam) for e in (1e-3, 1e-6, 1e-10)) == cf, lam


def test_bound_holds_on_random_games():
    """Certified error >= observed error against the exact rule, for one-sided and two-sided games."""
    rng = np.random.default_rng(0)
    core = ProductGamesShapleyNumpy()
    for d, n_games, spread in [(40, 5, 3.0), (200, 3, 1.5), (600, 2, 1.0)]:
        for kind in ("neutral", "two_sided", "signed"):
            U = np.exp(-rng.uniform(0, spread, size=(n_games, d)))
            if kind == "neutral":
                Ut = None
            elif kind == "two_sided":
                Ut = np.exp(-rng.uniform(0, spread, size=(n_games, d)))
            else:
                Ut = rng.uniform(-1, 1, size=(n_games, d)); U = rng.uniform(-1, 1, size=(n_games, d))
            w = rng.standard_normal(n_games)
            K = U - (1.0 if Ut is None else Ut)
            exact = core.phi_matrix_prefix_scan(K, (d + 1) // 2, Ut=Ut)
            phi_exact = (exact * w[:, None]).sum(0)
            summ = GameSummary(d=d).update(K, Ut, w)
            for eps in (1e-2, 1e-4, 1e-7):
                rep = budget_from_summary(summ, eps)
                phi = (core.phi_matrix_prefix_scan(K, rep.m_q, Ut=Ut) * w[:, None]).sum(0)
                err = np.abs(phi - phi_exact).max()
                assert err <= eps + 1e-12, (d, kind, eps, rep)
                assert rep.bound <= eps or rep.m_q == rep.exact_threshold
                assert err <= certify(summ, rep.m_q) + 1e-12
            # l2 norm variant
            rep2 = budget_from_summary(summ, 1e-4, norm="l2")
            phi = (core.phi_matrix_prefix_scan(K, rep2.m_q, Ut=Ut) * w[:, None]).sum(0)
            assert np.linalg.norm(phi - phi_exact) <= 1e-4 + 1e-12


def test_summary_quantities():
    """A_i and Lambda from the definitions, including zero factors and chunked accumulation."""
    rng = np.random.default_rng(1)
    d = 6
    U = rng.uniform(0, 2, size=(3, d)); Ut = rng.uniform(0, 2, size=(3, d)); Ut[0, 2] = 0.0; U[1, 4] = 0.0
    w = rng.standard_normal(3)
    M = np.maximum(np.abs(U), np.abs(Ut))
    A_ref = np.zeros(d); lam_ref = 0.0
    for r in range(3):
        lam_ref = max(lam_ref, (np.abs(U[r] - Ut[r]) / M[r]).sum())
        for i in range(d):
            A_ref[i] += abs(w[r]) * abs(U[r, i] - Ut[r, i]) * np.prod(np.delete(M[r], i))
    s1 = GameSummary(d=d).update(U - Ut, Ut, w)
    s2 = GameSummary(d=d)
    for r in range(3):
        s2.update((U - Ut)[r:r + 1], Ut[r:r + 1], w[r:r + 1])
    for s in (s1, s2):
        np.testing.assert_allclose(s.A, A_ref, rtol=1e-12)
        assert abs(s.lambda_max - lam_ref) < 1e-12
    rep = node_budget([((U - Ut)[r:r + 1], Ut[r:r + 1], w[r:r + 1]) for r in range(3)], d, eps=1e-3)
    assert 1 <= rep.m_q <= 3 and rep.exact_threshold == 3


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn(); print("ok", fn_name)
