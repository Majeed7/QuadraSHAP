"""QuadraSHAP on fitted product-kernel models vs. brute force through ``model.predict``.

Runs under pytest or directly (``python tests/test_quadrashap_kernels.py``).
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadrashap import QuadraSHAP
from quadrashap.kernels.explainer import RBFLocalExplainer


def _brute_force(f, x, refs):
    """Shapley values of v(S) = mean_b f(x_S, ref_b_{-S}) by enumerating all 2^d coalitions."""
    d = len(x); fact = [math.factorial(i) for i in range(d + 1)]
    cache = {}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            pts = np.array([np.where(np.isin(np.arange(d), list(S)), x, r) for r in refs])
            cache[key] = float(np.mean(f(pts)))
        return cache[key]

    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for k in range(d):
            wgt = fact[k] * fact[d - k - 1] / fact[d]
            for S in itertools.combinations(others, k):
                phi[i] += wgt * (v(S + (i,)) - v(S))
    return phi


def _data(seed=0, n=60, d=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)); y = np.sin(X[:, 0]) * X[:, 1] + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(n)
    return X, y


def _models():
    X, y = _data()
    yield "krr", KernelRidge(kernel="rbf", gamma=0.4, alpha=0.1).fit(X, y), X
    yield "svr", SVR(kernel="rbf", gamma=0.4, C=2.0, epsilon=0.05).fit(X, y), X
    yield "gpr", GaussianProcessRegressor(RBF(1.3), alpha=0.1).fit(X, y), X


def _decision(model):
    """The kernel expansion without intercept, i.e. the quantity QuadraSHAP attributes."""
    b = float(np.asarray(getattr(model, "intercept_", 0.0)).ravel()[0]) if hasattr(model, "intercept_") else 0.0
    return lambda P: model.predict(P) - b


def test_value_functions_vs_brute_force():
    for name, model, X in _models():
        f = _decision(model)
        ex = QuadraSHAP(model, backend="prefix_scan_numpy")
        x = X[0]; xb = X[1]; bg = X[2:9]
        # baseline and interventional are literally f on spliced points
        np.testing.assert_allclose(ex.explain(x, "baseline", baseline=xb, m_q="exact"),
                                   _brute_force(f, x, [xb]), atol=1e-8, err_msg=name)
        np.testing.assert_allclose(ex.explain(x, "interventional", background=bg, m_q="exact"),
                                   _brute_force(f, x, bg), atol=1e-8, err_msg=name)
        # neutral: v(S) = sum_r alpha_r prod_{j in S} k_j -- the restricted-kernel value function
        def v_neutral(S):
            K = np.exp(-ex.gamma * (ex.X_train[:, list(S)] - x[list(S)]) ** 2).prod(axis=1) if S else np.ones(ex.n)
            return float(ex.alpha @ K)
        d = len(x); fact = [math.factorial(i) for i in range(d + 1)]; phi_ref = np.zeros(d)
        for i in range(d):
            others = [j for j in range(d) if j != i]
            for k in range(d):
                for S in itertools.combinations(others, k):
                    phi_ref[i] += fact[k] * fact[d - k - 1] / fact[d] * (v_neutral(S + (i,)) - v_neutral(S))
        np.testing.assert_allclose(ex.explain(x, "neutral", m_q="exact"), phi_ref, atol=1e-8, err_msg=name)


def test_efficiency_and_base_values():
    for name, model, X in _models():
        f = _decision(model); ex = QuadraSHAP(model, backend="prefix_scan_numpy", background=X[:10])
        x = X[3]
        assert abs(ex.explain(x, m_q="exact").sum() - (f(x[None])[0] - ex.alpha.sum())) < 1e-8
        assert abs(ex.value_function_at_empty("neutral") - ex.alpha.sum()) < 1e-12
        assert abs(ex.explain(x, "baseline", baseline=X[4], m_q="exact").sum() - (f(x[None])[0] - f(X[4:5])[0])) < 1e-8
        assert abs(ex.value_function_at_empty("baseline", baseline=X[4]) - f(X[4:5])[0]) < 1e-8
        assert abs(ex.explain(x, "interventional", m_q="exact").sum() - (f(x[None])[0] - f(X[:10]).mean())) < 1e-8
        assert abs(ex.value_function_at_empty("interventional") - f(X[:10]).mean()) < 1e-8


def test_default_budget_certifies_eps():
    """m_q=None chooses the nodes a priori; the error against the exact rule must be within eps."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((150, 60)); y = X[:, 0] * X[:, 1] + rng.standard_normal(150) * 0.1
    model = KernelRidge(kernel="rbf", gamma=0.3, alpha=0.1).fit(X, y)
    for eps in (1e-2, 1e-3, 1e-5):
        ex = QuadraSHAP(model, eps=eps, backend="logspace_numpy", background=X[:8])
        for vf, kw in (("neutral", {}), ("interventional", {}), ("baseline", {"baseline": X[9]})):
            for x in X[:3]:
                phi, rep = ex.explain(x, vf, return_report=True, **kw)
                exact = ex.explain(x, vf, m_q="exact", **kw)
                assert np.abs(phi - exact).max() <= eps, (vf, eps, rep)
                assert rep.bound <= eps or rep.m_q == rep.exact_threshold
                assert rep.m_q < rep.exact_threshold  # the budget beats exactness here
        # explicit m_q with a report: certified error for the m_q actually used
        phi, rep = ex.explain(X[0], m_q=5, return_report=True)
        assert rep.m_q == 5 and np.abs(phi - ex.explain(X[0], m_q="exact")).max() <= rep.bound + 1e-15
    assert ex.node_budget(X[0]).seconds < 1.0


def test_shap_values_batch_and_backward_compat():
    X, y = _data(); model = KernelRidge(kernel="rbf", gamma=0.4, alpha=0.1).fit(X, y)
    ex = QuadraSHAP(model, backend="logspace_numpy")
    Phi = ex.shap_values(X[:4], m_q="exact")
    assert Phi.shape == (4, X.shape[1])
    old = RBFLocalExplainer(model)
    np.testing.assert_array_equal(old.explain(X[0], method="logspace_numpy"), Phi[0])
    np.testing.assert_array_equal(old.explain(X[0], method="prefix_scan_numpy", m_q=2),
                                  ex.explain(X[0], m_q=2, backend="prefix_scan_numpy"))


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn(); print("ok", fn_name)
