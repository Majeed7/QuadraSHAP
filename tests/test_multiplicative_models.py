"""Named explainers for log-link GLMs, odds-scale classifiers and Cox models vs. brute force through the model.

For each family the value function is computed by calling the fitted model on spliced points
``(x_S, xtilde_{-S})`` (mean response, odds from predict_proba, hazard ratio) and enumerating all
coalitions; this verifies that the factorization each adapter claims is the one the model computes.
Runs under pytest or directly (``python tests/test_multiplicative_models.py``).
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quadrashap as qs
from quadrashap import (Explainer, QuadraSHAP, RKHSExplainer, PoissonExplainer, GammaExplainer, TweedieExplainer,
                        GLMExplainer, CoxExplainer, LogisticExplainer, NaiveBayesExplainer, FactorModel, TreeExplainer)
from quadrashap.multiplicative.glm import CoxPH


def _brute_force(F, x, refs):
    """Shapley values of v(S) = mean_b F(x_S, ref_b_{-S}) by enumerating all 2^d coalitions."""
    d = len(x); fact = [math.factorial(i) for i in range(d + 1)]; cache = {}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            pts = np.array([np.where(np.isin(np.arange(d), list(S)), x, r) for r in refs])
            cache[key] = float(np.mean(F(pts)))
        return cache[key]

    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for k in range(d):
            wgt = fact[k] * fact[d - k - 1] / fact[d]
            for S in itertools.combinations(others, k):
                phi[i] += wgt * (v(S + (i,)) - v(S))
    return phi


def _check(explainer, F, x, refs, name, atol=1e-8):
    """baseline (one ref) and interventional (all refs) vs brute force; efficiency; budget certifies eps."""
    phi_b = explainer.explain(x, "baseline", baseline=refs[0], m_q="exact")
    np.testing.assert_allclose(phi_b, _brute_force(F, x, refs[:1]), atol=atol, rtol=1e-7, err_msg=f"{name} baseline")
    phi_i = explainer.explain(x, "interventional", background=refs, m_q="exact")
    np.testing.assert_allclose(phi_i, _brute_force(F, x, refs), atol=atol, rtol=1e-7, err_msg=f"{name} interventional")
    Fx = float(F(x[None, :])[0])
    assert abs(phi_b.sum() - (Fx - float(F(refs[:1])[0]))) < 1e-7 * max(1, abs(Fx)), name
    assert abs(phi_i.sum() - (Fx - float(F(refs).mean()))) < 1e-7 * max(1, abs(Fx)), name
    assert abs(explainer.value_function_at_empty("interventional", background=refs) - F(refs).mean()) < 1e-9 * max(1, abs(Fx))
    # default: interventional with the a priori budget
    phi, rep = explainer.explain(x, background=refs, return_report=True)
    assert np.abs(phi - phi_i).max() <= explainer.eps + 1e-12, (name, rep)
    assert rep.efficiency_residual is not None and rep.efficiency_residual <= rep.bound + 1e-9 * max(1, abs(Fx))
    # the neutral factor is refused outside product kernels
    if not explainer.model.supports_neutral:
        try:
            explainer.explain(x, "neutral"); raise AssertionError("neutral should be refused")
        except ValueError:
            pass


def _data(seed, n=300, d=8, kind="regression"):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)) * 0.7
    if kind == "counts":
        y = rng.poisson(np.exp(0.3 + X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]))
    elif kind == "positive":
        y = rng.gamma(2.0, np.exp(0.2 + 0.6 * X[:, 0] - 0.4 * X[:, 3]) / 2.0)
    else:
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 + X[:, 0] - X[:, 1] + 0.5 * X[:, 4])))).astype(int)
    return X, y


def test_log_link_glms():
    X, y = _data(0, kind="counts")
    for name, model, cls in [("poisson", PoissonRegressor(alpha=1e-3).fit(X, y), PoissonExplainer),
                             ("tweedie", TweedieRegressor(power=1.5, link="log", alpha=1e-3).fit(X, y + 0.1), TweedieExplainer)]:
        ex = cls(model, background=X[:6])
        assert ex.n == 1 and ex.d == X.shape[1]
        _check(ex, model.predict, X[10], X[:6], name)
        assert isinstance(Explainer(model), cls)
    Xg, yg = _data(1, kind="positive")
    gm = GammaRegressor(alpha=1e-3).fit(Xg, yg)
    _check(GammaExplainer(gm), gm.predict, Xg[3], Xg[:5], "gamma")
    # raw coefficients (e.g. from statsmodels): F = exp(b0 + beta.x)
    beta, b0 = np.array([0.4, -0.3, 0.2, 0.0, 0.1, -0.2, 0.05, 0.3]), 0.7
    ex = GLMExplainer(beta, b0)
    _check(ex, lambda P: np.exp(b0 + P @ beta), X[0], X[:4], "glm-coef")


def test_logistic_odds():
    X, y = _data(2, kind="binary")
    lr = LogisticRegression(C=1.0).fit(X, y)
    odds = lambda P: lr.predict_proba(P)[:, 1] / lr.predict_proba(P)[:, 0]
    _check(LogisticExplainer(lr), odds, X[5], X[:6], "logistic")
    assert isinstance(Explainer(lr), LogisticExplainer)
    # reversed class pair = reciprocal odds
    ex_rev = LogisticExplainer(lr, classes=(lr.classes_[0], lr.classes_[1]))
    _check(ex_rev, lambda P: 1 / odds(P), X[5], X[:6], "logistic-reversed")
    # multinomial: odds of class 2 against class 0
    rng = np.random.default_rng(3); Xm = rng.standard_normal((400, 6)); ym = np.argmax(Xm[:, :3] + 0.5 * rng.standard_normal((400, 3)), axis=1)
    lrm = LogisticRegression(C=1.0, max_iter=500).fit(Xm, ym)
    odds20 = lambda P: lrm.predict_proba(P)[:, 2] / lrm.predict_proba(P)[:, 0]
    _check(LogisticExplainer(lrm, classes=(2, 0)), odds20, Xm[1], Xm[:5], "multinomial")


def test_naive_bayes_odds():
    X, y = _data(4, kind="binary")
    gnb = GaussianNB().fit(X, y)
    odds = lambda m: (lambda P: m.predict_proba(P)[:, 1] / m.predict_proba(P)[:, 0])
    _check(NaiveBayesExplainer(gnb), odds(gnb), X[7], X[:6], "gaussian-nb", atol=1e-7)
    assert isinstance(Explainer(gnb), NaiveBayesExplainer)
    Xb = (X > 0).astype(float)
    bnb = BernoulliNB().fit(Xb, y)
    _check(NaiveBayesExplainer(bnb), odds(bnb), Xb[7], Xb[:6], "bernoulli-nb", atol=1e-9)
    rng = np.random.default_rng(5); Xc = rng.poisson(2.0, size=(300, 8)).astype(float)
    mnb = MultinomialNB().fit(Xc, y)
    _check(NaiveBayesExplainer(mnb), odds(mnb), Xc[2], Xc[:6], "multinomial-nb", atol=1e-8)
    # three classes: odds of class 2 vs class 0
    y3 = np.argmax(X[:, :3], axis=1); g3 = GaussianNB().fit(X, y3)
    _check(NaiveBayesExplainer(g3, classes=(2, 0)), lambda P: g3.predict_proba(P)[:, 2] / g3.predict_proba(P)[:, 0],
           X[9], X[:5], "gaussian-nb-3class", atol=1e-7)


def test_cox_hazard_ratio():
    rng = np.random.default_rng(6); X = rng.standard_normal((100, 7)) * 0.5
    beta = np.array([0.5, -0.3, 0.0, 0.8, -0.2, 0.1, 0.25]); mean = X.mean(axis=0)
    hr = lambda P: np.exp((P - mean) @ beta)               # lifelines-style partial hazard
    ex = CoxExplainer(beta, reference=mean, background=X[:6])
    _check(ex, hr, X[0], X[:6], "cox")
    ex2 = CoxExplainer(CoxPH(beta, mean, baseline_hazard=0.05))
    _check(ex2, lambda P: 0.05 * hr(P), X[1], X[:4], "cox-baseline-hazard")
    # duck-typed lifelines-like object
    class _Fit:  # minimal stand-in for lifelines.CoxPHFitter
        params_ = beta; _norm_mean = mean
        def predict_partial_hazard(self, P): return hr(P)
    _check(CoxExplainer(_Fit()), hr, X[2], X[:3], "cox-lifelines-like")
    assert isinstance(Explainer(_Fit()), CoxExplainer)


def test_factor_model_and_dispatch():
    # generic adapter: a hand-written multiplicative model with two components
    rng = np.random.default_rng(7); d = 6
    B = rng.standard_normal((2, d)) * 0.3; c = np.array([1.5, -0.7])
    fm = FactorModel(c, lambda x: np.exp(B * x), d, scale="test")
    ex = QuadraSHAP(fm, background=rng.standard_normal((5, d)))
    F = lambda P: (np.exp(P @ B.T) * c).sum(axis=1)
    _check(ex, F, rng.standard_normal(d), rng.standard_normal((5, d)), "factor-model")
    # dispatch: kernels and trees
    X, y = _data(8)
    krr = KernelRidge(kernel="rbf", gamma=0.3, alpha=0.1).fit(X, y)
    assert isinstance(Explainer(krr), RKHSExplainer) and isinstance(QuadraSHAP(krr), QuadraSHAP)
    assert Explainer(krr).model.supports_neutral and Explainer(krr).model.default_value_function == "neutral"
    rf = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0).fit(X, y)
    assert isinstance(Explainer(rf), TreeExplainer)
    for name in ("RKHSExplainer", "PoissonExplainer", "GammaExplainer", "TweedieExplainer", "GLMExplainer",
                 "CoxExplainer", "LogisticExplainer", "NaiveBayesExplainer", "TreeExplainer", "Explainer"):
        assert hasattr(qs, name), name


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn(); print("ok", fn_name)
