import numpy as np

from benchmarks.cox_float64_reference import reference_cox
from benchmarks.jax_metal_smoke import brute_force_cox
from quadrashap import CoxExplainer


def test_independent_reference_matches_exhaustive_predictions():
    rng = np.random.default_rng(13)
    beta = rng.normal(size=7) * 0.3
    beta[2] = 0
    X, bg = rng.normal(size=(2, 7)), rng.normal(size=(9, 7))
    actual = reference_cox(beta, X, bg, m_q=4, node_block=3)
    for i, x in enumerate(X):
        np.testing.assert_allclose(actual[i], brute_force_cox(beta, x, bg), rtol=1e-12, atol=1e-12)


def test_reference_with_large_hazard_scale_matches_public_cpu_backend():
    rng = np.random.default_rng(7)
    beta = rng.normal(size=80) * 0.1
    X = rng.normal(size=(2, 80))
    bg = rng.normal(size=(5, 80)) + 30 * np.sign(beta)
    actual = reference_cox(beta, X, bg, m_q=128)
    expected = CoxExplainer(beta, background=bg, backend="prefix_scan_numpy").shap_values(X, m_q=128)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
