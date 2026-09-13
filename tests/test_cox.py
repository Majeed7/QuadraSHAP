"""Validate the empirical-background game and the full ridge coordinate change."""
import itertools
import math

import numpy as np
import pytest

from quadrashap.cox import CoxPHExplainer
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy


def brute_force(coef, background, x):
    d = len(coef)
    values = np.zeros(d)
    def game(indices):
        rows = background.copy()
        rows[:, indices] = x[indices]
        return np.exp(rows @ coef).mean()
    for feature in range(d):
        others = [j for j in range(d) if j != feature]
        for size in range(d):
            for coalition in itertools.combinations(others, size):
                coalition = list(coalition)
                values[feature] += (game(coalition + [feature]) - game(coalition)) / (d * math.comb(d - 1, size))
    return values


def test_cox_matches_exhaustive_empirical_background_with_dummy_feature():
    rng = np.random.default_rng(24)
    coef = rng.normal(0, 0.25, 7)
    coef[2] = 0
    background, x = rng.normal(size=(4, 7)), rng.normal(size=7)
    explainer = CoxPHExplainer(coef, background, node_block_size=2)
    result = explainer.explain(x)
    np.testing.assert_allclose(result.values, brute_force(coef, background, x), rtol=2e-12, atol=2e-12)
    assert result.values[2] == 0
    assert result.exact_nodes == 3
    assert abs(result.efficiency_residual) < 1e-12
    np.testing.assert_array_equal(explainer.explain(x).values, result.values)
    assert explainer.explain(x).rule_seconds == 0


def test_positive_chunked_backend_matches_existing_product_backend():
    rng = np.random.default_rng(12)
    log_u = rng.normal(0, 0.2, size=(3, 19))
    multiplier = np.array([-2, 0, 2])
    backend = ProductGamesShapleyNumpy()
    expected = backend.phi_matrix_logspace(np.expm1(log_u), 10) * np.exp(multiplier[:, None])
    for block in (1, 4, 64):
        actual = backend.phi_matrix_positive_logspace(log_u, 10, log_multiplier=multiplier, node_block_size=block)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_extreme_positive_factors_do_not_overflow_intermediate_products():
    result = ProductGamesShapleyNumpy().phi_matrix_positive_logspace(
        np.array([[1000, -1000]]), 1, log_multiplier=[-1000],
    )
    np.testing.assert_allclose(result, [[0.5, -0.5]], rtol=1e-12)


def test_identical_background_and_constant_model():
    x = np.array([1.0, 2.0])
    for coef in ([0, 0], [0.1, 0.2]):
        result = CoxPHExplainer(coef, x[None]).explain(x)
        np.testing.assert_array_equal(result.values, [0, 0])
        assert result.efficiency_residual == 0


def test_progress_callback_does_not_change_attributions():
    explainer = CoxPHExplainer([0.2, -0.1], np.array([[0.0, 1.0], [2.0, 0.0]]))
    updates = []
    with_progress = explainer.explain([1.0, 2.0], progress_callback=lambda *args: updates.append(args))
    without_progress = explainer.explain([1.0, 2.0])
    np.testing.assert_array_equal(with_progress.values, without_progress.values)
    assert updates == [(0, 1, 2, 1), (1, 1, 2, 1)]


def test_sample_space_fit_matches_direct_ridge_cox():
    pytest.importorskip("sksurv")
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from benchmarks.cox_model import SampleSpaceRidgeCox
    rng = np.random.default_rng(32)
    X = rng.normal(size=(18, 27))
    X[:, 3] = X[:, 2]  # Include an exactly dependent original feature.
    X -= X.mean(axis=0)
    y = np.empty(18, dtype=[("event", "?"), ("time", "f8")])
    y["event"] = np.arange(18) % 3 != 0
    y["time"] = rng.integers(1, 12, size=18)  # Censoring and tied event times.
    direct = CoxPHSurvivalAnalysis(alpha=4.0, ties="breslow").fit(X, y)
    reduced = SampleSpaceRidgeCox(alpha=4.0).fit(X, y)
    np.testing.assert_allclose(reduced.coef_, direct.coef_, rtol=2e-6, atol=1e-7)
    held_out = rng.normal(size=(6, 27))
    np.testing.assert_allclose(reduced.predict(held_out), direct.predict(held_out), rtol=2e-6, atol=1e-7)
