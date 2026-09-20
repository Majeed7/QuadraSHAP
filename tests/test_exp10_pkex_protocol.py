"""Sanity checks for the shared game and exact PKeX repository adapter."""
import sys
from pathlib import Path

import numpy as np

from quadrashap.product_games.budget import GameSummary, budget_from_summary
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from exp10_pkex_metal_imdb import selected_model, stratified_indices  # noqa: E402


def test_stratified_selection_reproducible_and_balanced():
    y = np.repeat([0, 1], 100)
    idx = stratified_indices(y, 20, 42)
    np.testing.assert_array_equal(idx, stratified_indices(y, 20, 42))
    assert len(np.unique(idx)) == 20
    assert np.bincount(y[idx]).tolist() == [10, 10]


def test_multiclass_selection_and_predicted_score_component():
    y = np.repeat(np.arange(6), 100)
    idx = stratified_indices(y, 20, 42)
    assert sorted(np.bincount(y[idx]).tolist()) == [3, 3, 3, 3, 4, 4]

    class MultiClass:
        estimators_ = [object() for _ in range(6)]

    multi = MultiClass()
    assert selected_model({"model": multi}, 4) is multi.estimators_[4]
    binary = object()
    assert selected_model({"model": binary}, 1) is binary


def test_repository_pkex_matches_exact_rule_at_small_d():
    repo = ROOT.parent / "RKHS-ExactSHAP"
    if not (repo / "explainer" / "esp.py").exists():
        import pytest
        pytest.skip("optional cloned PKeX repository unavailable")
    sys.path.insert(0, str(repo / "explainer"))
    from esp import ESPComputer

    rng = np.random.default_rng(2)
    U = np.exp(-rng.uniform(0, 0.8, (5, 12)))
    alpha = rng.normal(size=5)
    omega = ESPComputer(method="quadratic", use_scaling=True).compute_weight_vectors(U)
    phi_pkex = (alpha[:, None] * (U - 1) * omega).sum(axis=0)
    phi_rule = (alpha[:, None] * ProductGamesShapleyNumpy().phi_matrix_prefix_scan(
        U - 1, 6, node_block=2)).sum(axis=0)
    np.testing.assert_allclose(phi_pkex, phi_rule, rtol=1e-11, atol=1e-12)

    summary = GameSummary(d=12).update(U - 1, None, alpha)
    budget = budget_from_summary(summary, eps=1e-5)
    assert 1 <= budget.m_q <= 6
    assert budget.bound <= 1e-5
