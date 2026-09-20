"""Explain a fitted Cox model's relative hazard using empirical background rows."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .product_games.shapley import ProductGamesShapleyNumpy


@dataclass
class CoxExplanation:
    values: np.ndarray
    base_value: float
    prediction: float
    m_q: int
    exact_nodes: int
    active_features: int
    rule_seconds: float
    integration_seconds: float
    total_seconds: float

    @property
    def efficiency_residual(self):
        return self.base_value + self.values.sum() - self.prediction


class CoxPHExplainer:
    """Attribute exp(x @ coef) on the same processed feature scale used to fit.

    Background rows define v(S) = mean_b exp(x_S @ coef_S + z_b,-S @ coef_-S).
    None for m_q selects the sufficient exact rule; an integer selects that many
    nodes. Exactness is algebraic and remains subject to floating-point error.
    Zero coefficients are removed internally and restored as zero attributions.
    """

    def __init__(self, coef, background, *, node_block_size=64):
        self.coef = np.array(coef, dtype=np.float64, copy=True)
        self.background = np.array(background, dtype=np.float64, copy=True)
        if self.coef.ndim != 1 or self.background.ndim != 2 or not len(self.background):
            raise ValueError("Expected a coefficient vector and nonempty background matrix")
        if self.background.shape[1] != len(self.coef):
            raise ValueError("Background and coefficient dimensions differ")
        if not np.isfinite(self.coef).all() or not np.isfinite(self.background).all():
            raise ValueError("Coefficients and background must be finite")
        self.active = self.coef != 0
        self.exact_nodes = max(1, (int(self.active.sum()) + 1) // 2)
        self.log_background_risk = self.background @ self.coef
        with np.errstate(over="raise"):
            self.expected_value = float(np.exp(self.log_background_risk).mean())
        self.node_block_size = node_block_size
        self._rules = {}
        self._backend = ProductGamesShapleyNumpy()

    def explain(self, x, m_q=None, *, timeout_seconds=None, progress_callback=None):
        started = perf_counter()
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.coef.shape or not np.isfinite(x).all():
            raise ValueError("x must be a finite vector matching the fitted coefficients")
        if m_q is None:
            m_q = self.exact_nodes
        if not isinstance(m_q, (int, np.integer)) or m_q < 1:
            raise ValueError("m_q must be a positive integer or None")
        rule_seconds = 0.0
        if m_q not in self._rules:
            from scipy.special import roots_legendre
            tick = perf_counter()
            nodes, weights = roots_legendre(m_q)
            self._rules[m_q] = ((nodes + 1) / 2, weights / 2)
            rule_seconds = perf_counter() - tick
        log_factors = (x[self.active] - self.background[:, self.active]) * self.coef[self.active]
        tick = perf_counter()
        phi = self._backend.phi_matrix_positive_logspace(
            log_factors, m_q, log_multiplier=self.log_background_risk,
            nodes_weights=self._rules[m_q], node_block_size=self.node_block_size,
            timeout_seconds=timeout_seconds, progress_callback=progress_callback,
        ).mean(axis=0)
        integration_seconds = perf_counter() - tick
        values = np.zeros_like(self.coef)
        values[self.active] = phi
        with np.errstate(over="raise"):
            prediction = float(np.exp(x @ self.coef))
        return CoxExplanation(values, self.expected_value, prediction, int(m_q), self.exact_nodes,
                              int(self.active.sum()), rule_seconds, integration_seconds,
                              perf_counter() - started)
