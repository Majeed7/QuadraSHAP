"""
Multiplicative models (Definition 2 of the paper).

A predictive model is *multiplicative* with ``p`` components if

    f(x) = intercept + sum_{r=1}^{p} c_r prod_{j=1}^{d} u_j^{(r)}(x_j),

with coefficients ``c_r`` and univariate factor functions ``u_j^{(r)}``.  Every
such model induces, under the value functions of Section 4, a weighted sum of
product games with present factors ``u_j^{(r)}(x_j)`` and absent factors
``u_j^{(r)}(xtilde_j)``, and the same Gauss--Legendre machinery serves them all.
An adapter therefore needs to expose only the coefficient vector and the table
of present factors for a point; :class:`quadrashap.multiplicative.engine.QuadraSHAP`
does the rest (value functions, node budget, blockwise evaluation).

The additive ``intercept`` (an SVM bias, for instance) is not part of the product
and is not attributed: the explainers attribute ``F(x) = f(x) - intercept``.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class MultiplicativeModel:
    """Base adapter. Subclasses set ``coef`` (p,), ``d`` and implement ``factors(x) -> (p, d)``.

    Attributes
    ----------
    coef : the component coefficients ``c_r`` on the multiplicative scale.
    d : number of features.
    intercept : additive constant outside the products (not attributed).
    scale : human-readable name of the quantity being attributed.
    default_value_function : used by the engine when none is given.
    supports_neutral : whether the neutral-factor value function (absent factor 1) is meaningful;
        it is the PKeX-Shapley value function for product kernels and is disabled for the other
        families, whose natural value functions are the baseline and empirical interventional ones.
    """

    scale: str = "multiplicative scale"
    default_value_function: str = "interventional"
    supports_neutral: bool = False
    intercept: float = 0.0

    coef: np.ndarray
    d: int

    def factors(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Present factors ``u_j^{(r)}(x_j)`` for one point ``x`` (d,), as a ``(p, d)`` table."""
        raise NotImplementedError

    @property
    def n_components(self) -> int:
        return int(np.asarray(self.coef).shape[0])

    def predict_scale(self, X: np.ndarray) -> np.ndarray:
        """``F(x) = sum_r c_r prod_j u_j^{(r)}(x_j)`` for each row of ``X`` (the attributed quantity)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.array([float(self.coef @ self.factors(x).prod(axis=1)) for x in X])

    def describe(self) -> str:
        return f"{type(self).__name__}(p={self.n_components}, d={self.d}, scale='{self.scale}')"


class FactorModel(MultiplicativeModel):
    """Generic adapter from user-supplied coefficients and a factor function.

    Parameters
    ----------
    coef : (p,) component coefficients.
    factor_fn : callable ``x -> (p, d)`` table of present factors for a point ``x`` (d,).
    d : number of features.
    """

    def __init__(self, coef, factor_fn: Callable[[np.ndarray], np.ndarray], d: int, *, intercept: float = 0.0,
                 scale: str = "multiplicative scale", supports_neutral: bool = False,
                 default_value_function: Optional[str] = None):
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        self._fn = factor_fn
        self.d = int(d)
        self.intercept = float(intercept)
        self.scale = scale
        self.supports_neutral = bool(supports_neutral)
        if default_value_function is not None:
            self.default_value_function = default_value_function

    def factors(self, x: np.ndarray) -> np.ndarray:
        U = np.asarray(self._fn(np.asarray(x, dtype=np.float64).reshape(-1)), dtype=np.float64)
        if U.ndim == 1:
            U = U[None, :]
        if U.shape != (self.n_components, self.d):
            raise ValueError(f"factor_fn must return shape {(self.n_components, self.d)}, got {U.shape}")
        return U
