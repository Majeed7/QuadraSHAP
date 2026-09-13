"""
Product-kernel machines (RKHS models): SVM/SVR, kernel ridge, Gaussian processes with an RBF kernel.

``f(x) = b + sum_r alpha_r prod_j k_j(x_j, x_j^{(r)})`` with ``k_j(a, c) = exp(-gamma (a - c)^2)``
is a multiplicative model with one component per training (or support) point:
``c_r = alpha_r`` and ``u_j^{(r)}(x_j) = k_j(x_j, x_j^{(r)})`` (Table 1 of the paper).
It is the only family with ``p > 1`` components, hence the one where blockwise
evaluation and the log-space arithmetic matter, and the only one for which the
neutral-factor value function (absent factor 1, i.e. the kernel restricted to the
present features -- the value function of PKeX-Shapley) is meaningful.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from quadrashap.multiplicative.base import MultiplicativeModel
from quadrashap.multiplicative.engine import QuadraSHAP


class ProductKernelModel(MultiplicativeModel):
    """Adapter for a fitted scikit-learn estimator with an RBF product kernel.

    Probes ``support_vectors_`` / ``X_fit_`` / ``X_train_`` for the components, ``dual_coef_`` /
    ``alpha_`` for the coefficients, ``intercept_`` for the additive bias, and infers ``gamma``
    from ``_gamma`` / ``gamma`` / a GP kernel's ``length_scale`` (``gamma = 1 / (2 l^2)``).
    """

    scale = "kernel expansion sum_r alpha_r k(x, x^(r)) (decision function without intercept)"
    default_value_function = "neutral"
    supports_neutral = True

    def __init__(self, model, gamma: Optional[float] = None):
        self.estimator = model
        X_train = self._get_X_train(model)
        if hasattr(X_train, "toarray"):  # scipy sparse training data (e.g. TF-IDF features)
            X_train = X_train.toarray()
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.coef, self.intercept = self._get_alpha(model)
        self.n, self.d = self.X_train.shape
        if self.coef.shape[0] != self.n:
            raise ValueError(f"{self.coef.shape[0]} dual coefficients for {self.n} training points")
        self.gamma = float(gamma) if gamma is not None else self._infer_gamma(model, self.d)

    @property
    def alpha(self) -> np.ndarray:
        return self.coef

    # ---- interrogation of the fitted estimator
    @staticmethod
    def _get_X_train(m) -> np.ndarray:
        if hasattr(m, "support_vectors_"):
            return m.support_vectors_
        if hasattr(m, "X_fit_"):
            return m.X_fit_
        if hasattr(m, "X_train_"):
            return m.X_train_
        if hasattr(m, "base_estimator_") and hasattr(m.base_estimator_, "X_train_"):
            return m.base_estimator_.X_train_
        raise ValueError("Unsupported model type: cannot locate the training inputs.")

    @staticmethod
    def _get_alpha(m) -> Tuple[np.ndarray, float]:
        if hasattr(m, "dual_coef_"):
            alpha = np.asarray(m.dual_coef_, dtype=np.float64).ravel()
            b = float(np.asarray(m.intercept_).ravel()[0]) if hasattr(m, "intercept_") else 0.0
            return alpha, b
        if hasattr(m, "alpha_"):
            return np.asarray(m.alpha_, dtype=np.float64).ravel(), 0.0
        raise ValueError("Unsupported model type: cannot locate the dual coefficients.")

    @staticmethod
    def _infer_gamma(m, d: int) -> float:
        if hasattr(m, "_gamma"):
            return float(m._gamma)
        if hasattr(m, "gamma"):
            g = m.gamma
            if g is not None and not isinstance(g, str):
                return float(g)
            if getattr(m, "kernel", None) == "rbf":
                return 1.0 / d
        if hasattr(m, "kernel_") and hasattr(m.kernel_, "length_scale"):
            ls2 = np.mean(np.asarray(m.kernel_.length_scale, dtype=np.float64) ** 2)
            return float(1.0 / (2.0 * ls2))
        raise ValueError("Cannot infer gamma; pass gamma=... explicitly.")

    # ---- the factor table
    def factors(self, x: np.ndarray) -> np.ndarray:
        """``u_j^{(r)} = exp(-gamma (x_j - x_j^{(r)})^2)``, shape ``(n, d)``."""
        return np.exp(-self.gamma * (self.X_train - x[None, :]) ** 2)

    def predict_scale(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty(X.shape[0])
        for i, x in enumerate(X):  # sum_r alpha_r prod_j k_j, in log-space for long products
            out[i] = self.coef @ np.exp((-self.gamma * (self.X_train - x[None, :]) ** 2).sum(axis=1))
        return out


class RKHSExplainer(QuadraSHAP):
    """QuadraSHAP for product-kernel (RKHS) models: SVR/SVC, KernelRidge, Gaussian processes with an RBF kernel.

    Not to be confused with ``shap.KernelExplainer`` (KernelSHAP), which is a model-agnostic
    sampling method: this explainer computes the exact (or certified-accuracy) Shapley values
    of the kernel expansion by Gauss--Legendre quadrature.  Default value function: neutral factor.

    Parameters
    ----------
    model : fitted estimator (see :class:`ProductKernelModel`).
    gamma : RBF parameter, inferred from the model when ``None``.
    **kwargs : forwarded to :class:`~quadrashap.multiplicative.engine.QuadraSHAP` (``background``,
        ``eps``, ``backend``, ``block_size``, ``memory_budget``, ...).
    """

    def __init__(self, model, *, gamma: Optional[float] = None, **kwargs):
        adapter = model if isinstance(model, ProductKernelModel) else ProductKernelModel(model, gamma=gamma)
        super().__init__(adapter, **kwargs)
