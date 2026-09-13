"""
Backward-compatibility shim.  The product-kernel explainer now lives in
:mod:`quadrashap.multiplicative.rkhs` as ``RKHSExplainer`` (engine: :mod:`quadrashap.multiplicative.engine`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quadrashap.multiplicative.engine import QuadraSHAP, VALUE_FUNCTIONS, BACKENDS  # noqa: F401
from quadrashap.multiplicative.rkhs import RKHSExplainer, ProductKernelModel  # noqa: F401


class ProductKernelLocalExplainer(RKHSExplainer):
    """Deprecated name; use :class:`RKHSExplainer`. Keeps the ``explain(x, gamma, method, m_q)`` signature
    with the exactness threshold as its default number of nodes."""

    def __init__(self, model):
        try:
            super().__init__(model)
        except ValueError:  # gamma not inferable: it must be passed to explain(), as before
            super().__init__(model, gamma=np.nan)
        self.null_game = self.model.intercept if hasattr(model, "intercept_") else float(self.coef.sum())

    def explain(self, x, gamma=None, method: str = "logspace_jax", m_q: Optional[int] = None, **kwargs):  # type: ignore[override]
        if gamma is not None:
            self.model.gamma = float(gamma)
        if not np.isfinite(self.model.gamma):
            raise ValueError("gamma could not be inferred from the model; pass gamma=...")
        m_q = "exact" if m_q is None else m_q
        return super().explain(x, backend=method, m_q=m_q, **kwargs)


class RBFLocalExplainer(ProductKernelLocalExplainer):
    """Deprecated name; use :class:`RKHSExplainer` (gamma is inferred automatically)."""

    def explain(self, x, method: str = "logspace_jax", m_q: Optional[int] = None, **kwargs):  # type: ignore[override]
        return super().explain(x, gamma=None, method=method, m_q=m_q, **kwargs)
