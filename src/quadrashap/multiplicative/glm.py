"""
Log-link generalized linear models and Cox proportional-hazards models.

A GLM with a log link predicts ``mu(x) = exp(beta_0 + sum_j beta_j x_j) = e^{beta_0} prod_j e^{beta_j x_j}``,
a multiplicative model with a single component (``p = 1``), coefficient ``c = e^{beta_0}`` and
factors ``u_j(x_j) = e^{beta_j x_j}`` (Table 1 of the paper: Poisson, Gamma, Tweedie and
negative-binomial regression, log-link GAMs with ``u_j = e^{s_j(x_j)}``).  The attributions are
on the response (mean) scale and sum to ``mu(x) - v(empty)``.

A Cox model has the same structure on the hazard-ratio scale: ``h(t | x) = h_0(t) exp(beta . x)``,
so the relative hazard ``exp(beta . x)`` (relative to a reference, typically the training mean,
as in lifelines' partial hazard) is attributed feature by feature; the baseline hazard ``h_0(t)``
is a time-dependent constant that can be multiplied in through ``baseline_hazard``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quadrashap.multiplicative.base import MultiplicativeModel
from quadrashap.multiplicative.engine import QuadraSHAP


class LogLinkGLM(MultiplicativeModel):
    """``F(x) = exp(intercept) * prod_j exp(beta_j x_j)``.

    Parameters
    ----------
    beta : (d,) linear coefficients on the log scale.
    intercept : scalar intercept on the log scale (``e^intercept`` becomes the component coefficient).
    scale : name of the attributed quantity.
    """

    default_value_function = "interventional"
    supports_neutral = False

    def __init__(self, beta, intercept: float = 0.0, *, scale: str = "mean response (log link)"):
        self.beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        self.log_intercept = float(intercept)
        self.coef = np.array([np.exp(self.log_intercept)])
        self.d = int(self.beta.shape[0])
        self.scale = scale

    def factors(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.beta * x)[None, :]

    def predict_scale(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.exp(self.log_intercept + X @ self.beta)

    # ---- constructors from fitted libraries
    @classmethod
    def from_sklearn(cls, model, **kw) -> "LogLinkGLM":
        """``PoissonRegressor``, ``GammaRegressor`` or ``TweedieRegressor`` with a log link (``link='log'``, or
        ``'auto'`` with ``power > 0``)."""
        name = type(model).__name__
        link = getattr(model, "link", "log")
        if name == "TweedieRegressor":
            if link == "auto":
                link = "log" if getattr(model, "power", 0) > 0 else "identity"
            if link != "log":
                raise ValueError(f"TweedieRegressor with link='{link}' is not multiplicative; need the log link")
        elif name not in ("PoissonRegressor", "GammaRegressor") and not hasattr(model, "coef_"):
            raise ValueError(f"cannot read log-link GLM coefficients from {name}")
        return cls(np.asarray(model.coef_).ravel(), float(np.asarray(getattr(model, "intercept_", 0.0)).ravel()[0]), **kw)

    @classmethod
    def from_statsmodels(cls, results, **kw) -> "LogLinkGLM":
        """A fitted ``statsmodels`` GLM/Poisson/NegativeBinomial result with a log link (``const`` column optional)."""
        params = np.asarray(results.params, dtype=np.float64).ravel()
        names = list(getattr(results.model, "exog_names", []))
        if names and names[0] in ("const", "Intercept"):
            return cls(params[1:], params[0], **kw)
        return cls(params, 0.0, **kw)


class CoxPH(LogLinkGLM):
    """Cox proportional hazards on the hazard-ratio scale: ``F(x) = baseline_hazard * exp(beta . (x - reference))``.

    Parameters
    ----------
    beta : (d,) log hazard ratios.
    reference : (d,) covariate values relative to which the hazard ratio is expressed (``None`` -> zero;
        lifelines uses the training means, so ``F`` then matches its ``predict_partial_hazard``).
    baseline_hazard : optional ``h_0(t)`` to obtain the hazard at time ``t`` instead of the ratio.
    """

    def __init__(self, beta, reference=None, *, baseline_hazard: float = 1.0):
        beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        ref = np.zeros_like(beta) if reference is None else np.asarray(reference, dtype=np.float64).reshape(-1)
        super().__init__(beta, float(np.log(baseline_hazard) - beta @ ref), scale="hazard ratio")
        self.reference = ref
        self.baseline_hazard = float(baseline_hazard)

    @classmethod
    def from_lifelines(cls, fitter, **kw) -> "CoxPH":
        """A fitted ``lifelines.CoxPHFitter`` (``params_``; the training means it centres on when available)."""
        beta = np.asarray(fitter.params_, dtype=np.float64).ravel()
        ref = getattr(fitter, "_norm_mean", None)
        return cls(beta, None if ref is None else np.asarray(ref, dtype=np.float64).ravel(), **kw)

    @classmethod
    def from_sksurv(cls, model, **kw) -> "CoxPH":
        """A fitted ``sksurv.linear_model.CoxPHSurvivalAnalysis`` (``coef_``)."""
        return cls(np.asarray(model.coef_, dtype=np.float64).ravel(), None, **kw)


# --------------------------------------------------------------------------- named explainers
def _glm_adapter(model, intercept, scale):
    if isinstance(model, LogLinkGLM):
        return model
    if hasattr(model, "coef_"):
        return LogLinkGLM.from_sklearn(model, scale=scale)
    if hasattr(model, "params") and hasattr(model, "model"):
        return LogLinkGLM.from_statsmodels(model, scale=scale)
    return LogLinkGLM(model, 0.0 if intercept is None else intercept, scale=scale)  # a coefficient array


class GLMExplainer(QuadraSHAP):
    """QuadraSHAP for any log-link GLM/GAM given as a fitted scikit-learn or statsmodels model, or as ``(beta, intercept)``.

    Attributions are on the mean-response scale and sum to ``mu(x) - v(empty)``.  Default value
    function: empirical interventional (pass ``background``); the neutral factor is not available.
    """

    scale_name = "mean response (log link)"

    def __init__(self, model, intercept: Optional[float] = None, **kwargs):
        super().__init__(_glm_adapter(model, intercept, self.scale_name), **kwargs)


class PoissonExplainer(GLMExplainer):
    """QuadraSHAP for Poisson regression with a log link (``sklearn.linear_model.PoissonRegressor`` or coefficients)."""
    scale_name = "expected count (Poisson mean)"


class GammaExplainer(GLMExplainer):
    """QuadraSHAP for Gamma regression with a log link (``sklearn.linear_model.GammaRegressor`` or coefficients)."""
    scale_name = "expected response (Gamma mean)"


class TweedieExplainer(GLMExplainer):
    """QuadraSHAP for Tweedie regression with a log link (``sklearn.linear_model.TweedieRegressor`` or coefficients)."""
    scale_name = "expected response (Tweedie mean)"


class CoxExplainer(QuadraSHAP):
    """QuadraSHAP for Cox proportional-hazards models on the hazard-ratio scale.

    Parameters
    ----------
    model : a fitted ``lifelines.CoxPHFitter``, ``sksurv`` ``CoxPHSurvivalAnalysis``, a :class:`CoxPH`
        adapter, or an array of log hazard ratios ``beta``.
    reference : covariate values the hazard ratio is relative to (arrays only; lifelines' means are read automatically).
    baseline_hazard : optional ``h_0(t)`` multiplier.
    """

    def __init__(self, model, reference=None, *, baseline_hazard: float = 1.0, **kwargs):
        if isinstance(model, CoxPH):
            adapter = model
        elif hasattr(model, "params_"):
            adapter = CoxPH.from_lifelines(model, baseline_hazard=baseline_hazard)
        elif hasattr(model, "coef_"):
            adapter = CoxPH.from_sksurv(model, baseline_hazard=baseline_hazard)
        else:
            adapter = CoxPH(model, reference, baseline_hazard=baseline_hazard)
        super().__init__(adapter, **kwargs)
