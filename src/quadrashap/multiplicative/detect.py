"""
Pick the right explainer (or adapter) for a fitted estimator, in the spirit of ``shap.Explainer``.
"""
from __future__ import annotations

from quadrashap.multiplicative.base import MultiplicativeModel


def _is_tree_model(model) -> bool:
    name = type(model).__name__
    return (hasattr(model, "tree_") or (hasattr(model, "estimators_") and hasattr(model, "apply"))
            or name in ("Booster", "XGBRegressor", "XGBClassifier", "LGBMRegressor", "LGBMClassifier"))


def _family(model) -> str:
    if isinstance(model, MultiplicativeModel):
        return "adapter"
    name = type(model).__name__
    if _is_tree_model(model):
        return "tree"
    if name in ("PoissonRegressor",):
        return "poisson"
    if name in ("GammaRegressor",):
        return "gamma"
    if name in ("TweedieRegressor",):
        return "tweedie"
    if name == "LogisticRegression":
        return "logistic"
    if name in ("GaussianNB", "BernoulliNB", "MultinomialNB"):
        return "naive_bayes"
    if name in ("CoxPHFitter", "CoxPHSurvivalAnalysis") or hasattr(model, "params_") and hasattr(model, "predict_partial_hazard"):
        return "cox"
    if (hasattr(model, "dual_coef_") or hasattr(model, "alpha_")) and (
            hasattr(model, "support_vectors_") or hasattr(model, "X_fit_") or hasattr(model, "X_train_")
            or hasattr(model, "base_estimator_")):
        return "rkhs"
    if hasattr(model, "params") and hasattr(model, "model"):  # statsmodels result
        return "glm"
    raise ValueError(f"QuadraSHAP does not recognise {name}; wrap it in a MultiplicativeModel adapter "
                     "(quadrashap.multiplicative.FactorModel) or use a named explainer explicitly.")


def adapter_for(model) -> MultiplicativeModel:
    """Return the :class:`MultiplicativeModel` adapter for a fitted (non-tree) estimator."""
    fam = _family(model)
    if fam == "adapter":
        return model
    if fam == "tree":
        raise ValueError("tree models are handled by quadrashap.TreeExplainer, not by the multiplicative engine")
    from quadrashap.multiplicative.rkhs import ProductKernelModel
    from quadrashap.multiplicative.glm import LogLinkGLM, CoxPH
    from quadrashap.multiplicative.odds import LogisticOdds, NaiveBayesOdds
    if fam == "rkhs":
        return ProductKernelModel(model)
    if fam in ("poisson", "gamma", "tweedie"):
        return LogLinkGLM.from_sklearn(model)
    if fam == "glm":
        return LogLinkGLM.from_statsmodels(model)
    if fam == "logistic":
        return LogisticOdds.from_sklearn(model)
    if fam == "naive_bayes":
        return NaiveBayesOdds(model)
    if fam == "cox":
        return CoxPH.from_lifelines(model) if hasattr(model, "params_") else CoxPH.from_sksurv(model)
    raise AssertionError(fam)  # pragma: no cover


def Explainer(model, **kwargs):
    """Construct the explainer matching ``model``: ``TreeExplainer`` for tree ensembles, otherwise the
    named multiplicative explainer (``RKHSExplainer``, ``PoissonExplainer``, ``GammaExplainer``,
    ``TweedieExplainer``, ``GLMExplainer``, ``LogisticExplainer``, ``NaiveBayesExplainer``, ``CoxExplainer``).
    Keyword arguments are forwarded to that explainer's constructor."""
    fam = _family(model)
    if fam == "tree":
        from quadrashap.treeshap.explainer import TreeExplainer
        return TreeExplainer(model, **kwargs)
    from quadrashap.multiplicative.engine import QuadraSHAP
    from quadrashap.multiplicative.rkhs import RKHSExplainer
    from quadrashap.multiplicative.glm import PoissonExplainer, GammaExplainer, TweedieExplainer, GLMExplainer, CoxExplainer
    from quadrashap.multiplicative.odds import LogisticExplainer, NaiveBayesExplainer
    cls = {"adapter": QuadraSHAP, "rkhs": RKHSExplainer, "poisson": PoissonExplainer, "gamma": GammaExplainer,
           "tweedie": TweedieExplainer, "glm": GLMExplainer, "logistic": LogisticExplainer,
           "naive_bayes": NaiveBayesExplainer, "cox": CoxExplainer}[fam]
    return cls(model, **kwargs)
