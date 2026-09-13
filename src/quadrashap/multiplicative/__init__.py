"""Explainers for multiplicative models (Definition 2): product kernels, log-link GLMs, odds-scale classifiers, Cox models."""
from .base import MultiplicativeModel, FactorModel
from .engine import QuadraSHAP, VALUE_FUNCTIONS, BACKENDS
from .rkhs import ProductKernelModel, RKHSExplainer
from .glm import LogLinkGLM, CoxPH, GLMExplainer, PoissonExplainer, GammaExplainer, TweedieExplainer, CoxExplainer
from .odds import LogisticOdds, NaiveBayesOdds, LogisticExplainer, NaiveBayesExplainer
from .detect import Explainer, adapter_for

__all__ = ["MultiplicativeModel", "FactorModel", "QuadraSHAP", "VALUE_FUNCTIONS", "BACKENDS",
           "ProductKernelModel", "RKHSExplainer", "LogLinkGLM", "CoxPH", "GLMExplainer", "PoissonExplainer",
           "GammaExplainer", "TweedieExplainer", "CoxExplainer", "LogisticOdds", "NaiveBayesOdds",
           "LogisticExplainer", "NaiveBayesExplainer", "Explainer", "adapter_for"]
