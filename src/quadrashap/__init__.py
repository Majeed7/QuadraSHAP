"""
QuadraSHAP: Shapley values of product games by Gauss--Legendre quadrature.

Explainers share one interface (``shap_values(X)``, ``expected_value``; the multiplicative
ones also ``explain(x, value_function, ...)`` and ``node_budget``):

    TreeExplainer         tree ensembles (path-dependent value function)
    RKHSExplainer         product-kernel machines: SVR/SVC, KernelRidge, Gaussian processes (RBF kernel)
    PoissonExplainer, GammaExplainer, TweedieExplainer, GLMExplainer   log-link GLMs / GAMs (mean-response scale)
    LogisticExplainer     logistic regression (odds scale)
    NaiveBayesExplainer   Gaussian / Bernoulli / multinomial naive Bayes (odds scale)
    CoxExplainer          Cox proportional hazards (hazard-ratio scale)
    Explainer(model)      picks one of the above from the fitted estimator
    QuadraSHAP(adapter)   the engine, for any MultiplicativeModel adapter (see FactorModel)
"""
from .treeshap.explainer import TreeExplainer
from .multiplicative import (QuadraSHAP, RKHSExplainer, GLMExplainer, PoissonExplainer, GammaExplainer,
                             TweedieExplainer, CoxExplainer, LogisticExplainer, NaiveBayesExplainer, Explainer,
                             MultiplicativeModel, FactorModel)
from .product_games.budget import BudgetReport, node_budget, ellipse_bound
from .product_games.blocks import BlockPlan, plan_blocks, available_memory_bytes

__all__ = ["TreeExplainer", "RKHSExplainer", "GLMExplainer", "PoissonExplainer", "GammaExplainer", "TweedieExplainer",
           "CoxExplainer", "LogisticExplainer", "NaiveBayesExplainer", "Explainer", "QuadraSHAP",
           "MultiplicativeModel", "FactorModel", "BudgetReport", "node_budget", "ellipse_bound",
           "BlockPlan", "plan_blocks", "available_memory_bytes"]
