"""
Odds-scale classifiers: logistic regression and naive Bayes.

On the odds scale both are multiplicative models with a single component (Table 1):

* logistic regression: ``odds(x) = P(y=1|x)/P(y=0|x) = e^{beta_0} prod_j e^{beta_j x_j}``;
* naive Bayes:         ``odds(x) = P(y=1)/P(y=0) prod_j p(x_j | y=1) / p(x_j | y=0)``,
  the per-feature likelihood ratios being the factors.

For more than two classes the odds of a positive class against a reference class are
attributed (``classes=(positive, reference)``).  Attributions sum to ``odds(x) - v(empty)``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from quadrashap.multiplicative.base import MultiplicativeModel
from quadrashap.multiplicative.engine import QuadraSHAP
from quadrashap.multiplicative.glm import LogLinkGLM


def _class_pair(classes_, classes: Optional[Sequence]) -> Tuple[int, int]:
    """Indices (positive, reference) into ``classes_``."""
    classes_ = list(classes_)
    if classes is None:
        if len(classes_) != 2:
            raise ValueError(f"model has {len(classes_)} classes; pass classes=(positive, reference)")
        return 1, 0
    pos, ref = classes
    return classes_.index(pos), classes_.index(ref)


class LogisticOdds(LogLinkGLM):
    """Odds of a fitted logistic regression: ``exp(b_pos - b_ref) prod_j exp((w_pos_j - w_ref_j) x_j)``."""

    def __init__(self, beta, intercept: float = 0.0):
        super().__init__(beta, intercept, scale="odds P(positive)/P(reference)")

    @classmethod
    def from_sklearn(cls, model, classes: Optional[Sequence] = None) -> "LogisticOdds":  # type: ignore[override]
        W = np.atleast_2d(np.asarray(model.coef_, dtype=np.float64))
        b = np.asarray(model.intercept_, dtype=np.float64).ravel()
        if W.shape[0] == 1:  # binary: sklearn stores the log-odds of classes_[1] vs classes_[0]
            if classes is not None and tuple(classes) != (model.classes_[1], model.classes_[0]):
                if tuple(classes) == (model.classes_[0], model.classes_[1]):
                    return cls(-W[0], -b[0])
                raise ValueError(f"classes must be a pair of {list(model.classes_)}")
            return cls(W[0], b[0])
        i, j = _class_pair(model.classes_, classes)  # multinomial: log-odds = difference of the two rows
        return cls(W[i] - W[j], b[i] - b[j])


class NaiveBayesOdds(MultiplicativeModel):
    """Odds of a fitted scikit-learn naive Bayes classifier (``GaussianNB``, ``BernoulliNB``, ``MultinomialNB``).

    ``coef = [P(pos)/P(ref)]``; ``factors(x)[0, j] = p(x_j | pos) / p(x_j | ref)``.
    """

    scale = "odds P(positive)/P(reference)"
    default_value_function = "interventional"
    supports_neutral = False

    def __init__(self, model, classes: Optional[Sequence] = None):
        self.estimator = model
        self.kind = type(model).__name__
        i, j = _class_pair(model.classes_, classes)
        self.pos, self.ref = i, j
        prior = np.asarray(model.class_prior_ if hasattr(model, "class_prior_") else np.exp(model.class_log_prior_), dtype=np.float64)
        self.coef = np.array([prior[i] / prior[j]])
        if self.kind == "GaussianNB":
            self.theta = np.asarray(model.theta_, dtype=np.float64)
            self.var = np.asarray(model.var_ if hasattr(model, "var_") else model.sigma_, dtype=np.float64)
            self.d = self.theta.shape[1]
        elif self.kind in ("BernoulliNB", "MultinomialNB"):
            self.log_prob = np.asarray(model.feature_log_prob_, dtype=np.float64)  # (n_classes, d): log p(feature | class)
            self.d = self.log_prob.shape[1]
            if self.kind == "BernoulliNB":
                self.log_1m = np.log1p(-np.exp(self.log_prob))
        else:
            raise ValueError(f"{self.kind} is not supported (GaussianNB, BernoulliNB, MultinomialNB are)")

    def factors(self, x: np.ndarray) -> np.ndarray:
        i, j = self.pos, self.ref
        if self.kind == "GaussianNB":
            ll = lambda c: -0.5 * np.log(2 * np.pi * self.var[c]) - 0.5 * (x - self.theta[c]) ** 2 / self.var[c]
            return np.exp(ll(i) - ll(j))[None, :]
        if self.kind == "BernoulliNB":
            xb = (x > 0).astype(np.float64) if getattr(self.estimator, "binarize", None) is None else (x > self.estimator.binarize).astype(np.float64)
            return np.exp(xb * (self.log_prob[i] - self.log_prob[j]) + (1 - xb) * (self.log_1m[i] - self.log_1m[j]))[None, :]
        return np.exp(x * (self.log_prob[i] - self.log_prob[j]))[None, :]  # MultinomialNB: counts as exponents


# --------------------------------------------------------------------------- named explainers
class LogisticExplainer(QuadraSHAP):
    """QuadraSHAP for logistic regression on the odds scale (``sklearn.linear_model.LogisticRegression`` or ``(beta, intercept)``).

    Attributions sum to ``odds(x) - v(empty)``; ``classes=(positive, reference)`` selects the odds for
    multinomial models.  Default value function: empirical interventional (pass ``background``).
    """

    def __init__(self, model, intercept: Optional[float] = None, *, classes: Optional[Sequence] = None, **kwargs):
        if isinstance(model, LogisticOdds):
            adapter = model
        elif hasattr(model, "coef_") and hasattr(model, "classes_"):
            adapter = LogisticOdds.from_sklearn(model, classes)
        else:
            adapter = LogisticOdds(model, 0.0 if intercept is None else intercept)
        super().__init__(adapter, **kwargs)


class NaiveBayesExplainer(QuadraSHAP):
    """QuadraSHAP for naive Bayes on the odds scale (``GaussianNB``, ``BernoulliNB``, ``MultinomialNB``).

    Attributions sum to ``odds(x) - v(empty)``; ``classes=(positive, reference)`` selects the odds for
    multiclass models.  Default value function: empirical interventional (pass ``background``).
    """

    def __init__(self, model, *, classes: Optional[Sequence] = None, **kwargs):
        adapter = model if isinstance(model, NaiveBayesOdds) else NaiveBayesOdds(model, classes)
        super().__init__(adapter, **kwargs)
