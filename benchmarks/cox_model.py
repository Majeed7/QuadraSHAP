"""Fit dense ridge Cox coefficients without forming a d-by-d Hessian.

All non-null training directions are retained. This is a change of coordinates
for the ridge model, not selection of a small set of original predictors.
"""
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import eigh
from sksurv.linear_model import CoxPHSurvivalAnalysis


class SampleSpaceRidgeCox:
    def __init__(self, alpha=1.0, *, block_size=4096, tol=1e-9):
        self.alpha, self.block_size, self.tol = alpha, block_size, tol

    def fit(self, X, y):
        started = perf_counter()
        if X.ndim != 2 or not len(X) or self.alpha <= 0:
            raise ValueError("Use a nonempty matrix and a positive ridge penalty")
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        n, d = X.shape
        gram = np.zeros((n, n))
        tick = perf_counter()
        for start in range(0, d, self.block_size):
            block = np.asarray(X[:, start:start + self.block_size], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("Fit preprocessing before the Cox model")
            gram += block @ block.T
        eigenvalues, U = eigh(gram)
        self.rank_tolerance_ = max(eigenvalues[-1], 0) * np.finfo(float).eps * n * 10
        keep = eigenvalues > self.rank_tolerance_
        if not keep.any():
            raise ValueError("No nonzero training direction")
        singular = np.sqrt(eigenvalues[keep])
        U = U[:, keep]
        reduced = U * singular
        basis_seconds = perf_counter() - tick
        tick = perf_counter()
        self.reduced_model_ = CoxPHSurvivalAnalysis(alpha=self.alpha, ties="breslow", n_iter=200, tol=self.tol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.reduced_model_.fit(reduced, y)
        self.fit_warnings_ = [str(w.message) for w in caught]
        if self.fit_warnings_:
            raise RuntimeError(f"Cox fitting issued warnings: {self.fit_warnings_}")
        optimizer_seconds = perf_counter() - tick
        tick = perf_counter()
        dual = U @ (self.reduced_model_.coef_ / singular)
        self.coef_ = np.empty(d)
        for start in range(0, d, self.block_size):
            self.coef_[start:start + self.block_size] = np.asarray(X[:, start:start + self.block_size], dtype=np.float64).T @ dual
        mapping_seconds = perf_counter() - tick
        self.rank_ = int(keep.sum())
        self.n_features_in_ = d
        self.timings_ = {"sample_basis_seconds": basis_seconds, "cox_optimizer_seconds": optimizer_seconds,
                         "map_coefficients_seconds": mapping_seconds, "fit_seconds": perf_counter() - started}
        np.testing.assert_allclose(self.predict(X), self.reduced_model_.predict(reduced), rtol=1e-7, atol=1e-7)
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_

    def predict_relative_hazard(self, X):
        with np.errstate(over="raise"):
            return np.exp(self.predict(X))
