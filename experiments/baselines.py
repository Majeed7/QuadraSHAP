"""
Baselines for the QuadraSHAP experiments.

Two kinds:

* :func:`pkex_shapley` -- the *exact* competitor for product-kernel models: the
  weighted elementary-symmetric-polynomial recursion of PKeX-Shapley, which
  evaluates the neutral-factor value function ``v(S) = sum_r alpha_r prod_{j in S} u_j^(r)``
  in ``O(n d^2)``.  This is a faithful reimplementation of the published
  algorithm (normalised so that the symmetric polynomials stay bounded), not the
  authors' code; timings are therefore indicative of the algorithm, not of their
  implementation.
* Sampling estimators of the *interventional* value function with a background
  dataset -- ``shap``'s ``PermutationExplainer`` and ``KernelExplainer`` when the
  package is installed, else the dependency-free implementations below.  Which
  one ran is always recorded in the ``implementation`` field of the result, so a
  table in the paper can state it.

All estimators share the signature ``(f, x, background, budget) -> phi`` with
``f: (m, d) -> (m,)`` the quantity being attributed, so that every method sees
exactly the same value function.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    SHAP_AVAILABLE = False


# --------------------------------------------------------------------------- PKeX-Shapley
def pkex_shapley(U: np.ndarray, alpha: np.ndarray, *, return_diagnostics: bool = False):
    """Exact Shapley values of ``v(S) = sum_r alpha_r prod_{j in S} u_j^(r)`` in ``O(n d^2)`` (PKeX-Shapley).

    Parameters
    ----------
    U : (n, d) present factors ``u_j^(r) = k_j(x_j, x_j^(r))``.
    alpha : (n,) dual coefficients.

    Notes
    -----
    Writing ``z_j = u_j - 1``, the value function is its own Moebius (Harsanyi) expansion,
    ``v(S) = sum_r alpha_r sum_{T subset S} prod_{j in T} z_j``, whose dividends are shared
    equally inside each ``T``, so that

        phi_i = sum_r alpha_r z_i^(r) sum_{k=0}^{d-1} e_k(z^(r)_{-i}) / (k + 1),

    a *weighted* elementary symmetric polynomial.  All ``e_k(z)`` follow from one forward
    recursion (``O(d^2)`` per component) and the leave-one-out polynomials from the deflation
    ``e_k(z_{-i}) = e_k(z) - z_i e_{k-1}(z_{-i})``, which is stable exactly because the
    perturbations ``z_j = k_j - 1`` are bounded by one: its error amplification per step is
    ``|z_i|``.  The same recursion in the raw factors ``u_j`` would amplify by ``k/(d-k)`` and
    blow up well before ``d = 200``.
    """
    U = np.asarray(U, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).ravel()
    n, d = U.shape
    if d == 0:
        return (np.zeros(0), {}) if return_diagnostics else np.zeros(0)
    Z = U - 1.0

    E = np.zeros((n, d + 1)); E[:, 0] = 1.0            # E[:, k] = e_k(z_1..z_m), one feature at a time
    for m in range(d):
        E[:, 1:m + 2] += Z[:, [m]] * E[:, 0:m + 1]

    F = np.ones((n, d))                                 # F = e_k(z_{-i}); weight 1/(k+1)
    acc = F.copy()
    for k in range(1, d):
        F = E[:, [k]] - Z * F
        acc += F / (k + 1)
    phi = (alpha[:, None] * Z * acc).sum(axis=0)
    if return_diagnostics:
        diag = {"max_abs_e": float(np.abs(E).max()), "finite": bool(np.isfinite(phi).all())}
        return phi, diag
    return phi


def pkex_value_at_empty(alpha: np.ndarray) -> float:
    return float(np.asarray(alpha).sum())


# --------------------------------------------------------------------------- sampling estimators
@dataclass
class Attribution:
    phi: np.ndarray
    seconds: float
    implementation: str
    evaluations: Optional[int] = None


def _as_f(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return lambda P: np.asarray(f(np.atleast_2d(P)), dtype=np.float64).ravel()


def permutation_shap(f, x: np.ndarray, background: np.ndarray, n_permutations: int = 16, *,
                     seed: int = 0, force_fallback: bool = False) -> Attribution:
    """Permutation sampling of the interventional value function (``shap.PermutationExplainer`` if available).

    ``n_permutations`` antithetic pairs; the cost is ``2 * n_permutations * (d + 1) * n_b`` model rows.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    background = np.atleast_2d(np.asarray(background, dtype=np.float64))
    d = x.shape[0]
    if SHAP_AVAILABLE and not force_fallback:
        import shap
        t0 = time.perf_counter()
        expl = shap.PermutationExplainer(f, background, seed=seed)
        ev = 2 * n_permutations * (d + 1)
        phi = np.asarray(expl(x[None, :], max_evals=max(ev, 2 * d + 2), silent=True).values).ravel()
        return Attribution(phi, time.perf_counter() - t0, f"shap.PermutationExplainer(max_evals={max(ev, 2*d+2)})", ev)

    fn = _as_f(f)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    phi = np.zeros(d); n_eval = 0
    for _ in range(n_permutations):
        order = rng.permutation(d)
        for perm in (order, order[::-1]):                       # antithetic
            for b in background:
                pts = np.repeat(b[None, :], d + 1, axis=0)      # walk from background to x along perm
                for t, j in enumerate(perm):
                    pts[t + 1:, j] = x[j]
                vals = fn(pts); n_eval += len(pts)
                phi[perm] += np.diff(vals) / (2 * n_permutations * len(background))
    return Attribution(phi, time.perf_counter() - t0, f"fallback permutation sampling ({n_permutations} antithetic pairs)", n_eval)


def kernel_shap(f, x: np.ndarray, background: np.ndarray, n_samples: int = 2048, *,
                seed: int = 0, force_fallback: bool = False) -> Attribution:
    """KernelSHAP (``shap.KernelExplainer`` if available, else a weighted least-squares implementation).

    The fallback samples coalition sizes from the Shapley kernel, evaluates the same interventional
    value function, and solves the efficiency-constrained weighted least-squares problem.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    background = np.atleast_2d(np.asarray(background, dtype=np.float64))
    d = x.shape[0]
    if SHAP_AVAILABLE and not force_fallback:
        import shap
        t0 = time.perf_counter()
        expl = shap.KernelExplainer(f, background)
        phi = np.asarray(expl.shap_values(x, nsamples=n_samples, silent=True)).ravel()
        return Attribution(phi, time.perf_counter() - t0, f"shap.KernelExplainer(nsamples={n_samples})", n_samples)

    fn = _as_f(f)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    v0 = float(np.mean(fn(background)))                          # v(empty)
    vd = float(np.mean(fn(np.repeat(x[None, :], len(background), axis=0))))  # v(full)
    sizes = np.arange(1, d)
    w_size = (d - 1) / (sizes * (d - sizes)); w_size /= w_size.sum()
    ks = rng.choice(sizes, size=n_samples, p=w_size)
    Z = np.zeros((n_samples, d), dtype=bool)
    for r, k in enumerate(ks):
        Z[r, rng.choice(d, size=int(k), replace=False)] = True
    vals = np.empty(n_samples); n_eval = 0
    for r in range(n_samples):                                   # v(S) = mean_b f(x_S, b_{-S})
        pts = np.where(Z[r][None, :], x[None, :], background)
        vals[r] = fn(pts).mean(); n_eval += len(pts)
    # efficiency-constrained WLS: phi_d = (v_full - v0) - sum_{j<d} phi_j
    y = vals - v0 - Z[:, -1] * (vd - v0)
    A = (Z[:, :-1].astype(np.float64) - Z[:, [-1]].astype(np.float64))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi = np.append(beta, (vd - v0) - beta.sum())
    return Attribution(phi, time.perf_counter() - t0, f"fallback KernelSHAP (n_samples={n_samples})", n_eval)


def random_attribution(d: int, seed: int = 0) -> Attribution:
    """Uniform random scores: the floor for any ranking-quality metric."""
    rng = np.random.default_rng(seed)
    return Attribution(rng.standard_normal(d), 0.0, "random", 0)


# --------------------------------------------------------------------------- value function helpers
def interventional_f(model_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a model prediction so that shap and the fallbacks see the same quantity."""
    return lambda P: np.asarray(model_fn(np.atleast_2d(P)), dtype=np.float64).ravel()


def exact_interventional_shapley(f, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Brute-force Shapley values by enumerating all ``2^d`` coalitions (small ``d`` only; for tests)."""
    import itertools
    fn = _as_f(f)
    x = np.asarray(x, dtype=np.float64).ravel(); d = x.shape[0]
    background = np.atleast_2d(background)
    fact = [math.factorial(i) for i in range(d + 1)]
    v = {}
    for k in range(d + 1):
        for S in itertools.combinations(range(d), k):
            mask = np.zeros(d, dtype=bool); mask[list(S)] = True
            v[S] = float(fn(np.where(mask[None, :], x[None, :], background)).mean())
    phi = np.zeros(d)
    for i in range(d):
        for k in range(d):
            for S in itertools.combinations([j for j in range(d) if j != i], k):
                phi[i] += fact[k] * fact[d - k - 1] / fact[d] * (v[tuple(sorted(S + (i,)))] - v[S])
    return phi
