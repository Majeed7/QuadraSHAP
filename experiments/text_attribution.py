"""Masked value function for a product-kernel text classifier, and the sampling estimators.

Everything here explains **the same game**, which is the only way the comparison means anything.
For a document ``x`` with support ``supp(x)`` (its distinct words) and an RBF kernel machine
``f(z) = sum_r alpha_r prod_j exp(-gamma (z_j - Z_rj)^2) + b``, masking word ``j`` sets its TF-IDF
entry to the baseline value 0 ("the word is not there").  Words outside the support are never
masked, so their factors are the same in every coalition and fold into a constant per support
vector.  What is left is exactly a weighted sum of product games over ``d = |supp(x)|`` features:

    v(S) = sum_r w_r prod_{j in S} u_rj prod_{j not in S} ut_rj + b,
    u_rj  = exp(-gamma (x_j - Z_rj)^2),   ut_rj = exp(-gamma Z_rj^2),
    w_r   = alpha_r * exp(-gamma (||Z_r||^2 - sum_{j in supp} Z_rj^2)).

QuadraSHAP integrates that family in closed form; the estimators below sample it.  They are given
the *fast* vectorised evaluator (one matrix product per batch of coalitions), which is far quicker
than calling ``decision_function`` on masked sparse vectors -- the comparison is deliberately
generous to the baselines.

Cost is reported as wall-clock seconds and, for the samplers, as the number of value-function
evaluations (their model calls; QuadraSHAP makes none).  The two are *not* interchangeable: a
quadrature node is several passes over the ``(games x d)`` factor table and returns all ``d``
leave-one-out products, an evaluation is one pass and returns a scalar, and measured end to end
the ratio between them depends on ``d`` and on the implementation.  Only seconds put the methods on
one axis without an assumption.

Every estimator is *nested*: it draws its largest sample stream once and returns an estimate at each
budget along the way, so the error-versus-budget curve of one method comes from one sample path
(and costs what the largest budget costs, not the sum).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

CHUNK_BYTES = 64 << 20            # size of the (chunk, n_games) temporary of one matrix product


# --------------------------------------------------------------------------- the game
@dataclass
class MaskGame:
    """``v(S)`` for one document, evaluated in batches.  ``n_calls`` counts coalitions."""

    U: np.ndarray                 # (n_sv, d) present factors
    Ut: np.ndarray                # (n_sv, d) absent (baseline) factors
    w: np.ndarray                 # (n_sv,)  weights, alpha_r times the constant outside the support
    intercept: float = 0.0
    n_calls: int = field(default=0, init=False)

    def __post_init__(self):
        """Build ``D = (log U - log Ut)^T`` and ``base = sum_j log Ut`` in row chunks.

        ``U`` may have fewer rows than ``Ut``: then row ``r`` of the game uses ``U[r % n_u]``,
        which is how the interventional games (one per background row and support vector) share
        their present factors without tiling a second (games x d) table in memory.
        """
        n, d = self.Ut.shape
        n_u = self.U.shape[0]
        self.d = d
        self.D = np.empty((d, n))
        self.base = np.empty(n)
        step = max(1, (64 << 20) // (8 * d))                        # ~64 MB of rows per chunk
        with np.errstate(divide="ignore"):
            for r0 in range(0, n, step):
                r1 = min(n, r0 + step)
                logUt = np.log(self.Ut[r0:r1])
                rows = np.arange(r0, r1) % n_u
                self.D[:, r0:r1] = (np.log(self.U[rows]) - logUt).T
                self.base[r0:r1] = logUt.sum(axis=1)
        self.U = self.Ut = None                                     # value() needs only D and base

    def value(self, M: np.ndarray) -> np.ndarray:
        """``M`` is ``(B, d)`` with 1 for present; returns ``(B,)``.

        ``log prod_j T_j = sum_j log ut_j + sum_j m_j (log u_j - log ut_j)``, so one batch is a
        single ``(B, d) @ (d, n_sv)`` product -- the same arithmetic a masked prediction would do,
        done for all the coalitions of the batch at once.
        """
        M = np.ascontiguousarray(M, dtype=np.float64)
        self.n_calls += M.shape[0]
        out = np.empty(M.shape[0], dtype=np.float64)
        chunk = max(1, CHUNK_BYTES // (8 * self.w.size))
        for s in range(0, M.shape[0], chunk):
            block = M[s:s + chunk]
            tmp = np.exp(self.base[None, :] + block @ self.D)
            out[s:s + chunk] = tmp @ self.w
        return out + self.intercept

    def reset(self) -> "MaskGame":
        self.n_calls = 0
        return self

    # the two coalitions every estimator needs
    def empty_full(self):
        v = self.value(np.array([np.zeros(self.d), np.ones(self.d)]))
        return float(v[0]), float(v[1])


# --------------------------------------------------------------------------- helpers
@dataclass
class Path:
    """Estimates along one nested sample path: ``phi[k]`` used ``n_evals[k]`` evaluations."""

    method: str
    budgets: List[int] = field(default_factory=list)
    n_evals: List[int] = field(default_factory=list)
    seconds: List[float] = field(default_factory=list)
    phi: List[np.ndarray] = field(default_factory=list)

    def add(self, budget: int, n_evals: int, seconds: float, phi: np.ndarray) -> None:
        self.budgets.append(int(budget))
        self.n_evals.append(int(n_evals))
        self.seconds.append(float(seconds))
        self.phi.append(np.asarray(phi, dtype=float))


def _shapley_sizes(d: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Subset sizes drawn with the Shapley-kernel probabilities ``(d-1)/(s(d-s))``."""
    s = np.arange(1, d)
    p = (d - 1) / (s * (d - s))
    return rng.choice(s, size=n, p=p / p.sum())


def _masks_of_sizes(sizes: np.ndarray, d: int, rng: np.random.Generator) -> np.ndarray:
    M = np.zeros((sizes.size, d), dtype=np.float64)
    for k, s in enumerate(sizes):
        M[k, rng.choice(d, size=int(s), replace=False)] = 1.0
    return M


# --------------------------------------------------------------------------- estimators
def kernel_shap_path(game: MaskGame, budgets: Sequence[int], seed: int = 0,
                     paired: bool = True, ridge: float = 1e-8, time_cap: float | None = None,
                     slice_size: int = 256) -> Path:
    """KernelSHAP: coalitions from the Shapley kernel, then constrained least squares.

    Sampling the coalitions from the kernel distribution turns the weighted regression into an
    unweighted one.  Efficiency is imposed exactly by eliminating the last coefficient, which is
    what the reference implementation does.  Coalitions are drawn and evaluated in slices, so a
    ``time_cap`` (seconds of sampling) stops the path anywhere; at every checkpoint the recorded
    time is the sampling time so far plus the one solve a real run would do there.  With fewer
    coalitions than features the regression is under-determined and the scaled ridge is what makes
    it solvable at all (``identifiable`` is left to the caller: ``n_evals > d``).
    """
    d = game.d
    rng = np.random.default_rng(seed)
    game.reset()
    t_sample = 0.0
    t0 = time.perf_counter()
    v0, vN = game.empty_full()
    t_sample += time.perf_counter() - t0
    T = vN - v0
    M_all, y_all = np.zeros((0, d)), np.zeros(0)
    out = Path("kernelshap")

    def solve():
        t1 = time.perf_counter()
        A = M_all[:, :-1] - M_all[:, [-1]]
        y = y_all - M_all[:, -1] * T
        AtA = A.T @ A
        AtA[np.diag_indices_from(AtA)] += ridge * max(np.trace(AtA) / max(d - 1, 1), 1e-300)
        phi = np.empty(d)
        phi[:-1] = np.linalg.solve(AtA, A.T @ y)
        phi[-1] = T - phi[:-1].sum()
        return phi, time.perf_counter() - t1

    capped = False
    for B in budgets:
        while M_all.shape[0] < B and not capped:
            take = min(slice_size, B - M_all.shape[0])
            t0 = time.perf_counter()
            M = _masks_of_sizes(_shapley_sizes(d, rng, take), d, rng)
            if paired:
                M[1::2] = 1.0 - M[0::2][: M[1::2].shape[0]]
            y_all = np.concatenate([y_all, game.value(M) - v0])
            M_all = np.concatenate([M_all, M], axis=0)
            t_sample += time.perf_counter() - t0
            capped = time_cap is not None and t_sample > time_cap
        phi, t_solve = solve()
        out.add(M_all.shape[0], game.n_calls, t_sample + t_solve, phi)
        if capped:
            break
    return out


def sampling_shap_path(game: MaskGame, budgets: Sequence[int], seed: int = 0,
                       time_cap: float | None = None) -> Path:
    """SamplingSHAP: the per-feature Monte-Carlo difference estimator (``shap.SamplingExplainer``).

    For feature ``i`` and round ``t`` an independent coalition ``S`` not containing ``i`` is drawn
    from the permutation-induced distribution (uniform over sizes), and ``v(S + i) - v(S)`` is
    averaged.  Two evaluations per feature per round.
    """
    d = game.d
    rng = np.random.default_rng(seed)
    game.reset()
    rounds = [max(1, b // (2 * d)) for b in budgets]
    n_rounds = max(rounds)
    total = np.zeros(d)
    out = Path("samplingshap")
    done = 0
    t_acc = 0.0
    capped = False
    for r_target, B in zip(rounds, budgets):
        while done < r_target and not capped:
            t0 = time.perf_counter()
            sizes = rng.integers(0, d, size=d)            # |S| uniform on {0..d-1}
            S = np.zeros((d, d))
            for i in range(d):
                others = np.delete(np.arange(d), i)
                if sizes[i]:
                    S[i, rng.choice(others, size=int(sizes[i]), replace=False)] = 1.0
            S_with = S.copy()
            S_with[np.arange(d), np.arange(d)] = 1.0
            total += game.value(S_with) - game.value(S)
            done += 1
            t_acc += time.perf_counter() - t0
            capped = time_cap is not None and t_acc > time_cap
        out.add(B, game.n_calls, t_acc, total / max(done, 1))
        if capped:
            break
    return out


def permutation_shap_path(game: MaskGame, budgets: Sequence[int], seed: int = 0,
                          time_cap: float | None = None) -> Path:
    """Permutation SHAP with antithetic pairs: one sweep of the chain gives every feature at once."""
    d = game.d
    rng = np.random.default_rng(seed)
    game.reset()
    per_pair = 2 * d                                    # forward and backward sweep share endpoints
    pairs = [max(1, b // per_pair) for b in budgets]
    total = np.zeros(d)
    out = Path("permutationshap")
    done = 0
    t_acc = 0.0
    capped = False
    for p_target, B in zip(pairs, budgets):
        while done < p_target and not capped:
            t0 = time.perf_counter()
            perm = rng.permutation(d)
            for order in (perm, perm[::-1]):
                # the chain 0 -> ... -> N along this order: row k has the first k features present
                rows = np.zeros((d + 1, d))
                rows[np.arange(d + 1)[:, None] > np.arange(d)[None, :]] = 1.0
                M = np.empty_like(rows)
                M[:, order] = rows
                total[order] += np.diff(game.value(M))
            done += 1
            t_acc += time.perf_counter() - t0
            capped = time_cap is not None and t_acc > time_cap
        out.add(B, game.n_calls, t_acc, total / (2 * max(done, 1)))
        if capped:
            break
    return out


def lime_path(game: MaskGame, budgets: Sequence[int], seed: int = 0,
              kernel_width: float = 25.0, alpha: float = 1.0, time_cap: float | None = None,
              slice_size: int = 256) -> Path:
    """LIME on the same masking: uniform word dropout, exponential kernel, weighted ridge."""
    d = game.d
    rng = np.random.default_rng(seed)
    game.reset()
    t_sample = 0.0
    M_all, y_all = np.zeros((0, d)), np.zeros(0)
    out = Path("lime")

    def solve():
        t1 = time.perf_counter()
        m = M_all
        cos = m.sum(axis=1) / np.sqrt(d * np.maximum(m.sum(axis=1), 1))
        wts = np.exp(-((1 - cos) ** 2) / (kernel_width / 100.0) ** 2)
        Xd = np.concatenate([m, np.ones((m.shape[0], 1))], axis=1)
        Amat = Xd.T @ (wts[:, None] * Xd) + alpha * np.eye(d + 1)
        coef = np.linalg.solve(Amat, Xd.T @ (wts * y_all))
        return coef[:d], time.perf_counter() - t1

    capped = False
    for B in budgets:
        while M_all.shape[0] < B and not capped:
            take = min(slice_size, B - M_all.shape[0])
            t0 = time.perf_counter()
            n_drop = rng.integers(1, d + 1, size=take)
            M = np.ones((take, d))
            for k, nd in enumerate(n_drop):
                M[k, rng.choice(d, size=int(nd), replace=False)] = 0.0
            if M_all.shape[0] == 0:
                M[0] = 1.0
            y_all = np.concatenate([y_all, game.value(M)])
            M_all = np.concatenate([M_all, M], axis=0)
            t_sample += time.perf_counter() - t0
            capped = time_cap is not None and t_sample > time_cap
        phi, t_solve = solve()
        out.add(M_all.shape[0], game.n_calls, t_sample + t_solve, phi)
        if capped:
            break
    return out


def random_path(d: int, budgets: Sequence[int], seed: int = 0) -> Path:
    """The floor: an attribution with no information, at zero cost."""
    rng = np.random.default_rng(seed)
    out = Path("random")
    for B in budgets:
        out.add(B, 0, 0.0, rng.standard_normal(d))
    return out


ESTIMATORS: Dict[str, Callable] = {
    "kernelshap": kernel_shap_path,
    "samplingshap": sampling_shap_path,
    "permutationshap": permutation_shap_path,
    "lime": lime_path,
}
