"""
A priori Gauss--Legendre node budget for a prescribed accuracy.

Below the exactness threshold ``m_q >= ceil(d/2)`` the quadrature error of
QuadraSHAP decays geometrically (Proposition 2(ii) of the paper).  This module
makes every constant in that statement explicit so that, given the factor
tables of the games and a tolerance ``eps``, the smallest certified number of
nodes can be computed *before* any quadrature is run.  It follows the appendix
"Choosing the Number of Nodes for a Prescribed Accuracy":

* For one product game with present factors ``u_j`` and absent factors ``ut_j``
  the integrand ``g_i(t) = prod_{j != i} ((1-t) ut_j + t u_j)`` is bounded on
  the Bernstein ellipse ``E_rho`` for [0, 1] by ``Pi_i * exp(Lambda_i * delta(rho))``
  with ``delta(rho) = (rho + 1/rho - 2)/4``,

      M_j = max(|u_j|, |ut_j|),   Pi_i = prod_{j != i} M_j,
      Lambda_i = sum_{j != i} |u_j - ut_j| / M_j     (total relative variation).

* Trefethen's bound for the ``m``-point rule on [0, 1] then gives

      |phi_hat_i - phi_i| <= |u_i - ut_i| Pi_i * B(m, Lambda_i),
      B(m, Lambda) = min_{rho > 1} (32/15) exp(Lambda delta(rho)) rho^{-2m} / (1 - rho^{-2}),

  where the minimiser rho = e^s solves ``(Lambda/2) sinh s = 2m + 2/(e^{2s} - 1)``.

* For a weighted sum of games ``v = sum_r w_r v_r`` evaluated with one shared
  node set (all value functions of Section 4),

      max_i |phi_hat_i - phi_i| <= A_max * B(m, Lambda_max),
      A_i = sum_r |w_r| |u_i^{(r)} - ut_i^{(r)}| Pi_i^{(r)},   Lambda_max = max_r Lambda^{(r)},

  and the node budget is the smallest ``m`` with ``B(m, Lambda_max) <= eps / A_max``,
  capped at the exactness threshold.

The certificate concerns the quadrature error for the specified games only
(not the estimation of the interventional expectation from a background
sample, nor floating-point rounding); it is an absolute error on the scale
``A_max`` of the largest marginal contribution, and it is sufficient, not
minimal.  Everything here is plain NumPy and costs one pass over the factor
tables plus a scalar root find per candidate ``m``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Optional, Tuple

import numpy as np

_C_GL = 32.0 / 15.0  # Trefethen (ATAP, Thm. 19.3) transferred to [0, 1] and the m-point rule
_C_CLOSED = 4.0 * (1.0 + math.sqrt(2.0)) / 15.0  # ~0.6438, constant of the closed-form bound


# ----------------------------------------------------------------------------
# The ellipse bound B(m, Lambda)
# ----------------------------------------------------------------------------
def _log_bound_at(s: float, m: int, lam: float) -> float:
    """log of (32/15) exp(Lambda sinh^2(s/2)) e^{-2ms} / (1 - e^{-2s})."""
    return math.log(_C_GL) - math.log1p(-math.exp(-2.0 * s)) + lam * math.sinh(0.5 * s) ** 2 - 2.0 * m * s


def optimal_ellipse(m: int, lam: float) -> float:
    """Return ``s = log(rho)`` minimising the bound: ``(Lambda/2) sinh s = 2m + 2/(e^{2s}-1)``.

    The left side increases from 0 and the right side decreases from +inf, so the
    root is unique; it is found by bisection (no SciPy dependency).
    """
    if lam <= 0.0:
        return math.inf

    def f(s: float) -> float:
        return 0.5 * lam * math.sinh(s) - 2.0 * m - 2.0 / math.expm1(2.0 * s)

    lo, hi = 1e-12, 1.0
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e3:  # unreachable for finite lam, guard anyway
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def ellipse_bound(m: int, lam: float) -> float:
    """``B(m, Lambda)``: certified quadrature error of the ``m``-point rule relative to ``|u_i - ut_i| Pi_i``.

    Nonincreasing in ``m`` and nondecreasing in ``Lambda``.  ``B(m, 0) = 0``
    (constant integrand, any rule is exact).
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if lam <= 0.0:
        return 0.0
    s = optimal_ellipse(m, lam)
    return math.exp(_log_bound_at(s, m, lam))


def closed_form_bound(m: int, lam: float) -> float:
    """Rigorous closed form ``0.644 (Lambda/m) exp(-11 m^2 / (3 Lambda))``, valid for ``2 <= m <= Lambda/4``.

    Raises ``ValueError`` outside its validity range; use :func:`ellipse_bound` there.
    """
    if lam <= 0.0:
        return 0.0
    if m < 2 or m > lam / 4.0:
        raise ValueError("closed_form_bound is valid only for 2 <= m <= Lambda/4")
    return _C_CLOSED * (lam / m) * math.exp(-11.0 * m * m / (3.0 * lam))


def closed_form_budget(eta: float, lam: float) -> Optional[int]:
    """``m_cf = max(2, ceil(sqrt(3 Lambda/11 * ln(0.65 Lambda/eta))))`` when ``m_cf <= Lambda/4``, else ``None``.

    This is the displayed O(sqrt(Lambda log(Lambda/eta))) sufficient budget; the
    practical certificate is :func:`node_budget`, which is sharper and unrestricted.
    """
    if lam <= 0.0:
        return 1
    arg = math.log(0.65 * lam / eta) if eta > 0 else math.inf
    m = max(2, int(math.ceil(math.sqrt(3.0 * lam / 11.0 * max(arg, 0.0)))))
    return m if m <= lam / 4.0 else None


# ----------------------------------------------------------------------------
# Summaries of the factor tables: A_i and Lambda
# ----------------------------------------------------------------------------
def _leave_one_out_products(M: np.ndarray) -> np.ndarray:
    """``Pi[r, i] = prod_{j != i} M[r, j]`` for a nonnegative table, exact with zeros, no division."""
    M = np.asarray(M, dtype=np.float64)
    m, d = M.shape
    if d == 0:
        return np.ones((m, 0))
    pref = np.concatenate([np.ones((m, 1)), np.cumprod(M[:, :-1], axis=1)], axis=1)
    suf = np.concatenate([np.cumprod(M[:, :0:-1], axis=1)[:, ::-1], np.ones((m, 1))], axis=1)
    return pref * suf


def _leave_one_out_products_log(M: np.ndarray) -> np.ndarray:
    """Same as :func:`_leave_one_out_products` but through logarithms, for tables of thousands of factors.

    Rows containing zeros fall back to the scan (exact); other rows use ``exp(sum log - log M_i)``.
    """
    M = np.asarray(M, dtype=np.float64)
    out = np.empty_like(M)
    has_zero = (M == 0).any(axis=1)
    if has_zero.any():
        out[has_zero] = _leave_one_out_products(M[has_zero])
    ok = ~has_zero
    if ok.any():
        L = np.log(M[ok])
        out[ok] = np.exp(L.sum(axis=1, keepdims=True) - L)
    return out


@dataclass
class GameSummary:
    """Accumulates ``A_i`` and ``Lambda_max`` over (chunks of) games sharing one node set."""

    d: int
    A: np.ndarray = field(default=None)  # (d,)
    lambda_max: float = 0.0
    n_games: int = 0

    def __post_init__(self):
        if self.A is None:
            self.A = np.zeros(self.d, dtype=np.float64)

    def update(self, K: np.ndarray, Ut, w: np.ndarray) -> "GameSummary":
        """Add games with ``K = u - ut`` (m, d), absent factors ``Ut`` (broadcastable) and weights ``w`` (m,)."""
        K = np.asarray(K, dtype=np.float64)
        m, d = K.shape
        if d != self.d:
            raise ValueError("inconsistent number of features")
        Ut_b = np.ones((1, d)) if Ut is None else np.asarray(Ut, dtype=np.float64).reshape(-1, d)
        U = K + Ut_b
        M = np.maximum(np.abs(U), np.abs(np.broadcast_to(Ut_b, U.shape)))  # (m, d)
        absK = np.abs(K)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(M > 0, absK / M, 0.0)  # relative variation per factor, in [0, 2]
        if m:
            self.lambda_max = max(self.lambda_max, float(a.sum(axis=1).max()))
        Pi = _leave_one_out_products_log(M)  # (m, d)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.A += (np.abs(w)[:, None] * absK * Pi).sum(axis=0)
        self.n_games += m
        return self

    @property
    def A_max(self) -> float:
        return float(self.A.max()) if self.d else 0.0

    @property
    def A_l2(self) -> float:
        return float(np.sqrt((self.A ** 2).sum()))


def summarize_games(games: Iterable[Tuple[np.ndarray, object, np.ndarray]], d: int) -> GameSummary:
    """Run :meth:`GameSummary.update` over an iterable of ``(K, Ut, w)`` chunks."""
    summ = GameSummary(d=d)
    for K, Ut, w in games:
        summ.update(K, Ut, w)
    return summ


# ----------------------------------------------------------------------------
# The budget
# ----------------------------------------------------------------------------
@dataclass
class BudgetReport:
    """Result of :func:`node_budget`.

    Attributes
    ----------
    m_q : certified number of Gauss--Legendre nodes (``<= exact_threshold``).
    eps : requested absolute tolerance on the attributions.
    norm : ``"max"`` (every feature) or ``"l2"`` (whole attribution vector).
    scale : ``A_max`` (or ``||A||_2``), the amplitude the tolerance is measured against.
    eta : normalised tolerance ``eps / scale``.
    lambda_max : total relative variation of the hardest game.
    bound : certified error ``scale * B(m_q, lambda_max)`` (0 when ``m_q`` is the exactness threshold).
    exact_threshold : ``ceil(d/2)``.
    closed_form_m : the displayed closed-form budget when within its validity range, else ``None``.
    seconds : wall-clock time spent selecting the budget (excluding the factor-table pass).
    efficiency_residual : a posteriori check ``|sum_i phi_hat_i - (f(x) - v(empty))|`` of the
        attributions actually computed (``None`` until an explanation has been run). The quadrature
        error of the sum is the error of integrating ``d/dt prod_j T_j(t)``, so the residual is
        exactly zero at or above the exactness threshold and nonzero below it; it is a necessary
        condition only, since signed per-feature errors can cancel in the sum, and it does not
        replace the a priori certificate ``bound``.
    """

    m_q: int
    eps: float
    norm: str
    scale: float
    eta: float
    lambda_max: float
    bound: float
    exact_threshold: int
    closed_form_m: Optional[int]
    seconds: float
    efficiency_residual: Optional[float] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"BudgetReport(m_q={self.m_q}, eps={self.eps:g} [{self.norm}], scale={self.scale:.3g}, "
                f"eta={self.eta:.3g}, Lambda={self.lambda_max:.3g}, certified error={self.bound:.3g}, "
                f"exact at {self.exact_threshold}, closed form m={self.closed_form_m}, {self.seconds*1e3:.2f} ms"
                + (f", efficiency residual={self.efficiency_residual:.2e}" if self.efficiency_residual is not None else "") + ")")


def budget_from_summary(summary: GameSummary, eps: float, norm: str = "max", m_max: Optional[int] = None) -> BudgetReport:
    """Smallest ``m`` with ``scale * B(m, Lambda_max) <= eps``, capped at ``ceil(d/2)`` (or ``m_max``)."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    t0 = time.perf_counter()
    d = summary.d
    exact = max(1, (d + 1) // 2)
    cap = exact if m_max is None else max(1, min(int(m_max), exact))
    scale = summary.A_max if norm == "max" else summary.A_l2 if norm == "l2" else None
    if scale is None:
        raise ValueError("norm must be 'max' or 'l2'")
    lam = summary.lambda_max
    if scale == 0.0 or lam == 0.0:
        # all attributions vanish, or every integrand is constant: one node is exact
        return BudgetReport(1, eps, norm, scale, math.inf if scale == 0 else eps / scale, lam, 0.0, exact,
                            1, time.perf_counter() - t0)
    eta = eps / scale
    m_sel, bound = cap, 0.0
    for m in range(1, cap):
        b = ellipse_bound(m, lam)
        if b <= eta:
            m_sel, bound = m, scale * b
            break
    return BudgetReport(m_sel, eps, norm, scale, eta, lam, bound, exact,
                        closed_form_budget(eta, lam), time.perf_counter() - t0)


def node_budget(games: Iterable[Tuple[np.ndarray, object, np.ndarray]], d: int, eps: float = 1e-3,
                norm: str = "max", m_max: Optional[int] = None) -> BudgetReport:
    """Certified node budget for ``max_i |phi_hat_i - phi_i| <= eps`` over an iterable of game chunks.

    Parameters
    ----------
    games : iterable of ``(K, Ut, w)`` with ``K = u - ut`` of shape (m, d), absent factors
        ``Ut`` (``None`` for the neutral factor, or broadcastable to (m, d)) and weights ``w`` (m,).
        All chunks share the node set; for the empirical interventional value function the
        weights already include ``1/n_b``.
    d : number of features.
    eps : absolute tolerance on the attributions (default ``1e-3``).
    norm : ``"max"`` certifies every feature, ``"l2"`` the Euclidean norm of the attribution vector.
    m_max : optional cap below the exactness threshold.
    """
    return budget_from_summary(summarize_games(games, d), eps, norm=norm, m_max=m_max)


def certify(summary: GameSummary, m_q: int) -> float:
    """Certified absolute error (max norm) of the ``m_q``-point rule for the summarised games."""
    exact = max(1, (summary.d + 1) // 2)
    if m_q >= exact:
        return 0.0
    return summary.A_max * ellipse_bound(m_q, summary.lambda_max)
