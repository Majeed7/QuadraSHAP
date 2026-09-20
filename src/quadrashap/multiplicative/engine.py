"""
QuadraSHAP: the explainer engine for multiplicative models.

Any model of Definition 2, ``f(x) = intercept + sum_r c_r prod_j u_j^{(r)}(x_j)``, induces
under the value functions of Section 4 a weighted sum of product games with present
factors ``u_j^{(r)}(x_j)`` and absent factors ``ut_j^{(r)}``:

* ``"neutral"``       : ``ut_j = 1`` -- the absent feature is dropped from the product
                        (product kernels only; the value function of PKeX-Shapley);
* ``"baseline"``      : ``ut_j^{(r)} = u_j^{(r)}(x^b_j)`` for a reference point ``x^b``;
* ``"interventional"``: the average of the baseline games over a background dataset
                        (the empirical interventional value function, Proposition 4).

The engine takes a :class:`~quadrashap.multiplicative.base.MultiplicativeModel` adapter
(or a fitted estimator, which is wrapped by :func:`quadrashap.multiplicative.detect.adapter_for`)
and provides the value functions, the a priori node budget (default ``eps = 1e-3``,
:mod:`quadrashap.product_games.budget`), the blockwise evaluation
(:mod:`quadrashap.product_games.blocks`) and the four numerical backends of
:mod:`quadrashap.product_games.shapley`.  The named explainers (``RKHSExplainer``,
``PoissonExplainer``, ...) are thin subclasses that fix the adapter.
"""
from __future__ import annotations

import time
import warnings
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from quadrashap.product_games.shapley import (
    ProductGamesShapleyNumpy,
    ProductGamesShapleyJax,
    JAX_AVAILABLE,
)
from quadrashap.product_games.budget import BudgetReport, GameSummary, budget_from_summary, certify
from quadrashap.product_games.blocks import BlockPlan, plan_blocks, pad_block
from quadrashap.multiplicative.base import MultiplicativeModel

VALUE_FUNCTIONS = ("neutral", "baseline", "interventional")
BACKENDS = ("logspace_numpy", "logspace_jax", "prefix_scan_numpy", "prefix_scan_jax")


class QuadraSHAP:
    """Shapley values of a multiplicative model by Gauss--Legendre quadrature.

    Parameters
    ----------
    model : a :class:`MultiplicativeModel` adapter, or a fitted estimator that
        :func:`quadrashap.multiplicative.detect.adapter_for` recognises (product-kernel machines,
        log-link GLMs, logistic regression, naive Bayes, Cox models).
    background : optional default background dataset ``(n_b, d)`` for the interventional value function.
    eps : default absolute tolerance used to choose the number of nodes (``m_q=None``).
    backend : ``"logspace_jax"``, ``"logspace_numpy"``, ``"prefix_scan_jax"``, ``"prefix_scan_numpy"`` or
        ``"auto"`` (log-space JAX when available, else log-space NumPy; the division-free prefix scan
        whenever a factor table contains zeros).
    block_size : blockwise evaluation of the component--background pairs (Section 4, memory
        paragraph). ``"auto"`` (default) blocks only when the full computation would exceed the
        memory budget or the cache-size cap; a positive integer fixes the number of pairs per block;
        ``-1`` (or ``None``) switches blocking off. The plan actually used is stored in ``last_block_plan``.
    memory_budget : bytes, a string such as ``"512MB"``, or ``None`` for ``memory_fraction`` of the
        memory currently available on the backend's device.
    memory_fraction : share of the available memory the automatic plan may use (default 0.5).
    target_block_bytes : soft performance cap on the working set of one core call in ``"auto"`` mode
        (``"auto"``: 16 MB for the NumPy prefix scan; none for the other backends); ``None`` disables it.

    Notes
    -----
    Attributions are on the model's multiplicative scale (``model.scale``): the kernel expansion
    without intercept, the mean response of a log-link GLM, the odds of a classifier, the hazard
    ratio of a Cox model.  They satisfy efficiency with respect to the chosen value function,
    ``sum_i phi_i = F(x) - v(empty)`` with ``F = model.predict_scale``, where ``v(empty)`` is
    ``sum_r c_r`` (neutral), ``F(x^b)`` (baseline) or the mean of ``F`` over the background
    (interventional); see :meth:`value_function_at_empty` and :attr:`expected_value`.
    """

    def __init__(self, model, *, background: Optional[np.ndarray] = None, eps: float = 1e-3, backend: str = "auto",
                 block_size: Union[int, str, None] = "auto", memory_budget: Union[int, float, str, None] = None,
                 memory_fraction: float = 0.5, target_block_bytes: Union[int, float, str, None] = "auto"):
        if not isinstance(model, MultiplicativeModel):
            from quadrashap.multiplicative.detect import adapter_for
            model = adapter_for(model)
        self.model: MultiplicativeModel = model
        self.coef = np.asarray(model.coef, dtype=np.float64).reshape(-1)
        self.n, self.d = int(self.coef.shape[0]), int(model.d)
        self.background = None if background is None else np.atleast_2d(np.asarray(background, dtype=np.float64))
        self.eps = float(eps)
        self.backend = backend
        self.block_size = block_size
        self.memory_budget = memory_budget
        self.memory_fraction = float(memory_fraction)
        self.target_block_bytes = target_block_bytes
        self.last_block_plan: Optional[BlockPlan] = None
        self._np = ProductGamesShapleyNumpy()
        self._jax = None

    def __getattr__(self, name):  # convenience: expose adapter attributes (X_train, gamma, intercept, beta, ...)
        if name.startswith("_") or name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    @property
    def alpha(self) -> np.ndarray:
        """The component coefficients ``c_r`` (dual coefficients for kernel machines)."""
        return self.coef

    @property
    def scale(self) -> str:
        return self.model.scale

    @property
    def expected_value(self) -> float:
        """``v(empty)`` of the default value function (interventional needs the constructor's ``background``)."""
        return self.value_function_at_empty(None)

    # ------------------------------------------------------------------ factor tables
    def factors(self, x: np.ndarray) -> np.ndarray:
        """Present factors ``u_j^{(r)}(x_j)`` as a ``(p, d)`` table."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x has {x.shape[0]} features, model has {self.d}")
        return np.asarray(self.model.factors(x), dtype=np.float64)

    def value_function_at_empty(self, value_function: Optional[str] = None, *, baseline=None, background=None) -> float:
        """``v(empty)`` of the chosen value function, on the scale of the attributions."""
        vf, ref = self._resolve(value_function, baseline, background)
        if vf == "neutral":
            return float(self.coef.sum())
        return float(np.mean(self.model.predict_scale(ref)))

    def _resolve(self, value_function: str, baseline, background) -> Tuple[str, Optional[np.ndarray]]:
        if value_function is None:
            value_function = self.model.default_value_function
        if value_function not in VALUE_FUNCTIONS:
            raise ValueError(f"value_function must be one of {VALUE_FUNCTIONS}")
        if value_function == "neutral":
            if not self.model.supports_neutral:
                raise ValueError(f"the neutral-factor value function is not defined for {type(self.model).__name__}; "
                                 "use 'baseline' (reference point) or 'interventional' (background dataset)")
            return "neutral", None
        if value_function == "baseline":
            if baseline is None:
                raise ValueError("value_function='baseline' needs a reference point `baseline`")
            return "baseline", np.atleast_2d(np.asarray(baseline, dtype=np.float64))
        bg = self.background if background is None else np.atleast_2d(np.asarray(background, dtype=np.float64))
        if bg is None:
            raise ValueError("value_function='interventional' needs a `background` dataset "
                             "(pass it here or at construction)")
        if bg.shape[1] != self.d:
            raise ValueError(f"background has {bg.shape[1]} features, model has {self.d}")
        return "interventional", bg

    def n_pairs(self, value_function: Optional[str] = None, *, baseline=None, background=None) -> int:
        """Number of component--background pairs of the value function (``p`` times the number of reference rows)."""
        vf, ref = self._resolve(value_function, baseline, background)
        return self.n if vf == "neutral" else self.n * ref.shape[0]

    def games(self, x: np.ndarray, value_function: Optional[str] = None, *, baseline=None, background=None,
              block_size: Optional[int] = None) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
        """Yield the product games of the value function in blocks of at most ``block_size`` pairs.

        Each block is ``(K, Ut, w)`` with ``K = u - ut`` of shape (b, d), the absent factors
        ``Ut`` (``None`` for the neutral factor, else (b, d)) and weights ``w`` (b,): ``alpha``
        for the neutral and baseline value functions and ``alpha / n_b`` per background row
        for the interventional one, so that the Shapley values are the sum over all blocks
        of ``(Phi(K, Ut) * w[:, None]).sum(0)``.  ``block_size=None`` yields one block per
        reference row (all ``p`` components at once).
        """
        U = self.factors(x)
        vf, ref = self._resolve(value_function, baseline, background)
        if vf == "neutral":
            bs = self.n if block_size is None else max(1, int(block_size))
            for r0 in range(0, self.n, bs):
                yield U[r0:r0 + bs] - 1.0, None, self.coef[r0:r0 + bs]
            return
        n_b = ref.shape[0]
        w_all = self.coef / n_b
        bs = self.n if block_size is None else max(1, int(block_size))
        buf_K, buf_Ut, buf_w, filled = [], [], [], 0
        for r in ref:
            Ut = self.factors(r)
            r0 = 0
            while r0 < self.n:
                take = min(bs - filled, self.n - r0)
                buf_K.append(U[r0:r0 + take] - Ut[r0:r0 + take]); buf_Ut.append(Ut[r0:r0 + take]); buf_w.append(w_all[r0:r0 + take])
                filled += take; r0 += take
                if filled == bs:
                    yield (np.concatenate(buf_K), np.concatenate(buf_Ut), np.concatenate(buf_w)) if len(buf_K) > 1 else (buf_K[0], buf_Ut[0], buf_w[0])
                    buf_K, buf_Ut, buf_w, filled = [], [], [], 0
        if filled:
            yield (np.concatenate(buf_K), np.concatenate(buf_Ut), np.concatenate(buf_w)) if len(buf_K) > 1 else (buf_K[0], buf_Ut[0], buf_w[0])

    def plan_blocks(self, m_q: int, value_function: Optional[str] = None, *, baseline=None, background=None,
                    backend: Optional[str] = None, block_size: Union[int, str, None, object] = "unset") -> BlockPlan:
        """Decide the block sizes for one ``explain`` call from the memory budget (see :mod:`quadrashap.product_games.blocks`)."""
        backend = self._resolve_backend(self.backend if backend is None else backend)
        bs = self.block_size if block_size == "unset" else block_size
        return plan_blocks(self.n_pairs(value_function, baseline=baseline, background=background), m_q, self.d, backend,
                           block_size=bs, memory_budget=self.memory_budget, memory_fraction=self.memory_fraction,
                           target_bytes=self.target_block_bytes)

    # ------------------------------------------------------------------ node budget
    def summarize(self, x, value_function: Optional[str] = None, *, baseline=None, background=None) -> GameSummary:
        """One pass over the factor tables: ``A_i`` and ``Lambda_max`` of the games for ``x``."""
        summ = GameSummary(d=self.d)
        # the summary pass keeps a few (b, d) tables per block: plan it like a log-space call with one node
        plan = self.plan_blocks(1, value_function, baseline=baseline, background=background, backend="logspace_numpy")
        for K, Ut, w in self.games(x, value_function, baseline=baseline, background=background, block_size=plan.block_size):
            summ.update(K, Ut, w)
        return summ

    def node_budget(self, x, eps: Optional[float] = None, value_function: Optional[str] = None, *,
                    baseline=None, background=None, norm: str = "max") -> BudgetReport:
        """A priori number of nodes certifying ``max_i |phi_hat_i - phi_i| <= eps`` for this instance."""
        eps = self.eps if eps is None else float(eps)
        t0 = time.perf_counter()
        summ = self.summarize(x, value_function, baseline=baseline, background=background)
        rep = budget_from_summary(summ, eps, norm=norm)
        rep.seconds = time.perf_counter() - t0  # include the factor-table pass
        return rep

    # ------------------------------------------------------------------ quadrature
    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "auto":
            return "logspace_jax" if JAX_AVAILABLE else "logspace_numpy"
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS} or 'auto'")
        return backend

    def _phi_fn(self, backend: str, has_zero: bool):
        """Return ``(fn, is_prefix_scan)`` for the resolved backend, switching to the scan when factors vanish."""
        if has_zero and backend.startswith("logspace"):
            if self.backend == "auto":
                backend = backend.replace("logspace", "prefix_scan")
            else:
                warnings.warn("log-space backend with vanishing factors: results are approximate; "
                              "use a prefix_scan backend for exactness", RuntimeWarning)
        if backend.endswith("_jax"):
            if not JAX_AVAILABLE:
                raise RuntimeError("JAX is not available on this system.")
            if self._jax is None:
                self._jax = ProductGamesShapleyJax()
            obj = self._jax
        else:
            obj = self._np
        if backend.startswith("logspace"):
            return obj.phi_matrix_logspace, False
        return obj.phi_matrix_prefix_scan, True

    def explain(self, x, value_function: Optional[str] = None, *, baseline=None, background=None,
                m_q: Union[int, str, None] = None, eps: Optional[float] = None, backend: Optional[str] = None,
                block_size: Union[int, str, None, object] = "unset", return_report: bool = False):
        """Shapley values ``phi`` (d,) of ``x`` under the chosen value function.

        Parameters
        ----------
        m_q : ``None`` (default) chooses the number of nodes a priori so that every attribution
            is within ``eps`` of exact; ``"exact"`` uses the exactness threshold ``ceil(d/2)``;
            an integer is used as given.
        eps : tolerance for the automatic budget (defaults to the explainer's ``eps``).
        block_size : overrides the explainer's ``block_size`` for this call (``"auto"``, a positive
            integer, or ``-1``/``None`` for off); the plan used is stored in ``last_block_plan``.
        return_report : also return the :class:`BudgetReport` (with the certified error of the
            ``m_q`` actually used).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x has {x.shape[0]} features, model has {self.d}")
        backend = self._resolve_backend(self.backend if backend is None else backend)
        exact = max(1, (self.d + 1) // 2)

        report = None
        if m_q is None or m_q == "auto":
            report = self.node_budget(x, eps, value_function, baseline=baseline, background=background)
            m_q = report.m_q
        else:
            m_q = exact if m_q == "exact" else int(m_q)
            if m_q < 1:
                raise ValueError("m_q must be >= 1")
            if return_report:  # report the certified error of the m_q actually used
                eps_ = self.eps if eps is None else float(eps)
                summ = self.summarize(x, value_function, baseline=baseline, background=background)
                report = budget_from_summary(summ, eps_)
                report.m_q, report.bound = m_q, certify(summ, m_q)

        plan = self.plan_blocks(m_q, value_function, baseline=baseline, background=background,
                                backend=backend, block_size=block_size)
        self.last_block_plan = plan
        jit_backend = backend.endswith("_jax")

        phi = np.zeros(self.d, dtype=np.float64)
        for K, Ut, w in self.games(x, value_function, baseline=baseline, background=background,
                                   block_size=plan.block_size):
            if jit_backend and plan.n_blocks > 1:
                K, Ut, w = pad_block(K, Ut, w, plan.block_size)  # one shape -> one compilation
            has_zero = bool((K + (1.0 if Ut is None else Ut) == 0).any() or (Ut is not None and (Ut == 0).any()))
            fn, is_scan = self._phi_fn(backend, has_zero)
            supports_node_block = is_scan or backend.endswith("_jax")
            Phi = fn(K, m_q, Ut=Ut, node_block=plan.node_block) if supports_node_block else fn(K, m_q, Ut=Ut)
            phi += (Phi * w[:, None]).sum(axis=0)
        if report is not None:
            # a posteriori necessary check: sum_i phi_i must equal f(x) - v(empty) (exactly so at the exactness threshold)
            f_x = float(self.model.predict_scale(x[None, :])[0])
            v0 = self.value_function_at_empty(value_function, baseline=baseline, background=background)
            report.efficiency_residual = abs(float(phi.sum()) - (f_x - v0))
        return (phi, report) if return_report else phi

    def shap_values(self, X, **kwargs) -> np.ndarray:
        """Explain each row of ``X`` (``(n, d)`` -> ``(n, d)``); keyword arguments as in :meth:`explain`."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.stack([self.explain(row, **kwargs) for row in X], axis=0)

