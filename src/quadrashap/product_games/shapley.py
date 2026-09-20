"""
Shapley values of product games by Gauss--Legendre quadrature.

A product game with present factors ``u_j`` and absent factors ``ut_j`` has
coalition value ``v(S) = prod_{j in S} u_j * prod_{j not in S} ut_j``.  Its
Shapley values admit the one-dimensional integral representation

    phi_i = (u_i - ut_i) * int_0^1 prod_{j != i} ((1 - t) ut_j + t u_j) dt,

whose integrand is a polynomial of degree at most ``d - 1`` in ``t``.  An
``m_q``-point Gauss--Legendre rule on [0, 1] evaluates it exactly whenever
``m_q >= ceil(d / 2)`` and with geometrically decaying error below that
threshold (Proposition 2 of the paper); the a priori node budget for a
prescribed accuracy lives in :mod:`quadrashap.product_games.budget`.

All routines below take the table ``K = u - ut`` of shape ``(m, d)``, one row
per game, and optionally the absent-factor table ``Ut`` (broadcastable to
``(m, d)``).  With ``Ut=None`` the absent factor is the neutral factor 1, i.e.
the product game of Definition 1, ``v(S) = prod_{j in S} u_j``.  Fixing
``Ut`` to the factors of a reference point gives the baseline value function,
and averaging over background rows gives the empirical interventional value
function (Section 4 of the paper); both are handled by the callers.
"""
from functools import lru_cache

import numpy as np

# Optional JAX support
try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


@lru_cache(maxsize=16)
def _gauss_legendre_01_numpy(m_q: int, dtype=np.float64):
    """
    Gauss–Legendre nodes/weights mapped from [-1,1] to [0,1].
    """
    if not isinstance(m_q, (int, np.integer)) or m_q < 1:
        raise ValueError("m_q must be a positive integer")
    if m_q <= 256:
        x, w = np.polynomial.legendre.leggauss(m_q)
    else:
        # NumPy's dense companion matrix is prohibitive at genomics-scale d/2.
        # SciPy's tridiagonal construction uses linear working memory instead.
        from scipy.special import roots_legendre
        x, w = roots_legendre(m_q)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    x, w = x.astype(dtype, copy=False), w.astype(dtype, copy=False)
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def _absent_factors_numpy(K: np.ndarray, Ut):
    """Return the absent-factor table broadcast against ``K`` (``None`` -> neutral factor 1)."""
    if Ut is None:
        return np.ones((1, K.shape[1]), dtype=K.dtype)
    Ut = np.asarray(Ut, dtype=K.dtype)
    if Ut.ndim == 1:
        Ut = Ut[None, :]
    if Ut.shape[-1] != K.shape[1] or Ut.ndim != 2 or Ut.shape[0] not in (1, K.shape[0]):
        raise ValueError(f"Ut must broadcast to K.shape={K.shape}; got {Ut.shape}")
    return Ut


def _jax_work_dtype(K: np.ndarray):
    """Choose a JAX compute dtype that matches backend capabilities."""
    dtype = np.result_type(K.dtype, np.float32)

    if not JAX_AVAILABLE:
        return np.dtype(dtype)

    try:
        platforms = {dev.platform for dev in jax.devices()}
    except Exception:
        platforms = set()

    if (jax.default_backend() != "cpu") or (platforms & {"gpu", "tpu", "metal"}):
        return np.dtype(np.float32)

    if not getattr(jax.config, "jax_enable_x64", False) and np.dtype(dtype) == np.dtype(np.float64):
        return np.dtype(np.float32)

    return np.dtype(dtype)


class ProductGamesShapleyNumpy:
    """
    Coefficient-free product-game Shapley factors in NumPy.

    This class operates on the table ``K = u - ut`` of shape (m, d), one product
    game per row, with optional absent factors ``Ut`` (default: the neutral
    factor 1).  It returns a matrix Phi of shape (m, d) with

        Phi[r, i] = (u_i - ut_i) * sum_q w_q prod_{j != i} (ut_j + tau_q (u_j - ut_j))

    for game r, so that for any coefficients alpha of shape (m,) the Shapley
    values of the weighted sum of games are

        shapley = (Phi * alpha[:, None]).sum(axis=0).

    The implementations do not depend on any model or kernel; they only
    require K (and Ut) and the quadrature size m_q.  ``phi_matrix_prefix_scan``
    is division-free and exact for vanishing factors; ``phi_matrix_logspace``
    forms one shared product per node in log-space and is the memory-lean
    choice when all factors are non-zero.
    """

    def phi_matrix_prefix_scan(self, K: np.ndarray, m_q: int, Ut=None, node_block: int | None = None) -> np.ndarray:
        """Division-free evaluation by exclusive prefix and suffix products.

        ``node_block`` limits how many quadrature nodes are processed at once: the
        working tensors have shape ``(node_block, m, d)`` and the sum over nodes is
        accumulated, which is exact and bounds peak memory.  ``None`` processes all
        nodes at once.
        """
        K = np.asarray(K, dtype=np.float64)
        m, d = K.shape
        Ut = _absent_factors_numpy(K, Ut)

        x, w = _gauss_legendre_01_numpy(m_q, dtype=np.float64)
        nb = m_q if node_block is None else max(1, int(node_block))
        acc = np.zeros((m, d), dtype=np.float64)
        for q0 in range(0, m_q, nb):
            X = x[q0:q0 + nb, None, None]  # (nb,1,1)
            B = Ut[None, :, :] + X * K[None, :, :]  # (nb, m, d): factors (1-t) ut + t u

            pref = np.cumprod(B, axis=2)
            pref = np.concatenate(
                [np.ones((B.shape[0], m, 1), dtype=B.dtype), pref[:, :, :-1]], axis=2
            )

            suf = np.cumprod(B[:, :, ::-1], axis=2)[:, :, ::-1]
            suf = np.concatenate(
                [suf[:, :, 1:], np.ones((B.shape[0], m, 1), dtype=B.dtype)], axis=2
            )

            pref *= suf  # leave-one-out products, (nb, m, d)
            acc += (w[q0:q0 + nb, None, None] * pref).sum(axis=0)
        return K * acc

    def phi_matrix_logspace(self, K: np.ndarray, m_q: int, Ut=None, eps: float = 1e-12) -> np.ndarray:
        """Compute Phi (m,d) using the log-space shared product.

        Memory-lean in the shared product, but still returns Phi (m,d).  Factors
        with magnitude below ``eps`` are clamped, so this routine is exact only
        when no factor vanishes; use ``phi_matrix_prefix_scan`` otherwise.
        """
        K = np.asarray(K, dtype=np.float64)
        m, d = K.shape
        Ut = _absent_factors_numpy(K, Ut)
        Ut_b = np.broadcast_to(Ut, (m, d))

        x, w = _gauss_legendre_01_numpy(m_q, dtype=np.float64)

        log_abs_P = np.zeros((m_q, m), dtype=np.float64)
        sign_P = np.ones((m_q, m), dtype=np.float64)

        for j in range(d):
            t = Ut_b[None, :, j] + np.outer(x, K[:, j])  # (m_q,m): (1-t) ut_j + t u_j
            sign_P *= np.sign(t)
            log_abs_P += np.log(np.maximum(np.abs(t), eps), dtype=np.float64)

        Qint = np.empty((m, d), dtype=np.float64)
        wa = w[:, None]
        for i in range(d):
            denom = Ut_b[None, :, i] + np.outer(x, K[:, i])  # (m_q,m)
            integrand_sign = sign_P * np.sign(denom)
            integrand_log = log_abs_P - np.log(np.maximum(np.abs(denom), eps), dtype=np.float64)
            Qint[:, i] = (wa * (integrand_sign * np.exp(integrand_log))).sum(axis=0)

        return K * Qint


if JAX_AVAILABLE:
    class ProductGamesShapleyJax:
        """
        Product-game Shapley factors in JAX (same contract as ``ProductGamesShapleyNumpy``).

        Returns Phi (m,d) in NumPy format.
        """

        def __init__(self):
            if not JAX_AVAILABLE:
                raise RuntimeError("JAX is not available on this system.")

        @staticmethod
        @jax.jit
        def _phi_prefix_core(K, Ut, x, w):
            # K: (m, d) = u - ut;  Ut: (1, d) or (m, d) absent factors
            B = Ut[None, :, :] + x[:, None, None] * K[None, :, :]  # (m_q, m, d)
            pref = lax.cumprod(B, axis=2)
            pref = jnp.concatenate(
                [jnp.ones((B.shape[0], B.shape[1], 1), dtype=B.dtype), pref[:, :, :-1]], axis=2
            )
            suf = lax.cumprod(B[:, :, ::-1], axis=2)[:, :, ::-1]
            suf = jnp.concatenate(
                [suf[:, :, 1:], jnp.ones((B.shape[0], B.shape[1], 1), dtype=B.dtype)], axis=2
            )
            Q = pref * suf
            acc = (w[:, None, None] * Q).sum(axis=0)  # (m, d)
            return K * acc

        def phi_matrix_prefix_scan(self, K: np.ndarray, m_q: int, Ut=None, node_block: int | None = None) -> np.ndarray:
            """Same contract as the NumPy version; ``node_block`` bounds the ``(node_block, m, d)`` working set.

            Every node block has the same shape (the last one is padded with zero-weight
            nodes), so the jitted core compiles once per ``(node_block, m, d)``.
            """
            K = np.asarray(K)
            Ut = _absent_factors_numpy(K, Ut)

            x_np, w_np = _gauss_legendre_01_numpy(m_q)

            dtype = _jax_work_dtype(K)
            if K.dtype != dtype:
                K = K.astype(dtype, copy=False)
            Kj = jnp.asarray(K, dtype=dtype)
            Utj = jnp.asarray(Ut, dtype=dtype)

            nb = m_q if node_block is None else max(1, min(int(node_block), m_q))
            out = None
            for q0 in range(0, m_q, nb):
                xb, wb = x_np[q0:q0 + nb], w_np[q0:q0 + nb]
                if xb.shape[0] < nb:  # pad the last block: zero weights contribute nothing
                    pad = nb - xb.shape[0]
                    xb = np.concatenate([xb, np.full(pad, 0.5)]); wb = np.concatenate([wb, np.zeros(pad)])
                blk = self._phi_prefix_core(Kj, Utj, jnp.asarray(xb, dtype=dtype), jnp.asarray(wb, dtype=dtype))
                out = blk if out is None else out + blk
            return np.asarray(out)

        @staticmethod
        @jax.jit
        def _phi_logspace_parallel_core(K, Ut, x, w, eps):
            # Reduce independent features in parallel on the accelerator. A
            # sequential lax.scan over hundreds of thousands of features can
            # exceed Metal's command-buffer execution limit.
            factors = Ut[None, :, :] + x[:, None, None] * K[None, :, :]
            signs = jnp.sign(factors)
            logs = jnp.log(jnp.maximum(jnp.abs(factors), eps))
            log_product = logs.sum(axis=2, keepdims=True)
            sign_product = signs.prod(axis=2, keepdims=True)
            leave_one_out = sign_product * signs * jnp.exp(log_product - logs)
            return K * (w[:, None, None] * leave_one_out).sum(axis=0)

        @staticmethod
        @jax.jit
        def _phi_logspace_core(K, Ut, x, w, eps):
            # K: (m, d) = u - ut;  Ut: (m, d) absent factors
            x = x[:, None]  # (m_q, 1)
            w = w[:, None]  # (m_q, 1)
            m, d = K.shape

            def scan_step(carry, cols):
                # cols = (k_j, ut_j): column j of K and of Ut, each (m,)
                k_j, ut_j = cols
                log_abs_P, sign_P = carry
                t = ut_j[None, :] + x * k_j[None, :]  # (1-t) ut_j + t u_j
                sign_P = sign_P * jnp.sign(t)
                log_abs_P = log_abs_P + jnp.log(jnp.maximum(jnp.abs(t), eps))
                return (log_abs_P, sign_P), None

            # Scan over d columns of K and Ut
            init = (
                jnp.zeros((x.shape[0], m), K.dtype),
                jnp.ones((x.shape[0], m), K.dtype),
            )
            (log_abs_P, sign_P), _ = lax.scan(scan_step, init, (K.T, Ut.T))  # scan over (d, m)

            def per_feature(k_i, ut_i):
                # k_i, ut_i: (m,) — column i of K and of Ut
                denom = ut_i[None, :] + x * k_i[None, :]
                integrand_sign = sign_P * jnp.sign(denom)
                integrand_log = log_abs_P - jnp.log(jnp.maximum(jnp.abs(denom), eps))
                Qint = jnp.sum(w * (integrand_sign * jnp.exp(integrand_log)), axis=0)  # (m,)
                return k_i * Qint  # (m,)

            # vmap over d columns, result (d, m), then transpose to (m, d)
            return jax.vmap(per_feature, in_axes=(0, 0))(K.T, Ut.T).T

        def phi_matrix_logspace(self, K: np.ndarray, m_q: int, Ut=None, eps: float = 1e-100,
                                node_block: int | None = None) -> np.ndarray:
            K = np.asarray(K)
            Ut = np.broadcast_to(_absent_factors_numpy(K, Ut), K.shape)

            x_np, w_np = _gauss_legendre_01_numpy(m_q)

            dtype = _jax_work_dtype(K)
            if K.dtype != dtype:
                K = K.astype(dtype, copy=False)
            Kj = jnp.asarray(K, dtype=dtype)
            Utj = jnp.asarray(Ut, dtype=dtype)
            nb = m_q if node_block is None else max(1, min(int(node_block), m_q))
            core = (self._phi_logspace_core if jax.default_backend().lower() == "cpu"
                    else self._phi_logspace_parallel_core)
            # GPU arithmetic remains float32 on Metal. Accumulate node blocks
            # in host float64 to avoid another rounding drift in very long rules.
            out = np.zeros(K.shape, dtype=np.float64)
            for q0 in range(0, m_q, nb):
                xb, wb = x_np[q0:q0 + nb], w_np[q0:q0 + nb]
                if xb.shape[0] < nb:
                    pad = nb - xb.shape[0]
                    xb = np.concatenate([xb, np.full(pad, 0.5)])
                    wb = np.concatenate([wb, np.zeros(pad)])
                blk = core(Kj, Utj, jnp.asarray(xb, dtype=dtype), jnp.asarray(wb, dtype=dtype), eps)
                out += np.asarray(blk)
            return out
else:
    class ProductGamesShapleyJax:
        def __init__(self):
            raise RuntimeError("JAX is not available on this system.")
