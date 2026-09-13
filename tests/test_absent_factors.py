"""Product games with arbitrary absent factors: the quadrature cores vs. exhaustive enumeration.

Runs under pytest or directly (``python tests/test_absent_factors.py``).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy, ProductGamesShapleyJax, JAX_AVAILABLE
from tests.naive_shapley import naive_shapley, naive_shapley_absent

ATOL = 1e-9


def _cores(include_logspace=True):
    np_obj = ProductGamesShapleyNumpy()
    yield "numpy_prefix_scan", np_obj.phi_matrix_prefix_scan
    if include_logspace:
        yield "numpy_logspace", np_obj.phi_matrix_logspace
    if JAX_AVAILABLE:
        import jax
        jax.config.update("jax_enable_x64", True)
        jx = ProductGamesShapleyJax()
        yield "jax_prefix_scan", jx.phi_matrix_prefix_scan
        if include_logspace:
            yield "jax_logspace", jx.phi_matrix_logspace


def test_neutral_factor_is_unchanged():
    """Ut=None must reproduce the original product game (Definition 1) bit for bit, and Ut=1 too."""
    rng = np.random.default_rng(0)
    for name, fn in _cores():
        for _ in range(20):
            d = int(rng.integers(1, 7))
            K = rng.uniform(-3, 3, size=(3, d))
            m_q = (d + 1) // 2
            ref = fn(K, m_q)
            np.testing.assert_array_equal(fn(K, m_q, Ut=None), ref, err_msg=name)
            np.testing.assert_allclose(fn(K, m_q, Ut=np.ones(d)), ref, rtol=1e-12, err_msg=name)
            for r in range(3):
                np.testing.assert_allclose(ref[r], naive_shapley(K[r] + 1.0), atol=ATOL, err_msg=name)


def test_absent_factors_vs_naive():
    """Present and absent factors of arbitrary sign and magnitude, exactness at ceil(d/2) nodes."""
    rng = np.random.default_rng(1)
    for name, fn in _cores():
        for _ in range(200):
            d = int(rng.integers(1, 7))
            U = rng.uniform(-4, 4, size=(2, d))
            Ut = rng.uniform(-4, 4, size=(2, d))
            K = U - Ut
            phi = fn(K, (d + 1) // 2, Ut=Ut)
            assert phi.shape == (2, d)
            for r in range(2):
                np.testing.assert_allclose(phi[r], naive_shapley_absent(U[r], Ut[r]), atol=ATOL, err_msg=name)
        # one shared absent row broadcast to every game
        d = 5
        U = rng.uniform(0, 2, size=(4, d)); ut = rng.uniform(0, 2, size=d)
        phi = fn(U - ut[None, :], 3, Ut=ut)
        for r in range(4):
            np.testing.assert_allclose(phi[r], naive_shapley_absent(U[r], ut), atol=ATOL, err_msg=name)


def test_vanishing_absent_factors_prefix_scan_exact():
    """Rule indicators inactive at a background point give ut_j = 0; the scan is exact there."""
    rng = np.random.default_rng(2)
    for name, fn in _cores(include_logspace=False):
        for _ in range(100):
            d = int(rng.integers(1, 7))
            U = rng.uniform(0, 2, size=(1, d))
            Ut = rng.uniform(0, 2, size=(1, d)) * (rng.random((1, d)) < 0.6)  # ~40% zeros
            U = U * (rng.random((1, d)) < 0.8)
            phi = fn(U - Ut, (d + 1) // 2, Ut=Ut)
            np.testing.assert_allclose(phi[0], naive_shapley_absent(U[0], Ut[0]), atol=ATOL, err_msg=name)


def test_efficiency_large_d():
    """sum_i phi_i = prod u - prod ut, checked in log-space for d in the thousands."""
    rng = np.random.default_rng(3)
    for name, fn in _cores():
        for d in (100, 1000):
            c = 3.0 / np.sqrt(d)
            U = np.exp(rng.uniform(-c, c, size=(2, d)))
            Ut = np.exp(rng.uniform(-c, c, size=(2, d)))
            phi = fn(U - Ut, (d + 1) // 2, Ut=Ut)
            np.testing.assert_allclose(phi.sum(axis=1), U.prod(axis=1) - Ut.prod(axis=1), rtol=1e-7, err_msg=name)


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn(); print("ok", fn_name)
