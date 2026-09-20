"""Large exact rules must avoid a dense eigenproblem and preserve integration."""
import numpy as np
import pytest

from quadrashap.product_games.shapley import _gauss_legendre_01_numpy


def test_large_rule_avoids_dense_generator_and_integrates_moments(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense leggauss must not be used for large rules")

    monkeypatch.setattr(np.polynomial.legendre, "leggauss", forbidden)
    _gauss_legendre_01_numpy.cache_clear()
    x, w = _gauss_legendre_01_numpy(1024)
    assert np.all(np.diff(x) > 0) and np.all(w > 0)
    assert 0 < x[0] < x[-1] < 1
    for k in (0, 1, 2, 8, 31):
        np.testing.assert_allclose(w @ x ** k, 1 / (k + 1), atol=1e-12, rtol=1e-12)
    assert _gauss_legendre_01_numpy(1024)[0] is x
    assert not x.flags.writeable and not w.flags.writeable


def test_parallel_logspace_core_matches_signed_product_games():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from quadrashap.product_games.shapley import ProductGamesShapleyJax, ProductGamesShapleyNumpy

    rng = np.random.default_rng(9)
    signs = rng.choice([-1, 1], size=(3, 37))
    absent = signs * rng.uniform(0.9, 1.1, size=(3, 37))
    present = signs * rng.uniform(0.9, 1.1, size=(3, 37))
    K = present - absent
    x, w = _gauss_legendre_01_numpy(19)
    actual = ProductGamesShapleyJax._phi_logspace_parallel_core(
        jnp.asarray(K, dtype=jnp.float32), jnp.asarray(absent, dtype=jnp.float32),
        jnp.asarray(x, dtype=jnp.float32), jnp.asarray(w, dtype=jnp.float32), 1e-30)
    expected = ProductGamesShapleyNumpy().phi_matrix_prefix_scan(K, 19, Ut=absent)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


def test_jax_logspace_node_blocks_match_unblocked():
    pytest.importorskip("jax")
    from quadrashap.product_games.shapley import ProductGamesShapleyJax

    rng = np.random.default_rng(12)
    K = rng.uniform(-0.03, 0.03, size=(2, 17)).astype(np.float32)
    absent = rng.uniform(0.9, 1.1, size=(2, 17)).astype(np.float32)
    backend = ProductGamesShapleyJax()
    expected = backend.phi_matrix_logspace(K, 9, Ut=absent)
    actual = backend.phi_matrix_logspace(K, 9, Ut=absent, node_block=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
