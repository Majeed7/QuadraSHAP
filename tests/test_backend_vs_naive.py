import pytest
import numpy as np

from pgshapley.treeshap.base import TreeShapBackend
from pgshapley.treeshap.product_games import ProductGamesTreeShapBackend
from pgshapley.treeshap.quadrature_tree import QuadratureTreeShapBackend
from pgshapley.treeshap.sklearn import sklearn_to_unified

from pgshapley.product_games.shapley import ProductGamesShapleyNumpy, ProductGamesShapleyJax, JAX_AVAILABLE

from tests.naive_tree_backend import naive_tree_shap_values


def _backend_factories():
    """All TreeSHAP backends to stress-test against the naive enumerator."""
    np_obj = ProductGamesShapleyNumpy()

    def _pg(fn):
        return lambda: ProductGamesTreeShapBackend(phi_matrix_fn=fn, batch_size=64)

    yield pytest.param(_pg(np_obj.phi_matrix_prefix_scan), id="pg_numpy_prefix_scan")
    yield pytest.param(_pg(np_obj.phi_matrix_logspace), id="pg_numpy_logspace")
    if JAX_AVAILABLE:
        jax_obj = ProductGamesShapleyJax()
        yield pytest.param(_pg(jax_obj.phi_matrix_prefix_scan), id="pg_jax_prefix_scan")
        yield pytest.param(_pg(jax_obj.phi_matrix_logspace), id="pg_jax_logspace")
    yield pytest.param(lambda: QuadratureTreeShapBackend(), id="quadrature_tree")


@pytest.mark.parametrize("backend_factory", list(_backend_factories()))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_backend_matches_naive_decision_tree_regression(backend_factory, seed):
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(seed)
    X, y = make_regression(
        n_samples=64,
        n_features=6,
        noise=0.1,
        random_state=seed,
    )
    model = DecisionTreeRegressor(max_depth=3, random_state=seed).fit(X, y)

    ens = sklearn_to_unified(model)

    backend: TreeShapBackend = backend_factory()
    backend.prepare(ens)

    X_test = X[rng.choice(X.shape[0], size=5, replace=False)]

    sv_fast = backend.explain(X_test)  # (n, f, out)
    sv_naive = naive_tree_shap_values(ens, X_test)

    np.testing.assert_allclose(sv_fast, sv_naive, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("backend_factory", list(_backend_factories()))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backend_matches_naive_random_forest_regression(backend_factory, seed):
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    X, y = make_regression(
        n_samples=80,
        n_features=6,
        noise=0.2,
        random_state=seed,
    )
    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=3,
        random_state=seed,
    ).fit(X, y)

    ens = sklearn_to_unified(model)

    backend: TreeShapBackend = backend_factory()
    backend.prepare(ens)

    X_test = X[rng.choice(X.shape[0], size=4, replace=False)]

    sv_fast = backend.explain(X_test)
    sv_naive = naive_tree_shap_values(ens, X_test)

    np.testing.assert_allclose(sv_fast, sv_naive, atol=1e-6, rtol=1e-6)
