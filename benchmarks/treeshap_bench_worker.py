"""Worker for treeshap_bench.py.

Invoked as a subprocess by the parent benchmark driver. Loads a cached
(model, X_test) pickle and runs a single SHAP method, then prints the result
as JSON on a single line:

    {"status": "ok", "elapsed_s": 0.123, "sv_sum": [...], "ev": 1.23}

or:

    {"status": "err", "message": "..."}

Each method is split into a ``build(model)`` step and an ``explain(expl, X)``
step so that explainer construction (which can take hundreds of ms for
sklearn-backed libraries even on tiny trees) is excluded from the timed
region. We do one untimed warmup call on ``explain``, then time N_REPEATS
timed calls and take the median.

The ``ev`` (expected value) field is the *library's own* baseline:

- SHAP / FastTreeSHAP / pgshapley all use ``tree_.value[0]`` (the
  bootstrap-*weighted* root mean) and expose it as
  ``explainer.expected_value``.
- ``linear_tree_shap`` doesn't expose a baseline and doesn't read sklearn's
  bootstrap weights, so for a bagged tree its baseline is the mean of leaf
  values weighted by ``tree_.n_node_samples`` (unique counts, not the
  bootstrap-weighted count). For a plain ``DecisionTreeRegressor`` the two
  coincide, but inside an sklearn RandomForest they differ by the bootstrap
  duplicate pattern. We compute the correct one in
  ``_linear_tree_shap_ev_single`` below.

With this, ``|ev + sum(phi) - pred|`` is the correct additivity residual
for every library, and the metric is fair across all of them.

The parent imposes a wall-clock timeout via subprocess.run(timeout=...).
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Per-method (build, explain, ev) triples. `build` returns an opaque handle
# that `explain` consumes, and `ev(model, handle)` returns the library's
# expected-value baseline.
# ---------------------------------------------------------------------------

def build_shap(model):
    import shap
    return shap.TreeExplainer(model)


def explain_shap(expl, X):
    return expl.shap_values(X, check_additivity=False)


def ev_shap(model, expl):
    return float(np.asarray(expl.expected_value).ravel()[0])


def build_fasttreeshap_v1(model):
    import fasttreeshap
    return fasttreeshap.TreeExplainer(model, algorithm="v1", n_jobs=1)


def build_fasttreeshap_v2(model):
    import fasttreeshap
    return fasttreeshap.TreeExplainer(model, algorithm="v2", n_jobs=1)


def explain_fasttreeshap(expl, X):
    return expl.shap_values(X, check_additivity=False)


def ev_fasttreeshap(model, expl):
    return float(np.asarray(expl.expected_value).ravel()[0])


def build_linear_tree_shap(model):
    import linear_tree_shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    if isinstance(model, RandomForestRegressor):
        expls = [linear_tree_shap.TreeExplainer(est) for est in model.estimators_]
        return ("rf", expls)
    if isinstance(model, DecisionTreeRegressor):
        return ("tree", linear_tree_shap.TreeExplainer(model))
    raise NotImplementedError("linear_tree_shap only handles sklearn trees")


def explain_linear_tree_shap(handle, X):
    kind, payload = handle
    if kind == "rf":
        svs = [e.shap_values(X) for e in payload]
        return np.mean(np.stack(svs, axis=0), axis=0)
    return payload.shap_values(X)


def _linear_tree_shap_ev_single(tree) -> float:
    """linear_tree_shap's baseline for one sklearn tree.

    Unlike SHAP, linear_tree_shap's internal tree adapter uses the
    *unique-sample* leaf counts (``tree_.n_node_samples``) rather than the
    bootstrap-weighted counts (``tree_.weighted_n_node_samples``). For a
    plain ``DecisionTreeRegressor`` the two coincide (no bootstrap, no
    sample weights) and everything matches SHAP's root-value baseline, but
    for a bootstrap-bagged tree (every tree in an sklearn RF) they differ
    by the bootstrap duplicate pattern, and linear_tree_shap's prediction
    equals

        sum(leaf_val * n_node_samples_leaf) / sum(n_node_samples_leaf)

    which is what we compute here. Verified empirically to match to 1e-13.
    """
    t = tree.tree_
    leaf_mask = (t.children_left == -1)
    leaf_vals = t.value[leaf_mask].ravel().astype(np.float64)
    leaf_counts = t.n_node_samples[leaf_mask].astype(np.float64)
    return float((leaf_vals * leaf_counts).sum() / leaf_counts.sum())


def ev_linear_tree_shap(model, handle):
    """Baseline for linear_tree_shap. See _linear_tree_shap_ev_single."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    if isinstance(model, RandomForestRegressor):
        # RF prediction = arithmetic mean across trees.
        evs = [_linear_tree_shap_ev_single(est) for est in model.estimators_]
        return float(np.mean(evs))
    if isinstance(model, DecisionTreeRegressor):
        return _linear_tree_shap_ev_single(model)
    raise NotImplementedError


def build_pg_quad_cpp(model):
    from pgshapley import TreeExplainer
    return TreeExplainer(model, tree_solver="quadrature_tree", use_cpp=True)


def build_pg_quad_cpp_mq_d_over_4(model):
    """Approximate quadrature variant with m_q = max(1, n_features // 4).

    This undershoots the exact order ``ceil(D/2)`` (where ``D`` is the longest
    distinct-feature root-to-leaf path) when ``n_features/4 < D/2`` — so for
    small-feature / deep-tree regimes it's a strict approximation. When
    ``n_features/4 > D/2`` it simply runs more quadrature nodes than needed
    (slower, still exact up to fp precision).
    """
    from pgshapley import TreeExplainer
    n_features = int(model.n_features_in_)
    m_q = max(1, n_features // 4)
    return TreeExplainer(
        model, tree_solver="quadrature_tree", use_cpp=True, m_q=m_q,
    )


def explain_pg_quad_cpp(expl, X):
    return expl.shap_values(X, check_additivity=False)


def ev_pg_quad_cpp(model, expl):
    return float(np.asarray(expl.expected_value).ravel()[0])


def build_shapiq(model):
    from shapiq.explainer.tree import TreeExplainer
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TreeExplainer(model=model, max_order=1)


def explain_shapiq(expl, X):
    results = expl.explain_X(X)
    return np.array([r.get_n_order_values(1) for r in results])


def ev_shapiq(model, expl):
    # Run on a dummy sample to get the baseline
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dummy = np.zeros((1, model.n_features_in_))
        result = expl.explain(dummy[0])
        return float(result.baseline_value)


METHODS = {
    "shap":                   (build_shap,              explain_shap,              ev_shap),
    "fasttreeshap_v1":        (build_fasttreeshap_v1,   explain_fasttreeshap,      ev_fasttreeshap),
    "fasttreeshap_v2":        (build_fasttreeshap_v2,   explain_fasttreeshap,      ev_fasttreeshap),
    "linear_tree_shap":       (build_linear_tree_shap,  explain_linear_tree_shap,  ev_linear_tree_shap),
    "shapiq":                 (build_shapiq,            explain_shapiq,            ev_shapiq),
    "pg_quadrature_tree_cpp": (build_pg_quad_cpp,       explain_pg_quad_cpp,       ev_pg_quad_cpp),
    "pg_quadrature_tree_cpp_mq_d_over_4": (
        build_pg_quad_cpp_mq_d_over_4, explain_pg_quad_cpp, ev_pg_quad_cpp,
    ),
}


def main():
    if len(sys.argv) != 4:
        print(json.dumps({
            "status": "err",
            "message": "usage: worker.py MODEL_PKL METHOD N_REPEATS",
        }))
        return 2

    model_pkl = sys.argv[1]
    method = sys.argv[2]
    n_repeats = int(sys.argv[3])

    try:
        with open(model_pkl, "rb") as f:
            model, X = pickle.load(f)
    except Exception as e:
        print(json.dumps({"status": "err", "message": f"load: {e}"}))
        return 1

    entry = METHODS.get(method)
    if entry is None:
        print(json.dumps({"status": "err", "message": f"unknown method {method}"}))
        return 1
    build, explain, get_ev = entry

    try:
        # --- untimed: construction ---
        handle = build(model)
        ev = get_ev(model, handle)

        # --- untimed: warmup (JIT, caches, any first-call work) ---
        sv = explain(handle, X)

        # --- timed: N_REPEATS calls; bail out of extra repeats if a single
        #     run already took >1 s (noise no longer matters at that scale). ---
        times = []
        for _ in range(max(1, n_repeats)):
            t0 = time.perf_counter()
            sv = explain(handle, X)
            times.append(time.perf_counter() - t0)
            if times[0] > 1.0:
                break

        sv_arr = np.asarray(sv).reshape(X.shape[0], -1).astype(np.float64)
        out = {
            "status": "ok",
            "elapsed_s": float(np.median(times)),
            "times": [float(t) for t in times],
            "sv_sum": [float(v) for v in sv_arr.sum(axis=1)],
            "ev": float(ev),
        }
        print(json.dumps(out))
        return 0
    except BaseException as e:
        print(json.dumps({"status": "err", "message": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
