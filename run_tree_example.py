"""
TreeExplainer: QuadraSHAP for tree ensembles, start to finish, for ONE instance.

Flat script, no functions: set the knobs at the top, run it, step through it in a
debugger.  It fits a scikit-learn random forest, explains one test point with the
path-dependent TreeSHAP value function (each root-to-leaf path is a product game
of rule indicators; the number of Gauss--Legendre nodes is set by the tree depth,
ceil(D_tree / 2), which is already exact), checks the efficiency identity, and
compares with the `shap` package when it is installed.

    python run_tree_example.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run from a checkout without installing
from quadrashap import TreeExplainer

np.set_printoptions(precision=5, suppress=True, linewidth=120)

# ----------------------------------------------------------------------------- knobs
SEED = 0
N_TRAIN = 500
D = 12
N_TREES = 50
MAX_DEPTH = 6
TREE_SOLVER = "product_games"   # or "quadrature_tree" (the sequential per-tree evaluator, with the optional C++ / CUDA backends)
TEST_INDEX = 0

# ----------------------------------------------------------------------------- data + model
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N_TRAIN + 10, D))
y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + np.where(X[:, 3] > 0, 1.0, -1.0) + 0.1 * rng.standard_normal(len(X))
X_train, y_train, X_test = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]

t0 = time.perf_counter()
model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=SEED).fit(X_train, y_train)
print(f"trained RandomForestRegressor({N_TREES} trees, depth {MAX_DEPTH}) on {N_TRAIN} x {D} in {time.perf_counter()-t0:.2f}s")

x = X_test[TEST_INDEX]

# ----------------------------------------------------------------------------- explainer
t0 = time.perf_counter()
explainer = TreeExplainer(model, tree_solver=TREE_SOLVER)
print(f"TreeExplainer prepared in {time.perf_counter()-t0:.2f}s; expected value (mean leaf value) = {np.ravel(explainer.expected_value)[0]:.6f}")

# ----------------------------------------------------------------------------- 1. explain one instance
t0 = time.perf_counter(); phi = np.ravel(explainer.shap_values(x[None, :])[0]); t_one = time.perf_counter() - t0
prediction = model.predict(x[None, :])[0]
print("\n[1] path-dependent TreeSHAP values")
print(f"    time {t_one*1e3:.1f} ms")
print(f"    efficiency: expected_value + sum phi = {np.ravel(explainer.expected_value)[0] + phi.sum():.6f},  prediction = {prediction:.6f}")
print(f"    top-5 features: {np.argsort(-np.abs(phi))[:5]}   phi = {phi[np.argsort(-np.abs(phi))[:5]]}")

# ----------------------------------------------------------------------------- 2. a batch
t0 = time.perf_counter(); Phi = explainer.shap_values(X_test); t_batch = time.perf_counter() - t0
Phi = Phi.reshape(len(X_test), D) if Phi.ndim == 3 else Phi
print(f"\n[2] batch of {len(X_test)} instances in {t_batch*1e3:.1f} ms; max efficiency residual "
      f"{np.abs(np.ravel(explainer.expected_value)[0] + Phi.sum(axis=1) - model.predict(X_test)).max():.2e}")

# ----------------------------------------------------------------------------- 3. compare with the shap package if available
print("\n[3] comparison with shap.TreeExplainer")
try:
    import shap
    ref = shap.TreeExplainer(model).shap_values(X_test)
    ref = np.asarray(ref).reshape(len(X_test), D)
    print(f"    max |QuadraSHAP - shap| = {np.abs(Phi - ref).max():.2e}")
except ImportError:
    print("    shap is not installed; skipped (pip install shap)")

print("\ndone.")
