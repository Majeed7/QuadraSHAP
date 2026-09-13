"""
Explainer(model): one entry point that picks the right QuadraSHAP explainer for a fitted model.

Flat script: fits one model per supported family on the same synthetic features, hands each to
`quadrashap.Explainer`, prints which explainer was chosen, the multiplicative scale it attributes,
the a priori node budget and the attributions of one instance under the default value function.

    python run_explainer_example.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor, LogisticRegression
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run from a checkout without installing
from quadrashap import Explainer

np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ----------------------------------------------------------------------------- knobs
SEED = 0
N_TRAIN = 400
D = 10
N_BACKGROUND = 20
EPS = 1e-3

# ----------------------------------------------------------------------------- data
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N_TRAIN + 10, D)) * 0.7
eta = 0.3 + X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
y_real = eta + 0.1 * rng.standard_normal(len(X))
y_count = rng.poisson(np.exp(eta))
y_pos = rng.gamma(2.0, np.exp(eta) / 2.0)
y_bin = (rng.random(len(X)) < 1 / (1 + np.exp(-eta))).astype(int)
tr, te = slice(0, N_TRAIN), slice(N_TRAIN, None)
x = X[te][0]
background = X[tr][:N_BACKGROUND]

# ----------------------------------------------------------------------------- one model per family
models = {
    "KernelRidge (RBF)":  KernelRidge(kernel="rbf", gamma=1.0 / D, alpha=0.1).fit(X[tr], y_real[tr]),
    "PoissonRegressor":   PoissonRegressor(alpha=1e-3).fit(X[tr], y_count[tr]),
    "GammaRegressor":     GammaRegressor(alpha=1e-3).fit(X[tr], y_pos[tr]),
    "TweedieRegressor":   TweedieRegressor(power=1.5, link="log", alpha=1e-3).fit(X[tr], y_pos[tr]),
    "LogisticRegression": LogisticRegression().fit(X[tr], y_bin[tr]),
    "GaussianNB":         GaussianNB().fit(X[tr], y_bin[tr]),
    "RandomForest":       RandomForestRegressor(n_estimators=20, max_depth=4, random_state=SEED).fit(X[tr], y_real[tr]),
}

for name, model in models.items():
    print(f"\n=== {name}")
    if name == "RandomForest":
        explainer = Explainer(model)                       # -> TreeExplainer (path-dependent value function)
        phi = np.ravel(explainer.shap_values(x[None, :])[0])
        print(f"    {type(explainer).__name__}: expected_value + sum phi = {np.ravel(explainer.expected_value)[0] + phi.sum():.5f}, "
              f"prediction = {model.predict(x[None, :])[0]:.5f}")
    else:
        explainer = Explainer(model, background=background, eps=EPS)   # -> RKHS / Poisson / ... explainer
        phi, report = explainer.explain(x, return_report=True)          # default value function of the family
        print(f"    {type(explainer).__name__}  attributes: {explainer.scale}")
        print(f"    default value function: {explainer.model.default_value_function};  {report.m_q} nodes certified for eps={EPS} "
              f"(exact at {report.exact_threshold}), certified error {report.bound:.1e}, efficiency residual {report.efficiency_residual:.1e}")
        print(f"    sum phi = {phi.sum():.5f} = F(x) - v(empty) = {explainer.model.predict_scale(x[None, :])[0] - explainer.expected_value:.5f}")
    print(f"    phi = {phi}")

print("\ndone.")
