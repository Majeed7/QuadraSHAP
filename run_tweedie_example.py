"""
TweedieExplainer: QuadraSHAP for Tweedie regression with a log link, start to finish, for ONE instance.

Flat script, no functions: set the knobs at the top, run it, step through it in a
debugger.  It fits a scikit-learn TweedieRegressor (power 1.5, log link) to synthetic non-negative data, chooses the number of Gauss--Legendre nodes a priori
from the tolerance eps, explains one point under the baseline and the empirical
interventional value functions, and verifies the result against the exact quadrature,
against exhaustive coalition enumeration through the model itself, and against the
efficiency identity.  Attributions are on the expected-response (mean) scale.

    python run_tweedie_example.py
"""
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import TweedieRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run from a checkout without installing
from quadrashap import TweedieExplainer

np.set_printoptions(precision=5, suppress=True, linewidth=120)

# ----------------------------------------------------------------------------- knobs
SEED = 0
N_TRAIN = 400
D = 10                 # keep small: the brute-force check below enumerates 2**D coalitions
EPS = 1e-3             # tolerance on every attribution -> number of nodes chosen a priori
N_BACKGROUND = 20      # background rows for the empirical interventional value function
BACKEND = "logspace_numpy"   # or "logspace_jax", "prefix_scan_numpy", "prefix_scan_jax", "auto"
TEST_INDEX = 0
POWER = 1.5             # Tweedie power in (1, 2): compound Poisson-Gamma
RIDGE = 1e-3

# ----------------------------------------------------------------------------- data + model
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N_TRAIN + 10, D)) * 0.7
beta_true = np.zeros(D); beta_true[:3] = [0.8, -0.5, 0.3]
mu = np.exp(0.2 + X @ beta_true)
y = np.where(rng.random(len(X)) < 0.2, 0.0, rng.gamma(2.0, mu / 2.0))   # zeros plus positive mass
X_train, y_train, X_test = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]
t0 = time.perf_counter()
model = TweedieRegressor(power=POWER, link="log", alpha=RIDGE).fit(X_train, y_train)
print(f"trained TweedieRegressor on {N_TRAIN} x {D} in {time.perf_counter()-t0:.2f}s")

# F(x): the multiplicative-scale quantity that is attributed (expected-response (mean))
F = model.predict                       # mu(x) = exp(b0 + beta.x)
x = X_test[TEST_INDEX]                 # the instance to explain
x_baseline = X_train.mean(axis=0)      # reference point for the baseline value function
background = X_train[:N_BACKGROUND]    # background dataset for the interventional value function

# ----------------------------------------------------------------------------- explainer
explainer = TweedieExplainer(model, background=background, eps=EPS, backend=BACKEND)
print(f"adapter: {explainer.model.describe()}   (p = {explainer.n} component(s), d = {explainer.d}, exactness threshold = {(D + 1) // 2} nodes)")
print(f"present factors u_j(x_j) for this instance: {explainer.factors(x)}")
print(f"coefficients beta = {explainer.beta}, intercept b0 = {explainer.log_intercept:.4f}")

# ----------------------------------------------------------------------------- 1. node budget
report = explainer.node_budget(x, eps=EPS, value_function="interventional")
print("\n[1] node budget (interventional)")
print("   ", report)

# ----------------------------------------------------------------------------- 2. interventional value function with the budget
print(f"\n[2] empirical interventional value function  v(S) = mean_b F(x_S, xtilde^b_{{-S}})  over {N_BACKGROUND} background rows")
t0 = time.perf_counter(); phi_int, rep = explainer.explain(x, return_report=True); t_int = time.perf_counter() - t0   # default = interventional
phi_int_exact = explainer.explain(x, m_q="exact")
print(f"    m_q used = {rep.m_q}   time {t_int*1e3:.1f} ms   max |phi - phi_exact| = {np.abs(phi_int - phi_int_exact).max():.2e}   certified <= {rep.bound:.2e}   efficiency residual {rep.efficiency_residual:.2e}")
print(f"    efficiency: sum phi = {phi_int.sum():.6f},  F(x) - mean_b F(xtilde^b) = {F(x[None, :])[0] - F(background).mean():.6f}")
print(f"    top-5 features: {np.argsort(-np.abs(phi_int))[:5]}   phi = {phi_int[np.argsort(-np.abs(phi_int))[:5]]}")

# ----------------------------------------------------------------------------- 3. baseline value function
print("\n[3] baseline value function  v(S) = F(x_S, x^b_{-S})  with x^b = training mean")
phi_base, rep_b = explainer.explain(x, "baseline", baseline=x_baseline, return_report=True)
phi_base_exact = explainer.explain(x, "baseline", baseline=x_baseline, m_q="exact")
print(f"    m_q used = {rep_b.m_q}   max |phi - phi_exact| = {np.abs(phi_base - phi_base_exact).max():.2e}   certified <= {rep_b.bound:.2e}")
print(f"    efficiency: sum phi = {phi_base.sum():.6f},  F(x) - F(x^b) = {F(x[None, :])[0] - F(x_baseline[None, :])[0]:.6f}")

# ----------------------------------------------------------------------------- 4. brute force through the model: all 2**D coalitions
print(f"\n[4] exhaustive check: v(S) = mean_b F(x_S, xtilde^b_{{-S}}) evaluated by calling the model on spliced points ({2**D:,} coalitions)")
fact = [math.factorial(k) for k in range(D + 1)]
value = {}
for k in range(D + 1):
    for S in itertools.combinations(range(D), k):
        mask = np.zeros(D, dtype=bool); mask[list(S)] = True
        value[S] = F(np.where(mask[None, :], x[None, :], background)).mean()
phi_brute = np.zeros(D)
for i in range(D):
    for k in range(D):
        for S in itertools.combinations([j for j in range(D) if j != i], k):
            phi_brute[i] += fact[k] * fact[D - k - 1] / fact[D] * (value[tuple(sorted(S + (i,)))] - value[S])
print(f"    max |QuadraSHAP (exact rule) - brute force| = {np.abs(phi_int_exact - phi_brute).max():.2e}")
print(f"    max |QuadraSHAP (budget)     - brute force| = {np.abs(phi_int - phi_brute).max():.2e}   (eps = {EPS})")

# ----------------------------------------------------------------------------- 5. error vs. number of nodes
print("\n[5] error vs. number of nodes (interventional), certified bound alongside")
from quadrashap.product_games.budget import certify
summary = explainer.summarize(x, "interventional")
for m_q in range(1, rep.exact_threshold):
    err = np.abs(explainer.explain(x, m_q=m_q) - phi_int_exact).max()
    print(f"    m_q = {m_q:3d}   observed error {err:9.2e}   certified bound {certify(summary, m_q):9.2e}")

print("\ndone.")
