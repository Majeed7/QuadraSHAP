"""
CoxExplainer: QuadraSHAP for a Cox proportional-hazards model on the hazard-ratio scale, start to finish, for ONE instance.

Flat script, no functions: set the knobs at the top, run it, step through it in a
debugger.  It fits a Cox model by maximising the Breslow partial likelihood with a few Newton steps written out in NumPy (lifelines / scikit-survival are not required; a fitted lifelines CoxPHFitter can be passed directly instead), chooses the number of Gauss--Legendre nodes a priori
from the tolerance eps, explains one point under the baseline and the empirical
interventional value functions, and verifies the result against the exact quadrature,
against exhaustive coalition enumeration through the model itself, and against the
efficiency identity.  Attributions are on the hazard-ratio scale.

    python run_cox_example.py
"""
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run from a checkout without installing
from quadrashap import CoxExplainer

np.set_printoptions(precision=5, suppress=True, linewidth=120)

# ----------------------------------------------------------------------------- knobs
SEED = 0
N_TRAIN = 300
D = 8                 # keep small: the brute-force check below enumerates 2**D coalitions
EPS = 1e-3             # tolerance on every attribution -> number of nodes chosen a priori
N_BACKGROUND = 20      # background rows for the empirical interventional value function
BACKEND = "logspace_numpy"   # or "logspace_jax", "prefix_scan_numpy", "prefix_scan_jax", "auto"
TEST_INDEX = 0

# ----------------------------------------------------------------------------- data + model
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N_TRAIN + 10, D)) * 0.5
beta_true = np.zeros(D); beta_true[:4] = [0.8, -0.5, 0.4, 0.3]
T = rng.exponential(1.0 / np.exp(X @ beta_true))       # event times, hazard h0 exp(beta.x) with h0 = 1
E = rng.random(len(X)) < 0.8                            # 20 % censoring
X_train, T_train, E_train, X_test = X[:N_TRAIN], T[:N_TRAIN], E[:N_TRAIN], X[N_TRAIN:]
t0 = time.perf_counter()
order = np.argsort(-T_train); Xs, Es = X_train[order], E_train[order]   # sorted by decreasing time
beta = np.zeros(D)
for _ in range(20):                                     # Newton steps on the Breslow partial log-likelihood
    r = np.exp(Xs @ beta); S0 = np.cumsum(r); S1 = np.cumsum(r[:, None] * Xs, axis=0)
    S2 = np.cumsum(r[:, None, None] * Xs[:, :, None] * Xs[:, None, :], axis=0)
    grad = ((Xs - S1 / S0[:, None]) * Es[:, None]).sum(0)
    hess = -((S2 / S0[:, None, None] - (S1[:, :, None] * S1[:, None, :]) / S0[:, None, None] ** 2) * Es[:, None, None]).sum(0)
    beta -= np.linalg.solve(hess - 1e-8 * np.eye(D), grad)
reference = X_train.mean(axis=0)                        # hazard ratio relative to the mean covariates (lifelines convention)
print(f"trained Cox PH (Newton on the Breslow partial likelihood) on {N_TRAIN} x {D} in {time.perf_counter()-t0:.2f}s")

# F(x): the multiplicative-scale quantity that is attributed (hazard-ratio)
F = lambda P: np.exp((np.atleast_2d(P) - reference) @ beta)      # partial hazard exp(beta.(x - xbar))
x = X_test[TEST_INDEX]                 # the instance to explain
x_baseline = X_train.mean(axis=0)      # reference point for the baseline value function
background = X_train[:N_BACKGROUND]    # background dataset for the interventional value function

# ----------------------------------------------------------------------------- explainer
explainer = CoxExplainer(beta, reference=reference, background=background, eps=EPS, backend=BACKEND)
print(f"adapter: {explainer.model.describe()}   (p = {explainer.n} component(s), d = {explainer.d}, exactness threshold = {(D + 1) // 2} nodes)")
print(f"present factors u_j(x_j) for this instance: {explainer.factors(x)}")
print(f"log hazard ratios beta = {beta}   (true: {beta_true})")
print(f"hazard ratio of x relative to the mean covariates: {F(x[None, :])[0]:.4f}")

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
