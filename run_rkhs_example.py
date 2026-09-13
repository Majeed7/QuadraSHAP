"""
RKHSExplainer: QuadraSHAP on a product-kernel (RKHS) model, start to finish, for ONE instance.

Flat script, no functions: set the knobs at the top, run it, step through it in a
debugger.  It trains an RBF kernel ridge regressor, picks the number of
Gauss--Legendre nodes a priori from the tolerance eps, explains one test point
under the three value functions, and checks everything against the exact
quadrature and the efficiency identity.

    python run_rkhs_example.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run from a checkout without installing
from quadrashap import RKHSExplainer

np.set_printoptions(precision=5, suppress=True, linewidth=120)

# ----------------------------------------------------------------------------- knobs
SEED = 0
N_TRAIN = 300          # number of training points = number of product-game components p
D = 200                # number of features
GAMMA = 1.0 / D        # RBF parameter k_j(a, b) = exp(-gamma (a - b)^2);  try 10/D or 100/D for narrower kernels
RIDGE = 0.1            # KernelRidge regularisation
EPS = 1e-3             # tolerance on every attribution -> number of nodes chosen a priori
N_BACKGROUND = 20      # background rows for the empirical interventional value function
BACKEND = "logspace_numpy"   # or "logspace_jax", "prefix_scan_numpy", "prefix_scan_jax", "auto"
BLOCK_SIZE = "auto"    # blockwise evaluation of the component-background pairs: "auto" (block only when the
                       # full computation exceeds the memory budget / cache-size cap), an int (pairs per block), or -1 (off)
MEMORY_BUDGET = None   # e.g. "512MB"; None = half of the memory currently available on this machine
TEST_INDEX = 0         # which held-out point to explain

# ----------------------------------------------------------------------------- data + model
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N_TRAIN + 10, D))
y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + 0.1 * rng.standard_normal(len(X))
X = StandardScaler().fit_transform(X)
X_train, y_train, X_test = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]

t0 = time.perf_counter()
model = KernelRidge(kernel="rbf", gamma=GAMMA, alpha=RIDGE).fit(X_train, y_train)
print(f"trained KernelRidge(rbf, gamma={GAMMA:.3g}, alpha={RIDGE}) on {N_TRAIN} x {D} in {time.perf_counter()-t0:.2f}s")

x = X_test[TEST_INDEX]                 # the instance to explain
x_baseline = X_train.mean(axis=0)      # reference point for the baseline value function
background = X_train[:N_BACKGROUND]    # background dataset for the interventional value function

# ----------------------------------------------------------------------------- explainer
explainer = RKHSExplainer(model, background=background, eps=EPS, backend=BACKEND,
                       block_size=BLOCK_SIZE, memory_budget=MEMORY_BUDGET)
print(f"p = {explainer.n} components, d = {explainer.d} features, gamma = {explainer.gamma:.3g}, "
      f"||alpha||_1 = {np.abs(explainer.alpha).sum():.3g}, exactness threshold = {(D + 1) // 2} nodes")

# the factor table for this instance: u_j^{(r)} = k_j(x_j, x_j^{(r)}), shape (p, d)
U = explainer.factors(x)
print(f"present factors u: shape {U.shape}, min {U.min():.3g}, mean {U.mean():.3g}")

# ----------------------------------------------------------------------------- 1. node budget
report = explainer.node_budget(x, eps=EPS, value_function="neutral")
print("\n[1] node budget (neutral factor)")
print("   ", report)
print(f"    -> {report.m_q} nodes certify max_i |phi_hat_i - phi_i| <= {report.bound:.2e} (eps = {EPS})")

# ----------------------------------------------------------------------------- 2. explain with the budget
print("\n[2] neutral-factor value function  v(S) = sum_r alpha_r prod_{j in S} k_j")
t0 = time.perf_counter(); phi_neutral, rep = explainer.explain(x, "neutral", return_report=True); t_budget = time.perf_counter() - t0
plan = explainer.last_block_plan               # how the pairs / nodes were blocked for this call
t0 = time.perf_counter(); phi_neutral_exact = explainer.explain(x, "neutral", m_q="exact"); t_exact = time.perf_counter() - t0
f_x = model.predict(x[None, :])[0]
print(f"    m_q used = {rep.m_q}   time {t_budget*1e3:.1f} ms   (exact rule: {rep.exact_threshold} nodes, {t_exact*1e3:.1f} ms)")
print(f"    block plan: {plan}")
print(f"    max |phi - phi_exact| = {np.abs(phi_neutral - phi_neutral_exact).max():.2e}   certified <= {rep.bound:.2e}")
print(f"    efficiency: sum phi = {phi_neutral.sum():.6f},  f(x) - v(empty) = {f_x - explainer.value_function_at_empty('neutral'):.6f}")
print(f"    top-5 features: {np.argsort(-np.abs(phi_neutral))[:5]}   phi = {phi_neutral[np.argsort(-np.abs(phi_neutral))[:5]]}")

print("\n[3] baseline value function  v(S) = f(x_S, x^b_{-S})")
phi_base, rep = explainer.explain(x, "baseline", baseline=x_baseline, return_report=True)
phi_base_exact = explainer.explain(x, "baseline", baseline=x_baseline, m_q="exact")
print(f"    m_q used = {rep.m_q}   max |phi - phi_exact| = {np.abs(phi_base - phi_base_exact).max():.2e}   certified <= {rep.bound:.2e}")
print(f"    efficiency: sum phi = {phi_base.sum():.6f},  f(x) - f(x^b) = {f_x - model.predict(x_baseline[None, :])[0]:.6f}")

print(f"\n[4] empirical interventional value function  v(S) = mean_b f(x_S, xtilde^b_{{-S}})  over {N_BACKGROUND} background rows")
t0 = time.perf_counter(); phi_int, rep = explainer.explain(x, "interventional", return_report=True); t_int = time.perf_counter() - t0
plan = explainer.last_block_plan
phi_int_exact = explainer.explain(x, "interventional", m_q="exact")
print(f"    m_q used = {rep.m_q}   time {t_int*1e3:.1f} ms   max |phi - phi_exact| = {np.abs(phi_int - phi_int_exact).max():.2e}   certified <= {rep.bound:.2e}")
print(f"    block plan: {plan}")
print(f"    efficiency: sum phi = {phi_int.sum():.6f},  f(x) - mean_b f(xtilde^b) = {f_x - model.predict(background).mean():.6f}")
# the interventional attributions are the average of the baseline attributions over the background rows
phi_avg_baselines = np.mean([explainer.explain(x, "baseline", baseline=b, m_q="exact") for b in background], axis=0)
print(f"    interventional == mean of baselines: max diff {np.abs(phi_avg_baselines - phi_int_exact).max():.2e}")

# ----------------------------------------------------------------------------- 5. how the error decays with the number of nodes
print("\n[5] error vs. number of nodes (neutral factor), certified bound alongside")
from quadrashap.product_games.budget import certify
summary = explainer.summarize(x, "neutral")
for m_q in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    if m_q >= rep.exact_threshold:
        break
    err = np.abs(explainer.explain(x, "neutral", m_q=m_q) - phi_neutral_exact).max()
    print(f"    m_q = {m_q:3d}   observed error {err:9.2e}   certified bound {certify(summary, m_q):9.2e}")

print("\ndone.")
