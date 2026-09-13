#!/usr/bin/env python3
"""A priori node budget on a trained RBF kernel ridge model.

For each test instance and tolerance ``eps`` the script (i) selects the number of
Gauss--Legendre nodes a priori from the factor tables (``QuadraSHAP.node_budget``),
(ii) runs the quadrature with that budget and with the exactness threshold
``ceil(d/2)``, and (iii) reports the certified bound, the observed error, the
smallest budget that empirically meets ``eps``, and the timings.  A bandwidth
sweep repeats this for fixed ``gamma`` values, since the tuned bandwidth of a
standardised high-dimensional problem is wide and needs only a handful of nodes.

Usage:  python benchmarks/kernel_node_budget_bench.py [--d 1000] [--n-train 500]
        [--value-function neutral|interventional] [--n-background 20]
Outputs: benchmarks/results/kernel_node_budget/{tuned,sweep}.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quadrashap import QuadraSHAP  # noqa: E402

OUT = ROOT / "benchmarks" / "results" / "kernel_node_budget"
FIELDS = ["setting", "gamma_times_d", "cv_rmse", "alpha_l1", "test_idx", "eps", "lambda_max", "A_max", "eta",
          "m_certified", "certified_bound", "observed_err", "holds", "m_empirical", "t_budget_ms",
          "t_quad_certified_s", "t_quad_exact_s", "exact_threshold"]


def make_data(d: int, n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train + n_test, d))
    beta = np.zeros(d); beta[:50] = rng.standard_normal(min(50, d))
    y = X @ beta + 2.0 * np.sin(X[:, 0] * X[:, 1]) + 0.5 * rng.standard_normal(len(X))
    X = StandardScaler().fit_transform(X)
    return X[:n_train], y[:n_train], X[n_train:]


def run_case(setting, krr, cv_rmse, Xtr, Xte, args, writer, backend):
    d = Xtr.shape[1]
    kw = {}
    if args.value_function == "interventional":
        kw["background"] = Xtr[: args.n_background]
    ex = QuadraSHAP(krr, backend=backend, **kw)
    for k, x in enumerate(Xte):
        t0 = time.perf_counter(); phi_exact = ex.explain(x, args.value_function, m_q="exact"); t_exact = time.perf_counter() - t0
        for eps in args.eps:
            rep = ex.node_budget(x, eps, args.value_function)
            t0 = time.perf_counter(); phi = ex.explain(x, args.value_function, m_q=rep.m_q); t_quad = time.perf_counter() - t0
            err = float(np.abs(phi - phi_exact).max())
            m_emp = next(m for m in range(1, rep.m_q + 1)
                         if np.abs(ex.explain(x, args.value_function, m_q=m) - phi_exact).max() <= eps)
            row = dict(setting=setting, gamma_times_d=krr.gamma * d, cv_rmse=cv_rmse, alpha_l1=np.abs(ex.alpha).sum(),
                       test_idx=k, eps=eps, lambda_max=rep.lambda_max, A_max=rep.scale, eta=rep.eta,
                       m_certified=rep.m_q, certified_bound=rep.bound, observed_err=err, holds=err <= eps,
                       m_empirical=m_emp, t_budget_ms=rep.seconds * 1e3, t_quad_certified_s=t_quad,
                       t_quad_exact_s=t_exact, exact_threshold=rep.exact_threshold)
            writer.writerow(row)
            print(f"{setting:>8s} g*d={krr.gamma*d:6.2f} pt={k} eps={eps:6.0e} Lam={rep.lambda_max:8.1f} A={rep.scale:8.3g} "
                  f"m*={rep.m_q:3d} bound={rep.bound:8.2e} err={err:8.2e} {'ok' if err<=eps else 'VIOLATION'} "
                  f"m_emp={m_emp:3d} t_budget={rep.seconds*1e3:6.2f}ms t_quad={t_quad:.3f}s t_exact={t_exact:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--value-function", default="neutral", choices=["neutral", "interventional"])
    ap.add_argument("--n-background", type=int, default=20)
    ap.add_argument("--backend", default="logspace_numpy")
    ap.add_argument("--sweep", type=float, nargs="*", default=[1.0, 3.0, 10.0, 30.0, 100.0],
                    help="fixed gamma*d values for the bandwidth sweep (empty to skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte = make_data(args.d, args.n_train, args.n_test, args.seed)
    d = args.d

    with open(OUT / f"tuned_{args.value_function}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        grid = {"alpha": [1e-2, 1e-1, 1.0], "gamma": [g / d for g in (0.3, 1.0, 3.0, 10.0)]}
        cv = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=5, scoring="neg_mean_squared_error").fit(Xtr, ytr)
        print(f"CV-tuned KRR: alpha={cv.best_estimator_.alpha}, gamma*d={cv.best_estimator_.gamma*d:.2f}, "
              f"RMSE={np.sqrt(-cv.best_score_):.3f}")
        run_case("tuned", cv.best_estimator_, float(np.sqrt(-cv.best_score_)), Xtr, Xte, args, w, args.backend)

    if args.sweep:
        with open(OUT / f"sweep_{args.value_function}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
            for gd in args.sweep:
                best = max(((cross_val_score(KernelRidge(kernel="rbf", alpha=a, gamma=gd / d), Xtr, ytr, cv=5,
                                             scoring="neg_mean_squared_error").mean(), a) for a in (1e-3, 1e-2, 1e-1, 1.0)))
                krr = KernelRidge(kernel="rbf", alpha=best[1], gamma=gd / d).fit(Xtr, ytr)
                run_case("sweep", krr, float(np.sqrt(-best[0])), Xtr, Xte[:1], args, w, args.backend)
    print("written to", OUT)


if __name__ == "__main__":
    main()
