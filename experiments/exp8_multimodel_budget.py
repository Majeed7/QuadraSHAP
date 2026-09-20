"""Reproducible multi-model quadrature study; see the adjacent protocol/report.

All tested node counts and attribution vectors are retained. No test instance is
filtered by its error, model prediction, certificate, or required node count.
The certificate concerns quadrature in exact arithmetic, not floating point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import expit, roots_legendre
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import r2_score, roc_auc_score, mean_poisson_deviance
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits

REPO = Path(os.environ.get("QUADRASHAP_REPO", "/Users/Majid/surfdrive/Research/ExplainableAI/QuadraSHAP"))
if (Path(__file__).resolve().parents[1] / "src/quadrashap").is_dir():
    REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quadrashap.multiplicative.glm import LogLinkGLM
from quadrashap.multiplicative.odds import LogisticOdds, NaiveBayesOdds
from quadrashap.multiplicative.rkhs import ProductKernelModel
from quadrashap.product_games.budget import GameSummary, budget_from_summary, certify
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy

FAMILIES = ("rbf", "poisson", "logistic", "naive_bayes")
TOLERANCES = (1e-3, 1e-5, 1e-6, 1e-9)
EVALUATOR = ProductGamesShapleyNumpy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(None)
def scipy_rule(m):
    t, w = roots_legendre(m)
    return (t + 1) / 2, w / 2


def independent_gl(K, Ut, w, m):
    """Independent SciPy nodes + positive log-product evaluation, blockwise.

Unlike the package, uses a shared log product and division (all factors in
this experiment are strictly positive). No package quadrature code is used.
"""
    t, q = scipy_rule(m)
    total = np.zeros(K.shape[1])
    for start in range(0, m, 8):
        factors = Ut[None] + t[start:start + 8, None, None] * K[None]
        if np.any(factors <= 0):
            raise ValueError("This independent evaluator requires positive factors")
        product = np.exp(np.log(factors).sum(axis=-1))
        total += np.einsum("q,qr,qri,ri,r->i", q[start:start + 8], product,
                           1 / factors, K, w, optimize=True)
    return total


def bernstein_reference(K, Ut, w):
    """Independent coefficient-space integral via positive Bernstein recursion.

The integral of a degree-n polynomial in Bernstein form is mean(b_0,...,b_n).
Reverse differentiation w.r.t. BOTH endpoints of each linear factor returns
integral(prod_{j!=i} T_j). This is O(p*d^2), with no quadrature nodes, no
monomial coefficients, and no subtractive polynomial division. Used for p<=8.
"""
    p, d = K.shape
    U = Ut + K
    history = [np.ones((p, 1))]
    for n in range(1, d + 1):
        b = history[-1]
        z = np.zeros((p, n + 1))
        k = np.arange(n + 1) / n
        z[:, :-1] += b * Ut[:, n - 1, None] * (1 - k[:-1])
        z[:, 1:] += b * U[:, n - 1, None] * k[1:]
        history.append(z)
    adj = np.full((p, d + 1), 1 / (d + 1))
    integrals = np.empty((p, d))
    for n in range(d, 0, -1):
        k = np.arange(n + 1) / n
        left = adj[:, :-1] * (1 - k[:-1])
        right = adj[:, 1:] * k[1:]
        integrals[:, n - 1] = ((left + right) * history[n - 1]).sum(axis=1)
        adj = left * Ut[:, n - 1, None] + right * U[:, n - 1, None]
    return (w[:, None] * K * integrals).sum(axis=0)


def fit_case(family, d, seed, regime, n_eval, n_train, n_rbf):
    """Train each model without consulting quadrature results.

Fixed-feature signal: beta_j=+/-0.20 on 30% of coordinates. Fixed-total
control: beta_j=+/-0.20*sqrt(100/d). NB has class means +/-beta/2.
RBF always uses gamma=1/d (easy, stable-bandwidth negative control).
"""
    rng = np.random.default_rng(np.random.SeedSequence([20260914, seed, d]))
    n = n_rbf if family == "rbf" else n_train
    n_test = max(500, n_eval)
    X = rng.normal(size=(n + n_test, d))
    informative = rng.permutation(d)[:round(.3 * d)]
    beta = np.zeros(d)
    strength = .2 if regime == "fixed_feature" else .2 * np.sqrt(100 / d)
    beta[informative] = strength * rng.choice([-1., 1.], len(informative))
    eta = X @ beta
    fit_start = time.perf_counter()
    caught = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if family == "rbf":
            # Unit-scale regression response, fixed-bandwidth per total dimension.
            mean = eta / np.linalg.norm(beta)
            y = mean + .2 * rng.normal(size=len(X))
            estimator = KernelRidge(kernel="rbf", gamma=1 / d, alpha=.1).fit(X[:n], y[:n])
            adapter = ProductKernelModel(estimator)
            prediction = estimator.predict(X[n:])
            metrics = dict(metric="R2", score=float(r2_score(y[n:], prediction)))
        elif family == "poisson":
            mean = np.exp(eta)
            y = rng.poisson(mean)
            estimator = PoissonRegressor(alpha=.1, solver="newton-cholesky", max_iter=100, tol=1e-8).fit(X[:n], y[:n])
            adapter = LogLinkGLM.from_sklearn(estimator)
            prediction = estimator.predict(X[n:])
            metrics = dict(metric="Poisson_deviance", score=float(mean_poisson_deviance(y[n:], prediction)),
                           null_deviance=float(mean_poisson_deviance(y[n:], np.full(len(y[n:]), y[:n].mean()))),
                           oracle_deviance=float(mean_poisson_deviance(y[n:], mean[n:])))
        elif family == "logistic":
            y = rng.binomial(1, expit(eta))
            estimator = LogisticRegression(C=.1, max_iter=1000, tol=1e-8).fit(X[:n], y[:n])
            adapter = LogisticOdds.from_sklearn(estimator)
            prediction = estimator.predict_proba(X[n:])[:, 1]
            metrics = dict(metric="AUROC", score=float(roc_auc_score(y[n:], prediction)))
        elif family == "naive_bayes":
            y = rng.integers(0, 2, len(X))
            X += (2 * y[:, None] - 1) * beta / 2
            estimator = GaussianNB().fit(X[:n], y[:n])
            adapter = NaiveBayesOdds(estimator)
            prediction = estimator.predict_proba(X[n:])[:, 1]
            metrics = dict(metric="AUROC", score=float(roc_auc_score(y[n:], prediction)))
        else:
            raise ValueError(family)
    metrics.update(n_train=n, n_test=n_test, n_informative=len(informative),
                   signal_strength=float(strength), fit_seconds=time.perf_counter() - fit_start,
                   convergence_warnings=[str(c.message) for c in caught],
                   params=estimator.get_params(),
                   n_iter=np.asarray(getattr(estimator, "n_iter_", 0)).tolist())
    # Zero is the population mean for all designs (balanced mixture for NB).
    # This shared baseline avoids mixing neutral RBF and baseline GLM games.
    extras = dict(beta_true=beta, informative=informative, baseline=np.zeros(d),
                  background=X[:8], X_eval=X[n:n+n_eval], y_eval=y[n:n+n_eval],
                  test_predictions=prediction, test_y=y[n:])
    if family == "rbf":
        extras.update(X_train=X[:n], y_train=y[:n], dual_coef=estimator.dual_coef_)
    elif family in ("poisson", "logistic"):
        extras.update(beta_fit=np.asarray(estimator.coef_), intercept=np.asarray(estimator.intercept_))
    else:
        extras.update(theta=estimator.theta_, var=estimator.var_, prior=estimator.class_prior_)
    return adapter, extras, metrics


def evaluate_instance(adapter, x, refs, vf, save_path, identifiers):
    U = adapter.factors(x)
    Us = [adapter.factors(r) for r in refs]
    Ut = np.concatenate(Us)
    K = np.tile(U, (len(refs), 1)) - Ut
    w = np.tile(adapter.coef, len(refs)) / len(refs)
    d = len(x)
    exact = (d + 1) // 2
    summary = GameSummary(d=d).update(K, Ut, w)
    budgets = {eps: budget_from_summary(summary, eps).m_q for eps in TOLERANCES}
    started = time.perf_counter()
    # Independent of NumPy's leggauss nodes used by the actual package.
    reference = (bernstein_reference(K, Ut, w) if len(w) <= 8
                 else independent_gl(K, Ut, w, exact))
    cross_reference = independent_gl(K, Ut, w, exact + 3)
    ref_disagreement = float(np.max(np.abs(reference - cross_reference)))
    endpoint_delta = float(w @ (np.prod(Ut + K, axis=1) - np.prod(Ut, axis=1)))
    reference_efficiency = abs(float(reference.sum()) - endpoint_delta)
    reference_seconds = time.perf_counter() - started
    # Pre-specified full integer sweep through the tightest certified budget + 5.
    # Keep the exact rule as a separate floating-point diagnostic, not an error=0 anchor.
    end = min(exact, max(budgets.values()) + 5)
    ms = np.unique(np.r_[np.arange(1, end + 1), exact]).astype(int)
    phis, errors, bounds, seconds = [], [], [], []
    for m in ms:
        started = time.perf_counter()
        phi = (w[:, None] * EVALUATOR.phi_matrix_prefix_scan(K, int(m), Ut=Ut, node_block=8)).sum(axis=0)
        seconds.append(time.perf_counter() - started)
        phis.append(phi)
        errors.append(float(np.max(np.abs(phi - reference))))
        bounds.append(float(certify(summary, int(m))))
    phis, errors, bounds = np.asarray(phis), np.asarray(errors), np.asarray(bounds)
    # Numerical-resolution flag, not a replacement bound or license to declare coverage.
    roundoff_diagnostic = max(ref_disagreement, reference_efficiency / d)
    np.savez_compressed(save_path, K=K, Ut=Ut, w=w, x=x, refs=refs, ms=ms,
                        phi=phis, phi_reference=reference, phi_cross_reference=cross_reference,
                        observed_error=errors, certified_bound=bounds, seconds=seconds,
                        A=summary.A, lambda_max=summary.lambda_max)
    rows = []
    for eps, m_cert in budgets.items():
        at = int(np.flatnonzero(ms == m_cert)[0])
        passing = ms[errors <= eps]
        # Hindsight minimum over the tested integer prefix; no monotonicity assumption.
        m_observed = int(passing[0]) if len(passing) else None
        resolved = roundoff_diagnostic <= eps / 100
        rows.append(dict(**identifiers, value_function=vf, epsilon=eps,
                         m_certified=m_cert, m_observed=m_observed,
                         error_at_certified=float(errors[at]), bound_at_certified=float(bounds[at]),
                         target_met=bool(errors[at] <= eps), reference_resolved=bool(resolved),
                         exact_threshold=exact, lambda_max=summary.lambda_max, A_max=summary.A_max,
                         u_max=float(U.max()), u_gt_one_fraction=float(np.mean(U > 1)),
                         attribution_max=float(np.max(np.abs(reference))),
                         reference_disagreement=ref_disagreement, reference_efficiency=reference_efficiency,
                         roundoff_diagnostic=roundoff_diagnostic, reference_seconds=reference_seconds,
                         exact_rule_error=float(errors[-1]), npz=str(Path(save_path).name)))
    return rows


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = dict(dimensions=args.dimensions, seeds=args.seeds, families=args.families,
                  regimes=args.regimes, instances=args.instances, n_train=args.n_train,
                  n_rbf=args.n_rbf, tolerances=TOLERANCES, informative_fraction=.30,
                  background_rows=8, supplementary_interventional=args.interventional,
                  numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                  python=sys.version, platform=platform.platform(), threads=1,
                  reference="Bernstein reverse differentiation for <=8 games; independent SciPy GL otherwise",
                  repo=str(REPO), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    config["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    config["source_hashes"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (REPO / "src/quadrashap").rglob("*.py")}
    existing = out / "config.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        for key in ("dimensions", "seeds", "families", "regimes", "instances", "n_train", "n_rbf", "source_sha256"):
            if old[key] != config[key]:
                raise ValueError(f"Refusing incompatible resume: {key}")
    else:
        write_json(existing, config)
    rows = []
    with threadpool_limits(limits=1):
        for regime in args.regimes:
            for family in args.families:
                if family == "rbf" and regime != "fixed_feature":
                    continue  # fixed-total response normalization already built into RBF control
                for d in args.dimensions:
                    for seed in args.seeds:
                        case = f"{regime}_{family}_d{d}_s{seed}"
                        folder = out / case
                        folder.mkdir(exist_ok=True)
                        records_path = folder / "records.json"
                        if records_path.exists():
                            rows.extend(json.loads(records_path.read_text()))
                            print(f"resume {case}", flush=True)
                            continue
                        adapter, extras, metrics = fit_case(family, d, seed, regime, args.instances, args.n_train, args.n_rbf)
                        np.savez_compressed(folder / "model_data.npz", **extras)
                        write_json(folder / "fit.json", metrics)
                        case_rows = []
                        for instance, x in enumerate(extras["X_eval"]):
                            identifiers = dict(family=family, regime=regime, d=d, seed=seed, instance=instance)
                            values = [("baseline", extras["baseline"][None])]
                            if args.interventional and family != "rbf" and d == max(args.dimensions):
                                values.append(("interventional", extras["background"]))
                            for vf, refs in values:
                                case_rows.extend(evaluate_instance(adapter, x, refs, vf,
                                                  folder / f"instance{instance:03d}_{vf}.npz", identifiers))
                        write_json(records_path, case_rows)
                        rows.extend(case_rows)
                        main = [r for r in case_rows if r["epsilon"] == 1e-6 and r["value_function"] == "baseline"]
                        observed = [r["m_observed"] for r in main if r["m_observed"] is not None]
                        print(f"{case}: obs={np.median(observed) if observed else None} cert={np.median([r['m_certified'] for r in main])}"
                              f" maxerr={max(r['error_at_certified'] for r in main):.2e}"
                              f" unresolved={sum(not r['reference_resolved'] for r in main)} score={metrics['score']:.3g}", flush=True)
                        write_csv(out / "records.csv", rows)
    write_json(out / "records.json", rows)
    write_csv(out / "records.csv", rows)
    print(f"Completed {len(rows)} tolerance records in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO / "experiments/results/exp8_multimodel_budget"))
    parser.add_argument("--dimensions", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--regimes", nargs="+", choices=["fixed_feature", "fixed_total"], default=["fixed_feature", "fixed_total"])
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-rbf", type=int, default=384)
    parser.add_argument("--interventional", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
