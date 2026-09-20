"""Paper-dimension synthetic KRR benchmark: PKeX versus Metal QuadraSHAP.

Data follow the manuscript's make_regression dimensions, n=1000,
n_informative=d/4, noise=.1, seed=42. The formerly unspecified split/model
choices are fixed and recorded here: 80/20 split, training-only standardisation
of X and y, KernelRidge RBF gamma=1/d and alpha=.1. Fifty held-out points are
selected without replacement. PKeX is attempted only for d<=1000, each in an
independent subprocess with a 300s timeout. QuadraSHAP uses the same four
requested eps levels as the text experiment; Metal is the timed backend, and
one float64 CPU result at eps=1e-16 is used solely for empirical error audit.
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import exp10_pkex_metal_imdb as shared
from quadrashap.product_games.budget import budget_from_summary
from quadrashap.product_games.shapley import ProductGamesShapleyJax, ProductGamesShapleyNumpy

OUT = ROOT / "experiments" / "results" / "exp15_synthetic_pkex_metal_precision"
DIMS = (50, 500, 1000, 2000, 5000)
EPS = shared.EPS_LEVELS


def case_dir(d):
    return OUT / f"d{d}"


def prepare(d, n_instances=50):
    out = case_dir(d)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "case.npz"
    if path.exists():
        return
    X, y = make_regression(n_samples=1000, n_features=d,
                           n_informative=max(1, d // 4), noise=0.1,
                           random_state=42)
    ids = np.arange(len(X))
    train, test = train_test_split(ids, test_size=0.2, random_state=42, shuffle=True)
    x_scaler = StandardScaler().fit(X[train])
    y_scaler = StandardScaler().fit(y[train, None])
    X_train = x_scaler.transform(X[train]).astype(np.float64)
    X_test = x_scaler.transform(X[test]).astype(np.float64)
    y_train = y_scaler.transform(y[train, None]).ravel().astype(np.float64)
    gamma = 1.0 / d
    model = KernelRidge(kernel="rbf", gamma=gamma, alpha=0.1).fit(X_train, y_train)
    chosen = np.random.default_rng(20260917).choice(len(test), size=n_instances, replace=False)
    np.savez_compressed(path, support=X_train, alpha=model.dual_coef_,
                        X=X_test[chosen], test_indices=test[chosen],
                        y_train=y_train, gamma=np.array(gamma),
                        train_indices=train)
    shared.atomic_json(out / "prepare_meta.json", dict(d=d, n_total=1000,
        n_train=len(train), n_test=len(test), n_explained=n_instances,
        n_informative=max(1, d // 4), noise=0.1, data_seed=42,
        split_seed=42, explanation_seed=20260917,
        preprocessing="training-fit StandardScaler on X and y",
        model="KernelRidge(RBF, alpha=0.1, gamma=1/d)",
        gamma=gamma, model_alpha=0.1, test_indices=test[chosen].tolist(),
        machine=platform.machine(), created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


def factors(support, x, gamma):
    return np.exp(-gamma * np.square(support - x[None, :]))


def evaluate(K, alpha, m_q, method, node_block=2):
    engine = ProductGamesShapleyJax() if method == "metal" else ProductGamesShapleyNumpy()
    if method == "metal":
        K = K.astype(np.float32)
    t0 = time.perf_counter()
    phi_matrix = engine.phi_matrix_prefix_scan(K, m_q, node_block=node_block)
    phi = alpha @ np.asarray(phi_matrix, dtype=np.float64)
    return phi, time.perf_counter() - t0


def quad_one(d, instance, force=False):
    import jax
    devices = [str(dev) for dev in jax.devices()]
    if not any("metal" in s.lower() for s in devices):
        raise RuntimeError(f"Apple Metal device required, got {devices}")
    out = case_dir(d)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    marker = raw / f"metal_instance{instance:03d}_eps{shared.eps_tag(EPS[-1])}.json"
    if marker.exists() and not force:
        return
    case = np.load(out / "case.npz")
    support = case["support"]
    alpha = case["alpha"].reshape(-1)
    x = case["X"][instance]
    gamma = float(case["gamma"])
    t0 = time.perf_counter()
    U = factors(support, x, gamma)
    K = U - 1.0
    summary = shared.GameSummary(d=d).update(K, None, alpha)
    summary_seconds = time.perf_counter() - t0
    budgets = {eps: budget_from_summary(summary, eps, norm="max") for eps in EPS}
    q_ref = budgets[1e-16].m_q
    phi_ref, ref_seconds = evaluate(K, alpha, q_ref, "cpu")
    extra_q = min((d + 1) // 2, q_ref + 2)
    phi_extra, _ = evaluate(K, alpha, extra_q, "cpu") if extra_q > q_ref else (phi_ref, 0.0)
    ref_gap = float(np.max(np.abs(phi_extra - phi_ref)))
    full_prediction = float(np.exp(-gamma * np.square(support - x).sum(axis=1)) @ alpha)
    target = full_prediction - float(alpha.sum())
    ref_path = raw / f"reference_instance{instance:03d}.npz"
    np.savez_compressed(ref_path, phi=phi_ref, phi_extra=phi_extra)
    shared.atomic_json(raw / f"reference_instance{instance:03d}.json", dict(
        d=d, instance=instance, m_q=q_ref, extra_nodes=extra_q,
        extra_node_gap=ref_gap, efficiency_residual=abs(float(phi_ref.sum())-target),
        seconds=ref_seconds, dtype="float64 CPU", phi_path=str(ref_path)))
    for eps in EPS:
        budget = budgets[eps]
        phi, seconds = evaluate(K, alpha, budget.m_q, "metal")
        err = float(np.max(np.abs(phi-phi_ref)))
        vec_path = raw / f"metal_instance{instance:03d}_eps{shared.eps_tag(eps)}.npz"
        np.savez_compressed(vec_path, phi=phi)
        row = dict(d=d, instance=instance, method="QuadraSHAP Metal",
                   value_function="neutral factor, same as PKeX-Shapley",
                   n_support_vectors=len(alpha), requested_eps=eps,
                   m_q=budget.m_q, certified_quadrature_bound=budget.bound,
                   certificate_scope="quadrature in exact arithmetic only",
                   exactness_threshold=budget.exact_threshold,
                   A_max=summary.A_max, lambda_max=summary.lambda_max,
                   seconds_summary=summary_seconds, seconds_core=seconds,
                   seconds_end_to_end=summary_seconds+seconds,
                   observed_max_abs_error_to_float64_reference=err,
                   observed_l2_error_to_float64_reference=float(np.linalg.norm(phi-phi_ref)),
                   efficiency_residual=abs(float(phi.sum())-target),
                   reference_extra_node_gap=ref_gap, finite=bool(np.isfinite(phi).all()),
                   dtype="float32 Metal; float64 weighted reduction",
                   phi_path=str(vec_path), reference_phi_path=str(ref_path))
        shared.atomic_json(raw / f"metal_instance{instance:03d}_eps{shared.eps_tag(eps)}.json", row)
        print(f"d={d} x={instance:02d} Metal eps={eps:.0e} m={budget.m_q} "
              f"time={seconds:.2f}s err={err:.2e}", flush=True)


def pkex_one(d, instance, repo):
    if d > 1000:
        raise ValueError("PKeX is intentionally not run above d=1000")
    esp_dir = repo / "explainer"
    if not (esp_dir / "esp.py").exists():
        raise FileNotFoundError(esp_dir / "esp.py")
    sys.path.insert(0, str(esp_dir))
    from esp import ESPComputer
    raw = case_dir(d) / "raw"
    case = np.load(case_dir(d) / "case.npz")
    support, alpha, x = case["support"], case["alpha"].reshape(-1), case["X"][instance]
    gamma = float(case["gamma"])
    esp = ESPComputer(method="quadratic", use_scaling=True)
    phi = np.zeros(d, dtype=np.float64)
    t0 = time.perf_counter()
    pkex_chunk = 16
    for start in range(0, len(alpha), pkex_chunk):
        stop = min(start + pkex_chunk, len(alpha))
        U = np.exp(-gamma * np.square(support[start:stop] - x[None, :]))
        omega = esp.compute_weight_vectors(U)
        phi += (alpha[start:stop, None] * ((U-1.0)*omega)).sum(axis=0)
    seconds = time.perf_counter()-t0
    ref_path = raw / f"reference_instance{instance:03d}.npz"
    phi_ref = np.load(ref_path)["phi"]
    target = float(np.exp(-gamma*np.square(support-x).sum(axis=1))@alpha-alpha.sum())
    vec_path = raw / f"pkex_instance{instance:03d}.npz"
    np.savez_compressed(vec_path, phi=phi)
    row = dict(d=d, instance=instance, method="PKeX-Shapley",
        value_function="neutral factor, same as QuadraSHAP", status="complete",
        n_support_vectors=len(alpha), requested_eps=None, m_q=None,
        certified_quadrature_bound=0.0,
        certificate_scope="algebraically exact in exact arithmetic",
        seconds_core=seconds, seconds_end_to_end=seconds,
        observed_max_abs_error_to_float64_reference=float(np.max(np.abs(phi-phi_ref))),
        observed_l2_error_to_float64_reference=float(np.linalg.norm(phi-phi_ref)),
        efficiency_residual=abs(float(phi.sum())-target), finite=bool(np.isfinite(phi).all()),
        dtype="float64 CPU", phi_path=str(vec_path), reference_phi_path=str(ref_path),
        pkex_chunk=pkex_chunk, pkex_repo=str(repo), pkex_commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
    shared.atomic_json(raw / f"pkex_instance{instance:03d}.json", row)
    print(f"d={d} x={instance:02d} PKeX time={seconds:.2f}s", flush=True)


def _pkex_subprocess(d, instance, repo, timeout):
    cmd = [sys.executable, str(Path(__file__)), "worker-pkex", "--d", str(d),
           "--instance", str(instance), "--pkex-repo", str(repo)]
    started = time.perf_counter()
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout,
                                env={**os.environ, "JAX_PLATFORMS": "cpu"})
        return instance, result.returncode, result.stdout, result.stderr, time.perf_counter()-started
    except subprocess.TimeoutExpired:
        return instance, None, "", "", timeout


def pkex_parent(d, n_instances, repo, timeout=300, force=False, parallel=1):
    raw = case_dir(d) / "raw"
    shared.atomic_json(case_dir(d) / "pkex_run_meta.json", dict(
        d=d, n_instances=n_instances, timeout_seconds=timeout,
        last_invocation_parallel_jobs=parallel, pkex_chunk=16,
        concurrency_note="records can include multiple batches; see summary/pkex_execution_batches.json",
        pkex_repo=str(repo), pkex_esp_sha256=shared.file_sha256(repo / "explainer" / "esp.py"),
        pkex_commit=subprocess.check_output(["git", "-C", str(repo),
                                              "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=shared.file_sha256(Path(__file__)),
        machine=platform.machine(), created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    def needs_run(i):
        path = raw / f"pkex_instance{i:03d}.json"
        return force or not path.exists() or json.loads(path.read_text()).get("status") == "failed"

    pending = [i for i in range(n_instances) if needs_run(i)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(_pkex_subprocess, d, i, repo, timeout) for i in pending]
        for future in concurrent.futures.as_completed(futures):
            instance, returncode, stdout, stderr, elapsed = future.result()
            json_path = raw / f"pkex_instance{instance:03d}.json"
            if returncode is None:
                row = dict(d=d, instance=instance, method="PKeX-Shapley", status="timeout",
                           timeout_seconds=timeout, seconds_end_to_end=timeout,
                           certificate_scope="no attribution returned")
                shared.atomic_json(json_path, row)
                print(f"d={d} x={instance:02d} PKeX timeout>{timeout}s", flush=True)
            elif returncode:
                row = dict(d=d, instance=instance, method="PKeX-Shapley", status="failed",
                           seconds_end_to_end=elapsed, error=stderr[-2000:])
                shared.atomic_json(json_path, row)
                print(f"d={d} x={instance:02d} PKeX failed: {stderr[-250:]}", flush=True)
            else:
                print(stdout.strip(), flush=True)
            assemble(d)


def assemble(d):
    raw = case_dir(d) / "raw"
    rows = [json.loads(p.read_text()) for p in sorted(raw.glob("metal_instance*_eps*.json"))]
    rows += [json.loads(p.read_text()) for p in sorted(raw.glob("pkex_instance*.json"))]
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (case_dir(d) / "records.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "quad", "pkex", "worker-pkex", "assemble"))
    ap.add_argument("--d", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--n-instances", type=int, default=50)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--parallel-pkex", type=int, default=1,
                    help="independent PKeX subprocesses (one for isolation)")
    ap.add_argument("--pkex-repo", type=Path, default=shared.DEFAULT_PKEX_REPO)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for d in args.d:
        if d not in DIMS:
            ap.error(f"d must be one of {DIMS}")
        if args.stage == "prepare":
            prepare(d, args.n_instances)
        elif args.stage == "quad":
            if not (case_dir(d)/"case.npz").exists():
                raise FileNotFoundError("prepare must run first")
            import jax
            shared.atomic_json(case_dir(d) / "metal_run_meta.json", dict(
                d=d, requested_eps_levels=EPS,
                devices=[str(dev) for dev in jax.devices()],
                backend="JAX Apple Metal float32",
                reference_backend="NumPy float64",
                certificate_scope="exact-arithmetic quadrature only",
                case_sha256=shared.file_sha256(case_dir(d) / "case.npz"),
                script_sha256=shared.file_sha256(Path(__file__)),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            for i in range(args.n_instances):
                quad_one(d, i, args.force)
                assemble(d)
        elif args.stage == "pkex":
            if d > 1000:
                print(f"d={d}: PKeX intentionally omitted", flush=True)
            else:
                pkex_parent(d, args.n_instances, args.pkex_repo, args.timeout,
                            args.force, args.parallel_pkex)
        elif args.stage == "worker-pkex":
            pkex_one(d, args.instance, args.pkex_repo)
        else:
            assemble(d)


if __name__ == "__main__":
    main()
