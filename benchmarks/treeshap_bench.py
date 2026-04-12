"""TreeSHAP benchmark.

Compares SHAP, FastTreeSHAP (v1, v2), linear_tree_shap, and pgshapley's C++
quadrature-tree kernel across 10..10^6 total leaves, for two feature widths
(10 and 100).

All methods run single-threaded: the worker subprocess is spawned with
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, etc., and
every library that exposes an n_jobs knob is pinned to 1. This matters
because FastTreeSHAP and SHAP both parallelise across samples internally,
which would otherwise make them look faster purely by using more cores.

Each (method, size) measurement runs in a subprocess with a hard wall-clock
timeout. The script caches trained models on disk so re-runs don't retrain.

For stability, N_MODEL_SEEDS (default 3) independently-seeded forests are
trained per (n_features, n_leaves) and every method is measured on each
one; the aggregated result reports the median time and max error across
those seeds.

Run with:
    .venv/bin/python benchmarks/treeshap_bench.py

Results are written to benchmarks/treeshap_bench_results.json.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "benchmarks" / "treeshap_bench_results.json"
MODELS_CACHE = REPO / "benchmarks" / "_treeshap_bench_models"
MODELS_CACHE.mkdir(parents=True, exist_ok=True)
WORKER = REPO / "benchmarks" / "treeshap_bench_worker.py"
PYTHON = sys.executable

TIMEOUT_S = 120
N_TEST_SAMPLES = 10      # samples explained per forest
N_MODEL_SEEDS = 3        # independent forests per (n_features, n_leaves)
N_FEATURE_CHOICES = [10, 100]
LEAF_SIZES = [10, 20, 50, 100, 1_000, 10_000, 100_000]
N_REPEATS = 3

METHOD_NAMES = [
    "shap",
    "fasttreeshap_v1",
    "fasttreeshap_v2",
    "linear_tree_shap",
    "shapiq",
    "pg_quadrature_tree_cpp",
]
# All methods run on the same sklearn RandomForestRegressor so the tree
# structure is identical across methods. (XGBoost SHAP was intentionally
# dropped because it needs its own XGBoost tree format.)

# Per-tree cap. Targets larger than this grow the forest instead of the tree.
MAX_LEAVES_PER_TREE = 10_000
# Training samples are only big enough to support the per-tree leaf cap.
N_TRAIN_SAMPLES = 50_000

# These env vars are forced on every worker subprocess to pin all numeric
# libraries to a single thread. Comparing single-threaded implementations
# isolates the algorithm from the parallelism bonus.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def _training_config(n_leaves: int):
    if n_leaves <= MAX_LEAVES_PER_TREE:
        return 1, n_leaves
    n_est = max(1, n_leaves // MAX_LEAVES_PER_TREE)
    return n_est, MAX_LEAVES_PER_TREE


# ---------------------------------------------------------------------------
# Model training / caching
# ---------------------------------------------------------------------------

def _make_data(n_features: int, seed: int):
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=N_TRAIN_SAMPLES,
        n_features=n_features,
        random_state=seed,
    )
    return X, y


def _build_sklearn_rf(n_features: int, n_leaves: int, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    n_est, max_leaves = _training_config(n_leaves)
    X, y = _make_data(n_features, seed)
    model = RandomForestRegressor(
        n_estimators=n_est,
        max_leaf_nodes=max_leaves,
        random_state=seed,
        n_jobs=1,  # training single-threaded for full determinism
    ).fit(X, y)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=N_TEST_SAMPLES, replace=False)
    X_test = X[idx]
    return model, X_test


def get_or_train(n_features: int, n_leaves: int, seed: int):
    key = f"sklearn_rf_f{n_features}_l{n_leaves}_s{seed}.pkl"
    path = MODELS_CACHE / key
    if path.exists():
        try:
            with open(path, "rb") as f:
                model, Xt = pickle.load(f)
            return path, model, Xt
        except Exception:
            pass  # rebuild below

    t0 = time.perf_counter()
    model, Xt = _build_sklearn_rf(n_features, n_leaves, seed)
    train_time = time.perf_counter() - t0
    print(f"  trained sklearn_rf f={n_features} l={n_leaves} s={seed} "
          f"in {train_time:.1f}s", flush=True)

    try:
        with open(path, "wb") as f:
            pickle.dump((model, Xt), f)
    except Exception as e:
        print(f"  warn: failed to cache model ({e})", flush=True)
    return path, model, Xt


# ---------------------------------------------------------------------------
# Subprocess worker invocation with wall-clock timeout
# ---------------------------------------------------------------------------

def measure_method(method_name: str, model_pkl: Path, pred: np.ndarray):
    """Invoke the worker subprocess with a wall-clock timeout.

    The worker returns both phi_sum and its own expected_value (``ev``). Error
    is computed as ``max_i |ev + phi_sum_i - pred_i|`` — the direct additivity
    residual using whichever baseline the library uses. For ``linear_tree_shap``
    the worker reports the alternative baseline (sum of leaf predictions per
    tree, averaged across the forest) so the metric is fair.

    Returns {status, time_s, error, [message]}.
    """
    cmd = [PYTHON, str(WORKER), str(model_pkl), method_name, str(N_REPEATS)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env={**os.environ, **SINGLE_THREAD_ENV},
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        return {"status": "timeout", "time_s": None, "error": None, "wall_s": wall}

    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "status": "err",
            "time_s": None,
            "error": None,
            "message": (proc.stderr or "").strip()[-300:],
            "wall_s": wall,
        }

    # Last JSON line on stdout (libraries may emit warnings before it).
    line = None
    for ln in reversed(proc.stdout.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            line = ln
            break
    if line is None:
        return {
            "status": "err",
            "time_s": None,
            "error": None,
            "message": (proc.stderr or proc.stdout or "")[-300:],
            "wall_s": wall,
        }
    try:
        payload = json.loads(line)
    except Exception as e:
        return {
            "status": "err",
            "time_s": None,
            "error": None,
            "message": f"parse: {e}",
            "wall_s": wall,
        }

    if payload.get("status") != "ok":
        return {
            "status": "err",
            "time_s": None,
            "error": None,
            "message": payload.get("message", ""),
            "wall_s": wall,
        }

    phi_sum = np.asarray(payload["sv_sum"], dtype=np.float64)
    ev = float(payload["ev"])
    err = float(np.max(np.abs(ev + phi_sum - pred)))
    return {
        "status": "ok",
        "time_s": float(payload["elapsed_s"]),
        "error": err,
        "times": payload.get("times", []),
        "wall_s": wall,
    }


# ---------------------------------------------------------------------------
# Aggregation across model seeds
# ---------------------------------------------------------------------------

def _aggregate(per_seed_results):
    """Aggregate a list of per-seed result dicts for one method.

    - If any seed ran OK, the aggregate is 'ok' with:
        time_s = median over OK seeds
        error  = max over OK seeds (worst case)
    - Otherwise the aggregate inherits the first failure status.
    """
    oks = [r for r in per_seed_results if r.get("status") == "ok"]
    if not oks:
        # Surface the first non-ok result unchanged, but add a per_seed list.
        head = dict(per_seed_results[0])
        head["per_seed"] = per_seed_results
        return head

    times = [r["time_s"] for r in oks]
    errs = [r["error"] for r in oks if r.get("error") is not None]
    return {
        "status": "ok",
        "time_s": float(np.median(times)),
        "time_s_min": float(np.min(times)),
        "time_s_max": float(np.max(times)),
        "error": float(np.max(errs)) if errs else None,
        "n_seeds_ok": len(oks),
        "n_seeds_total": len(per_seed_results),
        "per_seed": per_seed_results,
    }


def main():
    results = load_results()

    for n_features in N_FEATURE_CHOICES:
        results.setdefault(str(n_features), {})
        for n_leaves in LEAF_SIZES:
            results[str(n_features)].setdefault(str(n_leaves), {})
            done_methods = set(results[str(n_features)][str(n_leaves)].keys())
            remaining = [m for m in METHOD_NAMES if m not in done_methods]
            if not remaining:
                print(f"[skip] f={n_features} l={n_leaves} (cached)", flush=True)
                continue

            print(f"[run]  f={n_features} l={n_leaves}", flush=True)

            # Train all seed models up front and cache their (path, pred) info.
            seeds = list(range(42, 42 + N_MODEL_SEEDS))
            per_seed_ctx = []
            for seed in seeds:
                try:
                    path, model, X = get_or_train(n_features, n_leaves, seed)
                except BaseException as e:
                    print(f"  [train-fail s={seed}] {e}", flush=True)
                    per_seed_ctx.append(None)
                    continue
                pred = np.asarray(model.predict(X), dtype=np.float64)
                per_seed_ctx.append((path, pred))
                del model, X
                gc.collect()

            for method_name in remaining:
                per_seed_results = []
                for seed, ctx in zip(seeds, per_seed_ctx):
                    if ctx is None:
                        per_seed_results.append({
                            "status": "train-fail", "time_s": None, "error": None,
                        })
                        continue
                    path, pred = ctx
                    per_seed_results.append(measure_method(method_name, path, pred))

                agg = _aggregate(per_seed_results)
                results[str(n_features)][str(n_leaves)][method_name] = agg

                status = agg["status"]
                tstr = (f"{agg['time_s']*1e3:9.2f} ms"
                        if agg.get("time_s") is not None else "    -     ")
                estr = (f"{agg['error']:.2e}"
                        if agg.get("error") is not None else "   -    ")
                n_ok = agg.get("n_seeds_ok", 0)
                n_tot = agg.get("n_seeds_total", len(seeds))
                print(f"    {method_name:26s} {status:8s} {tstr}  "
                      f"err={estr}  ({n_ok}/{n_tot} seeds)", flush=True)

                save_results(results)

            gc.collect()

    save_results(results)
    print(f"\nResults written to {RESULTS_PATH}")


def load_results():
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_results(results):
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(RESULTS_PATH)


if __name__ == "__main__":
    main()
