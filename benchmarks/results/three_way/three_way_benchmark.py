"""Fresh-input benchmark for QuadraSHAP, QuadratureTreeSHAP, and TreeGrad-Shap.

TreeGrad's official implementation is CPU-only and supports individual
scikit-learn trees.  For a forest, this harness applies the unmodified
``treegrad_shap`` function to each estimator and averages the attributions,
which follows directly from Shapley linearity.  Models use ``bootstrap=False``
so TreeGrad's unweighted ``n_node_samples`` covers equal the weighted covers
used by the other two implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks"))

from xgboost_sklearn_bridge import sklearn_forest_to_xgboost  # noqa: E402


METHODS = (
    "quadrashap_cuda_exact",
    "xgb_qts_fresh_dmatrix",
    "treegrad_shap_cpu",
)
SEEDS = (42, 43, 44)
BATCHES = (1, 10, 64)
CONFIGS = (
    {"name": "single_f16_l256", "n_features": 16, "n_trees": 1, "leaves_per_tree": 256},
    {"name": "single_f16_l4096", "n_features": 16, "n_trees": 1, "leaves_per_tree": 4096},
    {"name": "single_f16_l16384", "n_features": 16, "n_trees": 1, "leaves_per_tree": 16384},
    {"name": "single_f100_l4096", "n_features": 100, "n_trees": 1, "leaves_per_tree": 4096},
    {"name": "forest32_f16_l128", "n_features": 16, "n_trees": 32, "leaves_per_tree": 128},
    {"name": "forest32_f100_l128", "n_features": 100, "n_trees": 32, "leaves_per_tree": 128},
)


def _json_line(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _timed(fn, *, warmups: int, repeats: int, synchronize=None):
    value = None
    for _ in range(warmups):
        value = fn()
        if synchronize is not None:
            synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = fn()
        if synchronize is not None:
            synchronize()
        times.append((time.perf_counter_ns() - start) * 1e-9)
    return value, times


def _expected_value(model) -> float:
    from quadrashap import TreeExplainer

    explainer = TreeExplainer(
        model,
        tree_solver="quadrature_tree",
        device="cpu",
        use_cpp=False,
    )
    return float(np.asarray(explainer.expected_value).ravel()[0])


def _treegrad_explain(model, X: np.ndarray, treegrad_shap) -> np.ndarray:
    estimators = list(np.ravel(model.estimators_))
    output = np.zeros((len(X), X.shape[1]), dtype=np.float64)
    weight = 1.0 / len(estimators)
    for row_index, row in enumerate(X):
        for estimator in estimators:
            output[row_index] += weight * treegrad_shap(
                estimator, row, (1, 1)
            )
    return output


def worker(args: argparse.Namespace) -> int:
    import cupy as cp
    import joblib
    import xgboost as xgb

    bundle = joblib.load(args.model)
    model = bundle["estimator"]
    pool = np.asarray(bundle["X"], dtype=np.float64)
    X = pool[np.arange(args.batch) % len(pool)] if len(pool) < args.batch else pool[: args.batch]
    X = np.ascontiguousarray(X, dtype=np.float64)
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    bias = _expected_value(model)
    bridge_error = None

    if args.method == "quadrashap_cuda_exact":
        from quadrashap import TreeExplainer

        explainer = TreeExplainer(model, tree_solver="quadrature_tree", device="cuda")
        used_mq = int(explainer._backend._cuda_solver.host.m_q)

        def explain():
            fresh_X = np.array(X, dtype=np.float64, order="C", copy=True)
            return explainer.shap_values(fresh_X, check_additivity=False)

        values, times = _timed(
            explain,
            warmups=args.warmups,
            repeats=args.repeats,
            synchronize=cp.cuda.runtime.deviceSynchronize,
        )
        phi = np.asarray(values, dtype=np.float64)

    elif args.method == "xgb_qts_fresh_dmatrix":
        booster = sklearn_forest_to_xgboost(model, device="cuda:0")

        def explain():
            fresh_X = np.array(X, dtype=np.float64, order="C", copy=True)
            return booster.predict(xgb.DMatrix(fresh_X), pred_contribs=True)

        values, times = _timed(
            explain,
            warmups=args.warmups,
            repeats=args.repeats,
            synchronize=cp.cuda.runtime.deviceSynchronize,
        )
        values = np.asarray(values, dtype=np.float64)
        phi = values[:, :-1]
        bias = float(values[0, -1])
        validation = xgb.DMatrix(np.array(X, copy=True, order="C"))
        xgb_prediction = np.asarray(
            booster.predict(validation, output_margin=True), dtype=np.float64
        )
        bridge_error = float(np.max(np.abs(xgb_prediction - prediction)))
        prediction = xgb_prediction
        used_mq = 8

    elif args.method == "treegrad_shap_cpu":
        if args.treegrad_repo is None:
            raise ValueError("--treegrad-repo is required for TreeGrad")
        sys.path.insert(0, str(args.treegrad_repo))
        from TreeGrad import treegrad_shap

        used_mq = max(
            (min(int(est.tree_.max_depth), X.shape[1]) + 1) // 2
            for est in np.ravel(model.estimators_)
        )

        def explain():
            fresh_X = np.array(X, dtype=np.float64, order="C", copy=True)
            return _treegrad_explain(model, fresh_X, treegrad_shap)

        phi, times = _timed(
            explain,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        phi = np.asarray(phi, dtype=np.float64)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    additivity = np.abs(bias + phi.sum(axis=1) - prediction)
    median_s = float(np.median(times))
    _json_line(
        {
            "status": "ok",
            "method": args.method,
            "device": "cpu" if args.method == "treegrad_shap_cpu" else "cuda",
            "batch": int(len(X)),
            "median_ms": 1e3 * median_s,
            "q1_ms": 1e3 * float(np.percentile(times, 25)),
            "q3_ms": 1e3 * float(np.percentile(times, 75)),
            "min_ms": 1e3 * float(min(times)),
            "times_ms": [1e3 * float(value) for value in times],
            "instances_per_second": float(len(X) / median_s),
            "max_additivity_error": float(np.max(additivity)),
            "mean_additivity_error": float(np.mean(additivity)),
            "used_mq": int(used_mq),
            "bridge_prediction_max_abs_error": bridge_error,
        }
    )
    return 0


def _training_rows(config: dict) -> int:
    leaves = int(config["leaves_per_tree"])
    return max(4096, 2 * leaves + 512) if int(config["n_trees"]) == 1 else max(8192, 8 * leaves)


def prepare_model(config: dict, seed: int, cache: Path, max_batch: int) -> Path:
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{config['name']}_s{seed}.joblib"
    if path.exists():
        bundle = joblib.load(path)
        if bundle.get("bootstrap") is False:
            return path

    rng = np.random.default_rng(seed)
    n_features = int(config["n_features"])
    n_train = _training_rows(config)
    X = rng.standard_normal((n_train, n_features), dtype=np.float32)
    width = min(8, n_features)
    y = (
        np.sin(X[:, :width]).sum(axis=1)
        + 0.35 * np.square(X[:, :width]).sum(axis=1)
        + rng.normal(0.0, 1.0, size=n_train)
    )
    model = RandomForestRegressor(
        n_estimators=int(config["n_trees"]),
        max_leaf_nodes=int(config["leaves_per_tree"]),
        max_features="sqrt",
        bootstrap=False,
        random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1),
    ).fit(X, y)
    X_test = rng.standard_normal((max_batch, n_features), dtype=np.float32)
    leaves = [int(est.tree_.n_leaves) for est in model.estimators_]
    joblib.dump(
        {
            "estimator": model,
            "X": np.asarray(X_test, dtype=np.float64),
            "config": config,
            "seed": seed,
            "bootstrap": False,
            "n_train": n_train,
            "actual_total_leaves": int(sum(leaves)),
            "actual_leaves_min": int(min(leaves)),
            "actual_leaves_max": int(max(leaves)),
            "max_depth": int(max(est.tree_.max_depth for est in model.estimators_)),
        },
        path,
        compress=3,
    )
    return path


def _last_payload(proc: subprocess.CompletedProcess) -> dict:
    for line in reversed(proc.stdout.splitlines()):
        if line.lstrip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {
        "status": "error",
        "returncode": proc.returncode,
        "message": (proc.stderr or proc.stdout)[-3000:],
    }


def run_worker(script: Path, model: Path, method: str, batch: int, args) -> dict:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--model",
        str(model),
        "--method",
        method,
        "--batch",
        str(batch),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--treegrad-repo",
        str(args.treegrad_repo),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_s": args.timeout_s}
    return _last_payload(proc)


def aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return {"status": rows[0].get("status", "error"), "per_seed": rows}
    result = {
        "status": "ok" if len(valid) == len(rows) else "partial",
        "median_ms": float(np.median([row["median_ms"] for row in valid])),
        "instances_per_second": float(
            np.median([row["instances_per_second"] for row in valid])
        ),
        "max_additivity_error": float(max(row["max_additivity_error"] for row in valid)),
        "per_seed": rows,
    }
    bridge = [row["bridge_prediction_max_abs_error"] for row in valid if row.get("bridge_prediction_max_abs_error") is not None]
    if bridge:
        result["bridge_prediction_max_abs_error"] = float(max(bridge))
    return result


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def metadata(args) -> dict:
    import cupy as cp
    import sklearn
    import xgboost

    props = cp.cuda.runtime.getDeviceProperties(0)
    try:
        cpu_model = subprocess.check_output(
            ["sh", "-c", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"],
            text=True,
        ).strip()
    except Exception:
        cpu_model = platform.processor() or "unknown"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": props["name"].decode(),
        "gpu_compute_capability": f"{props['major']}.{props['minor']}",
        "cpu": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "quadra_repository": "https://github.com/Majeed7/QuadraSHAP",
        "quadra_revision": _revision(REPO),
        "treegrad_repository": "https://github.com/watml/TreeGrad",
        "treegrad_revision": _revision(args.treegrad_repo),
        "versions": {
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }


def save_csv(payload: dict, path: Path) -> None:
    rows = []
    for config, batches in payload.get("results", {}).items():
        for batch, methods in batches.items():
            for method, result in methods.items():
                rows.append(
                    {
                        "config": config,
                        "batch": int(batch),
                        "method": method,
                        **{key: value for key, value in result.items() if key != "per_seed"},
                    }
                )
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args) -> int:
    import joblib

    configs = CONFIGS
    if args.configs:
        requested = set(args.configs)
        configs = tuple(item for item in configs if item["name"] in requested)
        missing = requested.difference(item["name"] for item in configs)
        if missing:
            raise ValueError(f"Unknown configurations: {sorted(missing)}")
    payload = {
        "metadata": metadata(args),
        "protocol": {
            "methods": list(METHODS),
            "fresh_host_copy_each_call": True,
            "fresh_xgboost_dmatrix_each_call": True,
            "cached_explanations": False,
            "prepared_models_reused": True,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seeds": list(args.seeds),
            "batches": list(args.batches),
            "bootstrap": False,
            "treegrad_device": "CPU (official implementation has no GPU backend)",
            "treegrad_forest_adapter": "apply official routine per estimator and average",
            "timed_worker_numeric_threads": 1,
        },
        "models": {},
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        save_csv(payload, args.output.with_suffix(".csv"))

    script = Path(__file__).resolve()
    for config in configs:
        name = config["name"]
        paths = {}
        payload["models"][name] = {"requested": config, "per_seed": {}}
        for seed in args.seeds:
            print(f"[prepare] {name} seed={seed}", flush=True)
            path = prepare_model(config, seed, args.cache, max(args.batches))
            paths[seed] = path
            bundle = joblib.load(path)
            payload["models"][name]["per_seed"][str(seed)] = {
                key: bundle[key]
                for key in (
                    "n_train",
                    "bootstrap",
                    "actual_total_leaves",
                    "actual_leaves_min",
                    "actual_leaves_max",
                    "max_depth",
                )
            }
        save()
        for batch in args.batches:
            print(f"[benchmark] {name} batch={batch}", flush=True)
            cell = payload["results"].setdefault(name, {}).setdefault(str(batch), {})
            order = list(METHODS)
            shift = (sum(map(ord, name)) + int(batch)) % len(order)
            order = order[shift:] + order[:shift]
            for method in order:
                rows = []
                for seed in args.seeds:
                    result = run_worker(script, paths[seed], method, int(batch), args)
                    result["seed"] = int(seed)
                    rows.append(result)
                summary = aggregate(rows)
                cell[method] = summary
                print(
                    f"  {method:27s} {summary.get('median_ms', float('nan')):12.3f} ms "
                    f"status={summary.get('status')}",
                    flush=True,
                )
                save()
    save()
    print(f"Wrote {args.output}", flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--treegrad-repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=REPO / "benchmarks" / "_gpu_three_way_models")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "benchmarks" / "results" / "three_way" / "three_way_results.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else benchmark(parsed))
