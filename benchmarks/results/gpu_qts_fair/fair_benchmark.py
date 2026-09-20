"""Fair fresh-input CUDA benchmark against XGBoost QuadratureTreeSHAP.

This script belongs to the exact ``Majeed7/QuadraSHAP`` repository.  It does
not reuse an XGBoost DMatrix and does not cache explanations.  Both methods
reuse a prepared model, receive a fresh contiguous NumPy copy on every timed
call, and return their attributions to host memory.  Construction/JIT and
warm-up are excluded symmetrically.
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
    "quadrashap_cuda_mq8",
    "xgb_qts_fresh_dmatrix",
)
SEEDS = (42, 43, 44)
BATCHES = (1, 64, 1024)
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


def _timed(fn, cp, *, warmups: int, repeats: int):
    value = None
    for _ in range(warmups):
        value = fn()
        cp.cuda.runtime.deviceSynchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = fn()
        cp.cuda.runtime.deviceSynchronize()
        times.append((time.perf_counter_ns() - start) * 1e-9)
    return value, times


def _exact_reference(model, X: np.ndarray):
    from quadrashap import TreeExplainer

    explainer = TreeExplainer(model, tree_solver="quadrature_tree", device="cuda")
    phi = np.asarray(explainer.shap_values(X, check_additivity=False), dtype=np.float64)
    bias = float(np.asarray(explainer.expected_value).ravel()[0])
    mq = int(explainer._backend._cuda_solver.host.m_q)
    return phi, bias, mq


def worker(args: argparse.Namespace) -> int:
    import cupy as cp
    import joblib
    import xgboost as xgb

    bundle = joblib.load(args.model)
    model = bundle["estimator"]
    pool = np.asarray(bundle["X"], dtype=np.float64)
    if len(pool) < args.batch:
        X = pool[np.arange(args.batch) % len(pool)]
    else:
        X = pool[: args.batch]
    X = np.ascontiguousarray(X, dtype=np.float64)
    sklearn_prediction = np.asarray(model.predict(X), dtype=np.float64)

    if args.method.startswith("quadrashap_cuda"):
        from quadrashap import TreeExplainer

        requested_mq = 8 if args.method.endswith("mq8") else None
        explainer = TreeExplainer(
            model,
            tree_solver="quadrature_tree",
            device="cuda",
            m_q=requested_mq,
        )
        used_mq = int(explainer._backend._cuda_solver.host.m_q)

        def explain():
            # Host copy is deliberately inside the timed region.
            fresh_X = np.array(X, dtype=np.float64, order="C", copy=True)
            return explainer.shap_values(fresh_X, check_additivity=False)

        values, times = _timed(explain, cp, warmups=args.warmups, repeats=args.repeats)
        phi = np.asarray(values, dtype=np.float64)
        bias = float(np.asarray(explainer.expected_value).ravel()[0])
        additivity = np.abs(bias + phi.sum(axis=1) - sklearn_prediction)
        bridge_error = None

    elif args.method == "xgb_qts_fresh_dmatrix":
        booster = sklearn_forest_to_xgboost(model, device="cuda:0")

        def explain():
            # Both the host copy and a brand-new DMatrix are timed.  Because
            # the DMatrix identity changes, XGBoost cannot reuse an input-page
            # or prediction cache from a preceding repetition.
            fresh_X = np.array(X, dtype=np.float64, order="C", copy=True)
            fresh_dmatrix = xgb.DMatrix(fresh_X)
            return booster.predict(fresh_dmatrix, pred_contribs=True)

        values, times = _timed(explain, cp, warmups=args.warmups, repeats=args.repeats)
        values = np.asarray(values, dtype=np.float64)
        phi = values[:, :-1]
        bias = float(values[0, -1])
        validation_dmatrix = xgb.DMatrix(np.array(X, copy=True, order="C"))
        xgb_prediction = np.asarray(
            booster.predict(validation_dmatrix, output_margin=True), dtype=np.float64
        )
        cp.cuda.runtime.deviceSynchronize()
        additivity = np.abs(values.sum(axis=1) - xgb_prediction)
        bridge_error = float(np.max(np.abs(xgb_prediction - sklearn_prediction)))
        used_mq = 8

    else:
        raise ValueError(f"Unknown method: {args.method}")

    median_s = float(np.median(times))
    result = {
        "status": "ok",
        "method": args.method,
        "batch": int(len(X)),
        "median_ms": 1e3 * median_s,
        "q1_ms": 1e3 * float(np.percentile(times, 25)),
        "q3_ms": 1e3 * float(np.percentile(times, 75)),
        "min_ms": 1e3 * float(min(times)),
        "times_ms": [1e3 * float(value) for value in times],
        "instances_per_second": float(len(X) / median_s),
        "max_additivity_error": float(np.max(additivity)),
        "mean_additivity_error": float(np.mean(additivity)),
        "used_mq": used_mq,
        "bridge_prediction_max_abs_error": bridge_error,
    }

    # Cross-method validation is deliberately outside the timed region.
    if args.method in ("quadrashap_cuda_mq8", "xgb_qts_fresh_dmatrix"):
        n_reference = min(64, len(X))
        exact_phi, exact_bias, exact_mq = _exact_reference(model, X[:n_reference])
        delta = np.abs(phi[:n_reference] - exact_phi)
        result.update(
            {
                "reference_rows": n_reference,
                "reference_exact_mq": exact_mq,
                "phi_vs_exact_max_abs_error": float(np.max(delta)),
                "phi_vs_exact_mean_abs_error": float(np.mean(delta)),
                "bias_vs_exact_abs_error": float(abs(bias - exact_bias)),
            }
        )

    _json_line(result)
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
        bootstrap=True,
        random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1),
    ).fit(X, y)
    X_test = rng.standard_normal((max_batch, n_features), dtype=np.float32)
    leaf_counts = [int(est.tree_.n_leaves) for est in model.estimators_]
    joblib.dump(
        {
            "estimator": model,
            "X": np.asarray(X_test, dtype=np.float64),
            "config": config,
            "seed": seed,
            "n_train": n_train,
            "actual_total_leaves": int(sum(leaf_counts)),
            "actual_leaves_min": int(min(leaf_counts)),
            "actual_leaves_max": int(max(leaf_counts)),
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


def run_worker(script: Path, model: Path, method: str, batch: int, args: argparse.Namespace) -> dict:
    proc = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return _last_payload(proc)


def aggregate(per_seed: list[dict]) -> dict:
    valid = [item for item in per_seed if item.get("status") == "ok"]
    if not valid:
        return {"status": "error", "per_seed": per_seed}
    output = {
        "status": "ok",
        "median_ms": float(np.median([item["median_ms"] for item in valid])),
        "instances_per_second": float(
            np.median([item["instances_per_second"] for item in valid])
        ),
        "max_additivity_error": float(max(item["max_additivity_error"] for item in valid)),
        "per_seed": per_seed,
    }
    for key in (
        "phi_vs_exact_max_abs_error",
        "phi_vs_exact_mean_abs_error",
        "bias_vs_exact_abs_error",
        "bridge_prediction_max_abs_error",
    ):
        values = [item[key] for item in valid if item.get(key) is not None]
        if values:
            output[key] = float(max(values))
    return output


def _revision() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def metadata() -> dict:
    import cupy as cp
    import sklearn
    import xgboost

    props = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_repository": "https://github.com/Majeed7/QuadraSHAP",
        "git_revision": _revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": props["name"].decode(),
        "gpu_compute_capability": f"{props['major']}.{props['minor']}",
        "gpu_memory_bytes": int(props["totalGlobalMem"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "versions": {
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }


def save_csv(results: dict, path: Path) -> None:
    rows = []
    for config, batches in results.get("results", {}).items():
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


def benchmark(args: argparse.Namespace) -> int:
    import joblib

    configs = CONFIGS
    if args.configs:
        requested = set(args.configs)
        configs = tuple(item for item in configs if item["name"] in requested)
        missing = requested.difference(item["name"] for item in configs)
        if missing:
            raise ValueError(f"Unknown configurations: {sorted(missing)}")

    output = {
        "metadata": metadata(),
        "protocol": {
            "primary_comparison": "fresh input end-to-end explanation call",
            "fresh_host_copy_each_call": True,
            "fresh_xgboost_dmatrix_each_call": True,
            "reused_xgboost_dmatrix": False,
            "cached_explanations": False,
            "prepared_model_reused_symmetrically": True,
            "construction_and_jit_excluded": True,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seeds": list(args.seeds),
            "batches": list(args.batches),
            "same_logical_tree_topology_and_covers": True,
            "xgboost_model_precision": "float32",
            "quadrashap_model_precision": "float64",
        },
        "models": {},
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        save_csv(output, args.output.with_suffix(".csv"))

    script = Path(__file__).resolve()
    for config in configs:
        name = config["name"]
        paths = {}
        output["models"][name] = {"requested": config, "per_seed": {}}
        for seed in args.seeds:
            print(f"[prepare] {name} seed={seed}", flush=True)
            path = prepare_model(config, seed, args.cache, max(args.batches))
            paths[seed] = path
            bundle = joblib.load(path)
            output["models"][name]["per_seed"][str(seed)] = {
                key: bundle[key]
                for key in (
                    "n_train",
                    "actual_total_leaves",
                    "actual_leaves_min",
                    "actual_leaves_max",
                    "max_depth",
                )
            }
        save()

        for batch in args.batches:
            print(f"[benchmark] {name} batch={batch}", flush=True)
            cell = output["results"].setdefault(name, {}).setdefault(str(batch), {})
            order = list(METHODS)
            shift = (sum(map(ord, name)) + int(batch)) % len(order)
            order = order[shift:] + order[:shift]
            for method in order:
                per_seed = []
                for seed in args.seeds:
                    result = run_worker(script, paths[seed], method, int(batch), args)
                    result["seed"] = int(seed)
                    per_seed.append(result)
                summary = aggregate(per_seed)
                cell[method] = summary
                print(
                    f"  {method:27s} {summary.get('median_ms', float('nan')):10.3f} ms "
                    f"status={summary.get('status')}",
                    flush=True,
                )
                save()

    save()
    print(f"Wrote {args.output}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--cache", type=Path, default=REPO / "benchmarks" / "_gpu_qts_fair_models")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "benchmarks" / "results" / "gpu_qts_fair" / "fair_results.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else benchmark(parsed))
