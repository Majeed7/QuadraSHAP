"""Reproducible Cox tolerance/GPU rerun using the collaborator's CoxExplainer.

Model fitting stays on CPU; explanations use an explicitly checked accelerator.
The full exact quadrature rule is timed separately from its generation on CPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games.shapley import _gauss_legendre_01_numpy

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmarks/results/cox_survival"
OUTPUT = ROOT / "benchmarks/results/cox_gpu_tolerance"
DATASETS = ("gse24080", "tcga_lgg_methylation")
TOLERANCES = (1e-1, 1e-2, 1e-3, 1e-4)
NODES = (1, 2, 4, 8, 16, 32, 64, 128)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(require_gpu=True):
    import jax
    devices = jax.devices()
    if require_gpu and (not devices or any(d.platform.lower() == "cpu" for d in devices)):
        raise RuntimeError("GPU required: refusing to publish CPU timings as GPU timings.")
    if require_gpu and jax.config.jax_disable_jit:
        raise RuntimeError("JIT must be enabled.")
    info = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "executable": sys.executable,
            "devices": [str(d) for d in devices], "backend": jax.default_backend(),
            "jit_disabled": bool(jax.config.jax_disable_jit), "blas": threadpool_info(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("jax", "jaxlib", "numpy", "scipy", "pandas", "scikit-survival")},
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                              (ROOT / "src/quadrashap/product_games/shapley.py",
                               ROOT / "src/quadrashap/product_games/blocks.py",
                               ROOT / "src/quadrashap/multiplicative/engine.py",
                               ROOT / "src/quadrashap/multiplicative/glm.py")}}
    try:
        info["packages"]["jax-metal"] = importlib.metadata.version("jax-metal")
    except importlib.metadata.PackageNotFoundError:
        pass
    return info


def fit_cpu():
    # Use the same NumPy/scikit-survival environment and BLAS limit as before,
    # so the archived float64 exact attributions remain a valid reference.
    from benchmarks.cox_experiment import fit_dataset
    with threadpool_limits(limits=4):
        for name in DATASETS:
            exp = fit_dataset(name, OUTPUT, explainer_backend="prefix_scan_numpy")
            previous = np.load(OLD / name / "model_and_preprocessing.npz")
            current = np.load(OUTPUT / name / "model_and_preprocessing.npz")
            old_inputs = np.load(OLD / name / "explanation_inputs.npz")
            new_inputs = np.load(OUTPUT / name / "explanation_inputs.npz")
            same = all(np.array_equal(current[k], previous[k]) for k in ("coef", "train_rows", "test_rows"))
            same = same and all(np.array_equal(new_inputs[k], old_inputs[k]) for k in ("background", "X", "patient_ids"))
            provenance = {"bitwise_identical_to_previous_fit_and_explanation_inputs": same,
                          "max_coefficient_difference": float(np.max(np.abs(current["coef"] - previous["coef"]))),
                          "reference_source": str(OLD / name),
                          "reference_sha256": {k: file_hash(OLD / name / k) for k in
                                               ("model_and_preprocessing.npz", "explanation_inputs.npz", "exact_attributions_patient_0.npy")},
                          "fit_environment": environment(require_gpu=False)}
            write_json(OUTPUT / name / "refit_provenance.json", provenance)
            if not same:
                raise RuntimeError(f"{name}: refit changed; do not compare against archived exact values without recomputing the reference.")
            print(f"{name}: refit and all explanation inputs match the previous run bit for bit", flush=True)


def launch_cpu_refit():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT), "JAX_PLATFORMS": "cpu"}
    t = perf_counter()
    subprocess.run([str(ROOT / ".venv/bin/python"), "-u", "-m", "benchmarks.cox_gpu_experiment", "--fit"],
                   cwd=ROOT, env=env, check=True)
    write_json(OUTPUT / "refit_wall_time.json", {"subprocess_seconds": perf_counter() - t,
                                               "includes": "startup, load, preprocessing, fit, prediction, artifact writes and verification"})


def load_experiment(name):
    folder = OUTPUT / name
    provenance = json.loads((folder / "refit_provenance.json").read_text())
    if not provenance["bitwise_identical_to_previous_fit_and_explanation_inputs"]:
        raise RuntimeError("Archived exact reference does not match.")
    for k, digest in provenance["reference_sha256"].items():
        if file_hash(OLD / name / k) != digest:
            raise RuntimeError(f"Reference changed since verification: {k}")
    with np.load(folder / "model_and_preprocessing.npz") as m:
        coef = m["coef"]
    with np.load(folder / "explanation_inputs.npz") as inputs:
        bg, X, ids = inputs["background"], inputs["X"], inputs["patient_ids"]
    return {"name": name, "folder": folder, "coef": coef, "background": bg, "X": X, "ids": ids,
            "reference_0": np.load(OLD / name / "exact_attributions_patient_0.npy"),
            "fit": json.loads((folder / "fit_summary.json").read_text())}


def error_metrics(phi, reference):
    diff = phi - reference
    k = min(20, len(phi))
    top = set(np.argsort(np.abs(reference))[-k:])
    proposed = set(np.argsort(np.abs(phi))[-k:])
    return {"relative_l2_error": float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-300)),
            "max_absolute_error": float(np.max(np.abs(diff))), "top20_overlap": len(top & proposed) / k}


def small_exhaustive_benchmark():
    from benchmarks.jax_metal_smoke import brute_force_cox

    rng = np.random.default_rng(42)
    d = 12
    beta = rng.normal(size=d) * 0.1
    x, bg = rng.normal(size=d), rng.normal(size=(4, d))
    t = perf_counter()
    reference = brute_force_cox(beta, x, bg)
    exhaustive_seconds = perf_counter() - t
    rows = []
    for backend in ("prefix_scan_numpy", "prefix_scan_jax", "logspace_jax"):
        ex = CoxExplainer(beta, background=bg, backend=backend)
        t = perf_counter()
        phi = ex.explain(x, m_q="exact")
        first = perf_counter() - t
        times = []
        for _ in range(3):
            t = perf_counter()
            phi = ex.explain(x, m_q="exact")
            times.append(perf_counter() - t)
        np.testing.assert_allclose(phi, reference, rtol=2e-5, atol=2e-6)
        rows.append({"d": d, "coalitions": 2 ** d, "m_q": (d + 1) // 2, "backend": backend,
                     "first_call_seconds": first, "median_seconds": float(np.median(times)),
                     "exhaustive_seconds": exhaustive_seconds,
                     "speedup_vs_exhaustive": exhaustive_seconds / np.median(times),
                     **error_metrics(phi, reference)})
    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "small_exhaustive.csv", index=False)
    return table


def benchmark_sweep(exp, repeats=3):
    rows, raw, arrays = [], [], {}
    cpu = CoxExplainer(exp["coef"], background=exp["background"], backend="prefix_scan_numpy", memory_budget="512MB")
    references = [exp["reference_0"]] + [cpu.explain(x, m_q=128) for x in exp["X"][1:]]
    configs = [("tolerance", eps, None) for eps in TOLERANCES] + [("fixed_nodes", None, m) for m in NODES]
    for backend in ("logspace_jax", "prefix_scan_jax", "prefix_scan_numpy"):
        ex = CoxExplainer(exp["coef"], background=exp["background"], backend=backend, memory_budget="512MB")
        for mode, eps, m in configs:
            for i, x in enumerate(exp["X"]):
                kwargs = {"eps": eps, "return_report": True} if mode == "tolerance" else {"m_q": m}
                t = perf_counter()
                out = ex.explain(x, **kwargs)
                first_seconds = perf_counter() - t  # includes JIT only if this shape is new
                phi, report = out if mode == "tolerance" else (out, None)
                times, budgets = [], []
                for rep in range(repeats):
                    t = perf_counter()
                    out = ex.explain(x, **kwargs)
                    seconds = perf_counter() - t  # numpy return has synchronized device work
                    phi, report = out if mode == "tolerance" else (out, None)
                    times.append(seconds)
                    if report is not None:
                        budgets.append(report.seconds)
                    raw.append({"backend": backend, "mode": mode, "eps": eps, "m_q": ex.last_block_plan.m_q,
                                "patient": i, "repeat": rep, "seconds": seconds,
                                "budget_seconds": report.seconds if report else None})
                rec = {"dataset": exp["name"], "backend": backend, "device": "CPU" if backend.endswith("numpy") else "GPU",
                       "mode": mode, "eps": eps, "m_q": ex.last_block_plan.m_q, "patient": i,
                       "patient_id": str(exp["ids"][i]), "first_call_seconds": first_seconds,
                       "median_seconds": float(np.median(times)), "min_seconds": min(times), "max_seconds": max(times),
                       "median_budget_seconds": float(np.median(budgets)) if budgets else 0.0,
                       "bound": report.bound if report else None,
                       "exact_nodes": (len(x) + 1) // 2,
                       "reference": "archived float64 exact, verified identical model and inputs" if i == 0 else "new float64 128-node approximation",
                       "efficiency_residual": float(abs(phi.sum() - (ex.model.predict_scale(x[None])[0] - ex.expected_value))),
                       "node_block": ex.last_block_plan.node_block, "pair_block": ex.last_block_plan.block_size,
                       **error_metrics(phi, references[i])}
                rec["observed_tolerance_met"] = rec["max_absolute_error"] <= eps if eps is not None else None
                rows.append(rec)
                key = f"{backend}_p{i}_{mode}_{eps if eps is not None else m}"
                arrays[key] = phi
            pd.DataFrame(rows).to_csv(exp["folder"] / "sweep.csv", index=False)
            pd.DataFrame(raw).to_csv(exp["folder"] / "raw_timings.csv", index=False)
            print(f"{exp['name']} {backend}: {mode}={eps if eps is not None else m}, 3 patients complete", flush=True)
    np.savez_compressed(exp["folder"] / "sweep_attributions.npz", **arrays)
    return pd.DataFrame(rows)


def benchmark_exact_gpu(exp):
    d = len(exp["coef"])
    m = (d + 1) // 2
    print(f"{exp['name']}: generating all {m:,} exact-rule nodes on CPU", flush=True)
    # Explicitly clear the rule cache: these are fresh rule-generation timings.
    _gauss_legendre_01_numpy.cache_clear()
    t = perf_counter()
    nodes, weights = _gauss_legendre_01_numpy(m)
    rule_seconds = perf_counter() - t
    moment_error = max(abs(weights @ nodes ** k - 1 / (k + 1)) for k in (0, 1, 2, 8, 16))
    if not (np.isfinite(nodes).all() and np.isfinite(weights).all() and (weights > 0).all() and moment_error < 1e-8):
        raise RuntimeError("Exact quadrature rule failed validation")
    write_json(exp["folder"] / "exact_progress.json", {"phase": "GPU integration", "m_q": m, "rule_seconds": rule_seconds})
    print(f"{exp['name']}: rule ready in {rule_seconds:.2f}s; starting full exact-node GPU integration", flush=True)
    ex = CoxExplainer(exp["coef"], background=exp["background"], backend="logspace_jax", memory_budget="512MB")
    t = perf_counter()
    phi = ex.explain(exp["X"][0], m_q="exact")
    integration_seconds = perf_counter() - t
    np.save(exp["folder"] / "exact_gpu_attributions_patient_0.npy", phi)
    record = {"dataset": exp["name"], "backend": "logspace_jax", "patient": 0,
              "patient_id": str(exp["ids"][0]), "d": d, "m_q": m, "repeats": 1,
              "rule_seconds": rule_seconds, "integration_seconds": integration_seconds,
              "total_compute_seconds": rule_seconds + integration_seconds,
              "rule_moment_max_error": float(moment_error), "block_plan": asdict(ex.last_block_plan),
              "reference": "archived float64 exact, verified identical model and inputs",
              "qualification": "Exact polynomial degree; GPU float32 arithmetic with host float64 block accumulation",
              "efficiency_residual": float(abs(phi.sum() - (ex.model.predict_scale(exp["X"][:1])[0] - ex.expected_value))),
              **error_metrics(phi, exp["reference_0"])}
    old = json.loads((OLD / exp["name"] / "exact_timing.json").read_text())
    record["historical_cpu_integration_seconds"] = old["integration_seconds"]
    record["speedup_vs_historical_cpu_integration"] = old["integration_seconds"] / integration_seconds
    write_json(exp["folder"] / "exact_gpu_timing.json", record)
    write_json(exp["folder"] / "exact_progress.json", {"phase": "complete", **record})
    print(f"{exp['name']}: exact GPU integration {integration_seconds:.3f}s; rule + integration {record['total_compute_seconds']:.3f}s", flush=True)
    return record


def summary_tables():
    frames = []
    for name in DATASETS:
        folder = OUTPUT / name
        df = pd.read_csv(folder / "sweep.csv")
        exact = json.loads((folder / "exact_gpu_timing.json").read_text())
        chosen = df[(df.patient == 0) & (df["mode"] == "tolerance")].copy()
        chosen["gpu_exact_integration_seconds"] = exact["integration_seconds"]
        chosen["speedup_vs_gpu_exact"] = exact["integration_seconds"] / chosen.median_seconds
        cpu_times = chosen[chosen.backend == "prefix_scan_numpy"].set_index("eps").median_seconds
        chosen["speedup_vs_cpu_prefix_same_tolerance"] = chosen.eps.map(cpu_times) / chosen.median_seconds
        with np.load(folder / "sweep_attributions.npz") as values:
            ref_gpu = np.load(folder / "exact_gpu_attributions_patient_0.npy")
            chosen["relative_l2_vs_gpu_exact"] = [error_metrics(values[f"{r.backend}_p0_tolerance_{r.eps}"], ref_gpu)["relative_l2_error"]
                                                 for r in chosen.itertuples()]
        frames.append(chosen)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUTPUT / "tolerance_summary.csv", index=False)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if args.fit:
        fit_cpu()
    else:
        raise SystemExit("Use tutorials/cox_gpu_tolerance.ipynb to execute and record the GPU experiment.")
