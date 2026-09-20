"""Rerun three Cox GPU tolerances and matched CPU FP64 accuracy diagnostics.

The completed B=30/five-patient degree-exact GPU results remain the baseline.
No old measurements or attributions are overwritten.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games import shapley
from benchmarks.cox_gpu_experiment import environment, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/results/cox_background30_gpu"
OUT = SOURCE / "tolerance_rerun"
DATASETS = ("gse24080", "tcga_lgg_methylation")
TOLERANCES = (1e-1, 1e-2, 1e-3)


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_inputs(name):
    folder = SOURCE / name
    design = read_json(SOURCE / "design.json")[name]
    assert file_hash(folder / "inputs.npz") == design["prepared_inputs_sha256"]
    with np.load(folder / "inputs.npz") as saved:
        result = {k: saved[k] for k in ("beta", "background", "X", "patient_ids")}
    assert len(result["background"]) == 30 and len(result["X"]) == 5
    assert len(set(result["patient_ids"])) == 5
    result["reference"] = np.load(folder / "reference_64.npy")
    return result


def error_pair(phi, reference):
    delta = phi - reference
    return float(np.max(np.abs(delta))), float(np.linalg.norm(delta) / np.linalg.norm(reference))


def gpu():
    if (OUT / "gpu_summary.csv").exists() or (OUT / "gpu_raw_timings.csv").exists():
        raise FileExistsError("Preserve existing rerun results before repeating")
    OUT.mkdir(exist_ok=True)
    begin = perf_counter()
    env = environment(require_gpu=True)
    assert env["backend"].lower() == "metal"
    old_env = read_json(SOURCE / "environment.json")
    assert env["source_sha256"] == old_env["source_sha256"]
    assert env["packages"] == old_env["packages"]
    assert read_json(SOURCE / "run_status.json")["phase"] == "complete"
    env["benchmark_source_sha256"] = file_hash(__file__)
    write_json(OUT / "gpu_environment.json", env)
    selected = pd.read_csv(SOURCE / "summary.csv")
    exact_rows = selected[selected["mode"] == "exact"]
    assert len(exact_rows) == 10 and exact_rows.B.eq(30).all()
    assert exact_rows.memory_budget.eq("8GB").all() and exact_rows.block_size.eq(1).all()
    write_json(OUT / "provenance.json", {
        "exact_source": "../summary.csv", "exact_attributions": "../attributions.npz",
        "exact_baseline_reused": True, "tolerances": list(TOLERANCES),
        "background_size": 30, "distinct_patients_per_dataset": 5,
        "measured_calls_per_patient_method": 1,
        "warmup": "Once per dataset and selected node count before first measured call",
        "core_hashes_match_exact_baseline": True, "packages_match_exact_baseline": True,
        "input_sha256": {name: file_hash(SOURCE / name / "inputs.npz") for name in DATASETS},
        "source_sha256": {f: file_hash(SOURCE / f) for f in
                          ("summary.csv", "attributions.npz", "design.json", "environment.json", "rule_setup.json")},
        "reference_sha256": {name: file_hash(SOURCE / name / "reference_64.npy") for name in DATASETS}})
    original_rule = shapley._gauss_legendre_01_numpy
    rules = {}

    def cached_rule(m, dtype=np.float64):
        if int(m) not in rules:
            rules[int(m)] = original_rule(int(m))
        return tuple(a.astype(dtype, copy=False) for a in rules[int(m)])

    raw, rows, arrays = [], [], {}
    state = {"phase": "starting", "pid": os.getpid()}
    shapley._gauss_legendre_01_numpy = cached_rule
    try:
        with np.load(SOURCE / "attributions.npz") as saved_exact:
            for name in DATASETS:
                data = checked_inputs(name)
                beta, bg, X, ids = (data[k] for k in ("beta", "background", "X", "patient_ids"))
                assert shapley._jax_work_dtype(beta) == np.dtype(np.float32)
                exact = [saved_exact[f"{name}_p{p}_exact_None"] for p in range(5)]
                ex = CoxExplainer(beta, background=bg, backend="logspace_jax", memory_budget="8GB", block_size=1)
                warmed = set()
                for eps in TOLERANCES:
                    for patient, x in enumerate(X):
                        m = ex.node_budget(x, eps=eps).m_q
                        warmup_seconds = 0.0
                        phases = ("warmup", "measured") if m not in warmed else ("measured",)
                        for phase in phases:
                            state = {"phase": "running", "pid": os.getpid(), "dataset": name,
                                     "patient": patient, "eps": eps, "call_phase": phase,
                                     "measured_calls": len(rows), "total_measured_calls": 30}
                            write_json(OUT / "gpu_status.json", state)
                            tick = perf_counter()
                            if phase == "warmup":
                                phi, report = ex.explain(x, m_q=m), None
                            else:
                                phi, report = ex.explain(x, eps=eps, return_report=True)
                            seconds = perf_counter() - tick
                            assert np.isfinite(phi).all() and ex.last_block_plan.m_q == m
                            raw.append({"dataset": name, "patient": patient, "eps": eps, "m_q": m,
                                        "phase": phase, "seconds": seconds,
                                        "budget_seconds": 0.0 if report is None else report.seconds})
                            if phase == "warmup":
                                warmup_seconds = seconds
                                warmed.add(m)
                            pd.DataFrame(raw).to_csv(OUT / "gpu_raw_timings.csv", index=False)
                        assert report.bound <= eps
                        exact_abs, exact_rel = error_pair(phi, exact[patient])
                        ref_abs, ref_rel = error_pair(phi, data["reference"][patient])
                        rows.append({"dataset": name, "patient": patient, "patient_id": str(ids[patient]),
                                     "B": 30, "d": len(beta), "eps": eps, "m_q": m,
                                     "seconds": seconds, "warmup_seconds": warmup_seconds, "bound": report.bound,
                                     "max_absolute_error_vs_gpu_exact": exact_abs,
                                     "relative_l2_error_vs_gpu_exact": exact_rel,
                                     "within_eps_vs_gpu_exact": exact_abs <= eps,
                                     "max_absolute_error_vs_fp64_reference": ref_abs,
                                     "relative_l2_error_vs_fp64_reference": ref_rel,
                                     "within_eps_vs_fp64_reference": ref_abs <= eps,
                                     "memory_budget": "8GB", "block_size": 1})
                        arrays[f"{name}_p{patient}_eps_{eps}"] = phi
                        pd.DataFrame(rows).to_csv(OUT / "gpu_summary.csv", index=False)
                        print(f"GPU {name} p{patient} eps={eps:g} m={m}: {seconds:.6f}s; "
                              f"difference vs GPU exact={exact_abs:.6g}; bound={report.bound:.6g}", flush=True)
        np.savez_compressed(OUT / "gpu_attributions.npz", **arrays)
    except BaseException:
        write_json(OUT / "gpu_status.json", {**state, "phase": "interrupted", "measured_calls": len(rows)})
        raise
    finally:
        shapley._gauss_legendre_01_numpy = original_rule
    setup = []
    for m in sorted(rules):
        times = []
        for _ in range(5):
            original_rule.cache_clear()
            tick = perf_counter(); original_rule(m); times.append(perf_counter() - tick)
        setup.append({"m_q": m, "median_seconds": float(np.median(times)),
                      "min_seconds": min(times), "max_seconds": max(times), "repeats": 5})
    pd.DataFrame(setup).to_csv(OUT / "small_rule_timings.csv", index=False)
    assert env["source_sha256"] == environment(require_gpu=True)["source_sha256"]
    write_json(OUT / "gpu_status.json", {"phase": "complete", "cases": len(rows),
               "measured_calls": len(rows), "completed_calls": len(raw), "wall_seconds": perf_counter() - begin})


def fp64():
    import jax
    import jax.numpy as jnp
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert read_json(OUT / "gpu_status.json")["phase"] == "complete"
    if (OUT / "fp64_summary.csv").exists():
        raise FileExistsError("Preserve existing FP64 diagnostic before repeating")
    env = environment(require_gpu=False)
    env.update(jax_enable_x64=True, benchmark_source_sha256=file_hash(__file__),
               purpose="CPU FP64 accuracy diagnostic using GPU parallel-core formula; not GPU timings")
    assert env["source_sha256"] == read_json(OUT / "gpu_environment.json")["source_sha256"]
    write_json(OUT / "fp64_environment.json", env)
    cases = pd.read_csv(OUT / "gpu_summary.csv")
    assert len(cases) == 30
    rows, arrays = [], {}
    begin = perf_counter()
    for name, subset in cases.groupby("dataset", sort=False):
        data = checked_inputs(name)
        ex = CoxExplainer(data["beta"], background=data["background"], backend="logspace_jax",
                          memory_budget="8GB", block_size=1)
        assert shapley._jax_work_dtype(data["beta"]) == np.dtype(np.float64)
        ex._jax = shapley.ProductGamesShapleyJax()
        ex._jax._phi_logspace_core = ex._jax._phi_logspace_parallel_core
        probe = ex._jax._phi_logspace_parallel_core(
            jnp.ones((1, 2), dtype=jnp.float64), jnp.ones((1, 2), dtype=jnp.float64),
            jnp.array([0.5], dtype=jnp.float64), jnp.array([1.0], dtype=jnp.float64), 1e-100)
        assert np.asarray(probe).dtype == np.float64
        for case in subset.itertuples():
            tick = perf_counter()
            phi = ex.explain(data["X"][case.patient], m_q=int(case.m_q))
            seconds = perf_counter() - tick
            assert phi.dtype == np.float64 and np.isfinite(phi).all()
            absolute, relative = error_pair(phi, data["reference"][case.patient])
            rows.append({"dataset": name, "patient": case.patient, "eps": case.eps, "m_q": case.m_q,
                         "max_absolute_error_vs_fp64_reference": absolute,
                         "relative_l2_error_vs_fp64_reference": relative,
                         "within_eps": absolute <= case.eps, "seconds": seconds})
            arrays[f"{name}_p{case.patient}_eps_{case.eps}"] = phi
            pd.DataFrame(rows).to_csv(OUT / "fp64_summary.csv", index=False)
            print(f"CPU FP64 {name} p{case.patient} eps={case.eps:g}: error={absolute:.6g}; "
                  f"passed={absolute <= case.eps}", flush=True)
    np.savez_compressed(OUT / "fp64_attributions.npz", **arrays)
    write_json(OUT / "fp64_status.json", {"phase": "complete", "cases": len(rows),
               "all_within_tolerance": all(r["within_eps"] for r in rows),
               "wall_seconds": perf_counter() - begin})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gpu", "fp64"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"gpu": gpu, "fp64": fp64}[args.action]()
