"""Matched warmed CPU FP64/FP32 timings for the saved B30 Cox GPU sweep.

Uses fixed inputs, the GPU parallel-core formula, identical node selection,
and synchronized public-API timings. Does not alter production dtype policy.
"""
import argparse
import os
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games import shapley
from benchmarks.cox_gpu_experiment import environment, file_hash, write_json
from benchmarks.cox_tolerance_rerun import (
    SOURCE, OUT as GPU_SOURCE, DATASETS, checked_inputs, error_pair, read_json,
)

OUT = GPU_SOURCE / "cpu_precision_timing"


def run(precision):
    import jax
    import jax.numpy as jnp

    prefix = f"cpu_{precision}"
    dtype = np.dtype(np.float64 if precision == "fp64" else np.float32)
    assert jax.default_backend() == "cpu"
    assert bool(jax.config.jax_enable_x64) == (precision == "fp64")
    assert not jax.config.jax_disable_jit
    for suffix in ("summary.csv", "raw_timings.csv", "attributions.npz"):
        if (OUT / f"{prefix}_{suffix}").exists():
            raise FileExistsError(f"Preserve existing {prefix} outputs before rerunning")
    OUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    env = environment(require_gpu=False)
    gpu_env = read_json(GPU_SOURCE / "gpu_environment.json")
    assert env["source_sha256"] == gpu_env["source_sha256"]
    assert env["packages"] == gpu_env["packages"]
    env.update(jax_enable_x64=bool(jax.config.jax_enable_x64), work_dtype=str(dtype),
               core_formula="parallel logspace core also used by GPU",
               host_accumulation_dtype="float64", benchmark_source_sha256=file_hash(__file__),
               purpose="Warmed CPU timings including automatic node selection and synchronized output",
               thread_environment={k: os.environ.get(k) for k in
                                   ("XLA_FLAGS", "OMP_NUM_THREADS", "JAX_NUM_THREADS")})
    write_json(OUT / f"{prefix}_environment.json", env)
    provenance = {
        "input_sha256": {n: file_hash(SOURCE / n / "inputs.npz") for n in DATASETS},
        "reference_sha256": {n: file_hash(SOURCE / n / "reference_64.npy") for n in DATASETS},
        "gpu_source_sha256": {f: file_hash(GPU_SOURCE / f) for f in
                              ("gpu_summary.csv", "gpu_attributions.npz", "gpu_environment.json")},
        "exact_source_sha256": {f: file_hash(SOURCE / f) for f in
                                ("summary.csv", "attributions.npz", "design.json", "environment.json")},
        "background_size": 30, "distinct_patients_per_dataset": 5,
        "measured_calls_per_patient_method": 1,
        "warmup": "Once per dataset and selected node count before the first measured call",
        "timing_includes": "Automatic node selection, explanation, host accumulation and report calculations",
        "timing_excludes": "Model fit, warmup/JIT compilation, quadrature-rule construction",
        "exact_cpu_measured": False,
        "gpu_timings_reused": True,
    }
    if (OUT / "provenance.json").exists():
        assert read_json(OUT / "provenance.json") == provenance
    else:
        write_json(OUT / "provenance.json", provenance)

    cases = pd.read_csv(GPU_SOURCE / "gpu_summary.csv")
    assert len(cases) == 30 and len(cases.groupby(["dataset", "patient", "eps"])) == 30
    original_rule = shapley._gauss_legendre_01_numpy
    rules = {int(m): original_rule(int(m)) for m in cases.m_q.unique()}

    def cached_rule(m, dtype=np.float64):
        return tuple(a.astype(dtype, copy=False) for a in rules[int(m)])

    raw, rows, arrays = [], [], {}
    state = {"phase": "starting", "pid": os.getpid(), "precision": precision}
    shapley._gauss_legendre_01_numpy = cached_rule
    try:
        for name, subset in cases.groupby("dataset", sort=False):
            data = checked_inputs(name)
            assert shapley._jax_work_dtype(data["beta"]) == dtype
            ex = CoxExplainer(data["beta"], background=data["background"],
                              backend="logspace_jax", memory_budget="8GB", block_size=1)
            ex._jax = shapley.ProductGamesShapleyJax()
            parallel = ex._jax._phi_logspace_parallel_core
            # Verify actual kernel precision independently of FP64 public outputs.
            probe = parallel(jnp.ones((1, 2), dtype=dtype), jnp.ones((1, 2), dtype=dtype),
                             jnp.array([0.5], dtype=dtype), jnp.array([1.0], dtype=dtype), 1e-100)
            assert np.asarray(probe).dtype == dtype
            ex._jax._phi_logspace_core = parallel
            warmed = set()
            for case in subset.itertuples():
                patient, eps, m = int(case.patient), float(case.eps), int(case.m_q)
                x = data["X"][patient]
                assert ex.node_budget(x, eps=eps).m_q == m
                warmup_seconds = 0.0
                for phase in (("warmup", "measured") if m not in warmed else ("measured",)):
                    state = {"phase": "running", "pid": os.getpid(), "precision": precision,
                             "dataset": name, "patient": patient, "eps": eps,
                             "call_phase": phase, "measured_calls": len(rows)}
                    write_json(OUT / f"{prefix}_status.json", state)
                    tick = perf_counter()
                    if phase == "warmup":
                        phi, report = ex.explain(x, m_q=m), None
                    else:
                        phi, report = ex.explain(x, eps=eps, return_report=True)
                    seconds = perf_counter() - tick
                    assert np.isfinite(phi).all() and phi.dtype == np.float64
                    assert ex.last_block_plan.m_q == m
                    raw.append({"dataset": name, "patient": patient, "eps": eps, "m_q": m,
                                "phase": phase, "seconds": seconds,
                                "budget_seconds": 0.0 if report is None else report.seconds})
                    if phase == "warmup":
                        warmed.add(m)
                        warmup_seconds = seconds
                    pd.DataFrame(raw).to_csv(OUT / f"{prefix}_raw_timings.csv", index=False)
                assert report.bound <= eps and np.isclose(report.bound, case.bound, rtol=1e-12, atol=0)
                absolute, relative = error_pair(phi, data["reference"][patient])
                rows.append({"dataset": name, "patient": patient,
                             "patient_id": str(data["patient_ids"][patient]),
                             "B": 30, "d": len(data["beta"]), "eps": eps, "m_q": m,
                             "seconds": seconds, "warmup_seconds": warmup_seconds,
                             "bound": report.bound,
                             "max_absolute_error_vs_fp64_reference": absolute,
                             "relative_l2_error_vs_fp64_reference": relative,
                             "within_eps": absolute <= eps, "memory_budget": "8GB",
                             "block_size": 1, "node_block": ex.last_block_plan.node_block,
                             "dtype": str(dtype), "core_dtype": str(dtype)})
                arrays[f"{name}_p{patient}_eps_{eps}"] = phi
                pd.DataFrame(rows).to_csv(OUT / f"{prefix}_summary.csv", index=False)
                print(f"CPU {precision} {name} p{patient} eps={eps:g} m={m}: "
                      f"{seconds:.6f}s; error={absolute:.6g}; pass={absolute <= eps}", flush=True)
        np.savez_compressed(OUT / f"{prefix}_attributions.npz", **arrays)
    except BaseException:
        write_json(OUT / f"{prefix}_status.json", {**state, "phase": "interrupted",
                                                "measured_calls": len(rows)})
        raise
    finally:
        shapley._gauss_legendre_01_numpy = original_rule
    assert environment(require_gpu=False)["source_sha256"] == env["source_sha256"]
    write_json(OUT / f"{prefix}_status.json", {
        "phase": "complete", "precision": precision, "cases": len(rows),
        "measured_calls": len(rows), "completed_calls": len(raw),
        "all_within_tolerance": all(r["within_eps"] for r in rows),
        "wall_seconds": perf_counter() - start,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=("fp64", "fp32"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.precision)
