"""CPU FP64 diagnostic using the GPU's log-space reduction formula.

This is an accuracy check at the saved 1e-6 node counts, not an FP64 GPU
benchmark or a new full exact-degree reference computation.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games.shapley import ProductGamesShapleyJax, _jax_work_dtype
from benchmarks.cox_gpu_experiment import environment, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/results/cox_background30_gpu"
OUT = SOURCE / "fp64_precision_check"


def run():
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    if (OUT / "results.csv").exists():
        raise FileExistsError("Preserve the existing precision diagnostic before rerunning")
    OUT.mkdir(exist_ok=True)
    env = environment(require_gpu=False)
    env.update(jax_enable_x64=bool(jax.config.jax_enable_x64),
               diagnostic_source_sha256=file_hash(__file__),
               purpose="CPU FP64 accuracy diagnostic; same parallel log-space formula used by GPU",
               comparison="Independent saved 64-node FP64 reference; not full degree-exact FP64")
    write_json(OUT / "environment.json", env)
    selected = pd.read_csv(SOURCE / "summary.csv")
    selected = selected[(selected["mode"] == "tolerance") & (selected.eps == 1e-6)]
    assert len(selected) == 10
    rows, arrays = [], {}
    begin = perf_counter()
    for name, cases in selected.groupby("dataset", sort=False):
        folder = SOURCE / name
        with np.load(folder / "inputs.npz") as saved:
            beta, background, X = (saved[k] for k in ("beta", "background", "X"))
        assert _jax_work_dtype(beta) == np.dtype(np.float64)
        reference = np.load(folder / "reference_64.npy")
        ex = CoxExplainer(beta, background=background, backend="logspace_jax",
                          memory_budget="8GB", block_size=1)
        ex._jax = ProductGamesShapleyJax()
        # CPU dispatch normally chooses the sequential core. For this diagnostic,
        # explicitly select the same parallel formula used on non-CPU backends.
        ex._jax._phi_logspace_core = ex._jax._phi_logspace_parallel_core
        probe = ex._jax._phi_logspace_parallel_core(
            jnp.ones((1, 2), dtype=jnp.float64), jnp.ones((1, 2), dtype=jnp.float64),
            jnp.array([0.5], dtype=jnp.float64), jnp.array([1.0], dtype=jnp.float64), 1e-100)
        assert np.asarray(probe).dtype == np.float64
        for case in cases.itertuples():
            patient, m = int(case.patient), int(case.m_q)
            tick = perf_counter()
            phi = ex.explain(X[patient], m_q=m)
            seconds = perf_counter() - tick
            assert phi.dtype == np.float64 and np.isfinite(phi).all()
            delta = phi - reference[patient]
            error = float(np.max(np.abs(delta)))
            rows.append({"dataset": name, "patient": patient, "B": len(background),
                         "d": len(beta), "eps": case.eps, "m_q": m,
                         "certified_quadrature_bound": case.bound,
                         "max_absolute_error_vs_fp64_reference": error,
                         "relative_l2_error_vs_fp64_reference": float(np.linalg.norm(delta) / np.linalg.norm(reference[patient])),
                         "within_requested_tolerance": error <= case.eps,
                         "gpu_fp32_error_vs_same_reference": case.max_absolute_error,
                         "cpu_diagnostic_seconds_including_compilation": seconds})
            arrays[f"{name}_patient_{patient}"] = phi
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
            print(f"{name} p{patient}: m={m}, FP64 error={error:.6g}, eps={case.eps:g}, "
                  f"passed={error <= case.eps}, CPU diagnostic={seconds:.3f}s", flush=True)
    np.savez_compressed(OUT / "attributions.npz", **arrays)
    write_json(OUT / "run_summary.json", {"phase": "complete", "cases": len(rows),
               "all_within_tolerance": all(r["within_requested_tolerance"] for r in rows),
               "wall_seconds": perf_counter() - begin,
               "scope": "Ten 1e-6 CPU FP64 calls with the GPU parallel formula; no GPU FP64 timing or full-degree FP64 rerun"})


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
