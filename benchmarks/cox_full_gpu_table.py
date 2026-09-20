"""Matched GPU approximation measurements for three held-out Cox patients."""
import argparse
from dataclasses import asdict
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games.shapley import _gauss_legendre_01_numpy
from benchmarks.cox_background_experiment import DATASETS, load
from benchmarks.cox_gpu_experiment import environment, error_metrics, write_json
from benchmarks.cox_gpu_table_followup import OUT


def run(memory_budget, block_size):
    OUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    write_json(OUT / "full_background_environment.json", environment(require_gpu=True))
    rows, raw, arrays, rules = [], [], {}, []
    for name in DATASETS:
        data = load(name)
        refs = np.load(data["folder"] / "reference_full_64.npy")
        ex = CoxExplainer(data["beta"], background=data["training"], backend="logspace_jax",
                          memory_budget=memory_budget, block_size=block_size)
        for eps in (1e-6, 1e-1, 1e-3):
            for patient, x in enumerate(data["X"]):
                times = []
                for repeat in (-1, 0, 1, 2):
                    tick = perf_counter()
                    phi, report = ex.explain(x, eps=eps, return_report=True)
                    seconds = perf_counter() - tick
                    assert np.isfinite(phi).all() and report.bound <= eps
                    raw.append({"dataset": name, "patient": patient, "eps": eps, "m_q": report.m_q,
                                "repeat": repeat, "seconds": seconds, "budget_seconds": report.seconds})
                    if repeat == -1:
                        first = seconds
                    else:
                        times.append(seconds)
                    print(f"{name} p{patient} eps={eps:g} repeat={repeat}: {seconds:.6f}s", flush=True)
                metrics = error_metrics(phi, refs[patient])
                rows.append({"dataset": name, "patient": patient, "patient_id": str(data["patient_ids"][patient]),
                             "B": len(data["training"]), "d": len(x), "eps": eps, "m_q": report.m_q,
                             "first_seconds": first, "median_seconds": float(np.median(times)),
                             "mean_seconds": float(np.mean(times)), "min_seconds": min(times), "max_seconds": max(times),
                             "bound": report.bound, "efficiency_residual": report.efficiency_residual,
                             "memory_budget": memory_budget, "block_size": block_size,
                             "plan": asdict(ex.last_block_plan),
                             "observed_tolerance_met": metrics["max_absolute_error"] <= eps, **metrics})
                arrays[f"{name}_p{patient}_eps{eps}"] = phi
                pd.DataFrame(rows).to_csv(OUT / "full_background_approximation.csv", index=False)
                pd.DataFrame(raw).to_csv(OUT / "full_background_raw.csv", index=False)
                np.savez_compressed(OUT / "full_background_approximation.npz", **arrays)
    for m in sorted({r["m_q"] for r in rows}):
        timings = []
        for repeat in range(5):
            _gauss_legendre_01_numpy.cache_clear()
            tick = perf_counter()
            nodes, weights = _gauss_legendre_01_numpy(m)
            timings.append(perf_counter() - tick)
            assert np.isfinite(nodes).all() and abs(weights.sum() - 1) < 1e-12
        rules.append({"m_q": m, "median_seconds": float(np.median(timings)),
                      "min_seconds": min(timings), "max_seconds": max(timings), "repeats": 5})
    pd.DataFrame(rules).to_csv(OUT / "full_background_rule_timings.csv", index=False)
    write_json(OUT / "full_background_approximation_summary.json", {
        "wall_seconds": perf_counter() - start, "cases": len(rows), "calls": len(raw),
        "patients_per_dataset": 3, "background": "all training patients, uniformly weighted",
        "memory_budget": memory_budget, "block_size": block_size,
        "table_aggregation": "mean and sample SD across the three per-patient median warm call times",
        "reference": "Independent FP64 64-node quadrature with certified bound <1e-10; patient0 crosschecked at96 nodes"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-budget", default="4GB")
    parser.add_argument("--block-size", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.memory_budget, args.block_size)
