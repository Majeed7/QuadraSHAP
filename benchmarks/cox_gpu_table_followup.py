"""Add eps=1e-6 GPU measurements without replacing the archived Cox experiments."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games.shapley import _gauss_legendre_01_numpy
from benchmarks.cox_gpu_experiment import (
    DATASETS, ROOT, environment, error_metrics, file_hash, load_experiment, write_json,
)
from benchmarks.cox_background_experiment import load as load_full

OUT = ROOT / "benchmarks/results/cox_gpu_tolerance/eps_1e6_followup"


def run():
    start = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    env = environment(require_gpu=True)
    # Reusing archived timings requires an unchanged numerical implementation.
    for folder in ("cox_gpu_tolerance", "cox_background_sensitivity"):
        archived = json.loads((ROOT / "benchmarks/results" / folder / "environment.json").read_text())
        assert env["source_sha256"] == archived["source_sha256"], folder
        assert env["packages"] == archived["packages"], folder
        assert env["devices"] == archived["devices"], folder
    env["followup_source_sha256"] = file_hash(Path(__file__))
    write_json(OUT / "environment.json", env)
    rows, raw, arrays, provenance = [], [], {}, {}
    for name in DATASETS:
        old = load_experiment(name)
        full = load_full(name)
        assert np.array_equal(old["coef"], full["beta"])
        assert np.array_equal(old["X"], full["X"])
        provenance[name] = {"historical_model": file_hash(old["folder"] / "model_and_preprocessing.npz"),
                            "historical_inputs": file_hash(old["folder"] / "explanation_inputs.npz"),
                            "full_inputs": file_hash(full["folder"] / "inputs.npz"),
                            "full_training_background": file_hash(full["folder"] / "training_background.npy")}
        configurations = [
            ("four", old["background"], old["reference_0"], "Archived float64 full exact-degree reference"),
            ("full", full["training"], np.load(full["folder"] / "reference_full_64.npy")[0],
             "Independent float64 64-node reference; quadrature bound <1e-10, crosschecked at 96 nodes"),
        ]
        for label, background, reference, qualification in configurations:
            ex = CoxExplainer(old["coef"], background=background, backend="logspace_jax", memory_budget="512MB")
            times, budgets = [], []
            for repeat in (-1, 0, 1, 2):
                t = perf_counter()
                phi, report = ex.explain(old["X"][0], eps=1e-6, return_report=True)
                seconds = perf_counter() - t
                assert np.isfinite(phi).all() and report.bound <= 1e-6
                raw.append({"dataset": name, "background": label, "B": len(background), "eps": 1e-6,
                            "repeat": repeat, "seconds": seconds, "budget_seconds": report.seconds})
                print(f"{name}, B={len(background)}, eps=1e-6, repeat={repeat}: {seconds:.6f}s", flush=True)
                if repeat == -1:
                    first = seconds
                else:
                    times.append(seconds)
                    budgets.append(report.seconds)
            metrics = error_metrics(phi, reference)
            rows.append({"dataset": name, "background": label, "B": len(background), "d": len(old["coef"]),
                         "patient": 0, "patient_id": str(old["ids"][0]), "eps": 1e-6,
                         "backend": "logspace_jax", "m_q": report.m_q, "first_seconds": first,
                         "median_seconds": float(np.median(times)), "min_seconds": min(times), "max_seconds": max(times),
                         "median_budget_seconds": float(np.median(budgets)), "bound": report.bound,
                         "efficiency_residual": report.efficiency_residual,
                         "observed_tolerance_met": metrics["max_absolute_error"] <= 1e-6,
                         "reference": qualification, "block_plan": asdict(ex.last_block_plan), **metrics})
            arrays[f"{name}_{label}"] = phi
            pd.DataFrame(rows).to_csv(OUT / "eps_1e6_results.csv", index=False)
            pd.DataFrame(raw).to_csv(OUT / "raw_timings.csv", index=False)
            np.savez_compressed(OUT / "attributions.npz", **arrays)
            write_json(OUT / "provenance.json", provenance)

    # These are CPU setup measurements, not GPU timings. Cache eviction forces
    # fresh construction and is performed outside each measured interval.
    old_rows = pd.read_csv(ROOT / "benchmarks/results/cox_gpu_tolerance/tolerance_summary.csv")
    full_rows = pd.read_csv(ROOT / "benchmarks/results/cox_background_sensitivity/full_background_timings.csv")
    nodes = {int(row["m_q"]) for row in rows}
    for frame in (old_rows, full_rows):
        selected = frame[frame.backend.eq("logspace_jax") & frame.patient.eq(0) & frame.eps.isin([1e-1, 1e-3])]
        nodes.update(selected.m_q.astype(int))
    rules, rule_raw = [], []
    for m in sorted(nodes):
        times = []
        for repeat in range(5):
            _gauss_legendre_01_numpy.cache_clear()
            t = perf_counter()
            x, w = _gauss_legendre_01_numpy(m)
            seconds = perf_counter() - t
            assert np.isfinite(x).all() and np.isfinite(w).all() and (w > 0).all()
            assert abs(w.sum() - 1) < 1e-12
            times.append(seconds)
            rule_raw.append({"m_q": m, "repeat": repeat, "seconds": seconds})
        rules.append({"m_q": m, "median_seconds": float(np.median(times)), "min_seconds": min(times),
                      "max_seconds": max(times), "repeats": 5, "device": "CPU", "cache": "cleared before each measurement"})
    pd.DataFrame(rules).to_csv(OUT / "rule_construction.csv", index=False)
    pd.DataFrame(rule_raw).to_csv(OUT / "rule_raw_timings.csv", index=False)
    write_json(OUT / "run_summary.json", {"wall_seconds": perf_counter() - start, "cases": len(rows),
                                         "recorded_calls": len(raw), "full_exact_rerun": False,
                                         "GPU": env["devices"], "all_attributions_finite": True})
    print(pd.DataFrame(rows)[["dataset", "B", "m_q", "median_seconds", "max_absolute_error", "observed_tolerance_met"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
