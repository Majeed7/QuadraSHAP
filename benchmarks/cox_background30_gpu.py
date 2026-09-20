"""Matched B=30 GPU benchmarks: five held-out patients, one timed call each."""
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
from quadrashap.product_games.budget import certify
from benchmarks.cox_background_experiment import DATASETS, load
from benchmarks.cox_float64_reference import reference_cox
from benchmarks.cox_full_exact_gpu import validate_rule
from benchmarks.cox_gpu_experiment import ROOT, environment, error_metrics, file_hash, write_json

OUT = ROOT / "benchmarks/results/cox_background30_gpu"
PREVIOUS = ROOT / "benchmarks/results/cox_gpu_tolerance/eps_1e6_followup"


def prepare():
    from benchmarks.experiment_data import TrainingPreprocessor, load_survival

    designs = {}
    for name in DATASETS:
        old = load(name)
        data = load_survival(name)
        model_path = ROOT / "benchmarks/results/cox_gpu_tolerance" / name / "model_and_preprocessing.npz"
        with np.load(model_path) as saved:
            assert np.array_equal(saved["coef"], old["beta"])
            assert np.array_equal(saved["test_rows"], old["test_rows"])
            preprocessor = TrainingPreprocessor(*(saved[k] for k in ("median", "mean", "scale", "keep")))
        test_rows = old["test_rows"][:5]
        X = np.asarray(preprocessor.transform(data.X[test_rows]), dtype=np.float64)
        ids = data.samples.patient_id.iloc[test_rows].to_numpy(dtype=str)
        assert len(set(ids)) == 5 and not np.intersect1d(test_rows, old["training_rows"]).size
        assert np.array_equal(X[:len(old["X"])], old["X"])
        assert np.array_equal(ids[:len(old["patient_ids"])], old["patient_ids"])
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        indices = old["background_order"][:30]
        background = np.asarray(old["training"][indices])
        assert len(np.unique(indices)) == 30
        np.savez_compressed(folder / "inputs.npz", beta=old["beta"], X=X,
                            background=background, patient_ids=ids, test_rows=test_rows,
                            background_indices=indices, training_rows=old["training_rows"][indices])
        pd.read_csv(old["folder"] / "background_selection.csv").iloc[indices].to_csv(
            folder / "background_selection.csv", index=False)
        designs[name] = {"n_train": len(old["training"]), "d": len(old["beta"]),
                         "background_size": 30, "selection_seed": 42,
                         "selection": "First 30 entries of the saved seed-42 permutation of training rows, without replacement",
                         "patient_ids": ids.tolist(), "test_rows": test_rows.tolist(),
                         "patient_selection": "First five held-out rows of the saved split, without outcome-dependent selection",
                         "timed_calls_per_patient_method": 1,
                         "warmup": "One untimed explicit-node explanation per distinct (dataset, node count), before its first measured call",
                         "model_refitted": False, "model_sha256": file_hash(model_path),
                         "source_sha256": {str(old["folder"] / f): file_hash(old["folder"] / f)
                                           for f in ("inputs.npz", "training_background.npy")},
                         "prepared_inputs_sha256": file_hash(folder / "inputs.npz")}
        ex = CoxExplainer(old["beta"], background=background, backend="prefix_scan_numpy", memory_budget="512MB")
        bounds = [certify(ex.summarize(x), 64) for x in X]
        assert max(bounds) < 1e-10
        tick = perf_counter()
        ref = reference_cox(old["beta"], X, background, m_q=64)
        seconds = perf_counter() - tick
        tick = perf_counter()
        higher = reference_cox(old["beta"], X[:1], background, m_q=96)[0]
        check_seconds = perf_counter() - tick
        difference = float(np.max(np.abs(higher - ref[0])))
        assert np.isfinite(ref).all() and difference < 1e-6
        np.save(folder / "reference_64.npy", ref)
        np.save(folder / "reference_96_patient_0.npy", higher)
        write_json(folder / "reference_validation.json", {"B": 30, "m_q": 64, "quadrature_bounds": bounds,
                   "reference_seconds": seconds, "crosscheck_seconds": check_seconds,
                   "crosscheck_max_absolute_difference": difference,
                   "reference_type": "Independent float64 tightly bounded quadrature; not full exact-degree reference"})
        print(f"{name}: prepared five patients, B=30; 64/96-node float64 difference {difference:.3g}", flush=True)
    write_json(OUT / "design.json", designs)


def run():
    if (OUT / "raw_timings.csv").exists() or (OUT / "summary.csv").exists():
        raise FileExistsError("Saved B=30 timings already exist; preserve them before starting a new run")
    start = perf_counter()
    env = environment(require_gpu=True)
    env["benchmark_source_sha256"] = file_hash(__file__)
    write_json(OUT / "environment.json", env)
    design = json.loads((OUT / "design.json").read_text())
    original_rule = shapley._gauss_legendre_01_numpy
    exact_rules, setup = {}, {}
    for name in DATASETS:
        d = design[name]["d"]
        m = (d + 1) // 2
        if name == "gse24080":
            path = PREVIOUS / name / "quadrature_rule.npz"
            source_timing = json.loads((PREVIOUS / name / "patient_0/exact_gpu_timing.json").read_text())
            original_seconds = source_timing["rule_preparation"]["fresh_rule_seconds"]
        else:
            path = ROOT / "benchmarks/results/cox_survival" / name / "exact_quadrature_rule.npz"
            source_timing = json.loads((ROOT / "benchmarks/results/cox_gpu_tolerance" / name / "exact_gpu_timing.json").read_text())
            original_seconds = source_timing["rule_seconds"]
        tick = perf_counter()
        with np.load(path) as rule:
            nodes, weights = rule["nodes"], rule["weights"]
        load_seconds = perf_counter() - tick
        validation = validate_rule(nodes, weights, m)
        nodes.setflags(write=False); weights.setflags(write=False)
        exact_rules[m] = nodes, weights
        setup[name] = {"exact_nodes": m, "source": str(path), "source_sha256": file_hash(path),
                       "original_construction_seconds": original_seconds,
                       "cached_rule_load_seconds": load_seconds, **validation}
    write_json(OUT / "rule_setup.json", setup)

    def cached_rule(m, dtype=np.float64):
        if int(m) not in exact_rules:
            exact_rules[int(m)] = original_rule(m)
        return tuple(a.astype(dtype, copy=False) for a in exact_rules[int(m)])

    rows, raw, arrays = [], [], {}
    status = {"phase": "initializing", "pid": os.getpid()}
    shapley._gauss_legendre_01_numpy = cached_rule
    try:
        for name in DATASETS:
            folder = OUT / name
            assert file_hash(folder / "inputs.npz") == design[name]["prepared_inputs_sha256"]
            with np.load(folder / "inputs.npz") as saved:
                beta, background, X, ids = (saved[k] for k in ("beta", "background", "X", "patient_ids"))
            refs = np.load(folder / "reference_64.npy")
            assert len(X) == len(ids) == len(refs) == 5
            ex = CoxExplainer(beta, background=background, backend="logspace_jax", memory_budget="8GB", block_size=1)
            warmed_nodes = set()
            for mode, eps in (("tolerance", 1e-6), ("tolerance", 1e-1), ("tolerance", 1e-3), ("exact", None)):
                for patient, x in enumerate(X):
                    m = (len(beta) + 1) // 2 if mode == "exact" else ex.node_budget(x, eps=eps).m_q
                    warmup_seconds = 0.0
                    phases = ("warmup", "measured") if m not in warmed_nodes else ("measured",)
                    for phase in phases:
                        status = {"phase": "running", "pid": os.getpid(), "dataset": name,
                                  "patient": patient, "mode": mode, "eps": eps, "call_phase": phase,
                                  "completed_calls": len(raw), "measured_calls": len(rows), "total_measured_calls": 40}
                        write_json(OUT / "run_status.json", status)
                        tick = perf_counter()
                        if phase == "warmup" or mode == "exact":
                            phi = ex.explain(x, m_q=m if phase == "warmup" else "exact")
                            report = None
                        else:
                            phi, report = ex.explain(x, eps=eps, return_report=True)
                        seconds = perf_counter() - tick
                        assert np.isfinite(phi).all()
                        assert ex.last_block_plan.m_q == m
                        raw.append({"dataset": name, "patient": patient, "mode": mode, "eps": eps,
                                    "m_q": m, "phase": phase, "seconds": seconds,
                                    "budget_seconds": report.seconds if report is not None else 0.0})
                        if phase == "warmup":
                            warmup_seconds = seconds
                            warmed_nodes.add(m)
                        pd.DataFrame(raw).to_csv(OUT / "raw_timings.csv", index=False)
                        print(f"{name} B=30 p{patient} {mode} eps={eps} {phase}: {seconds:.6f}s", flush=True)
                    metrics = error_metrics(phi, refs[patient])
                    if report is not None:
                        assert report.bound <= eps
                    rows.append({"dataset": name, "patient": patient, "patient_id": str(ids[patient]),
                                 "B": 30, "d": len(beta), "mode": mode, "eps": eps, "m_q": m,
                                 "seconds": seconds, "warmup_seconds": warmup_seconds,
                                 "bound": report.bound if report is not None else None,
                                 "observed_tolerance_met": metrics["max_absolute_error"] <= eps if eps is not None else None,
                                 "memory_budget": "8GB", "block_size": 1, **metrics})
                    arrays[f"{name}_p{patient}_{mode}_{eps}"] = phi
                    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
                    np.savez_compressed(OUT / "attributions.npz", **arrays)
    except BaseException:
        write_json(OUT / "run_status.json", {**status, "phase": "interrupted", "completed_calls": len(raw)})
        raise
    finally:
        shapley._gauss_legendre_01_numpy = original_rule

    rules = []
    for m in sorted({r["m_q"] for r in rows if r["mode"] == "tolerance"}):
        times = []
        for _ in range(5):
            original_rule.cache_clear()
            tick = perf_counter()
            original_rule(m)
            times.append(perf_counter() - tick)
        rules.append({"m_q": m, "median_seconds": float(np.median(times)),
                      "min_seconds": min(times), "max_seconds": max(times), "repeats": 5})
    pd.DataFrame(rules).to_csv(OUT / "small_rule_timings.csv", index=False)
    write_json(OUT / "run_status.json", {"phase": "complete", "pid": os.getpid(),
                                         "completed_calls": len(raw), "measured_calls": len(rows), "total_measured_calls": 40,
                                         "wall_seconds": perf_counter() - start, "cases": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"prepare": prepare, "run": run}[args.action]()
