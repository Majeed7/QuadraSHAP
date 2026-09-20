"""Study Cox explanation latency and sensitivity to the training background.

Saved four-row results remain untouched: changing the background changes the game.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games.shapley import ProductGamesShapleyJax
from quadrashap.product_games.budget import certify
from benchmarks.cox_gpu_experiment import environment, error_metrics, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmarks/results/cox_gpu_tolerance"
OUTPUT = ROOT / "benchmarks/results/cox_background_sensitivity"
DATASETS = ("gse24080", "tcga_lgg_methylation")
TOLERANCES = (1e-1, 1e-2, 1e-3, 1e-4)


def prepare():
    from benchmarks.experiment_data import TrainingPreprocessor, load_survival

    for name in DATASETS:
        folder = OUTPUT / name
        folder.mkdir(parents=True, exist_ok=True)
        old = OLD / name
        t = perf_counter()
        data = load_survival(name)
        with np.load(old / "model_and_preprocessing.npz") as saved:
            train, test, beta = saved["train_rows"], saved["test_rows"], saved["coef"]
            preprocessor = TrainingPreprocessor(*(saved[k] for k in ("median", "mean", "scale", "keep")))
        training = preprocessor.transform(data.X[train])
        with np.load(old / "explanation_inputs.npz") as saved:
            X, old_background, ids = saved["X"], saved["background"], saved["patient_ids"]
        old_indices = np.sort(np.random.default_rng(42).choice(len(train), size=4, replace=False))
        assert np.array_equal(training[old_indices], old_background)
        assert not np.intersect1d(train, test).size
        # Fixed random ordering allows nested 4/32/100/200/full comparisons.
        # The historical four rows are assessed separately and are not reused as
        # a reference for the changed game.
        order = np.random.default_rng(42).permutation(len(train))
        np.save(folder / "training_background.npy", training)
        np.savez_compressed(folder / "inputs.npz", beta=beta, X=X, patient_ids=ids,
                            background_order=order, training_rows=train, test_rows=test,
                            historical_background_indices=old_indices)
        selection = data.samples.iloc[train].copy()
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order))
        selection["background_selection_rank"] = rank
        selection["in_background_100"] = rank < 100
        selection.to_csv(folder / "background_selection.csv", index=False)
        fit = json.loads((old / "fit_summary.json").read_text())
        info = {"dataset": name, "n_total": fit["n"], "n_train": len(train), "n_test": len(test),
                "d": len(beta), "d_per_training_patient": len(beta) / len(train),
                "initial_pilot_background_size": 100, "primary_background_size": len(train),
                "primary_background_selection": "All training rows, uniformly weighted",
                "pilot_selection": "First 100 entries of seed-42 permutation of training rows; uniform sampling without replacement",
                "background_seed": 42, "patient_ids": ids.tolist(), "model_refitted": False,
                "historical_four_background_rows_reproduced_bitwise": True,
                "test_data_used_for_background": False, "preparation_seconds": perf_counter() - t,
                "source_sha256": {k: file_hash(old / k) for k in ("model_and_preprocessing.npz", "explanation_inputs.npz")}}
        write_json(folder / "design.json", info)
        print(f"{name}: n_train={len(train)}, d={len(beta):,}, all training-background patients prepared", flush=True)


def load(name):
    folder = OUTPUT / name
    with np.load(folder / "inputs.npz") as saved:
        result = {k: saved[k] for k in saved.files}
    result.update(name=name, folder=folder, design=json.loads((folder / "design.json").read_text()),
                  training=np.load(folder / "training_background.npy", mmap_mode="r"))
    return result


def explainer(exp, size=100, backend="logspace_jax", **kwargs):
    background = exp["training"][exp["background_order"][:size]]
    return CoxExplainer(exp["beta"], background=background, backend=backend, memory_budget="512MB", **kwargs)


def pilot():
    write_json(OUTPUT / "environment.json", environment(require_gpu=True))
    records = []
    for name in DATASETS:
        exp = load(name)
        ex = explainer(exp)
        for m in (32, 128, 1024):
            for repeat in range(2):
                t = perf_counter()
                phi = ex.explain(exp["X"][0], m_q=m)
                seconds = perf_counter() - t
                assert np.isfinite(phi).all()
                rec = {"dataset": name, "background_size": 100, "m_q": m, "repeat": repeat,
                       "seconds": seconds, "block_plan": asdict(ex.last_block_plan), "phi_norm": float(np.linalg.norm(phi))}
                records.append(rec)
                print(json.dumps(rec), flush=True)
        for eps in TOLERANCES:
            report = ex.node_budget(exp["X"][0], eps=eps)
            print(name, report, flush=True)
        write_json(OUTPUT / "pilot.json", records)


def stability():
    """Reuse each background-specific game for all subsets at the same 128 nodes.

    Differences here measure sensitivity to the empirical background, not error
    against an exact Shapley reference. GPU rounding is shared across comparisons.
    """
    write_json(OUTPUT / "environment.json", environment(require_gpu=True))
    records = []
    for name in DATASETS:
        exp = load(name)
        training = exp["training"]
        n, d = training.shape
        order = exp["background_order"]
        selections = {f"nested_{b}": order[:b] for b in (4, 32, 100, 200)}
        selections["full_training"] = np.arange(n)
        selections["historical_4"] = exp["historical_background_indices"]
        for seed in (43, 44, 45):
            selections[f"random100_seed{seed}"] = np.random.default_rng(seed).permutation(n)[:100]
        labels = list(selections)
        W = np.zeros((len(labels), n))
        for k, idx in enumerate(selections.values()):
            W[k, idx] = 1 / len(idx)
        risks = np.exp(training @ exp["beta"])
        full = labels.index("full_training")
        ex = CoxExplainer(exp["beta"], background=training, backend="logspace_jax", memory_budget="512MB")
        plan = ex.plan_blocks(128)
        backend = ProductGamesShapleyJax()
        for patient, x in enumerate(exp["X"]):
            values = np.zeros((len(labels), d))
            t = perf_counter()
            offset = 0
            for K, Ut, _ in ex.games(x, block_size=plan.block_size):
                phi = backend.phi_matrix_logspace(K, 128, Ut=Ut, node_block=plan.node_block)
                values += W[:, offset:offset + len(K)] @ phi
                offset += len(K)
            assert offset == n and np.isfinite(values).all()
            for k, label in enumerate(labels):
                idx = selections[label]
                records.append({"dataset": name, "patient": patient, "selection": label,
                                "background_size": len(idx), "background_mean_hazard": float(risks[idx].mean()),
                                "mean_hazard_ratio_to_full": float(risks[idx].mean() / risks.mean()),
                                "largest_hazard_fraction": float(risks[idx].max() / risks[idx].sum()),
                                "m_q": 128, "reference": "GPU 128-node attribution with all training background rows",
                                **error_metrics(values[k], values[full])})
            np.savez_compressed(exp["folder"] / f"stability_attributions_patient_{patient}.npz",
                                **dict(zip(labels, values)))
            print(f"{name}: background sensitivity for patient {patient} complete in {perf_counter()-t:.2f}s", flush=True)
            pd.DataFrame(records).to_csv(OUTPUT / "background_sensitivity.csv", index=False)


def references():
    """Tightly bounded float64 references for the complete training backgrounds."""
    from benchmarks.cox_float64_reference import reference_cox

    for name in DATASETS:
        exp = load(name)
        bg = exp["training"]
        ex = CoxExplainer(exp["beta"], background=bg, backend="prefix_scan_numpy", memory_budget="512MB")
        bounds = []
        for x in exp["X"]:
            bound = certify(ex.summarize(x), 64)
            assert np.isfinite(bound) and bound < 1e-10
            bounds.append(bound)
        def progress(k, total):
            if k % 50 == 0 or k == total:
                write_json(exp["folder"] / "reference_progress.json", {"completed_background_rows": k, "total": total})
                print(f"{name}: float64 reference {k}/{total} backgrounds", flush=True)
        t = perf_counter()
        phi = reference_cox(exp["beta"], exp["X"], bg, m_q=64, progress=progress)
        reference_seconds = perf_counter() - t
        np.save(exp["folder"] / "reference_full_64.npy", phi)
        t = perf_counter()
        higher = reference_cox(exp["beta"], exp["X"][:1], bg, m_q=96, progress=progress)[0]
        check_seconds = perf_counter() - t
        difference = float(np.max(np.abs(higher - phi[0])))
        assert difference < 1e-6, difference
        assert np.isfinite(phi).all()
        np.save(exp["folder"] / "reference_full_96_patient_0.npy", higher)
        write_json(exp["folder"] / "reference_validation.json", {
            "background_size": len(bg), "m_q": 64, "quadrature_bounds": bounds,
            "algorithm": "Independent float64 log1p/expm1 Cox quadrature",
            "reference_type": "Tightly bounded approximation, not full exact-degree quadrature",
            "crosscheck_m_q": 96, "crosscheck_patient": 0, "crosscheck_max_absolute_difference": difference,
            "reference_seconds": reference_seconds, "crosscheck_seconds": check_seconds})
        print(f"{name}: full-background float64 reference validated; 64/96-node max difference {difference:.3g}", flush=True)


def main_sweep():
    """Four GPU tolerances for three patients; CPU timing at eps=1e-3 for patient 0."""
    write_json(OUTPUT / "environment.json", environment(require_gpu=True))
    rows, raw = [], []
    for name in DATASETS:
        exp = load(name)
        reference = np.load(exp["folder"] / "reference_full_64.npy")
        bg = exp["training"]
        arrays = {}
        for backend in ("logspace_jax", "prefix_scan_numpy"):
            ex = CoxExplainer(exp["beta"], background=bg, backend=backend, memory_budget="512MB")
            configs = [(p, eps) for eps in TOLERANCES for p in range(3)] if backend.endswith("jax") else [(0, 1e-3)]
            for patient, eps in configs:
                times, budget_times = [], []
                for repeat in range(-1, 3):
                    t = perf_counter()
                    phi, report = ex.explain(exp["X"][patient], eps=eps, return_report=True)
                    seconds = perf_counter() - t
                    assert np.isfinite(phi).all()
                    raw.append({"dataset": name, "backend": backend, "patient": patient, "eps": eps,
                                "background_size": len(bg), "m_q": report.m_q, "repeat": repeat,
                                "seconds": seconds, "budget_seconds": report.seconds})
                    if repeat == -1:
                        first = seconds
                    else:
                        times.append(seconds); budget_times.append(report.seconds)
                metrics = error_metrics(phi, reference[patient])
                rows.append({"dataset": name, "n_train": len(bg), "d": len(exp["beta"]),
                             "background_size": len(bg), "backend": backend, "patient": patient, "eps": eps,
                             "m_q": report.m_q, "first_seconds": first, "median_seconds": float(np.median(times)),
                             "min_seconds": min(times), "max_seconds": max(times),
                             "median_budget_seconds": float(np.median(budget_times)),
                             "certified_quadrature_bound": report.bound, "efficiency_residual": report.efficiency_residual,
                             "reference": "Independent float64 64-node quadrature, certified bound <1e-10; patient 0 crosschecked at 96 nodes",
                             "observed_absolute_tolerance_met": metrics["max_absolute_error"] <= eps,
                             **metrics})
                arrays[f"{backend}_p{patient}_eps{eps}"] = phi
                pd.DataFrame(rows).to_csv(OUTPUT / "full_background_timings.csv", index=False)
                pd.DataFrame(raw).to_csv(OUTPUT / "raw_timings.csv", index=False)
                print(f"{name} {backend}, patient {patient}, eps={eps:g}: {np.median(times):.3f}s, "
                      f"{report.m_q} nodes, max error {metrics['max_absolute_error']:.3g}", flush=True)
        np.savez_compressed(exp["folder"] / "full_background_attributions.npz", **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "pilot", "stability", "references", "sweep"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"prepare": prepare, "pilot": pilot, "stability": stability,
         "references": references, "sweep": main_sweep}[args.action]()
