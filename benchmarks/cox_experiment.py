"""Reusable steps for the executed Cox notebook; timings use perf_counter."""
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from benchmarks.cox_model import SampleSpaceRidgeCox
from benchmarks.experiment_data import load_survival, TrainingPreprocessor
from quadrashap.cox import CoxPHExplainer


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    return {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "pandas", "scikit-learn", "scikit-survival", "matplotlib")}}


def fit_dataset(name, output, *, seed=42, background_size=4, n_explain=3, ridge_per_feature=0.01):
    started = perf_counter()
    output = Path(output) / name
    output.mkdir(parents=True, exist_ok=True)
    tick = perf_counter()
    data = load_survival(name)
    load_seconds = perf_counter() - tick
    rows = np.arange(len(data.y))
    if "split" in data.samples:
        train = rows[data.samples["split"].eq("training")]
        test = rows[data.samples["split"].eq("validation")]
        split_kind = "Provided training/validation split"
    else:
        train, test = train_test_split(rows, test_size=0.25, random_state=seed, stratify=data.y["event"])
        train, test = np.sort(train), np.sort(test)
        split_kind = "75/25 split stratified by event, seed 42"
    tick = perf_counter()
    preprocessor = TrainingPreprocessor.fit(data.X[train])
    X_train = preprocessor.transform(data.X[train])
    X_test = preprocessor.transform(data.X[test])
    preprocessing_seconds = perf_counter() - tick
    d = X_train.shape[1]
    alpha = ridge_per_feature * d
    model = SampleSpaceRidgeCox(alpha=alpha).fit(X_train, data.y[train])
    tick = perf_counter()
    train_scores, test_scores = model.predict(X_train), model.predict(X_test)
    train_c = concordance_index_censored(data.y["event"][train], data.y["time"][train], train_scores)[0]
    test_c = concordance_index_censored(data.y["event"][test], data.y["time"][test], test_scores)[0]
    scoring_seconds = perf_counter() - tick
    selected_background = np.sort(np.random.default_rng(seed).choice(len(train), size=background_size, replace=False))
    # Fixed first validation rows: selection does not depend on outcomes or risk.
    selected_test = np.arange(min(n_explain, len(test)))
    tick = perf_counter()
    explainer = CoxPHExplainer(model.coef_, X_train[selected_background],
                               node_block_size=max(1, min(64, 32 * 1024**2 // (8 * d))))
    explainer_seconds = perf_counter() - tick
    features = data.feature_names[preprocessor.keep]
    summary = {"dataset": name, "n": len(data.y), "n_train": len(train), "n_test": len(test),
               "input_features": data.X.shape[1], "used_features": d,
               "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
               "training_rank": model.rank_, "rank_tolerance": model.rank_tolerance_,
               "train_events": int(data.y["event"][train].sum()), "test_events": int(data.y["event"][test].sum()),
               "split": split_kind, "seed": seed, "ridge_alpha": alpha, "ridge_per_feature": ridge_per_feature,
               "penalty_tuning": "Fixed in advance; no hyperparameter search", "ties": "Breslow",
               "train_c_index": float(train_c), "test_c_index": float(test_c),
               "background_size": background_size, "n_explain": len(selected_test),
               "exact_nodes": explainer.exact_nodes, "node_block_size": explainer.node_block_size,
               "load_seconds": load_seconds, "preprocessing_seconds": preprocessing_seconds,
               **model.timings_, "scoring_seconds": scoring_seconds, "explainer_setup_seconds": explainer_seconds,
               "fit_workflow_seconds": perf_counter() - started}
    sample_table = data.samples.copy()
    sample_table["experiment_split"] = ""
    sample_table.loc[train, "experiment_split"] = "training"
    sample_table.loc[test, "experiment_split"] = "test"
    sample_table["background"] = False
    sample_table.loc[train[selected_background], "background"] = True
    sample_table["explained"] = False
    sample_table.loc[test[selected_test], "explained"] = True
    sample_table.to_csv(output / "sample_roles.csv", index=False)
    pd.DataFrame({"feature_id": features, "coefficient": model.coef_}).to_csv(output / "coefficients.csv", index=False)
    np.savez_compressed(output / "model_and_preprocessing.npz", coef=model.coef_,
                        median=preprocessor.median, mean=preprocessor.mean, scale=preprocessor.scale,
                        keep=preprocessor.keep, train_rows=train, test_rows=test,
                        baseline_times=model.reduced_model_.cum_baseline_hazard_.x,
                        baseline_cumulative_hazard=model.reduced_model_.cum_baseline_hazard_.y)
    inputs = np.asarray(X_test[selected_test], dtype=np.float64)
    np.savez_compressed(output / "explanation_inputs.npz", background=explainer.background, X=inputs,
                        patient_ids=data.samples.patient_id.iloc[test[selected_test]].to_numpy(dtype=str))
    predictions = data.samples.iloc[test][["patient_id", "OS_time", "OS_event"]].copy()
    predictions["log_relative_hazard"] = test_scores
    predictions["relative_hazard"] = np.exp(test_scores)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    write_json(output / "fit_summary.json", summary)
    print(f"{name}: {len(train)} training patients, {d:,} features; fit {summary['fit_seconds']:.3f}s; validation C-index {test_c:.3f}", flush=True)
    return {"summary": summary, "explainer": explainer, "X": inputs, "features": features, "output": output}


def benchmark_approximate(experiment, nodes=(1, 2, 4, 8, 16, 32, 64, 128), repeats=3):
    results, timings = {}, []
    for m_q in nodes:
        # Keep first calls separately; cached-rule repetitions are used for medians.
        for patient, x in enumerate(experiment["X"]):
            first = experiment["explainer"].explain(x, m_q=m_q)
            results[(patient, m_q)] = first
            record = asdict(first)
            record.pop("values")
            timings.append({**record, "patient": patient, "repeat": -1, "phase": "first_call"})
            for repeat in range(repeats):
                result = experiment["explainer"].explain(x, m_q=m_q)
                record = asdict(result)
                record.pop("values")
                timings.append({**record, "patient": patient, "repeat": repeat, "phase": "cached_rule"})
        print(f"{experiment['summary']['dataset']}: approximate m_q={m_q} completed for {len(experiment['X'])} patients", flush=True)
    pd.DataFrame(timings).to_csv(experiment["output"] / "approximation_timings.csv", index=False)
    np.savez_compressed(experiment["output"] / "approximate_attributions.npz",
                        **{f"patient_{patient}_mq_{m_q}": value.values for (patient, m_q), value in results.items()})
    experiment["approximations"], experiment["timings"] = results, pd.DataFrame(timings)
    return experiment["timings"]


def benchmark_exact(experiment, *, timeout_seconds=600):
    print(f"Starting full exact rule: {experiment['explainer'].exact_nodes:,} nodes, {experiment['summary']['used_features']:,} features, {experiment['summary']['background_size']} background patients", flush=True)
    result = experiment["explainer"].explain(experiment["X"][0], m_q=None, timeout_seconds=timeout_seconds)
    record = asdict(result)
    record.pop("values")
    record.update(patient=0, repeats=1, reference="Algebraically exact Gauss-Legendre rule, float64")
    np.save(experiment["output"] / "exact_attributions_patient_0.npy", result.values)
    write_json(experiment["output"] / "exact_timing.json", record)
    experiment["exact"] = result
    print(f"Exact completed: {result.total_seconds:.3f}s total; {result.rule_seconds:.3f}s rule generation; {result.integration_seconds:.3f}s integration", flush=True)
    return record


def comparison_table(experiment):
    rows = []
    for (patient, m_q), result in experiment["approximations"].items():
        has_exact = patient == 0 and "exact" in experiment
        reference = experiment["exact"] if has_exact else experiment["approximations"][(patient, 128)]
        difference = result.values - reference.values
        timing = experiment["timings"]
        timing = timing[(timing.patient == patient) & (timing.m_q == m_q) & (timing.phase == "cached_rule")]
        top = np.argsort(np.abs(reference.values))[-20:]
        candidate_top = np.argsort(np.abs(result.values))[-20:]
        rows.append({"patient": patient, "m_q": m_q,
                     "reference": "exact" if has_exact else "128-node approximation",
                     "median_seconds": float(timing.total_seconds.median()),
                     "min_seconds": float(timing.total_seconds.min()), "max_seconds": float(timing.total_seconds.max()),
                     "relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(reference.values), 1e-300)),
                     "max_absolute_error": float(np.max(np.abs(difference))),
                     "top20_overlap": len(set(top) & set(candidate_top)) / 20,
                     "efficiency_residual": float(result.efficiency_residual),
                     "relative_efficiency_residual": float(abs(result.efficiency_residual) / max(abs(result.prediction), abs(result.base_value), 1e-300)),
                     "base_value": result.base_value, "prediction": result.prediction})
    frame = pd.DataFrame(rows)
    frame.to_csv(experiment["output"] / "comparison.csv", index=False)
    reference = experiment.get("exact", experiment["approximations"][(0, 128)])
    order = np.argsort(np.abs(reference.values))[-20:][::-1]
    pd.DataFrame({"feature_id": experiment["features"][order], "shapley_value": reference.values[order],
                  "reference": "exact" if "exact" in experiment else "128-node approximation"}).to_csv(
                      experiment["output"] / "top20_patient_0.csv", index=False)
    return frame
