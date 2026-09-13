"""Run the full exact glioma game from saved coefficients and identical inputs."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import roots_legendre

from quadrashap.cox import CoxPHExplainer
from benchmarks.cox_experiment import environment, write_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def run_exact_followup(root):
    root = Path(root)
    output = root / "benchmarks/results/cox_survival/tcga_lgg_methylation"
    if (output / "exact_timing.json").exists():
        raise FileExistsError("An exact result already exists; preserve or move it before a new timed run")
    fit = json.loads((output / "fit_summary.json").read_text())
    with np.load(output / "model_and_preprocessing.npz", allow_pickle=False) as state:
        coef = state["coef"]
    with np.load(output / "explanation_inputs.npz", allow_pickle=False) as inputs:
        x, background, patient_id = inputs["X"][0], inputs["background"], str(inputs["patient_ids"][0])
    explainer = CoxPHExplainer(coef, background, node_block_size=fit["node_block_size"])
    assert coef.shape == x.shape == (396065,)
    assert background.shape == (4, 396065)
    assert explainer.exact_nodes == 198033
    provenance = {"patient_id": patient_id, "environment": environment(),
                  "source_files": {name: file_hash(output / name) for name in
                      ["model_and_preprocessing.npz", "explanation_inputs.npz", "approximate_attributions.npz", "approximation_timings.csv"]}}
    write_json(output / "exact_provenance.json", provenance)
    started = perf_counter()

    def progress(stage, **extra):
        record = {"stage": stage, "elapsed_seconds": perf_counter() - started,
                  "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
        write_json(output / "exact_progress.json", record)
        print(json.dumps(record), flush=True)

    progress("constructing_rule", nodes=explainer.exact_nodes)
    tick = perf_counter()
    nodes, weights = roots_legendre(explainer.exact_nodes)
    nodes, weights = (nodes + 1) / 2, weights / 2
    rule_seconds = perf_counter() - tick
    assert np.isfinite(nodes).all() and np.isfinite(weights).all()
    assert np.all(np.diff(nodes) > 0) and np.all(weights > 0)
    moments = np.array([np.dot(weights, nodes**k) for k in range(9)])
    np.testing.assert_allclose(moments, 1 / np.arange(1, 10), atol=2e-10, rtol=0)
    np.savez_compressed(output / "exact_quadrature_rule.npz", nodes=nodes, weights=weights)
    explainer._rules[explainer.exact_nodes] = (nodes, weights)
    progress("integrating", completed_fraction=0, rule_seconds=rule_seconds)

    def integration_progress(row, completed_nodes, games, m_q):
        progress("integrating", background_row=row, nodes_in_row=completed_nodes,
                 completed_fraction=(row * m_q + completed_nodes) / (games * m_q),
                 rule_seconds=rule_seconds)

    result = explainer.explain(x, progress_callback=integration_progress)
    computation_seconds = rule_seconds + result.total_seconds
    record = asdict(result)
    record.pop("values")
    record.update(rule_seconds=rule_seconds, total_seconds=computation_seconds,
                  followup_wall_seconds=perf_counter() - started,
                  patient=0, patient_id=patient_id, repeats=1, background_size=4,
                  efficiency_residual=float(result.efficiency_residual),
                  relative_efficiency_residual=float(abs(result.efficiency_residual) /
                      max(abs(result.base_value), abs(result.prediction), 1e-300)),
                  reference="Algebraically exact Gauss-Legendre rule, float64",
                  timing_definition="Rule construction plus full explanation call; intermediate rule validation and disk save excluded from total_seconds",
                  moment_max_absolute_error=float(np.max(np.abs(moments - 1 / np.arange(1, 10)))))
    np.save(output / "exact_attributions_patient_0.npy", result.values)
    write_json(output / "exact_timing.json", record)
    previous = pd.read_csv(output / "comparison.csv").query("patient == 0").copy()
    rows = []
    with np.load(output / "approximate_attributions.npz", allow_pickle=False) as approximate:
        exact_top = set(np.argsort(np.abs(result.values))[-20:])
        for row in previous.to_dict("records"):
            values = approximate[f"patient_0_mq_{int(row['m_q'])}"]
            difference = values - result.values
            row.update(reference="exact", relative_l2_error=float(np.linalg.norm(difference) / np.linalg.norm(result.values)),
                       max_absolute_error=float(np.abs(difference).max()),
                       top20_overlap=len(exact_top & set(np.argsort(np.abs(values))[-20:])) / 20,
                       speedup_vs_exact_cold=float(computation_seconds / row["median_seconds"]),
                       speedup_vs_exact_integration=float(result.integration_seconds / row["median_seconds"]))
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison_vs_exact.csv", index=False)
    coefficients = pd.read_csv(output / "coefficients.csv")
    top = np.argsort(np.abs(result.values))[-20:][::-1]
    pd.DataFrame({"feature_id": coefficients.feature_id.to_numpy()[top], "shapley_value": result.values[top],
                  "reference": "exact"}).to_csv(output / "top20_patient_0_exact.csv", index=False)
    progress("complete", completed_fraction=1, rule_seconds=rule_seconds,
             integration_seconds=result.integration_seconds, total_seconds=computation_seconds)
    return record, comparison
