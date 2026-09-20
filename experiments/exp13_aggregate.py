"""Aggregate saved matched-game references and SHAP errors, without reruns."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("rotten_tomatoes", "sst2", "sms_spam", "emotion")
OUT = ROOT / "results" / "exp13_interventional_accuracy_summary"


def read_csv(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    combined = []
    diagnostics = []
    summary = []
    for dataset in DATASETS:
        reference_dir = ROOT / "results" / f"exp13_interventional_accuracy_{dataset}"
        baseline_dir = ROOT / "results" / f"exp12_shap_background_{dataset}"
        rows = read_csv(reference_dir / "records.csv")
        baseline = read_csv(baseline_dir / "records.csv")
        baseline_lookup = {(r["method_key"], int(r["instance"])): r for r in baseline}
        if len(rows) != 40:
            raise ValueError(f"expected 40 matched rows for {dataset}, got {len(rows)}")
        for instance in range(20):
            info = json.loads((reference_dir / "raw" / f"reference_instance{instance:03d}.json").read_text())
            diagnostics.append(info)
        for row in rows:
            method_key = "kernel" if row["method"] == "KernelSHAP" else "sampling"
            prior = baseline_lookup[(method_key, int(row["instance"]))]
            if prior["status"] != "complete":
                raise ValueError(f"incomplete saved baseline for {dataset} text {row['instance']}")
            row["baseline_seconds_end_to_end"] = prior["seconds_end_to_end"]
            row["baseline_model_rows"] = prior["actual_model_rows"]
            row["baseline_nonzero_attributions"] = prior["n_nonzero_phi"]
            row["baseline_efficiency_residual"] = prior["efficiency_residual"]
            combined.append(row)
        for method in ("KernelSHAP", "SamplingSHAP"):
            own = [r for r in rows if r["method"] == method]
            other = {int(r["instance"]): r for r in rows if r["method"] != method}
            errors = np.array([float(r["max_abs_error_to_reference"]) for r in own])
            rel_l2 = np.array([float(r["relative_l2_error_to_reference"]) for r in own])
            baseline_times = np.array([float(baseline_lookup[("kernel" if method == "KernelSHAP" else "sampling", int(r["instance"]))]["seconds_end_to_end"]) for r in own])
            diags = diagnostics[-20:]
            summary.append({
                "dataset": dataset,
                "method": method,
                "n_instances": len(own),
                "n_background": 30,
                "requested_nsamples": 1000,
                "reference_requested_eps": 1e-6,
                "reference_median_m_q": float(np.median([r["m_q"] for r in diags])),
                "reference_m_q_min": min(r["m_q"] for r in diags),
                "reference_m_q_max": max(r["m_q"] for r in diags),
                "reference_max_certified_bound": max(r["certified_bound"] for r in diags),
                "reference_max_extra_node_gap": max((r["max_abs_change_at_extra_nodes"] or 0.0) for r in diags),
                "reference_max_efficiency_residual": max(r["reference_efficiency_residual"] for r in diags),
                "reference_median_seconds": float(np.median([r["seconds_reference"] for r in diags])),
                "median_max_abs_error_to_reference": float(np.median(errors)),
                "mean_max_abs_error_to_reference": float(np.mean(errors)),
                "p90_max_abs_error_to_reference": float(np.quantile(errors, .9)),
                "worst_max_abs_error_to_reference": float(np.max(errors)),
                "median_relative_l2_error_to_reference": float(np.median(rel_l2)),
                "median_baseline_seconds": float(np.median(baseline_times)),
                "n_lower_error_than_other_baseline": sum(
                    float(r["max_abs_error_to_reference"]) <
                    float(other[int(r["instance"])]["max_abs_error_to_reference"])
                    for r in own
                ),
            })
    write_csv(OUT / "all_instances.csv", combined)
    write_csv(OUT / "reference_diagnostics.csv", diagnostics)
    write_csv(OUT / "method_summary.csv", summary)
    for row in summary:
        print(f"{row['dataset']} {row['method']}: median {row['median_max_abs_error_to_reference']:.6g}, "
              f"worst {row['worst_max_abs_error_to_reference']:.6g}, "
              f"lower error on {row['n_lower_error_than_other_baseline']}/20 texts")
    print(f"Saved three replottable CSVs in {OUT}")


if __name__ == "__main__":
    main()
