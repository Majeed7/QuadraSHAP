"""Render the finished 30-background Cox GPU benchmark; never run experiments.

Each displayed time aggregates five calls, one for each of five distinct
held-out patients. Node-shape warm-ups are recorded separately and excluded.
Incomplete inputs or a different experimental protocol fail before any writes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/cox_background30_gpu"
COHORTS = {"gse24080": {"label": "GSE24080", "n_train": 339, "d": 54675},
           "tcga_lgg_methylation": {"label": "TCGA LGG methylation", "n_train": 383, "d": 396065}}
MODES = ("exact", "1e-1", "1e-3", "1e-6")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(value, label, positive=False):
    result = float(value)
    require(math.isfinite(result) and (result > 0 if positive else result >= 0),
            f"Nonfinite or invalid {label}: {value}")
    return result


def mode_key(row):
    if row["mode"] == "exact":
        require(row.get("eps", "") == "", "Exact rows must not contain a requested tolerance")
        return "exact"
    require(row["mode"] == "tolerance", f"Unexpected mode: {row['mode']}")
    eps = numeric(row["eps"], "requested tolerance", positive=True)
    matches = [mode for mode in MODES[1:] if math.isclose(eps, float(mode), rel_tol=1e-12)]
    require(len(matches) == 1, f"Unexpected requested tolerance: {eps}")
    return matches[0]


def case_key(row):
    return row["dataset"], int(row["patient"]), mode_key(row)


def node_range(nodes):
    """Plain text range, intentionally outside math mode in LaTeX."""
    low, high = min(nodes), max(nodes)
    return f"{low:,}" if low == high else f"{low:,}--{high:,}"


def check_inputs(folder):
    summaries = read_csv(folder / "summary.csv")
    raw = read_csv(folder / "raw_timings.csv")
    design = read_json(folder / "design.json")
    environment = read_json(folder / "environment.json")
    status = read_json(folder / "run_status.json")
    require(status["phase"] == "complete" and int(status["cases"]) == 40
            and int(status["measured_calls"]) == 40, "Benchmark has not completed all 40 measured cases")
    require(environment["backend"].lower() == "metal" and not environment["jit_disabled"],
            "This table requires JAX-Metal GPU execution with JIT enabled")
    require(environment.get("source_sha256") and environment.get("packages"),
            "Missing numerical-source/package provenance")
    for dataset, spec in COHORTS.items():
        details = design[dataset]
        require(int(details["background_size"]) == 30 and int(details["selection_seed"]) == 42,
                f"Expected 30 backgrounds selected with seed 42: {dataset}")
        require(int(details["n_train"]) == spec["n_train"] and int(details["d"]) == spec["d"],
                f"Training sample/feature count mismatch: {dataset}")
        require(len(details["patient_ids"]) == 5 and len(set(map(str, details["patient_ids"]))) == 5,
                f"Expected five distinct held-out patients: {dataset}")
        require(details["source_sha256"], f"Missing input provenance: {dataset}")
    expected = {(dataset, patient, mode) for dataset in COHORTS for patient in range(5) for mode in MODES}
    require(len(summaries) == 40 and sum(r["phase"] == "measured" for r in raw) == 40,
            "Results incomplete: require 40 cases and exactly 40 measured calls")
    grouped_raw, warmed, measured_shapes = {}, set(), set()
    for row in raw:
        key = case_key(row)
        require(key in expected, f"Unexpected raw timing case: {key}")
        numeric(row["seconds"], "GPU API time", positive=True)
        numeric(row["budget_seconds"], "node-budget selection time")
        shape = row["dataset"], int(row["m_q"])
        require(row["phase"] in ("warmup", "measured"), f"Unknown timing phase: {row['phase']}")
        if row["phase"] == "warmup":
            require(shape not in warmed and shape not in measured_shapes,
                    f"Duplicate or late node-count warm-up: {shape}")
            require(float(row["budget_seconds"]) == 0, "Warm-ups use fixed nodes without automatic selection")
            warmed.add(shape)
        else:
            require(shape in warmed, f"Missing prior node-count warm-up: {shape}")
            measured_shapes.add(shape)
        grouped_raw.setdefault(key, []).append(row)
    require(warmed == measured_shapes, "A warm-up has no corresponding measured node count")
    require(len(raw) == 40 + len(warmed), "Expected one warm-up per distinct dataset/node count")
    cases = {}
    for row in summaries:
        key = case_key(row)
        require(key in expected and key not in cases, f"Duplicate/unexpected summary: {key}")
        dataset, patient, mode = key
        spec = COHORTS[dataset]
        require(int(row["B"]) == 30 and int(row["d"]) == spec["d"], f"Background/dimension mismatch: {key}")
        require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                f"Expected the agreed 8 GiB / one-background-row plan: {key}")
        require(str(row["patient_id"]) == str(design[dataset]["patient_ids"][patient]),
                f"Patient ID mismatch: {key}")
        calls = grouped_raw.get(key, [])
        measured = [r for r in calls if r["phase"] == "measured"]
        warmups = [r for r in calls if r["phase"] == "warmup"]
        require(len(measured) == 1 and len(warmups) <= 1,
                f"Expected exactly one measured explanation for this patient/method: {key}")
        require(all(int(r["m_q"]) == int(row["m_q"]) for r in calls), f"Node count changed: {key}")
        nodes = int(row["m_q"])
        require(nodes > 0, f"Invalid node count: {key}")
        seconds = float(measured[0]["seconds"])
        expected_stats = {"seconds": seconds,
                          "warmup_seconds": float(warmups[0]["seconds"]) if warmups else 0.0}
        for field, target in expected_stats.items():
            observed = numeric(row[field], field)
            require(math.isclose(observed, target, rel_tol=1e-9, abs_tol=1e-10),
                    f"Summary {field} disagrees with raw calls: {key}")
        error = numeric(row["max_absolute_error"], "maximum absolute error")
        relative_error = numeric(row["relative_l2_error"], "relative L2 error")
        if mode == "exact":
            require(nodes == (spec["d"] + 1) // 2, f"Wrong exact-degree node count: {key}")
            require(all(float(r["budget_seconds"]) == 0 for r in calls),
                    f"Unexpected automatic node-budget calculation for exact mode: {key}")
            bound, eps, passed = "", "", ""
        else:
            eps = float(mode)
            bound = numeric(row["bound"], "certified quadrature bound")
            require(bound <= eps, f"Requested quadrature bound not satisfied: {key}")
            passed = error <= eps
            require(str(row["observed_tolerance_met"]).lower() == str(passed).lower(),
                    f"Saved observed tolerance status disagrees with error: {key}")
        cases[key] = {"dataset": dataset, "patient": patient, "patient_id": row["patient_id"],
                      "mode": mode, "B": 30, "d": spec["d"], "m_q": nodes, "seconds": seconds,
                      "warmup_seconds": expected_stats["warmup_seconds"],
                      "max_absolute_error": error, "relative_l2_error": relative_error,
                      "eps": eps, "quadrature_bound": bound, "observed_tolerance_met": passed}
    require(set(cases) == expected and set(grouped_raw) == expected, "Missing benchmark configurations")
    return cases, design, environment


def build(folder):
    cases, design, environment = check_inputs(folder)
    rule_setup = read_json(folder / "rule_setup.json")
    exact_setup = {}
    for dataset, spec in COHORTS.items():
        setup = rule_setup[dataset]
        require(int(setup["exact_nodes"]) == (spec["d"] + 1) // 2 and setup["source"],
                f"Wrong or missing exact-rule provenance: {dataset}")
        exact_setup[dataset] = numeric(setup["original_construction_seconds"], "historical rule construction", positive=True)
        numeric(setup["cached_rule_load_seconds"], "cached rule loading")
    small_rules = read_csv(folder / "small_rule_timings.csv")
    by_nodes = {int(row["m_q"]): row for row in small_rules}
    required_nodes = {row["m_q"] for row in cases.values() if row["mode"] != "exact"}
    require(len(by_nodes) == len(small_rules) and set(by_nodes) == required_nodes,
            "Fresh approximate-rule measurements are incomplete or duplicated")
    small_ms = []
    for row in small_rules:
        require(int(row["repeats"]) == 5, "Require five fresh constructions for each approximate rule")
        low, median, high = [numeric(row[field], field, positive=True)
                             for field in ("min_seconds", "median_seconds", "max_seconds")]
        require(low <= median <= high, "Inconsistent approximate-rule construction times")
        small_ms.append(1000 * median)

    aggregate, accuracy, indexed = [], [], {}
    for mode in MODES:
        for dataset in COHORTS:
            group = [cases[(dataset, p, mode)] for p in range(5)]
            times = [row["seconds"] for row in group]
            require(len(times) == 5, "Table entries must aggregate five distinct patients")
            result = {"dataset": dataset, "mode": mode, "background_size": 30,
                      "n_train": COHORTS[dataset]["n_train"], "features": COHORTS[dataset]["d"],
                      "patients": 5, "measured_calls_per_patient": 1, "timed_calls": 5,
                      "nodes_min": min(r["m_q"] for r in group), "nodes_max": max(r["m_q"] for r in group),
                      "gpu_seconds_mean": statistics.mean(times), "gpu_seconds_sample_sd": statistics.stdev(times),
                      "memory_budget": "8GB", "background_block_size": 1}
            aggregate.append(result)
            indexed[(dataset, mode)] = result
            accuracy.extend(dict(row) for row in group)
    for row in aggregate:
        exact_mean = indexed[(row["dataset"], "exact")]["gpu_seconds_mean"]
        row["speedup_vs_exact_mean"] = exact_mean / row["gpu_seconds_mean"]

    # A secondary matched-GPU comparison complements the independent FP64
    # accuracy measurements. Degree-exact FP32 output is not ground truth.
    gpu_comparison = []
    with np.load(folder / "attributions.npz", allow_pickle=False) as saved:
        for dataset, spec in COHORTS.items():
            for patient in range(5):
                exact = saved[f"{dataset}_p{patient}_exact_None"]
                require(exact.shape == (spec["d"],) and np.isfinite(exact).all(),
                        f"Invalid saved degree-exact attribution: {dataset}, patient {patient}")
                exact_norm = float(np.linalg.norm(exact))
                for mode in MODES[1:]:
                    eps = float(mode)
                    approximation = saved[f"{dataset}_p{patient}_tolerance_{eps}"]
                    require(approximation.shape == exact.shape and np.isfinite(approximation).all(),
                            f"Invalid saved approximate attribution: {dataset}, patient {patient}, {mode}")
                    difference = approximation - exact
                    gpu_comparison.append({
                        "dataset": dataset, "patient": patient,
                        "patient_id": cases[(dataset, patient, mode)]["patient_id"],
                        "background_size": 30, "eps": eps,
                        "approximate_nodes": cases[(dataset, patient, mode)]["m_q"],
                        "exact_degree_nodes": cases[(dataset, patient, "exact")]["m_q"],
                        "max_absolute_difference_vs_gpu_exact": float(np.max(np.abs(difference))),
                        "relative_l2_difference_vs_gpu_exact": float(np.linalg.norm(difference) / max(exact_norm, 1e-300)),
                        "comparison_reference": "Same-patient degree-exact GPU FP32 output; not ground truth"})
    failures = sum(not row["observed_tolerance_met"] for row in cases.values() if row["mode"] != "exact")
    latex_rows, markdown_rows = [], []
    for mode in MODES:
        latex_label = "Exact degree" if mode == "exact" else rf"$\varepsilon=10^{{{int(math.log10(float(mode)))}}}$"
        latex_cells, markdown_cells = [latex_label], ["Exact degree" if mode == "exact" else mode]
        for dataset in COHORTS:
            stats = indexed[(dataset, mode)]
            nodes = node_range([stats["nodes_min"], stats["nodes_max"]])
            mean, sd = stats["gpu_seconds_mean"], stats["gpu_seconds_sample_sd"]
            latex_cells += [nodes, f"${mean:.3f} \\pm {sd:.3f}$"]
            markdown_cells += [nodes.replace("--", "–"), f"{mean:.3f} ± {sd:.3f}"]
        latex_rows.append(" & ".join(latex_cells) + r" \\")
        markdown_rows.append("| " + " | ".join(markdown_cells) + " |")
    caption = r"""Cox explanation times on an Apple M4 Pro GPU (JAX-Metal, FP32),
using 30 fixed training-background samples. GSE24080 and TCGA LGG have
$(n_{\mathrm{train}},d)=(339,54{,}675)$ and $(383,396{,}065)$, respectively.
Times are mean $\pm$ sample standard deviation across five distinct held-out
patients, with one synchronized call per patient and method. Node ranges span
patients. Warm-up and cached rule construction are excluded; automatic node
selection is included. ``Exact degree'' denotes exact quadrature in real
arithmetic; the requested absolute quadrature tolerance $\varepsilon$ excludes FP32 rounding.
""" + f"Measured absolute errors exceeded the requested tolerance in {failures}/30 approximate cases.\n" + \
        f"Previously measured CPU construction of the cached exact rules took\n" + \
        f"{exact_setup['gse24080']:.2f}\\,s (GSE24080) and {exact_setup['tcga_lgg_methylation']:.2f}\\,s (TCGA).\n" + \
        f"Fresh approximate-rule construction took {min(small_ms):.3f}--{max(small_ms):.3f}\\,ms\n" + \
        "(range of per-rule medians over five constructions)."
    latex = r"""% Requires \usepackage{booktabs}. Node ranges are plain text outside math mode.
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{GSE24080} & \multicolumn{2}{c}{TCGA LGG methylation} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & Nodes & GPU time (s) & Nodes & GPU time (s) \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\caption{""" + caption + r"""}
\label{tab:cox-gpu-background30}
\end{table}
"""
    accuracy_rows = []
    for mode in MODES:
        for dataset, spec in COHORTS.items():
            group = [cases[(dataset, p, mode)] for p in range(5)]
            passed = "n/a" if mode == "exact" else f"{sum(r['observed_tolerance_met'] for r in group)}/5"
            accuracy_rows.append(f"| {spec['label']} | {mode} | {max(r['max_absolute_error'] for r in group):.6g} | "
                                 f"{max(r['relative_l2_error'] for r in group):.6g} | {passed} |")
    speedup_lines = [
        f"- {spec['label']}: " + "; ".join(
            f"{mode}: {indexed[(dataset, mode)]['speedup_vs_exact_mean']:.1f}×"
            for mode in MODES[1:]) + "."
        for dataset, spec in COHORTS.items()]
    readme = f"""# Cox GPU timings with 30 background samples and five held-out patients

Every configuration uses the same **30 training-background samples**, selected with seed 42,
and the same **five distinct held-out patients** per cohort. The full training sample counts remain 339
(GSE24080, 54,675 features) and 383 (TCGA LGG methylation, 396,065 features): B=30 is the
background count, not the training sample count. Saved models and preprocessing are reused.

| Method | GSE24080 nodes | GPU seconds, mean ± SD | TCGA LGG nodes | GPU seconds, mean ± SD |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(markdown_rows)}

Speedup is the **ratio of mean exact latency to mean approximation latency** (the exact row is 1×),
not the mean of per-patient ratios:

{chr(10).join(speedup_lines)}

Each entry is the mean and **sample SD across five patients**, each explained **once per method**.
There are no repeated measured explanations of a patient/method. One fixed-node warm-up per distinct
dataset/node count is recorded but excluded; it precedes that node count's first measured call.
The spread reflects the five observed patient timings, not a confidence interval or a population guarantee.
Node ranges span these patients. All 40 configurations use the same public synchronized explanation API, JAX-Metal FP32 on Apple M4 Pro,
8 GiB memory planning, and one background row per block. Calls include host factor preparation,
device transfers, GPU computation and conversion/accumulation on the host; these are not pure kernel times.
Approximation calls additionally include the automatic node-budget calculation and its report;
exact calls use the full degree-exact rule without that report. Both use cached rules.
Loading data, model fitting, explainer construction, warm-up and quadrature-rule construction are excluded.

## Separate quadrature setup

Both large exact rules were loaded from validated saved files. Their historical CPU construction times
were {exact_setup['gse24080']:.6f} s and {exact_setup['tcga_lgg_methylation']:.6f} s, respectively.
This cost is independent of patient/background count and can be amortized through reuse. Fresh small-rule
construction took {min(small_ms):.6f}–{max(small_ms):.6f} ms, the range of medians over five constructions
per distinct node count. This is construction of abscissae and weights; it is not the node-budget
selection already included in approximate API timing. `rule_setup.json` records rule sources and load time.

## Accuracy and interpretation

Degree-exact quadrature removes integration error in real arithmetic, not FP32 rounding. Likewise,
the requested absolute per-feature tolerance controls quadrature error only. **{failures}/30 approximate
cases exceeded the requested total absolute error**. More nodes alone do not remove a rounding floor.
The recorded errors use the benchmark driver's reference for this same 30-row background, never the old
four-row or full-training background. Background choice changes the explanation target; these results
are not a claim that 30 samples represent the full training distribution without error.
Independent FP64-reference errors remain the primary accuracy measurement. A separate
`comparison_to_gpu_exact.csv` compares each approximation to its same-patient GPU degree-exact
output: maximum absolute and relative L2 differences. That GPU output still uses FP32 and is **not
ground truth**; close agreement with it does not certify absolute Shapley accuracy.

| Dataset | Mode | Worst maximum absolute error | Worst relative L2 error | Patients meeting requested tolerance |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(accuracy_rows)}

The experiment covers five patients on an interactive laptop. Timing differences should be interpreted
within this scope. `design.json` and `environment.json` preserve input, source and package provenance.

## Files and table reproduction

- `cox_gpu_timing_table.tex`: manuscript table and caption; requires `booktabs`.
- `preview.tex`: standalone A4 wrapper; render with `pdflatex preview.tex` in this directory.
- `table.csv`: eight aggregate rows, each based on one call for each of five patients.
- `accuracy.csv`: all 40 patient/method cases with individual measured times and errors.
- `comparison_to_gpu_exact.csv`: 30 secondary same-patient comparisons against GPU degree-exact output.
- `summary.csv` / `raw_timings.csv`: the 40 measured calls and separately labeled warm-ups.
- `small_rule_timings.csv`: five fresh constructions per distinct approximate rule.

Once the benchmark finishes, rebuild the table from the repository root without running experiments:

```sh
.venv-metal/bin/python -m benchmarks.cox_background30_table
```

The builder requires completed run status, checks all 40 measured configurations, exactly one measured
call per patient/method, one prior warm-up per distinct dataset/node count, saved summaries against raw
timings, background counts, five distinct patient IDs, node counts,
GPU/JIT environment and matching memory/block settings. It does not replace incomplete runs with
historical results. Old four-background and full-training experiments remain separate.
"""
    # No artifact is emitted until every experimental and setup record has passed.
    write_csv(folder / "table.csv", aggregate)
    write_csv(folder / "accuracy.csv", accuracy)
    write_csv(folder / "comparison_to_gpu_exact.csv", gpu_comparison)
    (folder / "cox_gpu_timing_table.tex").write_text(latex)
    (folder / "preview.tex").write_text(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\pagestyle{empty}
\makeatletter
\setlength{\@fptop}{0pt}
\makeatother
\begin{document}
\input{cox_gpu_timing_table.tex}
\end{document}
""")
    (folder / "README.md").write_text(readme)
    print(f"Validated 40 measured configurations over five patients; wrote {folder / 'cox_gpu_timing_table.tex'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
