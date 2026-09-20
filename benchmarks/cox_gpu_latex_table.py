"""Build the matched, full-background Cox GPU timing table after all runs finish.

Refuses partial inputs, old four-background timings, mismatched numerical
sources, or settings other than the agreed 8 GiB / one-background-row plan.
This script reads results only; it never runs inference or generates timing data.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/cox_gpu_tolerance/eps_1e6_followup"
COHORTS = {"gse24080": {"label": "GSE24080", "B": 339, "d": 54675},
           "tcga_lgg_methylation": {"label": "TCGA LGG methylation", "B": 383, "d": 396065}}
TOLERANCES = (1e-1, 1e-3, 1e-6)
MODES = ("exact", "1e-1", "1e-3", "1e-6")


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


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def number(value, name, positive=False):
    value = float(value)
    require(math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"Invalid {name}: {value}")
    return value


def mode_for(eps):
    for tolerance, mode in zip(TOLERANCES, MODES[1:]):
        if math.isclose(float(eps), tolerance, rel_tol=1e-12):
            return mode
    raise ValueError(f"Unexpected tolerance: {eps}")


def validate_plan(plan, dataset):
    require(plan["backend"] == "logspace_jax", "Expected the JAX log-space implementation")
    require(int(plan["n_pairs"]) == COHORTS[dataset]["B"], "Background-count mismatch in plan")
    require(int(plan["block_size"]) == 1 and int(plan["budget_bytes"]) == 8 * 2 ** 30,
            "Expected 8 GiB memory budget and one background row per block")


def load_complete_results(folder):
    """Validate all 18 approximate cases, all six exact runs and timing repeats."""
    env = read_json(folder / "full_background_environment.json")
    require(env["backend"].lower() != "cpu" and not env["jit_disabled"], "GPU/JIT required")
    approx = read_csv(folder / "full_background_approximation.csv")
    raw = read_csv(folder / "full_background_raw.csv")
    expected = {(dataset, patient, mode) for dataset in COHORTS
                for patient in range(3) for mode in MODES[1:]}
    mapped = {}
    for row in approx:
        dataset, patient, mode = row["dataset"], int(row["patient"]), mode_for(row["eps"])
        key = dataset, patient, mode
        require(key in expected and key not in mapped, f"Unexpected or duplicate approximate case: {key}")
        require(int(row["B"]) == COHORTS[dataset]["B"] and int(row["d"]) == COHORTS[dataset]["d"],
                f"Wrong background or feature count: {key}")
        require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                f"Unmatched approximate settings: {key}")
        validate_plan(ast.literal_eval(row["plan"]), dataset)
        calls = [r for r in raw if r["dataset"] == dataset and int(r["patient"]) == patient
                 and mode_for(r["eps"]) == mode]
        require(len(calls) == 4 and sorted(int(r["repeat"]) for r in calls) == [-1, 0, 1, 2],
                f"Expected first call and three warm repetitions: {key}")
        warm = [number(r["seconds"], "warm time", positive=True)
                for r in calls if int(r["repeat"]) >= 0]
        seconds = number(row["median_seconds"], "median GPU time", positive=True)
        require(math.isclose(seconds, statistics.median(warm), rel_tol=1e-10, abs_tol=1e-12),
                f"Raw repetitions disagree with saved median: {key}")
        require(all(int(r["m_q"]) == int(row["m_q"]) for r in calls), f"Node counts changed: {key}")
        bound = number(row["bound"], "quadrature bound")
        eps = float(row["eps"])
        require(bound <= eps, f"Requested quadrature bound was not met: {key}")
        mapped[key] = {"dataset": dataset, "patient": patient, "patient_id": row["patient_id"],
                       "mode": mode, "B": int(row["B"]), "d": int(row["d"]),
                       "m_q": int(row["m_q"]), "seconds": seconds,
                       "max_absolute_error": number(row["max_absolute_error"], "absolute error"),
                       "relative_l2_error": number(row["relative_l2_error"], "relative error"),
                       "quadrature_bound": bound, "requested_tolerance": eps,
                       "observed_tolerance_met": float(row["max_absolute_error"]) <= eps,
                       "timing_basis": "Median of three warmed synchronized API calls"}
    require(set(mapped) == expected, f"Approximation results incomplete; missing {sorted(expected - set(mapped))}")
    require(len(raw) == 72, "Expected exactly 72 approximate first/warm calls")

    exact_records = {}
    for dataset in COHORTS:
        source_inputs, source_rule = None, None
        for patient in range(3):
            path = folder / dataset / f"patient_{patient}" / "exact_gpu_timing.json"
            row = read_json(path)  # No fallback to historical four-background runs.
            require(row["dataset"] == dataset and int(row["patient"]) == patient, f"Mislabeled exact result: {path}")
            require(row["exact_degree"] is True and int(row["m_q"]) == (COHORTS[dataset]["d"] + 1) // 2,
                    f"Exact-degree rule was not fully evaluated: {path}")
            require(int(row["background_size"]) == COHORTS[dataset]["B"]
                    and int(row["d"]) == COHORTS[dataset]["d"], f"Wrong exact background/dimension: {path}")
            require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1
                    and row["backend"] == "logspace_jax" and int(row["repeats"]) == 1,
                    f"Unmatched exact settings: {path}")
            validate_plan(row["block_plan"], dataset)
            require(row["packages"] == env["packages"] and row["device"] == env["devices"],
                    f"Exact/approximate environment mismatch: {path}")
            require(all(row["source_sha256"].get(k) == v for k, v in env["source_sha256"].items()),
                    f"Exact/approximate numerical sources differ: {path}")
            if source_inputs is None:
                source_inputs, source_rule = row["input_sha256"], row["rule_sha256"]
            require(row["input_sha256"] == source_inputs and row["rule_sha256"] == source_rule,
                    f"Inputs/rule changed across exact patients: {path}")
            patient_id = mapped[(dataset, patient, "1e-1")]["patient_id"]
            require(str(row["patient_id"]) == str(patient_id), f"Different exact and approximate patient: {path}")
            exact_records[(dataset, patient)] = row
            mapped[(dataset, patient, "exact")] = {
                "dataset": dataset, "patient": patient, "patient_id": patient_id, "mode": "exact",
                "B": int(row["background_size"]), "d": int(row["d"]), "m_q": int(row["m_q"]),
                "seconds": number(row["integration_seconds"], "exact GPU integration", positive=True),
                "max_absolute_error": number(row["max_absolute_error"], "exact-degree absolute error"),
                "relative_l2_error": number(row["relative_l2_error"], "exact-degree relative error"),
                "quadrature_bound": 0.0, "requested_tolerance": "", "observed_tolerance_met": "",
                "timing_basis": "One full integration; includes required JIT, excludes checkpoint IO"}
    return mapped, exact_records, env


def fresh_seconds(record):
    """Find the original generation time when later patients loaded the rule."""
    while record:
        value = record.get("fresh_rule_seconds")
        if value is not None:
            return number(value, "fresh exact rule construction", positive=True)
        record = record.get("original_preparation")
    raise ValueError("Missing fresh GSE rule-construction measurement")


def node_range(values, latex=False):
    low, high = min(values), max(values)
    if latex:
        fmt = lambda n: f"{n:,}".replace(",", r"{,}")
        return fmt(low) if low == high else f"{fmt(low)}--{fmt(high)}"
    return str(low) if low == high else f"{low}-{high}"


def build(folder):
    cases, exact, env = load_complete_results(folder)
    rules = read_csv(folder / "full_background_rule_timings.csv")
    by_nodes = {int(row["m_q"]): row for row in rules}
    needed_nodes = {case["m_q"] for case in cases.values() if case["mode"] != "exact"}
    require(len(by_nodes) == len(rules) and set(by_nodes) == needed_nodes,
            "Fresh approximate-rule measurements are missing or duplicated")
    require(all(int(row["repeats"]) == 5 for row in rules), "Expected five fresh constructions per approximate rule")
    setup_ms = [1000 * number(row["median_seconds"], "fresh approximate rule time", positive=True) for row in rules]
    gse_setup = fresh_seconds(exact[("gse24080", 0)]["rule_preparation"])
    tcga_history = read_json(ROOT / "benchmarks/results/cox_gpu_tolerance/tcga_lgg_methylation/exact_gpu_timing.json")
    require(int(tcga_history["m_q"]) == (COHORTS["tcga_lgg_methylation"]["d"] + 1) // 2,
            "Historical TCGA setup used a different rule size")
    tcga_setup = number(tcga_history["rule_seconds"], "historical TCGA rule construction", positive=True)

    aggregate = []
    for mode in MODES:
        for dataset in COHORTS:
            group = [cases[(dataset, p, mode)] for p in range(3)]
            times = [r["seconds"] for r in group]
            aggregate.append({"dataset": dataset, "mode": mode, "background_size": COHORTS[dataset]["B"],
                              "features": COHORTS[dataset]["d"], "patients": 3,
                              "nodes_min": min(r["m_q"] for r in group), "nodes_max": max(r["m_q"] for r in group),
                              "gpu_seconds_mean": statistics.mean(times),
                              "gpu_seconds_sample_sd": statistics.stdev(times),
                              "time_basis": group[0]["timing_basis"],
                              "memory_budget": "8GB", "background_block_size": 1})
    indexed = {(r["dataset"], r["mode"]): r for r in aggregate}
    table_rows = []
    for mode in MODES:
        label = "Exact degree" if mode == "exact" else rf"$\varepsilon=10^{{{int(math.log10(float(mode)))}}}$"
        cells = [label]
        for dataset in COHORTS:
            values = [cases[(dataset, p, mode)]["m_q"] for p in range(3)]
            stats = indexed[(dataset, mode)]
            cells.extend([node_range(values, latex=True),
                          f"${stats['gpu_seconds_mean']:.3f} \\pm {stats['gpu_seconds_sample_sd']:.3f}$"])
        table_rows.append(" & ".join(cells) + r" \\")
    failures = sum(not r["observed_tolerance_met"] for r in cases.values() if r["mode"] != "exact")
    tex = r"""% Requires \usepackage{booktabs}. All times are measured full-background results.
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
""" + "\n".join(table_rows) + r"""
\bottomrule
\end{tabular}
\caption{GPU-backed Cox relative-hazard explanations in the small-$n$, large-$d$ setting:
GSE24080 has $n_{\mathrm{train}}=339$, $d=54{,}675$; TCGA LGG methylation has
$n_{\mathrm{train}}=383$, $d=396{,}065$. All training patients form the uniformly weighted
background. Times are mean $\pm$ sample standard deviation across three held-out
patients on an Apple M4 Pro with JAX-Metal FP32 (8\,GiB memory budget,
one background row per block). Approximation times use each patient's median of
three warmed, synchronized API calls and include automatic node-count selection;
exact-degree times use one complete integration per patient, including required
JIT compilation and excluding checkpoint I/O. Node ranges reflect differences
between patients. ``Exact degree'' means polynomial integration is exact in real
arithmetic; $\varepsilon$ bounds quadrature error, not FP32 rounding.
""" + f"Measured absolute errors exceeded the requested tolerance in {failures}/18 approximate cases.\n" + r"""
CPU construction of quadrature nodes and weights is excluded and cached for reuse.
""" + f"Fresh exact-rule construction took {gse_setup:.2f}\\,s for GSE24080; the TCGA rule was reused,\n" + \
        f"with historical construction time {tcga_setup:.2f}\\,s. Fresh approximate-rule construction\n" + \
        f"took {min(setup_ms):.3f}--{max(setup_ms):.3f}\\,ms (range of medians over five constructions per rule)." + r"""}
\label{tab:cox-gpu-timing}
\end{table}
"""
    # Write only after every case, provenance check and setup measurement passed.
    write_csv(folder / "timing_table.csv", aggregate)
    accuracy = [{key: r[key] for key in ("dataset", "patient", "patient_id", "mode", "B", "d", "m_q",
                                         "requested_tolerance", "quadrature_bound", "max_absolute_error",
                                         "relative_l2_error", "observed_tolerance_met")}
                for mode in MODES for dataset in COHORTS for p in range(3)
                for r in [cases[(dataset, p, mode)]]]
    write_csv(folder / "accuracy_table.csv", accuracy)
    (folder / "cox_gpu_timing_table.tex").write_text(tex)
    preview = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
\input{cox_gpu_timing_table.tex}
\end{document}
"""
    (folder / "preview.tex").write_text(preview)

    display = ["| Method | GSE24080 nodes | GPU seconds, mean ± SD | TCGA LGG nodes | GPU seconds, mean ± SD |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        cells = ["Exact degree" if mode == "exact" else mode]
        for dataset in COHORTS:
            r = indexed[(dataset, mode)]
            cells += [node_range([r["nodes_min"], r["nodes_max"]]),
                      f"{r['gpu_seconds_mean']:.3f} ± {r['gpu_seconds_sample_sd']:.3f}"]
        display.append("| " + " | ".join(cells) + " |")
    accuracy_lines = []
    for mode in MODES:
        for dataset, spec in COHORTS.items():
            group = [cases[(dataset, p, mode)] for p in range(3)]
            passing = "n/a" if mode == "exact" else f"{sum(r['observed_tolerance_met'] for r in group)}/3"
            accuracy_lines.append(f"| {spec['label']} | {mode} | {max(r['max_absolute_error'] for r in group):.6g} | "
                                  f"{max(r['relative_l2_error'] for r in group):.6g} | {passing} |")
    readme = f"""# Matched Cox GPU timing table

All four modes use the complete training backgrounds (339 GSE24080 patients and 383 TCGA LGG patients),
the same three held-out patients, the same saved fitted models, JAX-Metal FP32, an 8 GiB planning budget,
and one background row per block. No four-background runtime is reused in this table.

{chr(10).join(display)}

Approximation results aggregate the three per-patient medians, each based on three warmed synchronized
API calls. The displayed spread is the **sample standard deviation across patients**, not a confidence
interval or the variation of all nine calls. Exact results aggregate one complete exact-degree run per
patient, including any required JIT compilation. All timings include host factor preparation, transfers,
GPU calculation, and synchronized host accumulation; they are not isolated GPU-kernel times.
Approximate calls additionally include automatic node-count selection and its report. Model fitting,
explainer initialization, rule construction and checkpoint disk I/O are outside integration timing.

## Quadrature-rule setup

Fresh GSE24080 exact-rule construction: **{gse_setup:.6f} s**. The exact TCGA rule was loaded from the
validated saved rule; its **historical CPU construction time was {tcga_setup:.6f} s**. This setup is
independent of patient/background count and reusable. Fresh approximate-rule generation took
**{min(setup_ms):.6f}–{max(setup_ms):.6f} ms**, the range of per-rule medians over five fresh constructions.
This means generating abscissae and weights, not the automatic node-budget calculation included in API time.

## Accuracy qualification

Exact degree removes polynomial quadrature error in real arithmetic, not floating-point rounding.
GPU kernels use FP32 and node/background accumulations use host FP64. The requested absolute
per-feature tolerance likewise controls quadrature error only. **{failures}/18 approximate cases exceeded
their requested total absolute error**. Increasing the number of nodes alone does not remove this rounding floor.
The error comparison uses the independent FP64 64-node reference with certified quadrature bound below
1e-10; it is a tightly bounded approximation, not a full exact-degree reference. Patient 0 also has a saved
96-node crosscheck. These references and unchanged model outputs come from `cox_background_sensitivity`.

| Dataset | Mode | Worst max absolute error | Worst relative L2 error | Patients meeting requested absolute tolerance |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(accuracy_lines)}

Three patients on an interactive laptop are a small timing sample, not a throughput guarantee.
The same blocking protocol was selected using a separate bounded batching pilot, and is applied to all modes.
Checkpoint-resumed timing adds recorded active work and excludes downtime or uncheckpointed work lost to interruption;
each exact JSON reports attempts and the precise timing scope.

## Artifacts and reproduction

- `cox_gpu_timing_table.tex`: manuscript-ready `booktabs` table and setup/precision caption.
- `preview.tex`: standalone A4 wrapper; run `pdflatex preview.tex` in this directory to render it.
- `timing_table.csv`: eight aggregate timing rows.
- `accuracy_table.csv`: all 24 patient/mode accuracy comparisons.
- `full_background_approximation.csv` / `full_background_raw.csv`: per-patient summaries and all raw repeats.
- `<dataset>/patient_<i>/exact_gpu_timing.json`: all six complete exact calculations, timing and provenance.
- `full_background_rule_timings.csv`: separately measured fresh small-rule construction.

From the repository root, reproduce the approximate measurements with:

```sh
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 .venv-metal/bin/python -u -m benchmarks.cox_full_gpu_table --memory-budget 8GB --block-size 1
```

For each dataset (`gse24080`, `tcga_lgg_methylation`) and patient (`0`, `1`, `2`), run:

```sh
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 .venv-metal/bin/python -u -m benchmarks.cox_full_exact_gpu --dataset gse24080 --patient 0 --memory-budget 8GB --block-size 1
```

Use `--resume` for an existing checkpoint; completed artifacts are deliberately protected from accidental replacement.
Once all runs exist, regenerate this table without running numerical experiments:

```sh
.venv-metal/bin/python -m benchmarks.cox_gpu_latex_table
```

The table builder validates all six exact records, all 18 approximate cases and 72 first/warm calls,
patient IDs, background counts, block plans, numerical source hashes and package/device compatibility.
"""
    (folder / "README.md").write_text(readme)
    print(f"Validated 24 cases; wrote {folder / 'cox_gpu_timing_table.tex'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Directory containing the finished full-background measurements")
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
