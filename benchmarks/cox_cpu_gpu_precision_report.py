"""Validate CPU precision controls and produce a shared-reference Cox report.

Fresh CPU FP32/FP64 timings are paired with the saved GPU FP32 experiment.
All displayed errors use the independent 64-node FP64 reference, rather than
the rounded GPU degree-exact result.  Full degree-exact CPU runs are unmeasured.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

import numpy as np

from benchmarks.cox_background30_table import COHORTS, node_range, numeric, read_csv, read_json, require, write_csv
from benchmarks.cox_tolerance_rerun_report import (
    MODES, OUT as GPU_OUT, ROOT, case_key, check_boolean, close, digest, sci,
    validate as validate_gpu,
)

OUT = GPU_OUT / "cpu_precision_timing"


def validate(folder):
    """Read-only validation, including independent recalculation of errors."""
    gpu_folder, original = folder.parent, folder.parent.parent
    gpu, _, exact, design = validate_gpu(gpu_folder)
    gpu_env = read_json(gpu_folder / "gpu_environment.json")
    provenance = read_json(folder / "provenance.json")
    require(provenance, "Missing CPU experiment provenance")
    require(provenance["gpu_timings_reused"] and not provenance["exact_cpu_measured"],
            "Unexpected GPU/CPU measurement provenance")
    for dataset in COHORTS:
        require(digest(original / dataset / "inputs.npz") == provenance["input_sha256"][dataset],
                f"CPU inputs changed since measurement: {dataset}")
        require(digest(original / dataset / "reference_64.npy") == provenance["reference_sha256"][dataset],
                f"CPU reference changed since measurement: {dataset}")
    for directory, field in ((gpu_folder, "gpu_source_sha256"), (original, "exact_source_sha256")):
        for filename, sha in provenance[field].items():
            require(digest(directory / filename) == sha, f"Matched source changed since CPU measurement: {filename}")
    expected = set(gpu)
    cpu_cases, differences = {}, {}
    for precision in ("fp64", "fp32"):
        prefix = f"cpu_{precision}"
        status = read_json(folder / f"{prefix}_status.json")
        require(status["phase"] == "complete" and int(status["cases"]) == 30,
                f"Incomplete {prefix} experiment")
        require(int(status["measured_calls"]) == 30, f"Require thirty measured {prefix} calls")
        env = read_json(folder / f"{prefix}_environment.json")
        require(env["backend"].lower() == "cpu" and not env["jit_disabled"],
                f"Expected CPU JIT execution: {precision}")
        if precision == "fp64":
            require(env["jax_enable_x64"], "FP64 control did not enable JAX x64")
        expected_dtype = "float64" if precision == "fp64" else "float32"
        require(env["work_dtype"] == expected_dtype and env["host_accumulation_dtype"] == "float64",
                f"CPU precision provenance differs from the requested control: {precision}")
        require(env["core_formula"] == "parallel logspace core also used by GPU",
                f"CPU precision control uses an unexpected formula: {precision}")
        require(env["source_sha256"] == gpu_env["source_sha256"],
                f"CPU/GPU numerical sources differ: {precision}")
        for package in ("jax", "jaxlib", "numpy", "scipy"):
            require(env["packages"][package] == gpu_env["packages"][package],
                    f"CPU/GPU package mismatch: {package}, {precision}")
        rows = read_csv(folder / f"{prefix}_summary.csv")
        require(len(rows) == 30, f"Expected thirty summaries: {precision}")
        indexed = {}
        for row in rows:
            key = case_key(row)
            require(key in expected and key not in indexed, f"Unexpected/duplicate CPU case: {key}")
            dataset, patient, _ = key
            require(int(row["B"]) == 30 and int(row["d"]) == COHORTS[dataset]["d"],
                    f"CPU background or dimension mismatch: {key}")
            require(row["patient_id"] == str(design[dataset]["patient_ids"][patient]),
                    f"CPU patient mismatch: {key}")
            require(int(row["m_q"]) == int(gpu[key]["m_q"]), f"CPU/GPU node counts differ: {key}")
            require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                    f"CPU/GPU memory plan differs: {key}")
            require(row["dtype"] == expected_dtype and row["core_dtype"] == expected_dtype,
                    f"Unexpected CPU work/core dtype: {key}")
            close(row["bound"], gpu[key]["bound"], f"CPU/GPU analytical bound: {key}")
            require(numeric(row["bound"], "CPU quadrature bound") <= float(row["eps"]),
                    f"CPU quadrature bound exceeds tolerance: {key}")
            numeric(row["seconds"], "CPU measured time", positive=True)
            indexed[key] = row
        require(set(indexed) == expected, f"Missing CPU cases: {precision}")
        raw = read_csv(folder / f"{prefix}_raw_timings.csv")
        warmed, warmups, measured = set(), {}, {}
        for row in raw:
            key = case_key(row)
            require(key in expected, f"Unexpected CPU raw case: {key}")
            shape = row["dataset"], int(row["m_q"])
            numeric(row["seconds"], "CPU raw time", positive=True)
            numeric(row["budget_seconds"], "CPU node-selection time")
            require(row["phase"] in ("warmup", "measured"), "Unknown CPU timing phase")
            if row["phase"] == "warmup":
                require(shape not in warmed and key not in warmups, f"Duplicate CPU warm-up: {shape}")
                require(float(row["budget_seconds"]) == 0, "CPU warm-up must use explicit nodes")
                warmed.add(shape)
                warmups[key] = row
            else:
                require(shape in warmed and key not in measured, f"Missing CPU warm-up or repeated call: {key}")
                measured[key] = row
        require(set(measured) == expected and len(raw) == 30 + len(warmed),
                f"CPU measured-call/warm-up count is inconsistent: {precision}")
        require(warmed == {(r["dataset"], int(r["m_q"])) for r in measured.values()},
                f"Unused CPU warm-up: {precision}")
        for key, row in indexed.items():
            close(row["seconds"], measured[key]["seconds"], f"CPU summary time: {key}")
            close(row["warmup_seconds"], warmups.get(key, {}).get("seconds", 0), f"CPU warm-up time: {key}")
            require(int(row["m_q"]) == int(measured[key]["m_q"]), f"CPU raw/summary nodes differ: {key}")
        with np.load(folder / f"{prefix}_attributions.npz", allow_pickle=False) as cpu_phi, \
                np.load(gpu_folder / "gpu_attributions.npz", allow_pickle=False) as gpu_phi:
            for dataset, spec in COHORTS.items():
                reference = np.load(original / dataset / "reference_64.npy", allow_pickle=False)
                for patient in range(5):
                    for mode in MODES:
                        key, eps = (dataset, patient, mode), float(mode)
                        name = f"{dataset}_p{patient}_eps_{eps}"
                        phi = cpu_phi[name]
                        require(phi.shape == (spec["d"],) and np.isfinite(phi).all(), f"Invalid CPU attribution: {key}")
                        delta = phi - reference[patient]
                        error = float(np.max(np.abs(delta)))
                        relative = float(np.linalg.norm(delta) / np.linalg.norm(reference[patient]))
                        close(indexed[key]["max_absolute_error_vs_fp64_reference"], error, f"CPU max error: {key}")
                        close(indexed[key]["relative_l2_error_vs_fp64_reference"], relative, f"CPU relative error: {key}")
                        check_boolean(indexed[key]["within_eps"], error <= eps, f"CPU accuracy flag: {key}")
                        differences[(precision, *key)] = float(np.max(np.abs(phi - gpu_phi[name])))
        cpu_cases[precision] = indexed
    return gpu, cpu_cases, exact, design, differences


def aggregate(gpu, cpu, exact, differences):
    results = []
    for dataset in COHORTS:
        for mode in ("exact", *MODES):
            exact_mode = mode == "exact"
            rows = [exact[(dataset, p)] if exact_mode else gpu[(dataset, p, mode)] for p in range(5)]
            times = [float(row["seconds"]) for row in rows]
            result = {"dataset": dataset, "method": mode, "B": 30, "patients": 5,
                      "calls_per_patient": 1, "nodes_min": min(int(r["m_q"]) for r in rows),
                      "nodes_max": max(int(r["m_q"]) for r in rows),
                      "C_max": 0 if exact_mode else max(float(r["bound"]) for r in rows),
                      "gpu_fp32_seconds_mean": statistics.mean(times),
                      "gpu_fp32_seconds_sample_sd": statistics.stdev(times),
                      "delta_gpu_fp32_vs_reference": max(float(r["max_absolute_error"] if exact_mode else
                                                                r["max_absolute_error_vs_fp64_reference"]) for r in rows),
                      "gpu_fp32_pass": "" if exact_mode else sum(float(r["max_absolute_error_vs_fp64_reference"]) <= float(mode) for r in rows),
                      "gpu_source": "reused saved exact-degree GPU run" if exact_mode else "reused saved GPU tolerance rerun",
                      "cpu_source": "unmeasured" if exact_mode else "fresh warmed CPU precision benchmark"}
            for precision in ("fp64", "fp32"):
                prefix = f"cpu_{precision}"
                if exact_mode:
                    for field in ("seconds_mean", "seconds_sample_sd", "delta_vs_reference", "pass",
                                  "max_discrepancy_vs_gpu_fp32", "latency_ratio_to_gpu_fp32"):
                        result[f"{prefix}_{field}"] = ""
                else:
                    selected = [cpu[precision][(dataset, p, mode)] for p in range(5)]
                    timings = [float(r["seconds"]) for r in selected]
                    result.update({f"{prefix}_seconds_mean": statistics.mean(timings),
                                   f"{prefix}_seconds_sample_sd": statistics.stdev(timings),
                                   f"{prefix}_delta_vs_reference": max(float(r["max_absolute_error_vs_fp64_reference"]) for r in selected),
                                   f"{prefix}_pass": sum(float(r["max_absolute_error_vs_fp64_reference"]) <= float(mode) for r in selected),
                                   f"{prefix}_max_discrepancy_vs_gpu_fp32": max(differences[(precision, dataset, p, mode)] for p in range(5)),
                                   f"{prefix}_latency_ratio_to_gpu_fp32": statistics.mean(timings) / result["gpu_fp32_seconds_mean"]})
            results.append(result)
    return results


def time_tex(mean, sd):
    return rf"${mean:.3f}\pm{sd:.3f}$"


def method_tex(mode):
    return "Exact degree" if mode == "exact" else rf"$10^{{{int(math.log10(float(mode)))}}}$"


def build(folder):
    gpu, cpu, exact, design, differences = validate(folder)
    results = aggregate(gpu, cpu, exact, differences)
    tex_rows, control_rows, markdown_rows, markdown_control = [], [], [], []
    for row in results:
        name = "GSE24080" if row["dataset"] == "gse24080" else "TCGA LGG"
        mode = row["method"]
        if row["dataset"] == "tcga_lgg_methylation" and mode == "exact":
            tex_rows.append(r"\midrule")
            control_rows.append(r"\midrule")
        nodes = node_range([row["nodes_min"], row["nodes_max"]])
        gpu_time = time_tex(row["gpu_fp32_seconds_mean"], row["gpu_fp32_seconds_sample_sd"])
        gpu_error = sci(row["delta_gpu_fp32_vs_reference"])
        gpu_pass = "---" if mode == "exact" else f"{row['gpu_fp32_pass']}/5"
        cpu_time = "---" if mode == "exact" else time_tex(row["cpu_fp64_seconds_mean"], row["cpu_fp64_seconds_sample_sd"])
        cpu_error = "---" if mode == "exact" else sci(row["cpu_fp64_delta_vs_reference"])
        cpu_pass = "---" if mode == "exact" else f"{row['cpu_fp64_pass']}/5"
        tex_rows.append(" & ".join([name if mode == "exact" else "", method_tex(mode), nodes, sci(row["C_max"]),
                                    gpu_time, gpu_error, gpu_pass, cpu_time, cpu_error, cpu_pass]) + r" \\")
        markdown_rows.append(f"| {name} | {mode} | {nodes.replace('--', '–')} | {row['C_max']:.3g} | "
                             f"{row['gpu_fp32_seconds_mean']:.3f} ± {row['gpu_fp32_seconds_sample_sd']:.3f} | "
                             f"{row['delta_gpu_fp32_vs_reference']:.3g} | {gpu_pass} | " +
                             ("— | — | — |" if mode == "exact" else
                              f"{row['cpu_fp64_seconds_mean']:.3f} ± {row['cpu_fp64_seconds_sample_sd']:.3f} | "
                              f"{row['cpu_fp64_delta_vs_reference']:.3g} | {cpu_pass} |"))
        if mode != "exact":
            control_rows.append(" & ".join([name, method_tex(mode),
                                            time_tex(row["cpu_fp32_seconds_mean"], row["cpu_fp32_seconds_sample_sd"]),
                                            sci(row["cpu_fp32_delta_vs_reference"]), f"{row['cpu_fp32_pass']}/5",
                                            sci(row["cpu_fp32_max_discrepancy_vs_gpu_fp32"])]) + r" \\")
            markdown_control.append(f"| {name} | {mode} | {row['cpu_fp32_seconds_mean']:.3f} ± "
                                    f"{row['cpu_fp32_seconds_sample_sd']:.3f} | {row['cpu_fp32_delta_vs_reference']:.3g} | "
                                    f"{row['cpu_fp32_pass']}/5 | {row['cpu_fp32_max_discrepancy_vs_gpu_fp32']:.3g} |")
    table = r"""% Requires \usepackage{amsmath,booktabs,graphicx}.
% Delta_GPU here uses the independent FP64 reference, unlike the earlier GPU-exact discrepancy table.
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrrrrr}
\toprule
 & & & & \multicolumn{3}{c}{GPU FP32} & \multicolumn{3}{c}{CPU FP64} \\
\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Dataset & $\varepsilon$ & Nodes & $C_{\max}$ & Time (s) & $\Delta_{\mathrm{GPU}}$ & Pass
& Time (s) & $\Delta_{\mathrm{CPU}}$ & Pass \\
\midrule
""" + "\n".join(tex_rows) + "\n" + r"""\bottomrule
\end{tabular}%
}
\caption{CPU/GPU Cox explanations using 30 fixed training-background patients
and five distinct held-out patients per cohort on an Apple M4 Pro.
Times are mean $\pm$ sample standard deviation across patients, with one
synchronized call per patient and method. CPU FP64 timings are newly measured;
GPU FP32 timings reuse the matched saved experiments.
Both $\Delta_{\mathrm{GPU}}$ and $\Delta_{\mathrm{CPU}}$ are maximum absolute
feature-wise errors over all five patients against the same independent,
tightly bounded 64-node FP64 reference. These errors differ in definition
from the earlier discrepancy against GPU degree-exact output.
Pass counts patients whose maximum error is at most $\varepsilon$.
$C_{\max}$ is the largest analytical quadrature bound across patients.
CPU degree-exact results are unmeasured (---); exact GPU output also contains
rounding error. API timings include automatic node selection for approximate rows, transfers,
and host accumulation, but exclude model fitting, compilation/warm-up,
and cached quadrature-rule construction. Previously measured construction
of the exact rules took 10.97\,s and 561.92\,s for GSE24080 and TCGA LGG;
approximate-rule construction took 0.386--0.563\,ms.
The analytical certificate bounds quadrature error in real arithmetic,
not floating-point error. Comparing GPU FP32 with CPU FP64 changes both
device and precision; their timing ratio is not a matched-precision GPU speedup.}
\label{tab:cox-cpu-gpu-precision}
\end{table*}
"""
    control = r"""% Requires \usepackage{booktabs}.
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Dataset & $\varepsilon$ & CPU FP32 time (s) & $\Delta_{\mathrm{CPU32}}$ & Pass
& CPU32--GPU32 gap \\
\midrule
""" + "\n".join(control_rows) + "\n" + r"""\bottomrule
\end{tabular}
\caption{Additional CPU FP32 control using the same patients, backgrounds,
node counts, formula, and timing protocol. The CPU32--GPU32 gap is the largest
absolute difference between their attribution outputs across patients and
features. $\Delta_{\mathrm{CPU32}}$ instead compares CPU FP32 with the same
independent FP64 reference used in Table~\ref{tab:cox-cpu-gpu-precision}.
Comparing CPU FP32 and CPU FP64 holds the device and formula fixed while
changing work precision; comparing CPU FP32 and GPU FP32 also exposes
backend-dependent numerical effects. Both FP32 configurations use FP64
host accumulation.}
\label{tab:cox-cpu-precision-control}
\end{table}
"""
    approx = [row for row in results if row["method"] != "exact"]
    passes = {label: sum(row[field] for row in approx) for label, field in
              (("GPU FP32", "gpu_fp32_pass"), ("CPU FP32", "cpu_fp32_pass"), ("CPU FP64", "cpu_fp64_pass"))}
    worst = {label: max(row[field] for row in approx) for label, field in
             (("GPU FP32", "delta_gpu_fp32_vs_reference"), ("CPU FP32", "cpu_fp32_delta_vs_reference"),
              ("CPU FP64", "cpu_fp64_delta_vs_reference"))}
    settings = r"""\paragraph{Separating approximation error from numerical precision.}
To determine whether the observed discrepancies arise from the quadrature
approximation or its numerical evaluation, we compare GPU FP32, CPU FP32,
and CPU FP64 explanations for the same fitted Cox models and empirical
background games. We retain the 30 fixed training-background patients,
five distinct held-out patients per dataset, and automatic node choices for
$\varepsilon\in\{10^{-1},10^{-2},10^{-3}\}$.
The GPU results are reused from the saved experiments; both CPU controls
are newly measured. Inputs, numerical sources, and package versions were
verified against the GPU experiment. The CPU controls evaluate the same
parallel product-game formula at the same selected node counts. CPU FP32
provides a control on the same hardware as CPU FP64, so that changing the
device is not necessary to investigate the effect of precision.

Across the three tolerances, mean CPU FP64 latencies are 156--164\,ms
for GSE24080 and 992--1,035\,ms for TCGA LGG, compared with
85--90\,ms and 310--320\,ms for the saved GPU FP32 runs.
These are measured precision/device tradeoffs, not predictions of
FP64 GPU performance.

All three configurations are compared with the same independent 64-node
FP64 reference $\phi^{\mathrm{ref}}_p$:
\[
\Delta_a=\max_{p=1,\ldots,5}
\|\widehat\phi^{a}_{m_p,p}-\phi^{\mathrm{ref}}_p\|_\infty,
\qquad a\in\{\mathrm{GPU32},\mathrm{CPU32},\mathrm{CPU64}\}.
\]
This reference has analytical quadrature bounds below $10^{-10}$ and a
96-node cross-check for the first patient of each dataset. It is a
high-accuracy numerical reference, not a full degree-exact FP64 computation.
In particular, $\Delta_{\mathrm{GPU}}$ in
Table~\ref{tab:cox-cpu-gpu-precision} now uses this common reference;
it is not the discrepancy against rounded GPU degree-exact output reported
in the earlier table. All automatically selected rules satisfy their
analytical bounds $C_p\le\varepsilon$ in real arithmetic.

"""
    settings += (f"Against the common reference, GPU FP32 meets the requested threshold in\n"
                 f"{passes['GPU FP32']}/30 patient--tolerance cases, CPU FP32 in {passes['CPU FP32']}/30,\n"
                 f"and CPU FP64 in {passes['CPU FP64']}/30. The worst errors over these cases are\n"
                 f"{sci(worst['GPU FP32'])}, {sci(worst['CPU FP32'])}, and {sci(worst['CPU FP64'])}, respectively.\n")
    if worst["CPU FP64"] < worst["CPU FP32"]:
        settings += r"""The reduction in error from CPU FP32 to CPU FP64, with the device and
formula held fixed, supports limited work precision as an important source
of the discrepancies. Differences between the CPU FP32 and GPU FP32
outputs additionally show the role of backend-specific evaluation.
These numerical controls support this explanation but do not constitute
a formal floating-point error bound.
"""
    else:
        settings += r"""The controls must be interpreted as empirical precision/backend
comparisons; they do not establish that work precision alone explains
the observed discrepancy and do not provide a floating-point error bound.
"""
    settings += r"""
For timing, each distinct dataset/node-count configuration is warmed up
once before its first measured call. We record one synchronized API call
per patient and method, including automatic node selection, data movement,
and host accumulation. Model fitting, compilation/warm-up, and cached rule
construction are excluded. The CPU and GPU runs use an 8\,GiB planning
budget and one background row per block; this is a planning budget, not
measured memory consumption. Reported standard deviations describe
variation across patients rather than repeated-call uncertainty.
The saved GPU timings and new CPU timings were recorded in separate runs.
The CPU FP64/GPU FP32 latency ratio changes both device and precision,
so it should not be interpreted as a matched-precision GPU speedup.
Full degree-exact CPU runtimes were not measured and are left blank.

The analytical guarantee remains a bound on quadrature error for the
fixed fitted model and chosen 30-row empirical background game.
It excludes floating-point rounding, uncertainty in the fitted model,
and error from replacing a population background with this empirical
background. Small FP64 discrepancies are an empirical accuracy check,
not a stronger analytical certificate. The observed accuracy of CPU FP64
must therefore be paired with its CPU runtime rather than with the faster
GPU FP32 runtime.
"""
    readme = ("# CPU/GPU precision and latency comparison\n\n"
              "CPU FP32 and FP64 measurements are new. All GPU results are reused from the matched saved "
              "experiments; no GPU timing was rerun. Model, inputs, backgrounds, patients, node counts, "
              "numerical sources and package versions are matched. CPU exact-degree experiments were "
              "not measured and are blank below.\n\n"
              "**The error reference has changed from the earlier main table.** All Delta columns here "
              "compare with the same independent 64-node FP64 reference, not GPU degree-exact output. "
              "The reference has analytical quadrature bounds below 1e-10 and a 96-node cross-check for "
              "the first patient in each cohort; it is not a full degree-exact FP64 computation.\n\n"
              "| Dataset | Method | Nodes | C_max | GPU FP32 time (s) | Delta_GPU | GPU pass | CPU FP64 time (s) | Delta_CPU | CPU pass |\n"
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(markdown_rows) + "\n\n"
              "Times are mean ± sample standard deviation across five distinct patients, with one "
              "measured explanation per patient. Errors are maxima over all five patients and all features. "
              "Pass counts use the unrounded per-patient maximum error. All analytical quadrature bounds "
              "satisfy their requested tolerance, independently of the measured floating-point error.\n\n"
              "## CPU FP32 control\n\n"
              "| Dataset | Tolerance | CPU FP32 time (s) | Delta_CPU32 | Pass | Max CPU32–GPU32 gap |\n"
              "|---|---|---:|---:|---:|---:|\n" + "\n".join(markdown_control) + "\n\n"
              f"Against the common reference: GPU FP32 passes {passes['GPU FP32']}/30, CPU FP32 "
              f"passes {passes['CPU FP32']}/30, and CPU FP64 passes {passes['CPU FP64']}/30. "
              f"Their worst approximate errors are {worst['GPU FP32']:.3g}, {worst['CPU FP32']:.3g}, "
              f"and {worst['CPU FP64']:.3g}, respectively.\n\n"
              "CPU FP32 versus CPU FP64 holds hardware and formula fixed while changing work precision. "
              "CPU FP32 versus GPU FP32 exposes backend-dependent numerical effects as well. The controls "
              "are empirical evidence about numerical precision, not a floating-point error certificate. "
              "Both FP32 implementations accumulate blocks on the host in FP64.\n\n"
              "## Timing scope and interpretation\n\n"
              "Each distinct dataset/node-count configuration receives one untimed warm-up. Measured "
              "API wall times include automatic node selection, transfers, and host accumulation; they "
              "exclude model fitting, compilation/warm-up, and cached rule construction. Memory planning "
              "uses 8 GiB with one background row per block. GPU results and CPU results were recorded in "
              "separate runs, so ratios are not contemporaneous paired benchmarks. The CPU FP64/GPU FP32 "
              "ratio additionally changes precision and must not be described as a matched-precision "
              "GPU speedup. The summary CSV also provides the same-work-precision CPU FP32/GPU FP32 ratio.\n\n"
              "The quadrature certificate applies to the fixed fitted relative-hazard model and 30-row "
              "empirical background game. It excludes rounding, fitted-model uncertainty, and error from "
              "approximating a population background. CPU FP64 accuracy must not be assigned to GPU FP32 "
              "timings. The five-patient standard deviations do not establish repeated-run variability "
              "or statistical significance.\n\n"
              "## Outputs\n\n"
              "- `cpu_gpu_table.tex`: main comparison with a common error reference.\n"
              "- `cpu_fp32_control_table.tex`: additional same-device precision control.\n"
              "- `cpu_gpu_analysis.tex`: interpretation and reproducibility details.\n"
              "- `table.csv`: full-precision aggregate values, bounds, pass counts, and latency ratios.\n"
              "- `preview.tex`: standalone LaTeX preview.\n")
    preview = r"""\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,booktabs,graphicx}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\begin{document}
\input{cpu_gpu_table.tex}
\input{cpu_fp32_control_table.tex}
\input{cpu_gpu_analysis.tex}
\end{document}
"""
    require("10^{-6}" not in table + control + settings, "Obsolete tolerance in report")
    write_csv(folder / "table.csv", results)
    (folder / "cpu_gpu_table.tex").write_text(table)
    (folder / "cpu_fp32_control_table.tex").write_text(control)
    (folder / "cpu_gpu_analysis.tex").write_text(settings)
    (folder / "README.md").write_text(readme)
    (folder / "preview.tex").write_text(preview)
    print(f"Validated 30 GPU, 60 new CPU, and 10 reused exact GPU cases; wrote {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT)
    build(parser.parse_args().folder)
