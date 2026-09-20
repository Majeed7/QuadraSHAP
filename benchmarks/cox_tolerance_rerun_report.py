"""Validate and report the fresh 0.1/0.01/0.001 Cox tolerance experiments.

This script never runs an experiment.  It requires all thirty new GPU calls
and all thirty matched FP64 diagnostics before producing any report.  The
degree-exact GPU baseline is explicitly reused from the original benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import statistics
from pathlib import Path

import numpy as np

from benchmarks.cox_background30_table import (
    COHORTS, node_range, numeric, read_csv, read_json, require, write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "benchmarks/results/cox_background30_gpu"
OUT = PARENT / "tolerance_rerun"
MODES = ("1e-1", "1e-2", "1e-3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_key(row):
    eps = numeric(row["eps"], "requested tolerance", positive=True)
    modes = [mode for mode in MODES if math.isclose(eps, float(mode), rel_tol=1e-12)]
    require(len(modes) == 1, f"Unexpected tolerance: {eps}")
    return row["dataset"], int(row["patient"]), modes[0]


def close(actual, expected, description):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-13),
            f"Inconsistent {description}: {actual} != {expected}")


def check_boolean(value, expected, description):
    require(str(value).lower() == str(bool(expected)).lower(), f"Inconsistent {description}")


def sci(value):
    if value == 0:
        return "$0$"
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def validate(folder):
    original = folder.parent
    design = read_json(original / "design.json")
    old_env = read_json(original / "environment.json")
    gpu_env = read_json(folder / "gpu_environment.json")
    provenance = read_json(folder / "provenance.json")
    require(provenance, "Missing provenance for reused exact baseline")
    require(provenance["exact_baseline_reused"] and provenance["core_hashes_match_exact_baseline"]
            and provenance["packages_match_exact_baseline"], "Matched exact-baseline provenance failed")
    require(provenance["exact_source"] == "../summary.csv" and provenance["exact_attributions"] == "../attributions.npz",
            "Unexpected source for the reused exact baseline")
    for filename, sha in provenance["source_sha256"].items():
        require(digest(original / filename) == sha, f"Reused source changed since the rerun: {filename}")
    require(gpu_env["backend"].lower() == "metal" and not gpu_env["jit_disabled"],
            "The timing table requires JAX-Metal with JIT enabled")
    require(gpu_env["source_sha256"] == old_env["source_sha256"],
            "The numerical sources differ from the reused exact GPU baseline")
    for package in ("jax", "jaxlib", "numpy", "scipy", "jax-metal"):
        require(gpu_env["packages"][package] == old_env["packages"][package],
                f"GPU package differs from the exact baseline: {package}")
    for source, sha in gpu_env["source_sha256"].items():
        require(digest(ROOT / source) == sha, f"Numerical source changed since measurement: {source}")
    expected = {(dataset, patient, mode) for dataset in COHORTS
                for patient in range(5) for mode in MODES}
    for prefix in ("gpu", "fp64"):
        status = read_json(folder / f"{prefix}_status.json")
        require(status["phase"] == "complete" and int(status["cases"]) == 30,
                f"The {prefix} experiment is incomplete")
        if prefix == "gpu":
            require(int(status["measured_calls"]) == 30, "Expected thirty measured GPU calls")
    for dataset, spec in COHORTS.items():
        details = design[dataset]
        require(int(details["background_size"]) == 30 and int(details["selection_seed"]) == 42,
                f"Background protocol changed: {dataset}")
        require(int(details["n_train"]) == spec["n_train"] and int(details["d"]) == spec["d"],
                f"Cohort dimensions changed: {dataset}")
        require(len(details["patient_ids"]) == 5 and len(set(details["patient_ids"])) == 5,
                f"Expected five distinct held-out patients: {dataset}")
        require(digest(original / dataset / "inputs.npz") == details["prepared_inputs_sha256"],
                f"Prepared model, background, or patient inputs changed: {dataset}")
        require(provenance["input_sha256"][dataset] == details["prepared_inputs_sha256"],
                f"Rerun inputs differ from the exact baseline: {dataset}")
        require(digest(original / dataset / "reference_64.npy") == provenance["reference_sha256"][dataset],
                f"Independent reference changed since the rerun: {dataset}")

    gpu_rows, fp64_rows = read_csv(folder / "gpu_summary.csv"), read_csv(folder / "fp64_summary.csv")
    require(len(gpu_rows) == len(fp64_rows) == 30, "Expected thirty GPU and thirty FP64 cases")
    gpu, fp64 = {}, {}
    for rows, indexed in ((gpu_rows, gpu), (fp64_rows, fp64)):
        for row in rows:
            key = case_key(row)
            require(key in expected and key not in indexed, f"Unexpected or duplicated case: {key}")
            require(int(row["m_q"]) > 0, f"Invalid node count: {key}")
            numeric(row["seconds"], "explanation time", positive=True)
            indexed[key] = row
        require(set(indexed) == expected, "Missing experiment cases")
    raw = read_csv(folder / "gpu_raw_timings.csv")
    measured, warmups, warmed = {}, {}, set()
    for row in raw:
        key = case_key(row)
        require(key in expected, f"Unexpected raw timing: {key}")
        shape = row["dataset"], int(row["m_q"])
        numeric(row["seconds"], "raw GPU time", positive=True)
        numeric(row["budget_seconds"], "node selection time")
        require(row["phase"] in ("warmup", "measured"), "Unknown timing phase")
        if row["phase"] == "warmup":
            require(shape not in warmed and key not in warmups, f"Duplicated warm-up: {shape}")
            require(float(row["budget_seconds"]) == 0, "Warm-up must use fixed nodes")
            warmed.add(shape)
            warmups[key] = row
        else:
            require(shape in warmed and key not in measured, f"Missing warm-up or repeated timing: {key}")
            measured[key] = row
    require(set(measured) == expected, "Require exactly one measured GPU call per patient/method")
    require(warmed == {(row["dataset"], int(row["m_q"])) for row in measured.values()},
            "Unused warm-up or missing measured shape")
    require(len(raw) == 30 + len(warmed), "Unexpected number of warm-up/measured calls")

    exact = {}
    for row in read_csv(original / "summary.csv"):
        if row["mode"] != "exact":
            continue
        key = row["dataset"], int(row["patient"])
        require(key not in exact, f"Duplicate reused exact baseline: {key}")
        dataset, patient = key
        require(dataset in COHORTS and patient in range(5), f"Unexpected exact baseline: {key}")
        require(int(row["B"]) == 30 and int(row["d"]) == COHORTS[dataset]["d"],
                f"Mismatched exact background/dimension: {key}")
        require(int(row["m_q"]) == (COHORTS[dataset]["d"] + 1) // 2,
                f"Incorrect exact-degree node count: {key}")
        require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                f"Exact baseline uses a different memory plan: {key}")
        require(row["patient_id"] == str(design[dataset]["patient_ids"][patient]),
                f"Exact baseline uses another patient: {key}")
        numeric(row["seconds"], "reused exact time", positive=True)
        exact[key] = row
    require(set(exact) == {(dataset, patient) for dataset in COHORTS for patient in range(5)},
            "Missing reused exact-degree GPU baseline")

    with np.load(original / "attributions.npz", allow_pickle=False) as old_phi, \
            np.load(folder / "gpu_attributions.npz", allow_pickle=False) as gpu_phi, \
            np.load(folder / "fp64_attributions.npz", allow_pickle=False) as fp64_phi:
        for dataset, spec in COHORTS.items():
            reference = np.load(original / dataset / "reference_64.npy", allow_pickle=False)
            require(reference.shape == (5, spec["d"]) and np.isfinite(reference).all(),
                    f"Invalid independent FP64 reference: {dataset}")
            ref_validation = read_json(original / dataset / "reference_validation.json")
            require(int(ref_validation["B"]) == 30 and int(ref_validation["m_q"]) == 64,
                    "FP64 reference uses a different background/rule")
            require(max(ref_validation["quadrature_bounds"]) < 1e-10,
                    "Independent reference quadrature bound is not sufficiently small")
            for patient in range(5):
                exact_phi = old_phi[f"{dataset}_p{patient}_exact_None"]
                require(exact_phi.shape == (spec["d"],) and np.isfinite(exact_phi).all(),
                        f"Invalid exact-degree GPU attribution: {dataset}, {patient}")
                close(exact[(dataset, patient)]["max_absolute_error"],
                      np.max(np.abs(exact_phi - reference[patient])), "reused exact reference error")
                for mode in MODES:
                    key, eps = (dataset, patient, mode), float(mode)
                    row, cpu = gpu[key], fp64[key]
                    require(int(row["B"]) == 30 and int(row["d"]) == spec["d"],
                            f"GPU background/dimension mismatch: {key}")
                    require(row["patient_id"] == str(design[dataset]["patient_ids"][patient]),
                            f"GPU patient mismatch: {key}")
                    require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                            f"GPU memory plan changed: {key}")
                    require(int(row["m_q"]) == int(cpu["m_q"]) == int(measured[key]["m_q"]),
                            f"GPU and FP64 node counts differ: {key}")
                    close(row["seconds"], measured[key]["seconds"], f"raw timing {key}")
                    close(row["warmup_seconds"], warmups.get(key, {}).get("seconds", 0), f"warm-up time {key}")
                    bound = numeric(row["bound"], "analytical quadrature bound")
                    require(bound <= eps, f"Quadrature certificate exceeds requested tolerance: {key}")
                    array_key = f"{dataset}_p{patient}_eps_{eps}"
                    for arrays, target, suffix in ((gpu_phi, row, "gpu"), (fp64_phi, cpu, "fp64")):
                        phi = arrays[array_key]
                        require(phi.shape == (spec["d"],) and np.isfinite(phi).all(),
                                f"Invalid saved {suffix} attribution: {key}")
                        delta = phi - reference[patient]
                        error = float(np.max(np.abs(delta)))
                        relative = float(np.linalg.norm(delta) / np.linalg.norm(reference[patient]))
                        close(target["max_absolute_error_vs_fp64_reference"], error, f"{suffix} reference error {key}")
                        close(target["relative_l2_error_vs_fp64_reference"], relative, f"{suffix} relative error {key}")
                        field = "within_eps_vs_fp64_reference" if suffix == "gpu" else "within_eps"
                        check_boolean(target[field], error <= eps, f"{suffix} reference threshold {key}")
                    delta = gpu_phi[array_key] - exact_phi
                    error = float(np.max(np.abs(delta)))
                    close(row["max_absolute_error_vs_gpu_exact"], error, f"GPU degree-exact discrepancy {key}")
                    close(row["relative_l2_error_vs_gpu_exact"],
                          np.linalg.norm(delta) / np.linalg.norm(exact_phi), f"GPU relative discrepancy {key}")
                    check_boolean(row["within_eps_vs_gpu_exact"], error <= eps, f"GPU degree-exact threshold {key}")
    return gpu, fp64, exact, design


def build(folder):
    gpu, fp64, exact, design = validate(folder)
    original = folder.parent
    rules = read_json(original / "rule_setup.json")
    setup = {dataset: numeric(rules[dataset]["original_construction_seconds"], "exact rule setup", positive=True)
             for dataset in COHORTS}
    small_rules = read_csv(folder / "small_rule_timings.csv")
    needed = {int(row["m_q"]) for row in gpu.values()}
    require(len(small_rules) == len(needed) and {int(row["m_q"]) for row in small_rules} == needed,
            "Small-rule construction timing is incomplete")
    small_ms = []
    for row in small_rules:
        require(int(row["repeats"]) == 5, "Require five fresh constructions per approximate rule")
        low, med, high = [numeric(row[field], field, positive=True)
                          for field in ("min_seconds", "median_seconds", "max_seconds")]
        require(low <= med <= high, "Inconsistent small-rule times")
        small_ms.append(1000 * med)
    aggregates, by_case = [], {}
    for dataset in COHORTS:
        for mode in ("exact", *MODES):
            group = [exact[(dataset, p)] if mode == "exact" else gpu[(dataset, p, mode)] for p in range(5)]
            times = [float(row["seconds"]) for row in group]
            result = {"dataset": dataset, "method": mode, "B": 30, "patients": 5,
                      "calls_per_patient": 1, "nodes_min": min(int(r["m_q"]) for r in group),
                      "nodes_max": max(int(r["m_q"]) for r in group),
                      "gpu_seconds_mean": statistics.mean(times),
                      "gpu_seconds_sample_sd": statistics.stdev(times),
                      "timing_source": "reused matched exact baseline" if mode == "exact" else "fresh tolerance rerun",
                      "C_max": 0 if mode == "exact" else max(float(r["bound"]) for r in group),
                      "delta_gpu": "" if mode == "exact" else max(float(r["max_absolute_error_vs_gpu_exact"]) for r in group),
                      "pass_gpu_exact": "" if mode == "exact" else sum(float(r["max_absolute_error_vs_gpu_exact"]) <= float(mode) for r in group),
                      "worst_gpu_error_vs_fp64_reference": max(float(r["max_absolute_error"] if mode == "exact" else r["max_absolute_error_vs_fp64_reference"]) for r in group),
                      "pass_gpu_fp64_reference": "" if mode == "exact" else sum(float(r["max_absolute_error_vs_fp64_reference"]) <= float(mode) for r in group),
                      "worst_fp64_error_vs_fp64_reference": "" if mode == "exact" else max(float(fp64[(dataset, p, mode)]["max_absolute_error_vs_fp64_reference"]) for p in range(5)),
                      "pass_fp64_reference": "" if mode == "exact" else sum(float(fp64[(dataset, p, mode)]["max_absolute_error_vs_fp64_reference"]) <= float(mode) for p in range(5))}
            by_case[(dataset, mode)] = result
            aggregates.append(result)
    for row in aggregates:
        row["speedup_vs_reused_exact_mean"] = by_case[(row["dataset"], "exact")]["gpu_seconds_mean"] / row["gpu_seconds_mean"]

    latex_rows, markdown_rows, check_rows, markdown_checks = [], [], [], []
    for dataset, spec in COHORTS.items():
        label = "GSE24080" if dataset == "gse24080" else "TCGA LGG"
        if latex_rows:
            latex_rows.append(r"\midrule")
        for mode in ("exact", *MODES):
            row = by_case[(dataset, mode)]
            method = "Exact degree" if mode == "exact" else rf"$\varepsilon=10^{{{int(math.log10(float(mode)))}}}$"
            nodes = node_range([row["nodes_min"], row["nodes_max"]])
            time = f"${row['gpu_seconds_mean']:.3f}\\pm{row['gpu_seconds_sample_sd']:.3f}$"
            delta = "---" if mode == "exact" else sci(row["delta_gpu"])
            passes = "---" if mode == "exact" else f"{row['pass_gpu_exact']}/5"
            latex_rows.append(" & ".join([label if mode == "exact" else "", method, nodes, time,
                                           sci(row["C_max"]), delta, passes]) + r" \\")
            markdown_rows.append(f"| {label} | {mode} | {nodes.replace('--', '–')} | "
                                 f"{row['gpu_seconds_mean']:.3f} ± {row['gpu_seconds_sample_sd']:.3f} | "
                                 f"{row['C_max']:.3g} | " + ("— | — |" if mode == "exact" else
                                 f"{row['delta_gpu']:.3g} | {passes} |"))
            if mode != "exact":
                check_rows.append(" & ".join([label, method, sci(row["worst_gpu_error_vs_fp64_reference"]),
                                               f"{row['pass_gpu_fp64_reference']}/5",
                                               sci(row["worst_fp64_error_vs_fp64_reference"]),
                                               f"{row['pass_fp64_reference']}/5"]) + r" \\")
                markdown_checks.append(f"| {label} | {mode} | {row['worst_gpu_error_vs_fp64_reference']:.3g} | "
                                       f"{row['pass_gpu_fp64_reference']}/5 | {row['worst_fp64_error_vs_fp64_reference']:.3g} | "
                                       f"{row['pass_fp64_reference']}/5 |")
    setup_text = (f"Previously measured CPU construction of the cached exact rules took {setup['gse24080']:.2f}\\,s "
                  f"(GSE24080) and {setup['tcga_lgg_methylation']:.2f}\\,s (TCGA LGG); fresh approximate-rule "
                  f"construction took {min(small_ms):.3f}--{max(small_ms):.3f}\\,ms "
                  "(range of medians over five constructions).")
    caption = r"""Cox explanations on an Apple M4 Pro GPU using JAX-Metal FP32,
30 fixed training-background samples, and five distinct held-out patients.
Time is mean $\pm$ sample standard deviation across patients, with one
synchronized call per patient and method; node ranges span patients.
Approximate results are newly measured; exact-degree rows reuse the matching
previously measured baseline with identical model, inputs, numerical sources,
software versions, and memory plan.
$C_{\max}$ is the largest analytical absolute quadrature-error bound across
patients. $\Delta_{\mathrm{GPU}}$ is the largest absolute difference from
the matching GPU degree-exact output over all patients and features.
Pass counts patients whose maximum feature-wise difference is at most
$\varepsilon$. The certificate excludes floating-point error; GPU
degree-exact output is itself subject to rounding.
Warm-up and rule construction are excluded; automatic node selection is included.
""" + setup_text
    table = r"""% Requires \usepackage{booktabs}.
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrc}
\toprule
Dataset & Method & Nodes & GPU time (s) & $C_{\max}$ & $\Delta_{\mathrm{GPU}}$ & Pass \\
\midrule
""" + "\n".join(latex_rows) + "\n" + r"""\bottomrule
\end{tabular}
\caption{""" + caption + r"""}
\label{tab:cox-high-dimensional}
\end{table}
"""

    # Reuse the reviewed experimental settings, but replace the old tolerance set
    # and make the provenance of the exact baseline explicit.
    settings = (original / "manuscript/cox_high_dimensional_subsection.tex").read_text().split(
        "Table~\\ref{tab:cox-high-dimensional}", 1)[0]
    settings = settings.replace(r"\{10^{-1},10^{-3},10^{-6}\}", r"\{10^{-1},10^{-2},10^{-3}\}")
    settings = settings.replace("All methods use JAX-Metal", "The timing experiments use JAX-Metal")
    settings += ("The approximate GPU experiments were rerun for these three tolerances.\n"
                 "The degree-exact baseline is reused from the matching five-patient experiment;\n"
                 "the model, background rows, patient inputs, numerical source hashes,\n"
                 "software versions, and GPU memory plan were verified to match.\n\n")
    gse_exact, tcga_exact = [by_case[(name, "exact")]["gpu_seconds_mean"] for name in COHORTS]
    ranges = {name: [by_case[(name, mode)]["gpu_seconds_mean"] * 1000 for mode in MODES] for name in COHORTS}
    settings += (f"Table~\\ref{{tab:cox-high-dimensional}} shows that degree-exact quadrature remains\n"
                 f"manageable even at these dimensions: 27,338 and 198,033 nodes require\n"
                 f"{gse_exact:.2f}\\,s and {tcga_exact:.2f}\\,s per patient on average, respectively.\n"
                 f"For latency-sensitive use, the certified node budgets reduce the measured\n"
                 f"latencies to {min(ranges['gse24080']):.0f}--{max(ranges['gse24080']):.0f}\\,ms for GSE24080\n"
                 f"and {min(ranges['tcga_lgg_methylation']):.0f}--{max(ranges['tcga_lgg_methylation']):.0f}\\,ms for TCGA LGG.\n"
                 "These subsecond latencies support interactive, on-demand Shapley\n"
                 "explanations once the model and quadrature rules are ready. At\n"
                 "$\\varepsilon=10^{-1}$, the speedups relative to the reused exact means are\n"
                 f"${by_case[('gse24080', '1e-1')]['speedup_vs_reused_exact_mean']:.1f}\\times$ and\n"
                 f"${by_case[('tcga_lgg_methylation', '1e-1')]['speedup_vs_reused_exact_mean']:.1f}\\times$, respectively.\n"
                 "The exact-rule construction costs are substantial but can be amortized\n"
                 "across patients by caching the nodes and weights.\n\n")
    settings += r"""The analytical certificate and measured numerical discrepancy answer
different questions. For patient $p$, let $\phi_p$ be the exact Shapley
vector for the fixed background game and $\phi_{m_p,p}$ its quadrature
approximation in real arithmetic. The selected node count satisfies
\[
\|\phi_{m_p,p}-\phi_p\|_\infty\le C_p\le\varepsilon,
\qquad C_{\max}=\max_{p=1,\ldots,5}C_p.
\]
Thus $C_{\max}$ is an evaluated analytical bound, not an observed error.
The table's empirical discrepancy instead compares two floating-point outputs:
\[
\Delta_{\mathrm{GPU}}=
\max_p\|\widehat\phi^{\mathrm{GPU}}_{m_p,p}
-\widehat\phi^{\mathrm{GPU}}_{\mathrm{exact},p}\|_\infty.
\]
Every selected rule satisfies its analytical quadrature bound.
"""
    for mode in MODES:
        g, t = by_case[("gse24080", mode)], by_case[("tcga_lgg_methylation", mode)]
        settings += (f"At $\\varepsilon=10^{{{int(math.log10(float(mode)))}}}$, the observed GPU\n"
                     f"discrepancy is within the requested threshold for {g['pass_gpu_exact']}/5 GSE24080\n"
                     f"and {t['pass_gpu_exact']}/5 TCGA LGG patients.\n")
    settings += r"""
Agreement with the GPU degree-exact result alone does not establish exact
numerical accuracy, because that baseline also contains FP32 rounding error.
We therefore additionally compare both the GPU approximation and a separate
FP64 CPU evaluation at the same selected nodes with an independent,
tightly bounded 64-node FP64 reference. This reference has analytical
quadrature bounds below $10^{-10}$; a 96-node cross-check is also available
for the first patient of each cohort. It is a high-accuracy numerical
reference, rather than a full degree-exact FP64 computation.
The FP64 CPU evaluation is an accuracy diagnostic; its errors are never
paired with the GPU timings.

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
 & & \multicolumn{2}{c}{GPU FP32} & \multicolumn{2}{c}{CPU FP64} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Dataset & Tolerance & Worst error & Pass & Worst error & Pass \\
\midrule
""" + "\n".join(check_rows) + "\n" + r"""\bottomrule
\end{tabular}
\caption{Separate precision checks against the same independent FP64
reference. Worst error is the maximum absolute feature-wise error over all
five patients; Pass counts patients whose maximum error is at most their
requested tolerance. CPU FP64 entries are accuracy diagnostics and are
not the errors of the GPU runs timed in Table~\ref{tab:cox-high-dimensional}.}
\label{tab:cox-precision-checks}
\end{table}

"""
    fp64_passes = sum(row["pass_fp64_reference"] for row in aggregates if row["method"] != "exact")
    gpu_passes = sum(row["pass_gpu_fp64_reference"] for row in aggregates if row["method"] != "exact")
    fp64_worst = max(row["worst_fp64_error_vs_fp64_reference"] for row in aggregates if row["method"] != "exact")
    settings += (f"The separate FP64 evaluation meets its requested threshold in {fp64_passes}/30\n"
                 f"patient--tolerance cases, with a worst absolute difference of {sci(fp64_worst)},\n"
                 f"whereas the timed FP32 GPU evaluations meet it in {gpu_passes}/30 cases\n"
                 "against that same reference.\n")
    for dataset, spec in COHORTS.items():
        for mode in MODES:
            row = by_case[(dataset, mode)]
            if row["pass_gpu_exact"] == 5 and row["pass_gpu_fp64_reference"] < 5:
                settings += (f"For {spec['label']} at $\\varepsilon=10^{{{int(math.log10(float(mode)))}}}$,\n"
                             "all five GPU approximations agree with their GPU degree-exact outputs\n"
                             f"within the threshold, but only {row['pass_gpu_fp64_reference']}/5 agree with the independent\n"
                             f"FP64 reference within that threshold (worst error {sci(row['worst_gpu_error_vs_fp64_reference'])}).\n"
                             "This illustrates why agreement between two FP32 computations alone\n"
                             "cannot establish the requested numerical accuracy.\n")
    settings += r"""These comparisons distinguish certified quadrature approximation from
end-to-end floating-point accuracy. Higher precision can reduce numerical
error, but the analytical certificate itself does not include rounding.
It also does not bound uncertainty from replacing a population background
distribution with the chosen 30-row empirical background.
"""
    readme = ("# Cox tolerance rerun: 30 backgrounds, five distinct patients\n\n"
              "All approximation rows are fresh GPU measurements. Exact-degree GPU rows reuse the matched "
              "saved baseline; input, numerical-source, and software-version checks passed. "
              "Each entry is the mean ± sample standard deviation of five calls, one per patient. "
              "The standard deviation describes variation across patients, not repeated-run uncertainty.\n\n"
              "| Dataset | Method | Nodes | GPU time (s) | C_max | Delta_GPU | Pass |\n"
              "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(markdown_rows) + "\n\n"
              "C_max is the largest analytical absolute quadrature bound; every patient's bound is at most "
              "its requested epsilon. Delta_GPU is the largest absolute discrepancy from the same-patient "
              "degree-exact GPU result. Pass counts use unrounded per-patient errors. The degree-exact GPU "
              "result is subject to rounding and is not numerical ground truth.\n\n"
              "## Independent precision checks\n\n"
              "The following errors use a separate 64-node FP64 reference with quadrature bounds below 1e-10. "
              "The first patient in each cohort additionally has a 96-node cross-check. The CPU FP64 "
              "diagnostic uses the same node count as the corresponding GPU run, but these are separate "
              "evaluations: CPU errors are never attributed to GPU timings.\n\n"
              "| Dataset | Tolerance | GPU FP32 worst error | GPU pass | CPU FP64 worst error | CPU pass |\n"
              "|---|---|---:|---:|---:|---:|\n" + "\n".join(markdown_checks) + "\n\n"
              f"Against the FP64 reference, GPU FP32 passes {gpu_passes}/30 cases and CPU FP64 passes "
              f"{fp64_passes}/30. The largest CPU FP64 difference is {fp64_worst:.3g}. "
              "These are empirical checks, not floating-point error certificates.\n\n"
              "In particular, at 1e-2 the TCGA GPU approximations agree with their degree-exact GPU "
              f"counterparts for {by_case[('tcga_lgg_methylation', '1e-2')]['pass_gpu_exact']}/5 patients, "
              "but meet the same threshold against the independent FP64 reference for only "
              f"{by_case[('tcga_lgg_methylation', '1e-2')]['pass_gpu_fp64_reference']}/5 patients. "
              "Agreement with a rounded GPU baseline alone therefore does not demonstrate absolute accuracy.\n\n"
              "## Timing scope and setup\n\n"
              "Apple M4 Pro, JAX-Metal FP32, an 8 GiB memory-planning budget, and one background row per "
              "block. One untimed explicit-node warm-up precedes the first measured call for each distinct "
              "dataset/node-count configuration. Synchronized API timings include automatic node selection, "
              "data transfers, and host accumulation; model fitting, warm-up, and rule construction are excluded.\n\n"
              f"Historical exact-rule CPU construction: GSE24080 {setup['gse24080']:.2f} s; TCGA LGG "
              f"{setup['tcga_lgg_methylation']:.2f} s. Fresh approximate-rule CPU construction: "
              f"{min(small_ms):.3f}–{max(small_ms):.3f} ms, the range of medians from five fresh constructions "
              "per distinct rule. Rules are cached for the timing measurements.\n\n"
              "## Files\n\n"
              "- `cox_tolerance_table.tex`: seven-column main GPU table.\n"
              "- `cox_tolerance_analysis.tex`: experimental subsection and separate precision-check table.\n"
              "- `table.csv`: full-precision aggregate data, including speedups and both error comparisons.\n"
              "- `preview.tex`: standalone LaTeX document; use BibTeX for the included source references.\n"
              "- Raw GPU timings, GPU/FP64 summaries, attribution arrays, status, environment, and provenance "
              "files are retained alongside these reports.\n")
    preview = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,booktabs,url}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\renewcommand{\thesubsection}{\arabic{subsection}}
\begin{document}
\input{cox_tolerance_table.tex}
\input{cox_tolerance_analysis.tex}
\begingroup
\raggedright
\bibliographystyle{plain}
\bibliography{cox_high_dimensional_sources}
\endgroup
\end{document}
"""
    require("10^{-6}" not in table + settings, "Obsolete tolerance leaked into the new report")
    write_csv(folder / "table.csv", aggregates)
    (folder / "cox_tolerance_table.tex").write_text(table)
    (folder / "cox_tolerance_analysis.tex").write_text(settings)
    (folder / "README.md").write_text(readme)
    (folder / "preview.tex").write_text(preview)
    (folder / "cox_high_dimensional_sources.bib").write_text(
        (original / "manuscript/cox_high_dimensional_sources.bib").read_text())
    print(f"Validated 30 fresh GPU cases, 30 FP64 diagnostics, and 10 matched reused exact cases; wrote {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT)
    build(parser.parse_args().folder)
