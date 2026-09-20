"""Report Cox approximations against degree-exact output on the same backend.

The GPU reference is the saved FP32 GPU degree-exact run.  The CPU reference
is the newly completed FP64 CPU degree-exact run.  No partial report is written.
Earlier independent-reference comparisons are validated only as provenance;
they are not substituted for either discrepancy in the new main table.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import statistics

import numpy as np

from benchmarks.cox_background30_table import COHORTS, node_range, numeric, read_csv, read_json, require, write_csv
from benchmarks.cox_tolerance_rerun_report import MODES, close, digest, sci
from benchmarks.cox_cpu_gpu_precision_report import OUT as CPU_OUT, method_tex, time_tex, validate as validate_controls

OUT = CPU_OUT / "cpu_exact_degree"


def validate(folder):
    """Read-only check of complete exact runs and matched approximation inputs."""
    cpu_folder, gpu_folder, original = folder.parent, folder.parent.parent, folder.parent.parent.parent
    gpu, cpu, gpu_exact, design, _ = validate_controls(cpu_folder)
    status = read_json(folder / "status.json")
    require(status["phase"] == "complete", "CPU exact experiment is not complete; partial reports are forbidden")
    require(int(status["cases"]) == 10, "Require all ten CPU degree-exact patients")
    if "measured_calls" in status:
        require(int(status["measured_calls"]) == 10, "Require one measured CPU exact call per patient")
    env = read_json(folder / "environment.json")
    approx_env = read_json(cpu_folder / "cpu_fp64_environment.json")
    require(env["backend"].lower() == "cpu" and env["jax_enable_x64"] and not env["jit_disabled"],
            "CPU exact baseline must use JIT-enabled FP64 CPU execution")
    require(env["source_sha256"] == approx_env["source_sha256"], "CPU exact and approximate numerical sources differ")
    for package in ("jax", "jaxlib", "numpy", "scipy"):
        require(env["packages"][package] == approx_env["packages"][package],
                f"CPU exact and approximate package versions differ: {package}")
    provenance = read_json(folder / "provenance.json")
    require(provenance, "Missing CPU degree-exact provenance")
    require(provenance["exact_cpu_measured"], "CPU full-degree reference has not been measured")
    for directory, group in ((cpu_folder, "cpu_approx_sha256"), (gpu_folder, "gpu_approx_sha256"),
                             (original, "gpu_exact_sha256")):
        for filename, sha in provenance[group].items():
            require(digest(directory / filename) == sha, f"Matched source changed since CPU exact execution: {filename}")
    validation = read_json(folder / "stream_validation/validation.json")
    require(validation["bitwise_equal_to_public_api"], "Checkpointed CPU loop lacks a successful public-API check")
    for dataset in COHORTS:
        require(digest(original / dataset / "inputs.npz") == provenance["input_sha256"][dataset],
                f"CPU degree-exact uses different model/background/patient inputs: {dataset}")
    rows = read_csv(folder / "summary.csv")
    require(len(rows) == 10, "All ten CPU exact timing rows are required")
    expected = {(dataset, patient) for dataset in COHORTS for patient in range(5)}
    exact = {}
    for row in rows:
        key = row["dataset"], int(row["patient"])
        require(key in expected and key not in exact, f"Unexpected/duplicate CPU exact case: {key}")
        dataset, patient = key
        d = COHORTS[dataset]["d"]
        require(int(row["B"]) == 30 and int(row["d"]) == d, f"CPU exact background/dimension mismatch: {key}")
        require(row["patient_id"] == str(design[dataset]["patient_ids"][patient]), f"CPU exact patient mismatch: {key}")
        require(int(row["m_q"]) == (d + 1) // 2, f"CPU exact rule has incorrect degree: {key}")
        require(row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                f"CPU exact uses a different memory/background block plan: {key}")
        require(row["core_dtype"] == "float64", f"CPU degree-exact core is not FP64: {key}")
        numeric(row["seconds"], "CPU degree-exact runtime", positive=True)
        numeric(row["warmup_seconds"], "CPU degree-exact warm-up time")
        require(0 < int(row["node_block"]) <= int(row["m_q"]), f"Invalid CPU exact node block: {key}")
        phi = np.load(folder / f"{dataset}_p{patient}_exact.npy", allow_pickle=False)
        require(phi.dtype == np.dtype(np.float64) and phi.shape == (d,) and np.isfinite(phi).all(),
                f"Invalid CPU degree-exact attribution vector: {key}")
        require(digest(folder / f"{dataset}_p{patient}_exact.npy") == row["attribution_sha256"],
                f"CPU exact attribution changed after measurement: {key}")
        progress = read_json(folder / dataset / f"patient_{patient}" / "exact_progress.json")
        require(progress["phase"] == "complete", f"CPU degree-exact integration is incomplete: {key}")
        numeric(row["checkpoint_io_seconds"], "excluded checkpoint IO time")
        exact[key] = row
    require(set(exact) == expected, "CPU exact patients are incomplete")
    raw_path = folder / "raw_timings.csv"
    if raw_path.exists():
        measured = {}
        for row in read_csv(raw_path):
            key = row["dataset"], int(row["patient"])
            require(key in expected, f"Unexpected CPU exact raw patient: {key}")
            require(row["phase"] in ("warmup", "measured"), "Unknown CPU exact timing phase")
            numeric(row["seconds"], "CPU exact raw time", positive=True)
            if row["phase"] == "measured":
                require(key not in measured, f"Repeated CPU exact measured call: {key}")
                require(int(row["m_q"]) == int(exact[key]["m_q"]), f"CPU exact raw node count mismatch: {key}")
                close(row["seconds"], exact[key]["seconds"], f"CPU exact raw/summary timing: {key}")
                measured[key] = row
        require(set(measured) == expected, "CPU exact raw timings are incomplete")

    errors, independent_audit = [], []
    with np.load(original / "attributions.npz", allow_pickle=False) as saved_gpu_exact, \
            np.load(gpu_folder / "gpu_attributions.npz", allow_pickle=False) as gpu_phi, \
            np.load(cpu_folder / "cpu_fp64_attributions.npz", allow_pickle=False) as cpu_phi:
        for dataset, spec in COHORTS.items():
            independent = np.load(original / dataset / "reference_64.npy", allow_pickle=False)
            for patient in range(5):
                cpu_ref = np.load(folder / f"{dataset}_p{patient}_exact.npy", allow_pickle=False)
                gpu_ref = saved_gpu_exact[f"{dataset}_p{patient}_exact_None"]
                close(exact[(dataset, patient)]["max_absolute_error_vs_auxiliary_reference"],
                      np.max(np.abs(cpu_ref - independent[patient])), f"CPU exact auxiliary reference diagnostic: {dataset}, {patient}")
                independent_audit.append({"dataset": dataset, "patient": patient,
                                          "cpu_degree_exact_vs_independent_fp64_reference": float(np.max(np.abs(cpu_ref - independent[patient]))),
                                          "gpu_degree_exact_vs_independent_fp64_reference": float(np.max(np.abs(gpu_ref - independent[patient]))),
                                          "purpose": "Separate numerical audit; not either Delta column in the main table"})
                for mode in MODES:
                    key, eps = (dataset, patient, mode), float(mode)
                    name = f"{dataset}_p{patient}_eps_{eps}"
                    gpu_difference, cpu_difference = gpu_phi[name] - gpu_ref, cpu_phi[name] - cpu_ref
                    gpu_error, cpu_error = float(np.max(np.abs(gpu_difference))), float(np.max(np.abs(cpu_difference)))
                    # Recompute the GPU discrepancy from arrays and cross-check its
                    # original saved value; the new CPU discrepancy has no proxy.
                    close(gpu_error, gpu[key]["max_absolute_error_vs_gpu_exact"], f"GPU same-device exact discrepancy: {key}")
                    errors.append({"dataset": dataset, "patient": patient,
                                   "patient_id": str(design[dataset]["patient_ids"][patient]),
                                   "B": 30, "d": spec["d"], "eps": eps, "m_q": int(gpu[key]["m_q"]),
                                   "bound": float(gpu[key]["bound"]),
                                   "delta_gpu_vs_gpu_fp32_degree_exact": gpu_error,
                                   "delta_cpu_vs_cpu_fp64_degree_exact": cpu_error,
                                   "gpu_within_eps": gpu_error <= eps, "cpu_within_eps": cpu_error <= eps,
                                   "gpu_relative_l2_vs_gpu_degree_exact": float(np.linalg.norm(gpu_difference) / np.linalg.norm(gpu_ref)),
                                   "cpu_relative_l2_vs_cpu_degree_exact": float(np.linalg.norm(cpu_difference) / np.linalg.norm(cpu_ref))})
    return gpu, cpu["fp64"], gpu_exact, exact, design, errors, independent_audit, provenance


def aggregate(gpu, cpu, gpu_exact, cpu_exact, errors):
    indexed_errors = {(r["dataset"], r["patient"], float(r["eps"])): r for r in errors}
    rows = []
    for dataset in COHORTS:
        for mode in ("exact", *MODES):
            is_exact = mode == "exact"
            gpu_rows = [gpu_exact[(dataset, p)] if is_exact else gpu[(dataset, p, mode)] for p in range(5)]
            cpu_rows = [cpu_exact[(dataset, p)] if is_exact else cpu[(dataset, p, mode)] for p in range(5)]
            gpu_times, cpu_times = [[float(row["seconds"]) for row in group] for group in (gpu_rows, cpu_rows)]
            errors_here = [] if is_exact else [indexed_errors[(dataset, p, float(mode))] for p in range(5)]
            row = {"dataset": dataset, "method": mode, "B": 30, "patients": 5, "calls_per_patient": 1,
                   "nodes_min": min(int(r["m_q"]) for r in gpu_rows), "nodes_max": max(int(r["m_q"]) for r in gpu_rows),
                   "C_max": 0 if is_exact else max(float(r["bound"]) for r in gpu_rows),
                   "gpu_fp32_seconds_mean": statistics.mean(gpu_times),
                   "gpu_fp32_seconds_sample_sd": statistics.stdev(gpu_times),
                   "cpu_fp64_seconds_mean": statistics.mean(cpu_times),
                   "cpu_fp64_seconds_sample_sd": statistics.stdev(cpu_times),
                   "delta_gpu_vs_gpu_fp32_degree_exact": 0 if is_exact else max(r["delta_gpu_vs_gpu_fp32_degree_exact"] for r in errors_here),
                   "delta_cpu_vs_cpu_fp64_degree_exact": 0 if is_exact else max(r["delta_cpu_vs_cpu_fp64_degree_exact"] for r in errors_here),
                   "gpu_pass": "" if is_exact else sum(r["gpu_within_eps"] for r in errors_here),
                   "cpu_pass": "" if is_exact else sum(r["cpu_within_eps"] for r in errors_here),
                   "gpu_timing_source": "reused matched GPU experiment",
                   "cpu_timing_source": "new full degree-exact CPU run" if is_exact else "saved warmed CPU approximation run",
                   "gpu_error_reference": "same-patient GPU FP32 degree-exact output",
                   "cpu_error_reference": "same-patient CPU FP64 degree-exact output"}
            require(all(int(g["m_q"]) == int(c["m_q"]) for g, c in zip(gpu_rows, cpu_rows)),
                    f"CPU/GPU quadrature node counts differ: {dataset}, {mode}")
            rows.append(row)
    indexed = {(r["dataset"], r["method"]): r for r in rows}
    for row in rows:
        exact = indexed[(row["dataset"], "exact")]
        row["gpu_speedup_vs_gpu_exact"] = exact["gpu_fp32_seconds_mean"] / row["gpu_fp32_seconds_mean"]
        row["cpu_speedup_vs_cpu_exact"] = exact["cpu_fp64_seconds_mean"] / row["cpu_fp64_seconds_mean"]
    return rows


def build(folder):
    gpu, cpu, gpu_exact, cpu_exact, design, errors, audit, provenance = validate(folder)
    rows = aggregate(gpu, cpu, gpu_exact, cpu_exact, errors)
    by_case = {(r["dataset"], r["method"]): r for r in rows}
    latex_rows, md_rows = [], []
    for row in rows:
        dataset, mode = row["dataset"], row["method"]
        label = "GSE24080" if dataset == "gse24080" else "TCGA LGG"
        if dataset == "tcga_lgg_methylation" and mode == "exact":
            latex_rows.append(r"\midrule")
        nodes = node_range([row["nodes_min"], row["nodes_max"]])
        gpu_pass = "---" if mode == "exact" else f"{row['gpu_pass']}/5"
        cpu_pass = "---" if mode == "exact" else f"{row['cpu_pass']}/5"
        latex_rows.append(" & ".join([label if mode == "exact" else "", method_tex(mode), nodes,
                                       sci(row["C_max"]), time_tex(row["gpu_fp32_seconds_mean"], row["gpu_fp32_seconds_sample_sd"]),
                                       sci(row["delta_gpu_vs_gpu_fp32_degree_exact"]), gpu_pass,
                                       time_tex(row["cpu_fp64_seconds_mean"], row["cpu_fp64_seconds_sample_sd"]),
                                       sci(row["delta_cpu_vs_cpu_fp64_degree_exact"]), cpu_pass]) + r" \\")
        md_rows.append(f"| {label} | {mode} | {nodes.replace('--', '–')} | {row['C_max']:.3g} | "
                       f"{row['gpu_fp32_seconds_mean']:.3f} ± {row['gpu_fp32_seconds_sample_sd']:.3f} | "
                       f"{row['delta_gpu_vs_gpu_fp32_degree_exact']:.3g} | {gpu_pass} | "
                       f"{row['cpu_fp64_seconds_mean']:.3f} ± {row['cpu_fp64_seconds_sample_sd']:.3f} | "
                       f"{row['delta_cpu_vs_cpu_fp64_degree_exact']:.3g} | {cpu_pass} |")
    table = r"""% Requires \usepackage{amsmath,booktabs,graphicx}.
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
""" + "\n".join(latex_rows) + "\n" + r"""\bottomrule
\end{tabular}%
}
\caption{Cox explanations with 30 fixed training-background samples and five
distinct held-out patients per cohort on an Apple M4 Pro. Times are mean
$\pm$ sample standard deviation across patients, with one synchronized
measurement per patient and method. $C_{\max}$ is the largest analytical
absolute quadrature-error bound across patients.
$\Delta_{\mathrm{GPU}}$ is the largest absolute feature-wise difference
from the matching GPU FP32 degree-exact output;
$\Delta_{\mathrm{CPU}}$ uses the matching CPU FP64 degree-exact output.
Both maxima range over all five patients and features. Pass counts
patients whose difference from their same-backend reference is at most
$\varepsilon$. Exact-row differences are zero by construction, not evidence
of zero floating-point error. CPU degree-exact measurements are new;
GPU results and CPU approximate measurements reuse the matched saved runs.
Warm-up and quadrature-rule construction are excluded; approximate API
times include automatic node selection. Previously measured exact-rule
construction required 10.97\,s for GSE24080 and 561.92\,s for TCGA;
approximate-rule construction required 0.386--0.563\,ms.
CPU exact timings measure active
checkpointed integration with the same factor/block order as the public
API, excluding checkpoint I/O and preliminary kernel-shape pilot calls.
The certificate excludes rounding,
and agreement with a rounded reference is not an absolute numerical guarantee.}
\label{tab:cox-device-exact}
\end{table*}
"""
    gpu_ms = {dataset: [1000 * by_case[(dataset, mode)]["gpu_fp32_seconds_mean"] for mode in MODES]
              for dataset in COHORTS}
    cpu_ms = {dataset: [1000 * by_case[(dataset, mode)]["cpu_fp64_seconds_mean"] for mode in MODES]
              for dataset in COHORTS}
    cpu_passes = sum(row["cpu_pass"] for row in rows if row["method"] != "exact")
    gpu_passes = sum(row["gpu_pass"] for row in rows if row["method"] != "exact")
    worst_cpu = max(row["delta_cpu_vs_cpu_fp64_degree_exact"] for row in rows if row["method"] != "exact")
    main = r"""\subsection{High-dimensional experiments with QuadraSHAP}
\label{sec:high-dimensional-quadrashap}

To assess whether Shapley attribution remains practical when the number of
features greatly exceeds the number of patients, we consider GSE24080
multiple-myeloma gene expression and TCGA lower-grade glioma DNA methylation
\cite{geo_gse24080,tcga_lgg_methylation}. Genome-wide assays yield 54,675
expression probe sets and 396,065 methylation sites, respectively, whereas
the training cohorts contain only 339 and 383 patients. We fit ridge-penalized
Cox models and explain their relative hazard, whose feature-wise product
structure makes QuadraSHAP directly applicable. Each explanation uses the
same 30 training-background patients; results summarize one timed call for
each of five distinct held-out patients. We compare degree-exact quadrature
with automatically selected node counts for
$\varepsilon\in\{10^{-1},10^{-2},10^{-3}\}$ on an Apple M4 Pro, using
GPU FP32 and CPU FP64 arithmetic.

"""
    main += (f"Table~\\ref{{tab:cox-device-exact}} shows that degree-exact GPU computation\n"
             f"remains manageable: 27,338 and 198,033 nodes require\n"
             f"{by_case[('gse24080', 'exact')]['gpu_fp32_seconds_mean']:.2f}\\,s and\n"
             f"{by_case[('tcga_lgg_methylation', 'exact')]['gpu_fp32_seconds_mean']:.2f}\\,s per patient, respectively.\n"
             f"The certified approximation reduces these latencies to\n"
             f"{min(gpu_ms['gse24080']):.0f}--{max(gpu_ms['gse24080']):.0f}\\,ms and\n"
             f"{min(gpu_ms['tcga_lgg_methylation']):.0f}--{max(gpu_ms['tcga_lgg_methylation']):.0f}\\,ms,\n"
             "supporting interactive, on-demand explanations after warm-up and\n"
             "quadrature-rule setup. CPU FP64 provides a measured alternative when\n"
             f"higher numerical precision is required, with approximate latencies of\n"
             f"{min(cpu_ms['gse24080']):.0f}--{max(cpu_ms['gse24080']):.0f}\\,ms and\n"
             f"{min(cpu_ms['tcga_lgg_methylation']):.0f}--{max(cpu_ms['tcga_lgg_methylation']):.0f}\\,ms, respectively.\n\n")
    main += ("For a matched numerical comparison, each approximation is compared with\n"
             "the degree-exact output from its own device and precision. All selected\n"
             "rules satisfy their analytical quadrature bounds, but FP32 arithmetic\n"
             "limits agreement at the tightest tolerance: at $\\varepsilon=10^{-3}$,\n"
             f"{by_case[('gse24080', '1e-3')]['gpu_pass']}/5 GSE24080 and\n"
             f"{by_case[('tcga_lgg_methylation', '1e-3')]['gpu_pass']}/5 TCGA GPU cases meet that threshold.\n"
             f"CPU FP64 meets its same-backend threshold in {cpu_passes}/30 cases.\n"
             "The guarantee concerns quadrature for the fixed fitted model and\n"
             "empirical background; it excludes rounding and population-background\n"
             "approximation. Agreement with degree-exact floating-point output is\n"
             "therefore an empirical consistency check, not an absolute error proof.\n"
             "Full settings, timing provenance, and error definitions are given in\n"
             "Appendix~\\ref{app:cox-device-exact}.\n")
    original = folder.parent.parent.parent
    source_settings = (original / "manuscript/cox_high_dimensional_subsection.tex").read_text().split(
        "Table~\\ref{tab:cox-high-dimensional}", 1)[0]
    source_settings = source_settings.replace(r"\subsection{High-dimensional experiments with QuadraSHAP}",
                                               r"\subsection{Additional details for high-dimensional Cox experiments}")
    source_settings = source_settings.replace(r"\label{sec:high-dimensional-quadrashap}", r"\label{app:cox-device-exact}")
    source_settings = source_settings.replace(r"\{10^{-1},10^{-3},10^{-6}\}", r"\{10^{-1},10^{-2},10^{-3}\}")
    # Replace the historical GPU-only implementation/timing paragraph. Exact CPU
    # warm-up and timing scope are supplied by the new experiment's provenance.
    source_settings = source_settings.split("All methods use JAX-Metal", 1)[0]
    appendix = source_settings + r"""
\paragraph{Matched degree-exact references.}
To assess approximation accuracy without changing the numerical backend,
we use a separate degree-exact reference for each device/precision setting.
Write $\widehat\phi^{a}_{m_p,p}$ for patient $p$'s computed attribution
vector using $m_p$ nodes, with $a=\mathrm{GPU}$ denoting GPU FP32 and
$a=\mathrm{CPU}$ denoting CPU FP64. Let $m_\star=\lceil d/2\rceil$.
The table reports
\[
\Delta_a=\max_{p=1,\ldots,5}
\|\widehat\phi^{a}_{m_p,p}-\widehat\phi^{a}_{m_\star,p}\|_\infty.
\]
Thus GPU approximation is compared only with GPU FP32 degree-exact output,
and CPU approximation only with CPU FP64 degree-exact output. These are
full degree-exact rules with 27,338 and 198,033 nodes, respectively; the
CPU references are newly measured rather than replaced with smaller-node
FP64 proxies. A patient passes if its maximum absolute feature-wise
discrepancy is at most its requested tolerance. Exact-row discrepancies
are zero by self-comparison and do not assert zero rounding error.

For the exact real-arithmetic Shapley vector $\phi_p$ and its quadrature
approximation $\phi_{m_p,p}$, the analytical node selector instead guarantees
\[
\|\phi_{m_p,p}-\phi_p\|_\infty\le C_p\le\varepsilon,
\qquad C_{\max}=\max_{p=1,\ldots,5}C_p.
\]
This bound can be conservative: the actual quadrature error need not be
close to either $C_p$ or $\varepsilon$. However, both computed vectors in
$\Delta_a$ also contain floating-point error. Consequently, a discrepancy
above $\varepsilon$ need not contradict the quadrature certificate, while
a discrepancy below $\varepsilon$ does not independently establish an
absolute error guarantee for either computed vector.

To avoid choosing unnecessarily many nodes, the implementation searches
integer node counts $m=1,2,\ldots$ and returns the first whose analytical
bound meets the requested tolerance; the degree-exact threshold is the
fallback cap. Evaluating the bound for each candidate requires optimizing
an auxiliary scalar that controls the bound. This inner optimization solves
its scalar stationarity equation by bisection, repeatedly halving an interval
containing the root. The separate \texttt{closed\_form\_budget} calculation
is recorded as a diagnostic and does not select the node count used in
these experiments. Timed automatic node selection includes both the
patient/background factor-summary pass and this integer search.

\paragraph{Implementation and timing.}
GPU explanations use JAX-Metal FP32; CPU explanations use JAX with FP64
enabled. Both evaluate the same parallel product-game formula, use an
8\,GiB memory-planning budget and one background row per block, and
accumulate returned blocks on the host in FP64. The memory setting is a
planning budget, not a measurement of actual allocation.
Each table entry summarizes five distinct held-out patients, with one
timed explanation per patient and method. We report the arithmetic mean
and sample standard deviation across those patients. This variation is
not repeated-run uncertainty and does not establish statistical significance.

GPU exact and approximate measurements are reused from the matched saved
experiments, and CPU approximate timings reuse the warmed CPU FP64 sweep.
The full degree-exact CPU runs are new. Model coefficients, preprocessing,
background and held-out rows, numerical source hashes, and package versions
are verified to match. Timings exclude rule construction and recorded
warm-up. CPU degree-exact computation uses a checkpointed integration loop
with the same factors, node order, and background/block accumulation order
as the public API; a small synthetic comparison was bitwise identical.
Before each dataset, one actual node block with the full exact-core shape
is evaluated for warm-up, followed by three pilot core calls. These calls
are excluded; the warm-up is not a full degree-exact explanation.
The reported CPU exact time sums active integration, including factor
preparation, block planning, transfers, synchronized core evaluation, and
host accumulation. Checkpoint I/O is measured separately and excluded;
if resumed, inactive downtime and uncheckpointed work lost to interruption
are also excluded. Thus this timing is not an undifferentiated process
wall-clock measurement.
Approximate API timings include automatic node selection as well as
explanation, synchronization, transfers and host work. Measurements were
recorded in separate runs. GPU FP32 versus CPU FP64 changes both device
and precision, so their latency ratio is not a matched-precision GPU speedup.

The quadrature nodes and weights are cached across patients. Previously
measured CPU construction of the exact GSE24080 and TCGA rules required
10.97\,s and 561.92\,s, respectively; the latest approximate rules required
0.386--0.563\,ms (range of medians over five fresh constructions per rule).
These setup costs are separate from explanation latency. Caching rules
does not remove evaluations of the patient- and background-dependent
integrands at those nodes.

\paragraph{Numerical interpretation and scope.}
"""
    appendix += (f"All 30 selected approximate rules satisfy $C_p\\le\\varepsilon$.\n"
                 f"GPU FP32 meets its same-backend discrepancy threshold in {gpu_passes}/30\n"
                 f"cases, while CPU FP64 meets it in {cpu_passes}/30 cases.\n"
                 f"The largest CPU discrepancy across patients, features, and the three\n"
                 f"tolerances is {sci(worst_cpu)}.\n")
    appendix += r"""The CPU FP64 results provide empirical evidence that the selected rules
agree closely with full degree-exact quadrature at higher precision.
They must be paired with the CPU runtime, not assigned to GPU FP32 timing.
The numerical-reference convention in this table is distinct from earlier
checks against a shared, independently implemented 64-node FP64 reference.
Those checks and the separate CPU FP32 control are retained as diagnostic
artifacts; they are not the definitions of either discrepancy reported here.

Finally, the explanations concern the fitted relative-hazard model and the
fixed uniform distribution on the 30 selected background rows. The
certificate does not quantify model-estimation uncertainty, uncertainty
from choosing a finite background sample, or discrepancy from a population
background distribution. The survival datasets demonstrate computational
feasibility in a small-$n$, large-$d$ setting; these experiments do not
establish clinical usefulness or causal interpretation of the attributions.
"""
    readme = ("# Same-backend degree-exact Cox comparison\n\n"
              "The requested reference convention is used throughout the main table: **Delta_GPU compares "
              "GPU FP32 approximation only with GPU FP32 degree-exact output; Delta_CPU compares CPU FP64 "
              "approximation only with CPU FP64 degree-exact output.** No smaller-node proxy is used. "
              "Exact-row discrepancies are zero by construction, not evidence of numerically exact values.\n\n"
              "All ten full CPU degree-exact runs are complete. GPU exact/approximate and CPU approximate "
              "timings reuse the matched saved runs. Every entry aggregates one measured call for each "
              "of five distinct patients, using the same 30 training-background samples.\n\n"
              "| Dataset | Method | Nodes | C_max | GPU FP32 time (s) | Delta_GPU | GPU pass | CPU FP64 time (s) | Delta_CPU | CPU pass |\n"
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(md_rows) + "\n\n"
              f"All analytical quadrature bounds are below the requested tolerance. Against their respective "
              f"degree-exact references, GPU FP32 passes {gpu_passes}/30 cases and CPU FP64 passes {cpu_passes}/30; "
              f"the worst CPU discrepancy is {worst_cpu:.3g}. Pass counts use unrounded per-patient values.\n\n"
              "## Interpretation\n\n"
              "C_max is the maximum analytical absolute quadrature bound across patients, assuming exact "
              "arithmetic. Observed discrepancies include numerical errors in both the approximation and "
              "its degree-exact reference. A same-backend pass is therefore a consistency check, not an "
              "absolute floating-point certificate. The fixed fitted model and empirical background "
              "define the game; population-background approximation and fitted-model uncertainty are outside "
              "the certificate. CPU FP64 errors must not be paired with GPU FP32 runtimes.\n\n"
              "Times are mean ± sample standard deviation across patients. Rule construction and warm-up "
              "are excluded. Approximate API measurements include automatic node selection. Exact CPU "
              "timing is accumulated active checkpointed integration, including factor preparation, "
              "planning, transfers, synchronized core evaluation and host accumulation. Checkpoint I/O "
              "and one kernel-shape warm-up plus three pilot calls per dataset are excluded; this is "
              "not a full-explanation warm-up. Resumed-run downtime and uncheckpointed work lost to "
              "interruption are excluded. The checkpoint loop preserves public-API factor/block order "
              "and passed a bitwise small-problem check. Detailed provenance is in `provenance.json`. "
              "GPU/CPU and exact/approximate measurements "
              "were recorded in separate runs. Comparing GPU FP32 with CPU FP64 also changes precision, "
              "so their latency ratio is not a matched-precision GPU speedup.\n\n"
              "## Outputs\n\n"
              "- `cox_device_exact_table.tex`: table with each backend's own degree-exact reference.\n"
              "- `cox_main_text.tex` and `cox_appendix.tex`: concise discussion and detailed experimental settings.\n"
              "- `table.csv`: unrounded aggregate statistics.\n"
              "- `case_errors.csv`: per-patient discrepancies calculated directly from saved attributions.\n"
              "- `independent_reference_audit.csv`: additional numerical diagnostic, explicitly separate from "
              "the main table's Delta definitions.\n"
              "- `preview.tex`: standalone LaTeX preview.\n")
    preview = r"""\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,booktabs,graphicx,url}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\begin{document}
\setcounter{section}{1}
\input{cox_main_text.tex}
\input{cox_device_exact_table.tex}
\clearpage
\appendix
\section{Additional experimental details}
\input{cox_appendix.tex}
\begingroup
\raggedright
\bibliographystyle{plain}
\bibliography{cox_high_dimensional_sources}
\endgroup
\end{document}
"""
    require("10^{-6}" not in table + main + appendix, "Obsolete tolerance in report")
    write_csv(folder / "table.csv", rows)
    write_csv(folder / "case_errors.csv", errors)
    write_csv(folder / "independent_reference_audit.csv", audit)
    (folder / "cox_device_exact_table.tex").write_text(table)
    (folder / "cox_main_text.tex").write_text(main)
    (folder / "cox_appendix.tex").write_text(appendix)
    (folder / "README.md").write_text(readme)
    (folder / "preview.tex").write_text(preview)
    bibliography = (original / "manuscript/cox_high_dimensional_sources.bib").read_text()
    # Inside \url, percent and ampersand characters are already handled literally.
    bibliography = re.sub(r"\\url\{([^}]*)\}",
                          lambda match: "\\url{" + match.group(1).replace(r"\%", "%").replace(r"\&", "&") + "}",
                          bibliography)
    (folder / "cox_high_dimensional_sources.bib").write_text(bibliography)
    print(f"Validated complete same-backend references and wrote {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT)
    build(parser.parse_args().folder)
