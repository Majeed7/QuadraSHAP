# Cox tolerance rerun: 30 backgrounds, five distinct patients

All approximation rows are fresh GPU measurements. Exact-degree GPU rows reuse the matched saved baseline; input, numerical-source, and software-version checks passed. Each entry is the mean ± sample standard deviation of five calls, one per patient. The standard deviation describes variation across patients, not repeated-run uncertainty.

| Dataset | Method | Nodes | GPU time (s) | C_max | Delta_GPU | Pass |
|---|---|---:|---:|---:|---:|---:|
| GSE24080 | exact | 27,338 | 1.172 ± 0.035 | 0 | — | — |
| GSE24080 | 1e-1 | 44–53 | 0.090 ± 0.020 | 0.0769 | 0.00034 | 5/5 |
| GSE24080 | 1e-2 | 45–54 | 0.085 ± 0.007 | 0.00709 | 0.000371 | 5/5 |
| GSE24080 | 1e-3 | 46–54 | 0.086 ± 0.006 | 0.000882 | 0.00131 | 4/5 |
| TCGA LGG | exact | 198,033 | 66.377 ± 1.196 | 0 | — | — |
| TCGA LGG | 1e-1 | 37–49 | 0.320 ± 0.028 | 0.0701 | 0.00447 | 5/5 |
| TCGA LGG | 1e-2 | 38–50 | 0.312 ± 0.006 | 0.00651 | 0.0038 | 5/5 |
| TCGA LGG | 1e-3 | 39–51 | 0.310 ± 0.011 | 0.00058 | 0.00595 | 2/5 |

C_max is the largest analytical absolute quadrature bound; every patient's bound is at most its requested epsilon. Delta_GPU is the largest absolute discrepancy from the same-patient degree-exact GPU result. Pass counts use unrounded per-patient errors. The degree-exact GPU result is subject to rounding and is not numerical ground truth.

## Independent precision checks

The following errors use a separate 64-node FP64 reference with quadrature bounds below 1e-10. The first patient in each cohort additionally has a 96-node cross-check. The CPU FP64 diagnostic uses the same node count as the corresponding GPU run, but these are separate evaluations: CPU errors are never attributed to GPU timings.

| Dataset | Tolerance | GPU FP32 worst error | GPU pass | CPU FP64 worst error | CPU pass |
|---|---|---:|---:|---:|---:|
| GSE24080 | 1e-1 | 0.00203 | 5/5 | 1.21e-11 | 5/5 |
| GSE24080 | 1e-2 | 0.00215 | 5/5 | 1.28e-11 | 5/5 |
| GSE24080 | 1e-3 | 0.00215 | 1/5 | 1.28e-11 | 5/5 |
| TCGA LGG | 1e-1 | 0.0305 | 5/5 | 1.4e-10 | 5/5 |
| TCGA LGG | 1e-2 | 0.0352 | 3/5 | 1.31e-10 | 5/5 |
| TCGA LGG | 1e-3 | 0.0391 | 0/5 | 1.59e-10 | 5/5 |

Against the FP64 reference, GPU FP32 passes 19/30 cases and CPU FP64 passes 30/30. The largest CPU FP64 difference is 1.59e-10. These are empirical checks, not floating-point error certificates.

In particular, at 1e-2 the TCGA GPU approximations agree with their degree-exact GPU counterparts for 5/5 patients, but meet the same threshold against the independent FP64 reference for only 3/5 patients. Agreement with a rounded GPU baseline alone therefore does not demonstrate absolute accuracy.

## Timing scope and setup

Apple M4 Pro, JAX-Metal FP32, an 8 GiB memory-planning budget, and one background row per block. One untimed explicit-node warm-up precedes the first measured call for each distinct dataset/node-count configuration. Synchronized API timings include automatic node selection, data transfers, and host accumulation; model fitting, warm-up, and rule construction are excluded.

Historical exact-rule CPU construction: GSE24080 10.97 s; TCGA LGG 561.92 s. Fresh approximate-rule CPU construction: 0.386–0.563 ms, the range of medians from five fresh constructions per distinct rule. Rules are cached for the timing measurements.

## Files

- `cox_tolerance_table.tex`: seven-column main GPU table.
- `cox_tolerance_analysis.tex`: experimental subsection and separate precision-check table.
- `table.csv`: full-precision aggregate data, including speedups and both error comparisons.
- `preview.tex`: standalone LaTeX document; use BibTeX for the included source references.
- Raw GPU timings, GPU/FP64 summaries, attribution arrays, status, environment, and provenance files are retained alongside these reports.
