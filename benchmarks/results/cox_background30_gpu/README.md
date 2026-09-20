# Cox GPU timings with 30 background samples and five held-out patients

Every configuration uses the same **30 training-background samples**, selected with seed 42,
and the same **five distinct held-out patients** per cohort. The full training sample counts remain 339
(GSE24080, 54,675 features) and 383 (TCGA LGG methylation, 396,065 features): B=30 is the
background count, not the training sample count. Saved models and preprocessing are reused.

| Method | GSE24080 nodes | GPU seconds, mean ± SD | TCGA LGG nodes | GPU seconds, mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| Exact degree | 27,338 | 1.172 ± 0.035 | 198,033 | 66.377 ± 1.196 |
| 1e-1 | 44–53 | 0.085 ± 0.003 | 37–49 | 0.375 ± 0.009 |
| 1e-3 | 46–54 | 0.089 ± 0.010 | 39–51 | 0.374 ± 0.019 |
| 1e-6 | 48–57 | 0.086 ± 0.008 | 41–53 | 0.385 ± 0.018 |

Speedup is the **ratio of mean exact latency to mean approximation latency** (the exact row is 1×),
not the mean of per-patient ratios:

- GSE24080: 1e-1: 13.8×; 1e-3: 13.2×; 1e-6: 13.7×.
- TCGA LGG methylation: 1e-1: 177.2×; 1e-3: 177.5×; 1e-6: 172.6×.

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
were 10.974699 s and 561.923277 s, respectively.
This cost is independent of patient/background count and can be amortized through reuse. Fresh small-rule
construction took 0.418291–0.625500 ms, the range of medians over five constructions
per distinct node count. This is construction of abscissae and weights; it is not the node-budget
selection already included in approximate API timing. `rule_setup.json` records rule sources and load time.

## Accuracy and interpretation

Degree-exact quadrature removes integration error in real arithmetic, not FP32 rounding. Likewise,
the requested absolute per-feature tolerance controls quadrature error only. **19/30 approximate
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
| GSE24080 | exact | 0.00187369 | 7.14514e-06 | n/a |
| TCGA LGG methylation | exact | 0.0331517 | 2.33443e-05 | n/a |
| GSE24080 | 1e-1 | 0.00202565 | 7.8864e-06 | 5/5 |
| TCGA LGG methylation | 1e-1 | 0.0305136 | 1.90655e-05 | 5/5 |
| GSE24080 | 1e-3 | 0.00215283 | 8.22163e-06 | 1/5 |
| TCGA LGG methylation | 1e-3 | 0.0391052 | 2.40642e-05 | 0/5 |
| GSE24080 | 1e-6 | 0.00226743 | 1.1278e-05 | 0/5 |
| TCGA LGG methylation | 1e-6 | 0.0224293 | 2.80116e-05 | 0/5 |

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
