# Cox survival experiments: JAX-Metal rerun

The repaired JAX log-space implementation runs successfully on the laptop's **Apple M4 Pro GPU**, with JIT enabled and the backend explicitly verified as `METAL:0`. All eight requested-tolerance comparisons for the first patients passed against the saved float64 exact references. The full exact-node GPU integrations were rerun for both cohorts.

These results include two changes needed for genomic dimensions: a parallel GPU reduction in the log-space kernel, and scalable generation of large quadrature rules. They are results for this revised implementation, rather than the unmodified collaborator commit. The original implementation's small GPU smoke tests passed, but its full methylation log-space pilot stalled; details are in [validation.json](validation.json).

## Cohorts and model fits

| Cohort | Train / test | Features | CPU fit (s) | Test C-index |
| --- | --- | --- | --- | --- |
| GSE24080 | 339 / 214 | 54,675 | 0.580 | 0.657 |
| TCGA LGG methylation | 383 / 128 | 396,065 | 1.138 | 0.854 |

The same ridge Cox fitting routine runs on CPU and retains all original features. It changes coordinates to the non-null training sample space to avoid a feature-by-feature Hessian. The ridge penalty remains `0.01 * d`, with Breslow handling of tied survival times. No penalty tuning was added. The new `CoxExplainer` consumes the fitted coefficients and explains relative hazard, `exp(x @ beta)`, against the same four training-background patients.

Both refits, train/test splits, and explanation inputs match the previous experiments **bit for bit**. This is checked before using saved exact attributions. Per-cohort `refit_provenance.json` files record source hashes and the separate CPU fitting environment. Fit time excludes preprocessing; the full breakdown remains in `fit_summary.json`. C-index measures how well predicted risk orders comparable patients; it is not an explanation-accuracy score.

## Requested tolerances: first patient of each cohort

`eps` is an **absolute per-feature quadrature tolerance**, not a relative tolerance. The reported maximum absolute error and relative L2 error use a float64 exact reference for an identical model, background and patient. Relative L2 error is the length of the error vector divided by the length of the reference vector.

| Cohort | Tolerance | Nodes | Warm API (ms) | Max abs. error | Relative L2 error | Speedup vs current CPU | Speedup vs GPU exact integration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSE24080 | 1e-01 | 44 | 9.15 | 1.88e-06 | 3.50e-06 | 7.11x | 131.6x |
| GSE24080 | 1e-02 | 45 | 8.73 | 1.68e-06 | 3.04e-06 | 7.55x | 137.9x |
| GSE24080 | 1e-03 | 46 | 8.33 | 6.65e-07 | 1.23e-06 | 7.99x | 144.5x |
| GSE24080 | 1e-04 | 47 | 8.35 | 6.50e-07 | 1.19e-06 | 8.23x | 144.1x |
| TCGA LGG methylation | 1e-01 | 27 | 64.02 | 4.62e-07 | 1.51e-05 | 4.80x | 2332.7x |
| TCGA LGG methylation | 1e-02 | 28 | 61.80 | 5.53e-07 | 1.77e-05 | 5.13x | 2416.5x |
| TCGA LGG methylation | 1e-03 | 29 | 51.67 | 3.60e-07 | 1.16e-05 | 6.34x | 2890.5x |
| TCGA LGG methylation | 1e-04 | 30 | 64.76 | 4.00e-07 | 1.30e-05 | 5.19x | 2306.1x |

Warm API time is the median of three repetitions and includes factor preparation, automatic node selection, host/device transfers, GPU work, and synchronization when the result returns to NumPy. It is not pure kernel time. First-call and individual repeated times are also recorded. First calls include JIT compilation only for shapes not already compiled. The CPU comparison uses the current NumPy prefix-scan implementation, the same requested tolerance, and four BLAS threads. It is a comparison between implementations and precisions, rather than an isolated hardware-only speedup.

The chosen node counts change only slightly across the requested tolerances. Small runtime differences need not be monotone: there are only three repeats, and this was an interactive laptop run. The saved min/max times show the variability. The bound is conservative on these fits; the fixed-node sweep also contains accurate results with fewer nodes.

## Full exact-node GPU calculations

| Cohort | Exact nodes | Fresh CPU rule (s) | GPU integration (s) | Combined (s) | Historical CPU integration (s) | Integration speedup | Max abs. error vs float64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSE24080 | 27,338 | 11.38 | 1.204 | 12.58 | 38.49 | 32.0x | 1.23e-06 |
| TCGA LGG methylation | 198,033 | 561.92 | 149.346 | 711.27 | 2014.41 | 13.5x | 4.20e-07 |

For a `d`-feature product game, each Shapley integrand has degree at most `d - 1`; `ceil(d / 2)` Gauss–Legendre nodes are sufficient for exact integration in real arithmetic. Every node in these full rules was evaluated, with no feature selection or early stopping. The GPU uses float32; node-block results are accumulated on the host in float64. Consequently, an exact node count does **not** imply float64-level numerical accuracy.

Each exact run was measured once. Rule construction is fresh and runs on CPU. GPU integration includes the explanation API with an already prepared rule, transfers and any needed compilation, and excludes the CPU rule-generation time. The previous CPU exact integrations are historical measurements from the earlier implementation; they were not rerun here. Their input equivalence was verified, and the table separates them explicitly. Rule generation can dominate combined runtime even when GPU integration is fast.

More nodes need not improve measured accuracy once rounding dominates. For example, GSE24080's 47-node GPU result has maximum error about 6.5e-7, compared with about 1.23e-6 for its full exact-node GPU result. This is a numerical-precision effect, not a failure of the polynomial exactness theorem.

## True exhaustive Shapley check

The small validation model has 12 features, four background points, and all **4,096 coalitions** were enumerated through the hazard prediction function. Exhaustive enumeration took 20.426 ms. Six quadrature nodes are sufficient for this fixture.

| Backend | First call (ms) | Warm API (ms) | Speedup vs exhaustive | Max abs. error |
| --- | --- | --- | --- | --- |
| prefix_scan_numpy | 12.406 | 0.051 | 399.5x | 1.46e-16 |
| prefix_scan_jax | 150.091 | 2.417 | 8.5x | 2.26e-08 |
| logspace_jax | 20.513 | 1.462 | 14.0x | 1.15e-08 |

At genomic dimensions, enumerating all `2^d` coalitions is infeasible. The large reference is the complete quadrature rule, not exhaustive coalition enumeration. GPU overhead also makes the CPU preferable for this tiny fixture despite the GPU's advantage on the large workloads.

## Rounding and the other explained patients

Both cohorts were evaluated for three patients, four tolerances, eight fixed node counts (1 through 128 by powers of two), and three backends: `logspace_jax`, `prefix_scan_jax`, and `prefix_scan_numpy`. That is 216 configurations and 648 recorded warm repetitions. Full exact references and exact GPU runs cover the first patient only, matching the previous scope. The other two patients use freshly computed float64 128-node references and are labeled as approximation comparisons.

Observed tolerance failures in those comparisons:

- TCGA LGG methylation, patient index 1, `prefix_scan_jax`, eps=0.0001: maximum error 0.000110664, compared with new float64 128-node approximation.

The theoretical certificate concerns quadrature error in real arithmetic. It excludes float32 rounding and uncertainty from representing the background distribution with four samples. The repaired log-space GPU path met all requested tolerances in the recorded comparisons; the original prefix-scan GPU results are retained to make the precision difference visible. No claim about NVIDIA/CUDA throughput follows from an Apple-Metal run.

## Implementation and checks

- `CoxExplainer` is the collaborator's current public API. The existing CPU fitter now optionally constructs this explainer; historical notebook outputs remain in `cox_survival`.
- The GPU log-space kernel uses a parallel reduction over features. The original sequential `lax.scan` remains the CPU JAX path.
- The engine now forwards the planned node-block size to JAX log-space evaluation. The planner accounts for its parallel working arrays; these experiments used a planned 512 MiB budget.
- Large rules use SciPy's `roots_legendre` instead of NumPy's dense companion-matrix routine. Immutable rules are cached in memory. The exact benchmark explicitly clears the cache before timing fresh construction.
- 18 focused tests passed: Cox behavior, absent factors, signed product games, node budgets, blocking, and large-rule moments. GPU correctness was also checked against exhaustive predictions, and every saved sweep attribution was checked for finite values.

## Reproduce and inspect

The executed notebook is [cox_gpu_tolerance.ipynb](../../../tutorials/cox_gpu_tolerance.ipynb). The reusable driver is [cox_gpu_experiment.py](../../cox_gpu_experiment.py), and the tested Metal environment is pinned in [requirements-cox-metal.txt](../../requirements-cox-metal.txt). A matching original CPU `.venv` is used only for reproducible refits.

```sh
uv venv --python 3.12 .venv-metal
uv pip install --python .venv-metal/bin/python -r benchmarks/requirements-cox-metal.txt
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  .venv-metal/bin/python benchmarks/jax_metal_smoke.py --case logspace_jax
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  MPLCONFIGDIR=/private/tmp/quadrashap-mpl .venv-metal/bin/python \
  benchmarks/run_cox_notebook.py --notebook tutorials/cox_gpu_tolerance.ipynb --timeout 7200
```

Set `REFIT = True` in the notebook to repeat fitting as well. No CPU fallback is accepted by the GPU benchmark. Fresh exact-node generation can take several minutes for methylation. This is local laptop computation.

[tolerance_summary.csv](tolerance_summary.csv) contains the comparison table. Each cohort directory contains `sweep.csv`, `raw_timings.csv`, `fit_summary.json`, `refit_provenance.json`, `exact_gpu_timing.json`, and local attribution arrays. [environment.json](environment.json) records versions and source hashes; [metal-environment-lock.txt](metal-environment-lock.txt) lists every installed package version. [validation.json](validation.json) records initial smoke results and test evidence.

Apple documents the experimental backend and its precision limitations in [Accelerated JAX on Mac](https://developer.apple.com/metal/jax/). JAX explains compilation, asynchronous execution and timing in [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html).
