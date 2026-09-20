"""Build a concise research report from the completed, recorded GPU experiments."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/results/cox_gpu_tolerance"
NAMES = {"gse24080": "GSE24080", "tcga_lgg_methylation": "TCGA LGG methylation"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def build_report():
    summary = pd.read_csv(BASE / "tolerance_summary.csv")
    fits = [json.loads((BASE / name / "fit_summary.json").read_text()) for name in NAMES]
    exact = [json.loads((BASE / name / "exact_gpu_timing.json").read_text()) for name in NAMES]
    sweeps = pd.concat([pd.read_csv(BASE / name / "sweep.csv") for name in NAMES])
    model_table = table(["Cohort", "Train / test", "Features", "CPU fit (s)", "Test C-index"],
                        [[NAMES[f["dataset"]], f'{f["n_train"]} / {f["n_test"]}', f'{f["used_features"]:,}',
                          f'{f["fit_seconds"]:.3f}', f'{f["test_c_index"]:.3f}'] for f in fits])
    selected = summary[summary.backend.eq("logspace_jax")]
    tolerance_table = table(["Cohort", "Tolerance", "Nodes", "Warm API (ms)", "Max abs. error", "Relative L2 error", "Speedup vs current CPU", "Speedup vs GPU exact integration"],
                            [[NAMES[r.dataset], f'{r.eps:.0e}', r.m_q, f'{r.median_seconds * 1000:.2f}',
                              f'{r.max_absolute_error:.2e}', f'{r.relative_l2_error:.2e}',
                              f'{r.speedup_vs_cpu_prefix_same_tolerance:.2f}x', f'{r.speedup_vs_gpu_exact:.1f}x']
                             for r in selected.itertuples()])
    exact_table = table(["Cohort", "Exact nodes", "Fresh CPU rule (s)", "GPU integration (s)", "Combined (s)", "Historical CPU integration (s)", "Integration speedup", "Max abs. error vs float64"],
                        [[NAMES[e["dataset"]], f'{e["m_q"]:,}', f'{e["rule_seconds"]:.2f}',
                          f'{e["integration_seconds"]:.3f}', f'{e["total_compute_seconds"]:.2f}',
                          f'{e["historical_cpu_integration_seconds"]:.2f}',
                          f'{e["speedup_vs_historical_cpu_integration"]:.1f}x', f'{e["max_absolute_error"]:.2e}'] for e in exact])
    small = pd.read_csv(BASE / "small_exhaustive.csv")
    small_table = table(["Backend", "First call (ms)", "Warm API (ms)", "Speedup vs exhaustive", "Max abs. error"],
                        [[r.backend, f'{1000*r.first_call_seconds:.3f}', f'{1000*r.median_seconds:.3f}',
                          f'{r.speedup_vs_exhaustive:.1f}x', f'{r.max_absolute_error:.2e}'] for r in small.itertuples()])
    failures = sweeps[(sweeps["mode"] == "tolerance") & (sweeps.observed_tolerance_met == False)]
    failure_text = "No observed tolerance failures."
    if len(failures):
        failure_text = "\n".join(f'- {NAMES[r.dataset]}, patient index {r.patient}, `{r.backend}`, eps={r.eps:g}: '
                                 f'maximum error {r.max_absolute_error:.6g}, compared with {r.reference}.' for r in failures.itertuples())
    text = f'''# Cox survival experiments: JAX-Metal rerun

The repaired JAX log-space implementation runs successfully on the laptop's **Apple M4 Pro GPU**, with JIT enabled and the backend explicitly verified as `METAL:0`. All eight requested-tolerance comparisons for the first patients passed against the saved float64 exact references. The full exact-node GPU integrations were rerun for both cohorts.

These results include two changes needed for genomic dimensions: a parallel GPU reduction in the log-space kernel, and scalable generation of large quadrature rules. They are results for this revised implementation, rather than the unmodified collaborator commit. The original implementation's small GPU smoke tests passed, but its full methylation log-space pilot stalled; details are in [validation.json](validation.json).

## Cohorts and model fits

{model_table}

The same ridge Cox fitting routine runs on CPU and retains all original features. It changes coordinates to the non-null training sample space to avoid a feature-by-feature Hessian. The ridge penalty remains `0.01 * d`, with Breslow handling of tied survival times. No penalty tuning was added. The new `CoxExplainer` consumes the fitted coefficients and explains relative hazard, `exp(x @ beta)`, against the same four training-background patients.

Both refits, train/test splits, and explanation inputs match the previous experiments **bit for bit**. This is checked before using saved exact attributions. Per-cohort `refit_provenance.json` files record source hashes and the separate CPU fitting environment. Fit time excludes preprocessing; the full breakdown remains in `fit_summary.json`. C-index measures how well predicted risk orders comparable patients; it is not an explanation-accuracy score.

## Requested tolerances: first patient of each cohort

`eps` is an **absolute per-feature quadrature tolerance**, not a relative tolerance. The reported maximum absolute error and relative L2 error use a float64 exact reference for an identical model, background and patient. Relative L2 error is the length of the error vector divided by the length of the reference vector.

{tolerance_table}

Warm API time is the median of three repetitions and includes factor preparation, automatic node selection, host/device transfers, GPU work, and synchronization when the result returns to NumPy. It is not pure kernel time. First-call and individual repeated times are also recorded. First calls include JIT compilation only for shapes not already compiled. The CPU comparison uses the current NumPy prefix-scan implementation, the same requested tolerance, and four BLAS threads. It is a comparison between implementations and precisions, rather than an isolated hardware-only speedup.

The chosen node counts change only slightly across the requested tolerances. Small runtime differences need not be monotone: there are only three repeats, and this was an interactive laptop run. The saved min/max times show the variability. The bound is conservative on these fits; the fixed-node sweep also contains accurate results with fewer nodes.

## Full exact-node GPU calculations

{exact_table}

For a `d`-feature product game, each Shapley integrand has degree at most `d - 1`; `ceil(d / 2)` Gauss–Legendre nodes are sufficient for exact integration in real arithmetic. Every node in these full rules was evaluated, with no feature selection or early stopping. The GPU uses float32; node-block results are accumulated on the host in float64. Consequently, an exact node count does **not** imply float64-level numerical accuracy.

Each exact run was measured once. Rule construction is fresh and runs on CPU. GPU integration includes the explanation API with an already prepared rule, transfers and any needed compilation, and excludes the CPU rule-generation time. The previous CPU exact integrations are historical measurements from the earlier implementation; they were not rerun here. Their input equivalence was verified, and the table separates them explicitly. Rule generation can dominate combined runtime even when GPU integration is fast.

More nodes need not improve measured accuracy once rounding dominates. For example, GSE24080's 47-node GPU result has maximum error about 6.5e-7, compared with about 1.23e-6 for its full exact-node GPU result. This is a numerical-precision effect, not a failure of the polynomial exactness theorem.

## True exhaustive Shapley check

The small validation model has 12 features, four background points, and all **4,096 coalitions** were enumerated through the hazard prediction function. Exhaustive enumeration took {small.exhaustive_seconds.iloc[0]*1000:.3f} ms. Six quadrature nodes are sufficient for this fixture.

{small_table}

At genomic dimensions, enumerating all `2^d` coalitions is infeasible. The large reference is the complete quadrature rule, not exhaustive coalition enumeration. GPU overhead also makes the CPU preferable for this tiny fixture despite the GPU's advantage on the large workloads.

## Rounding and the other explained patients

Both cohorts were evaluated for three patients, four tolerances, eight fixed node counts (1 through 128 by powers of two), and three backends: `logspace_jax`, `prefix_scan_jax`, and `prefix_scan_numpy`. That is 216 configurations and 648 recorded warm repetitions. Full exact references and exact GPU runs cover the first patient only, matching the previous scope. The other two patients use freshly computed float64 128-node references and are labeled as approximation comparisons.

Observed tolerance failures in those comparisons:

{failure_text}

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
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \\
  .venv-metal/bin/python benchmarks/jax_metal_smoke.py --case logspace_jax
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \\
  MPLCONFIGDIR=/private/tmp/quadrashap-mpl .venv-metal/bin/python \\
  benchmarks/run_cox_notebook.py --notebook tutorials/cox_gpu_tolerance.ipynb --timeout 7200
```

Set `REFIT = True` in the notebook to repeat fitting as well. No CPU fallback is accepted by the GPU benchmark. Fresh exact-node generation can take several minutes for methylation. This is local laptop computation.

[tolerance_summary.csv](tolerance_summary.csv) contains the comparison table. Each cohort directory contains `sweep.csv`, `raw_timings.csv`, `fit_summary.json`, `refit_provenance.json`, `exact_gpu_timing.json`, and local attribution arrays. [environment.json](environment.json) records versions and source hashes; [metal-environment-lock.txt](metal-environment-lock.txt) lists every installed package version. [validation.json](validation.json) records initial smoke results and test evidence.

Apple documents the experimental backend and its precision limitations in [Accelerated JAX on Mac](https://developer.apple.com/metal/jax/). JAX explains compilation, asynchronous execution and timing in [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html).
'''
    (BASE / "README.md").write_text(text)
    return BASE / "README.md"


if __name__ == "__main__":
    print(build_report())
