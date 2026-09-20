# Full-review IMDB product-kernel benchmark

This is an independent 20-review experiment comparing QuadraSHAP on CPU and Apple Metal
against the `ESPComputer(method="quadratic", use_scaling=True)` implementation from the
[PKeX-Shapley repository](https://github.com/Majeed7/RKHS-ExactSHAP). It does not use
`experiments/baselines.py` or the older 6,000-review length-filtered CSV.

## Exact protocol

- Source: the canonical Hugging Face `imdb/plain_text` cache, **all 25,000 training and all
  25,000 test reviews**, no length filter, no 400-character truncation.
- The vectorizer is fitted to all 25,000 training reviews: lower-case English-stopword
  TF-IDF; unigrams and bigrams; `min_df=3`, `max_df=0.95`, `sublinear_tf=True`,
  `max_features=5000`. The fitted RBF-SVC has `C=2`, `gamma="scale"`, 15,177 support
  vectors and 88.03% accuracy on the full test split. This **fixed-parameter model is
  not the manuscript's separate Optuna-tuned SVC**. Model training is excluded from
  explainer timings.
- Test sample: seed 20260916, a fixed shuffled sample of 10 positive and 10 negative
  reviews from the canonical test split. Word counts range from 118 to 582 (median 216);
  all are used whole. Indices and numeric TF-IDF inputs are retained in `instances.npz`.
- Identical **neutral-factor** value function for all methods:
  `v_x(S) = sum_r alpha_r * prod_{j in S} exp(-gamma*(x_j-z_rj)^2)`, without the SVC
  intercept. Each of the 15,177 support vectors contributes one product game.
- Requested **absolute** max-coordinate tolerances: 1e-3, 1e-5, 1e-10 and 1e-16.
  The prior certified quadrature bound selects `m_q` per review. The 1e-16 CPU
  float64 run is the stipulated numerical reference for *observed* errors; it is not
  a high-precision proof of exactness. At d=5000 the algebraically exact Gauss-Legendre
  threshold is 2500 nodes; selected budgets are 4, 5, 7 and 9 nodes for all 20 reviews.
- CPU uses the project division-free NumPy prefix scan in float64. GPU uses the same
  project's JAX prefix scan on **Apple M5 Max Metal, float32**. The factor summaries
  and node budgets are computed in float64 for both. The per-instance time reported
  for a requested tolerance includes factor construction/summation for the certificate
  and a median of two post-warm-up evaluations. One-time JIT compilation and model
  fitting are excluded. Warm-up costs and both evaluation repeats are saved.
- PKeX uses the repository ESP code, applied in one-support-vector chunks to bound
  memory, then summed with the SVC dual coefficients. The repository code has no
  approximation-tolerance parameter; in exact arithmetic its approximation error
  is zero. A timed-out run has **no observed error** to report. The pilot was run alone;
  19 remaining independent CPU-only PKeX runs were scheduled four at a time on the
  multicore M5 Max. Each subprocess had its own 300-second hard cap (including startup).

The a-priori certificate bounds the **quadrature error in exact arithmetic**, *not*
floating-point error of either backend. In particular, it is not a numerical error
certificate for Metal float32 at 1e-10 or 1e-16. PKeX's `0` exact-arithmetic
approximation error likewise does not assert floating-point accuracy. The efficiency
residual `|sum(phi) - (f_kernel(x)-sum(alpha))|` is a necessary stability check, not
a sufficient correctness test.

## Results across 20 reviews

| Method | Requested eps | m_q | Median certified bound | Worst observed max-absolute error | Median end-to-end time | Completed |
|---|---:|---:|---:|---:|---:|---:|
| CPU float64 | 1e-3 | 4 | 4.98e-4 | 1.66e-7 | 4.55 s | 20/20 |
| CPU float64 | 1e-5 | 5 | 1.54e-6 | 4.27e-10 | 5.26 s | 20/20 |
| CPU float64 | 1e-10 | 7 | 4.73e-12 | 1.91e-13 | 6.64 s | 20/20 |
| CPU float64 | 1e-16 | 9 | 4.58e-18 | 0 (self-reference) | 7.98 s | 20/20 |
| Metal float32 | 1e-3 | 4 | 4.98e-4* | 2.16e-6 | 3.18 s | 20/20 |
| Metal float32 | 1e-5 | 5 | 1.54e-6* | 2.18e-6 | 3.85 s | 20/20 |
| Metal float32 | 1e-10 | 7 | 4.73e-12* | 2.18e-6 | 4.40 s | 20/20 |
| Metal float32 | 1e-16 | 9 | 4.58e-18* | 2.19e-6 | 4.82 s | 20/20 |
| PKeX repository | exact algebraically | N/A | 0 exact-arithmetic approximation | unavailable | >300 s | 0/20 |

`*` The Metal rows share the float64-derived **quadrature** certificate, not a
float32 end-to-end numerical guarantee. Every CPU row meets its requested tolerance
*relative to the stipulated reference*. Metal meets 1e-3 and 1e-5 on all 20 reviews,
but neither 1e-10 nor 1e-16 on any review. The maximum CPU efficiency residual for
the 1e-16 reference is 5.93e-12, so the label `1e-16` must not be presented as
realized end-to-end accuracy. All CPU/Metal values were finite; two repeated
evaluations per review/tolerance agreed exactly at the stored precision.

## Saved outputs and reproduction

- `results/exp10_pkex_metal_imdb/records.csv`: all 180 review-method-tolerance
  records, with certified bounds, observed max-absolute and relative-L2 errors,
  timing, completion, efficiency residual and repeatability. Baseline timeouts
  retain missing observed-error fields; they are **not** set to zero.
- `summary.csv`: population aggregates; `raw/*.json`, `raw/*.npz`, `raw/log_*.txt`:
  individual output vectors, repeats, diagnostics and timeout logs.
- `prepare_meta.json`, `run_meta.json`, `result_manifest.json`: dataset, model,
  machine and repository provenance; `imdb_full_tfidf5000_svc.joblib` and
  `instances.npz`: the cached model and fixed inputs.
- `figures/imdb_accuracy_runtime.pdf` and `figures/imdb_numerical_stability.pdf`:
  vector paper figures. `exp10_plot.py` regenerates both without rerunning methods.

From the repository root, the complete execution is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/exp10_pkex_metal_imdb.py prepare
ENABLE_PJRT_COMPATIBILITY=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp10_pkex_metal_imdb.py run --n-instances 20 --timeout 300
MPLCONFIGDIR=/private/tmp/quadra-mpl /usr/local/bin/python3 experiments/exp10_plot.py
```

The benchmark expects the cloned PKeX repository beside QuadraSHAP by default;
`--pkex-repo PATH` overrides this. The recorded run used commit
`f74de21548ae23fabd467d7dcae3b3fe5ea76428`. Apple JAX-Metal requires a
compatible plug-in and direct hardware access; a restricted sandbox that reports
`No supported GPU was found` must **not** be counted as a GPU benchmark.
