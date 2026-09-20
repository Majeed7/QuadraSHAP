# Synthetic KRR precision and PKeX benchmark (main-text candidate)

This is the **neutral-factor** half of the product-kernel experiment. The
coalition game is

`v_x(S) = sum_r alpha_r product_{j in S} exp(-gamma (x_j-z_rj)^2)`.

PKeX-Shapley and QuadraSHAP explain **the same game**, so attribution vectors
and numerical errors can be compared directly. These results must not be
combined into one error ranking with the interventional KernelSHAP and
SamplingSHAP comparison in `README_exp14.md`.

## Protocol

The dimensions and data-generation parameters follow the manuscript's
synthetic product-kernel benchmark: `d in {50,500,1000,2000,5000}`,
`make_regression(n_samples=1000, n_informative=floor(d/4), noise=0.1,
random_state=42)`, and **50 held-out explanations per dimension**. Because the
current manuscript does not specify the train/test split or KRR parameters,
this new precision study fixes them explicitly: shuffled 80/20 split at seed
42; training-fit standardization of features **and targets**; RBF-KRR with
`gamma=1/d` and ridge `alpha=0.1`. The same 50 test indices are selected from
each 200-point test split at seed 20260917. The full matrices, coefficients,
selected indices, and fitted-model protocol are saved in each `d*/case.npz`
and `d*/prepare_meta.json`.

QuadraSHAP is run on Apple Metal at requested maximum-coordinate quadrature
tolerances `1e-3`, `1e-5`, `1e-10`, and `1e-16`. The a-priori certificate covers
quadrature **in exact arithmetic**, not the float32 Metal computation. One
separate float64 CPU QuadraSHAP result at requested `1e-16` is used solely as
an empirical reference, checked against two additional nodes. PKeX uses the
specified `RKHS-ExactSHAP` repository's quadratic, scaling-enabled ESP method
in 16-support-vector chunks (bounded memory). It is run only for `d<=1000`,
as requested. **No PKeX timeout result is claimed for d=2000 or d=5000**:
these cases were intentionally not attempted. Each PKeX call has a 300-second
process cap.

The factor table and final weighted sum are formed in float64 on the CPU;
the quadrature prefix scan executes on Metal in float32. This is a Metal
backend comparison, not an assertion that every preprocessing step runs on
the GPU.

The primary timing includes per-explanation factor construction, certificate
summary and Metal evaluation for QuadraSHAP; for PKeX, it includes factor
construction and the repository ESP calculation. Training, file loading,
reference evaluation and process startup are excluded. JIT compilation is not
separately subtracted from the first Metal point, but medians over 50 dilute
that one-off cost. `d*/pkex_run_meta.json` and
`summary/pkex_execution_batches.json` record concurrency; independently
scheduled PKeX processes may experience machine-level contention. Five
`d=1000` inputs had overlapping serial and parallel executions; dropping
them from the timing calculation changes the PKeX median from 79.35 to
79.13 seconds.

Error is `max_j |phi_method,j - phi_float64_reference,j|`. We also retain the
efficiency residual and every attribution vector. A `1e-16` request does **not**
mean `1e-16` total accuracy on float32 Metal; its observed error is limited by
rounding. Conversely, PKeX is algebraically exact in exact arithmetic, while
its observed difference from the reference measures numerical agreement.

The `gamma=1/d` RBF bandwidth deliberately keeps total kernel variation
roughly controlled as dimension grows. Node counts in this single model
family therefore need not grow with `d`; the separate multi-model budget
study (`README_exp8.md`) addresses when they do. This benchmark is about
neutral-game method runtime and numerical accuracy, not a general law for
node scaling.

## Results

All **50/50** explanations completed for PKeX at each attempted dimension;
all **50/50** completed for QuadraSHAP at every dimension and tolerance. The
table shows the `1e-5` QuadraSHAP setting (medians over the same 50 inputs):

| Features | QuadraSHAP nodes | QuadraSHAP time (s) | QuadraSHAP error | PKeX time (s) | PKeX error |
|---:|---:|---:|---:|---:|---:|
| 50 | 5 | 0.00629 | `9.03e-8` | 0.171 | `1.11e-15` |
| 500 | 5 | 0.0183 | `1.93e-7` | 17.5 | `7.49e-16` |
| 1000 | 4 | 0.0241 | `2.72e-7` | 79.4 | `4.02e-16` |
| 2000 | 4 | 0.0359 | `3.35e-7` | not attempted | not attempted |
| 5000 | 4 | 0.0945 | `4.05e-7` | not attempted | not attempted |

At the `1e-16` request, median certified nodes are 10, 9, 9, 8, and 8
respectively, but observed Metal errors remain around `1e-7` to `4e-7`.
The largest float64 reference change at two extra nodes is `1.78e-15`.
PKeX stays at float64-level numerical agreement, about `1e-15` or better,
on its attempted dimensions. At 5,000 features and `1e-16`, Metal's median
efficiency residual is `1.03e-5`; the much smaller per-coordinate error does
not imply an equally small error after summing 5,000 attributions.

## Saved outputs

- `d*/case.npz`: synthetic model factors, coefficients and all selected inputs;
- `d*/raw/`: every Metal and PKeX vector (`.npz`) plus timing/certification
  diagnostics (`.json`), and the float64 reference vectors;
- `d*/records.csv`: one row per method/input/tolerance, updated incrementally;
- `summary/all_instances.csv`, `metal_summary.csv`, `pkex_summary.csv`:
  cross-dimensional tables once all PKeX runs complete;
- `summary/synthetic_runtime_nodes.pdf`, `synthetic_accuracy.pdf`: paper-width
  figures generated only from completed, audited records.

Reproduction:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/exp15_synthetic_pkex_metal_precision.py prepare
ENABLE_PJRT_COMPATIBILITY=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp15_synthetic_pkex_metal_precision.py quad
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp15_synthetic_pkex_metal_precision.py pkex \
  --d 50 500 1000 --timeout 300
MPLCONFIGDIR=/private/tmp/quadrashap-mpl python3 experiments/exp15_aggregate_plot.py
```

This command uses serial PKeX jobs. For the recorded multi-process timing
batches, see `summary/pkex_execution_batches.json`; concurrency can change
wall-clock time without changing the value function or attribution vectors.

The present results are a new, fully specified precision study. They should
not be presented as a literal rerun of the manuscript's existing timing table
until that table's omitted split, preprocessing and KRR settings are verified
or aligned with this documented protocol.
