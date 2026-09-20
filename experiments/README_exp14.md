# Matched interventional text comparison (appendix candidate)

This is the **interventional** half of the paper's product-kernel experiment.
It must not be placed in the same attribution-error comparison as PKeX-Shapley:
PKeX uses a *neutral absent factor*, while KernelSHAP and SamplingSHAP use
feature replacement from a background set. The main-text PKeX comparison is
handled by `exp10`, `exp11`, and the synthetic `exp15` experiment.

For each of four 5,000-feature TF-IDF RBF-SVC models (Rotten Tomatoes, SST-2,
SMS spam, emotion), 20 fixed held-out texts are explained under the **same**
finite-background value function

`v_x(S) = (1/30) sum_{b in B} f(x_S, b_not-S)`.

The 30 training backgrounds and text/model files are *identical* to the saved
`exp12` KernelSHAP/SamplingSHAP runs. Their SHAP 0.51 attributions are reused,
not rerun. Both methods had `nsamples=1000`; KernelSHAP used the installed
package's default `l1_reg="num_features(10)"`. SamplingSHAP at this sample
budget could not run on the longer IMDB reviews, so IMDB is excluded from the
fully paired four-dataset figure. No short-review filtering was introduced.

Only **Metal** is timed for QuadraSHAP. It runs at requested maximum-coordinate
quadrature tolerances `1e-3, 1e-5, 1e-10, 1e-16`, selecting one certified node
count per text. The reference is a separate **float64 CPU** evaluation at
requested quadrature tolerance `1e-16`, used solely to measure observed error.
Factor tables and the final weighted reduction are formed on the CPU in
float64; the quadrature prefix scan itself executes on Metal in float32.
The reference is checked at two additional nodes (maximum change across the
80 texts: `4.87e-14`). The certificate bounds *quadrature error in exact
arithmetic* for this finite-background game, **not** Metal float32 rounding or
sampling error relative to a population expectation. In particular, requesting
`1e-16` does not give a `1e-16`-accurate Metal attribution.

Features with `x_j=b_j` are dummy players for that background row. We factor
their common RBF term into each product game's coefficient exactly, reducing
the integrand dimensions without altering any Shapley value. The Metal
implementation pads each reduced table with dummy factors solely to reuse JIT
shapes. The underlying reduction and certificate were tested against direct
full-game enumeration in `tests/test_exp13_interventional_accuracy.py`.

The reported error is `max_j |phi_method,j - phi_float64_reference,j|` on the
same game. Method time excludes model training and reference computation.
QuadraSHAP time adds its per-text factor-table/certificate pass to its Metal
evaluation; SHAP baseline time includes explainer construction and explanation.
JIT compilation can affect the first text of a dataset but is not separately
removed; paper summaries use medians over 20 texts.

| Dataset | QuadraSHAP Metal at `1e-5`: median nodes / error / s | KernelSHAP: median error / s | SamplingSHAP: median error / s |
|---|---:|---:|---:|
| RT | 5 / `2.48e-8` / 0.286 | 0.0966 / 9.07 | 0.129 / 0.850 |
| SST-2 | 6 / `3.62e-8` / 0.675 | 0.0475 / 19.72 | 0.140 / 1.63 |
| SMS | 5 / `2.95e-8` / 0.208 | 0.0696 / 3.22 | 0.147 / 0.408 |
| Emotion | 5 / `3.66e-8` / 0.345 | 0.0736 / 10.81 | 0.122 / 1.00 |

The four requested tolerances give median node counts of 4-5, 5-6, 7-8,
and 9-10 respectively, but the observed Metal error stays near `1e-8` because
float32 rounding dominates once quadrature error is small. This is a useful
precision-limit result, not evidence that the exact-arithmetic certificate
failed.

Saved records are in `results/exp14_interventional_metal_precision/`:

- `DATASET/raw/`: one vector (`.npz`) and diagnostics (`.json`) for every
  text/tolerance, plus the float64 reference vector and diagnostics;
- `DATASET/records.csv`: 80 Metal rows per dataset;
- `summary/all_instances.csv`: 480 paired method/text/tolerance rows, with
  KernelSHAP and SamplingSHAP re-compared to the tighter reference;
- `summary/method_summary.csv`: 24 publication-table groups;
- `summary/interventional_comparison.pdf`: paper-width two-row comparison;
- `summary/interventional_precision_limit.pdf`: certified versus observed
  error across the four QuadraSHAP tolerances.

In that figure, pale dots are individual texts, the diamonds are medians, and
the short vertical strokes span the 25th--75th percentiles. The upper row is
maximum-coordinate attribution error; the lower row is time per text.

Reproduce after `exp12` has prepared the same models, texts, and backgrounds:

```bash
for DATASET in rotten_tomatoes sst2 sms_spam emotion; do
  ENABLE_PJRT_COMPATIBILITY=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    experiments/exp14_interventional_metal_precision.py --dataset "$DATASET" --n-instances 20
done
MPLCONFIGDIR=/private/tmp/quadrashap-mpl python3 experiments/exp14_aggregate_plot.py
```
