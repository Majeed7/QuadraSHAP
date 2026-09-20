# Matched-game accuracy of KernelSHAP and SamplingSHAP

This experiment compares the **saved** 1,000-sample KernelSHAP and SamplingSHAP
attributions from `exp12` to a **new** QuadraSHAP float64 reference on exactly
the same finite, 30-background interventional game. Neither standard SHAP
baseline was rerun. There are 20 fixed held-out texts for each of Rotten
Tomatoes, SST-2, SMS spam and emotion, all with 5,000 TF-IDF coordinates.
The matched model, inputs, background values and selected emotion OVR score
are identical: saved model and instance hashes agree, and direct model
scores/background means agree to floating-point rounding.

For a test text `x`, fixed training-background texts `b_1,...,b_30`, and
model score `f`, the value function is

`v_x(S) = (1/30) sum_b f(x_S, b_not-S)`.

The QuadraSHAP reference uses the existing product-game engine's arbitrary
absent factors, with `u_rj = exp(-gamma*(x_j-z_rj)^2)` and
`ut_brj = exp(-gamma*(b_j-z_rj)^2)` for each support vector `z_r`. Model
intercepts are constant across coalitions and contribute zero Shapley value.
Coordinates satisfying `x_j=b_j` are dummy players for that background;
their common RBF factor is absorbed into the product game's weight. This is
an **algebraic reduction**, not a change in the game. A unit test compares
the reduced result with exhaustive full-game enumeration and independently
checks that its certificate amplitude equals that of the unreduced game.

One shared Gauss–Legendre node set is selected per text, using the summed
`A_i` amplitudes and largest relative variation over all support-vector and
background games. Its reported bound is a maximum-coordinate **quadrature**
error certificate in exact arithmetic for the *finite* 30-background game.
Requested `eps=1e-6` is therefore not an exact Shapley solution or a
floating-point guarantee, but is accurate enough to resolve the much larger
baseline errors below. Re-evaluating with two more nodes changed any
coordinate by at most `1.67e-12` over all 80 texts. The largest reference
efficiency residual was `4.30e-11`. The certificate does not cover the error
of using 30 backgrounds to approximate any population distribution.

For each baseline vector, the reported error is
`max_j |phi_j^baseline - phi_j^QuadraSHAP-reference|`. It is the empirical
maximum-coordinate difference from the matched numerical reference. The
corresponding exact-arithmetic error to the finite-background Shapley vector
can differ by at most the reference's per-text certified bound (at most
`9.78e-7` here), apart from unbounded floating-point rounding.

| Dataset (20 texts) | Median nodes in reference | Largest reference bound | KernelSHAP median / worst error | SamplingSHAP median / worst error | Lower error: Kernel / Sampling |
|---|---:|---:|---:|---:|---:|
| Rotten Tomatoes | 6 | 1.40e-8 | 0.0966 / 0.4421 | 0.1291 / 0.2569 | 13 / 7 |
| SST-2 | 6 | 1.03e-7 | 0.0475 / 0.3106 | 0.1397 / 0.3010 | 15 / 5 |
| SMS spam | 5 | 3.75e-7 | 0.0696 / 0.2683 | 0.1469 / 0.4918 | 19 / 1 |
| Emotion (OVR) | 6 | 9.78e-7 | 0.0736 / 0.2778 | 0.1221 / 0.3774 | 14 / 6 |

Pooling the 80 texts, the median maximum-coordinate error was `0.0716` for
KernelSHAP and `0.1381` for SamplingSHAP. KernelSHAP had the smaller error
on 61/80 texts. This is
specific to SHAP 0.51.0 with `nsamples=1000`, 30 saved backgrounds, and its
default `l1_reg="num_features(10)"` feature selection; it is not a general
ranking of estimators. Both baselines' 1,000-sample budget and their much
different model-evaluation counts remain as described in `README_exp12.md`.

## Saved data and reproduction

`results/exp13_interventional_accuracy_DATASET/` contains each text's
reference attribution vector (`raw/reference_instanceNNN.npz`), its
per-text certificate, node count, efficiency and extra-node diagnostics
(`raw/reference_instanceNNN.json`), and its two paired baseline errors
(`raw/comparison_instanceNNN.json`). The local `records.csv` has one row per
baseline and text. `run_meta.json` stores model, input and background hashes.

`results/exp13_interventional_accuracy_summary/` contains three files for
future plots without recomputation: `all_instances.csv` (160 paired error
rows plus saved baseline times), `reference_diagnostics.csv` (80 reference
rows) and `method_summary.csv` (eight dataset-method summaries).

From the repository root, using the already prepared models and exp12
background/baseline files:

```bash
for DATASET in rotten_tomatoes sst2 sms_spam emotion; do
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    experiments/exp13_interventional_accuracy.py \
    --dataset "$DATASET" --n-instances 20 --eps 1e-6
done
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/exp13_aggregate.py
```

The runner reuses completed references unless `--force` is given. Unit
checks are in `tests/test_exp13_interventional_accuracy.py`.
