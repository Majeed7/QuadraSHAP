# Cox experiments with the complete training background

Using every training patient removes the random background subsampling present in the earlier four-patient experiment. A 100-patient pilot was insufficiently stable on these fitted models, so the primary backgrounds now contain **339 and 383 training patients**. This changes the empirical Shapley game: old four-background attributions and timings are not references for the new results.

## Small n, extremely large d

| Cohort | Eligible patients | Training / test | Features | Features / training patient | Background |
| --- | --- | --- | --- | --- | --- |
| GSE24080 | 553 | 339 / 214 | 54,675 | 161.3 | 339 |
| TCGA LGG methylation | 511 | 383 / 128 | 396,065 | 1034.1 | 383 |

Here n counts patients and d counts gene-expression probes or methylation features. We retain every feature and reuse the saved dense ridge Cox coefficients, preprocessing, splits and three held-out patients; no model refitting or tuning was introduced. Background rows come exclusively from training data and have equal weights. All backgrounds refer to the model's relative-hazard prediction exp(x @ beta), with its original scale unchanged.

## GPU timings: first held-out patient

| Cohort | Tolerance | Nodes | Median API (s) | Min--max (s) | Node selection (s) | Max abs. error | Relative L2 error | Observed abs. tolerance met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSE24080 | 1e-01 | 51 | 0.712 | 0.708--0.735 | 0.230 | 0.00372 | 4.33e-06 | True |
| GSE24080 | 1e-02 | 52 | 0.704 | 0.702--0.709 | 0.221 | 0.00192 | 2.46e-06 | True |
| GSE24080 | 1e-03 | 53 | 0.708 | 0.705--0.723 | 0.220 | 0.00255 | 2.96e-06 | False |
| GSE24080 | 1e-04 | 54 | 0.717 | 0.711--0.828 | 0.234 | 0.00331 | 3.83e-06 | False |
| TCGA LGG methylation | 1e-01 | 46 | 7.304 | 7.274--7.411 | 2.002 | 0.0656 | 9.54e-06 | True |
| TCGA LGG methylation | 1e-02 | 47 | 7.220 | 6.903--7.405 | 1.949 | 0.0671 | 1e-05 | False |
| TCGA LGG methylation | 1e-03 | 48 | 7.796 | 7.387--8.084 | 2.508 | 0.106 | 1.65e-05 | False |
| TCGA LGG methylation | 1e-04 | 49 | 7.261 | 7.202--7.659 | 1.946 | 0.0907 | 1.39e-05 | False |

These are local **Apple M4 Pro / JAX-Metal float32** results. Each time is the median of three repeated, synchronized calls to the current `CoxExplainer.explain(..., eps=...)` API. Timing includes factor preparation, automatic node selection, transfers, GPU integration and conversion back to NumPy. Loading data, fitting models and constructing the explainer occur outside the timer. First-call time and individual repetitions are preserved in the CSV files. Node-selection time is included in API time, not added to it. JIT compilation may occur in the separately recorded first call; warmed shapes and quadrature rules are reused.

Three repetitions on an interactive laptop describe the observed runtime, not a throughput guarantee. Small nonmonotonic timing changes across tolerances should not be overinterpreted.

At tolerance 1e-3, using the same full training background:

| Cohort | GPU API (s) | CPU prefix API (s) | CPU / GPU time | CPU max abs. error | Sufficient exact nodes |
| --- | --- | --- | --- | --- | --- |
| GSE24080 | 0.708 | 6.791 | 9.60x | 4.22e-11 | 27338 |
| TCGA LGG methylation | 7.796 | 49.140 | 6.30x | 3.48e-10 | 198033 |

The CPU baseline is the current NumPy prefix implementation in float64 with four BLAS threads. The ratio compares implementations and precision as well as hardware. The independent stable float64 validator below is a separate reference calculation, not this CPU timing baseline.

The earlier construction of the 198,033-node Gauss--Legendre rule took 561.92 s on CPU. This is about 200,000 **nodes** for 396,065 features, not 200,000 features. Rule construction is independent of the background size, so that historical setup measurement is still informative here. It adds substantial startup cost when the rule is not already available, but a precomputed rule can be cached and reused across patients. No full-size rule was regenerated in this rerun.

The sufficient exact node counts are ceil(d/2). **Full exact-degree quadrature was not rerun with the enlarged backgrounds**, so no new exact runtime or speedup versus exact computation is claimed. The older 1.2 s / 149 s integrations and 8 / 52 ms approximations used four background patients and must not be attached to this experiment.

## Accuracy and finite precision

The requested tolerance is an absolute per-feature quadrature error bound in exact arithmetic. It excludes rounding error and uncertainty about the background distribution. Across 24 GPU configurations, **16** exceeded their requested absolute tolerance when compared with the independent float64 reference. The new background includes some very large fitted hazards, so small relative rounding errors can become appreciable absolute errors. Increasing the number of nodes does not by itself resolve this issue.

| Cohort | Tolerance | Nodes across patients | Median latency range (s) | Worst max abs. error | Worst relative L2 error | Patients meeting abs. tolerance |
| --- | --- | --- | --- | --- | --- | --- |
| GSE24080 | 1e-01 | 46--51 | 0.668--0.712 | 0.00417 | 4.33e-06 | 3/3 |
| GSE24080 | 1e-02 | 47--52 | 0.672--0.716 | 0.00599 | 3.88e-06 | 3/3 |
| GSE24080 | 1e-03 | 48--53 | 0.666--0.711 | 0.00299 | 3.33e-06 | 0/3 |
| GSE24080 | 1e-04 | 48--54 | 0.703--0.721 | 0.00435 | 3.83e-06 | 0/3 |
| TCGA LGG methylation | 1e-01 | 45--47 | 7.304--7.369 | 0.1 | 9.54e-06 | 2/3 |
| TCGA LGG methylation | 1e-02 | 46--48 | 7.207--7.234 | 0.158 | 1.22e-05 | 0/3 |
| TCGA LGG methylation | 1e-03 | 47--48 | 7.139--7.796 | 0.158 | 1.65e-05 | 0/3 |
| TCGA LGG methylation | 1e-04 | 48--49 | 7.127--7.354 | 0.216 | 1.58e-05 | 0/3 |

The reference uses an independent, stable float64 implementation based on log1p/expm1, with 64 quadrature nodes and a certified quadrature bound below 1e-10 for each of the six patient/cohort cases. It is a tightly bounded approximation, **not exhaustive Shapley enumeration or a full exact-degree rule**. For patient 0, an additional 96-node calculation differed by at most 2.5e-12 for GSE24080 and 1.41e-11 for TCGA LGG. These checks support its use as a numerical reference; they do not certify all float64 rounding. The validator also passes a tiny exhaustive-coalition test and a large-hazard comparison against the public float64 implementation.

Relative L2 error is the norm of the attribution error divided by the norm of the reference attribution vector. Maximum absolute error is the largest error over features. Neither is a prediction-performance metric.

## Why use the whole training background?

The background represents the empirical population over which missing features are averaged. A larger count alone does not ensure that a random subset adequately represents this fitted model. Its predicted training hazards are concentrated in a few rows.

For the first held-out patient:

| Cohort | Selection | B | Mean hazard / full | Relative L2 difference | Top-20 overlap |
| --- | --- | --- | --- | --- | --- |
| GSE24080 | nested_100 | 100 | 2.5 | 1.6 | 80% |
| GSE24080 | nested_200 | 200 | 1.25 | 0.361 | 80% |
| GSE24080 | full_training | 339 | 1 | 0 | 100% |
| GSE24080 | historical_4 | 4 | 0.0001 | 1 | 25% |
| TCGA LGG methylation | nested_100 | 100 | 0.0938 | 0.945 | 10% |
| TCGA LGG methylation | nested_200 | 200 | 1.49 | 0.628 | 90% |
| TCGA LGG methylation | full_training | 383 | 1 | 0 | 100% |
| TCGA LGG methylation | historical_4 | 4 | 7.56e-09 | 1 | 5% |

These differences compare backgrounds at a common 128-node GPU rule. They measure changes in the explanation target, not approximation error relative to an exact Shapley value. Top-20 overlap is the fraction of the 20 largest-magnitude full-background attributions also selected with the smaller background. The sensitivity CSV includes all three held-out patients, nested sizes 4/32/100/200/full, the historical four rows, and three additional 100-row seeds. Whole-training explanations remove this particular source of subsampling variability; they do not establish stability against a new patient population or model refitting.

## Reproduction and files

The executed notebook is `tutorials/cox_background_sensitivity.ipynb`. It runs the GPU/CPU tolerance sweep and displays the resulting tables. The preprocessing, background sensitivity and independent references were computed separately and saved; the notebook records their provenance instead of silently recomputing them.

```sh
PYTHONPATH=src:. JAX_PLATFORMS=cpu .venv/bin/python -m benchmarks.cox_background_experiment prepare
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 .venv-metal/bin/python -m benchmarks.cox_background_experiment stability
PYTHONPATH=src:. JAX_PLATFORMS=cpu .venv/bin/python -m benchmarks.cox_background_experiment references
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 .venv-metal/bin/python benchmarks/run_cox_notebook.py --notebook tutorials/cox_background_sensitivity.ipynb --timeout 7200
```

`full_background_timings.csv` contains all 26 configurations (24 GPU, 2 CPU); `raw_timings.csv` contains the first and three repeated times. `background_sensitivity.csv` contains 54 background comparisons. Per-cohort `design.json` records source hashes, cohort counts and selection; `reference_validation.json` records bounds, crosschecks and reference wall time. Large local input and attribution arrays are ignored by Git. Original four-background results remain unchanged in `cox_gpu_tolerance`.

The proposed manuscript text is in [experiment_paragraph.tex](experiment_paragraph.tex). It replaces the earlier four-background latency claims and explicitly separates the quadrature certificate from finite-precision evaluation.
