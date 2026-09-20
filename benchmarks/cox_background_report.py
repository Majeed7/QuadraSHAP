"""Build paper-ready summaries of the full-training-background Cox experiment."""
import json

import pandas as pd

from benchmarks.cox_background_experiment import DATASETS, OUTPUT
from benchmarks.cox_gpu_report import table

NAMES = {"gse24080": "GSE24080", "tcga_lgg_methylation": "TCGA LGG methylation"}


def build_report():
    timings = pd.read_csv(OUTPUT / "full_background_timings.csv")
    sensitivity = pd.read_csv(OUTPUT / "background_sensitivity.csv")
    gpu = timings[timings.backend.eq("logspace_jax")]
    first = gpu[gpu.patient.eq(0)]
    selected = first[first.eps.eq(1e-3)].set_index("dataset")
    designs = {name: json.loads((OUTPUT / name / "design.json").read_text()) for name in DATASETS}
    refs = {name: json.loads((OUTPUT / name / "reference_validation.json").read_text()) for name in DATASETS}

    cohorts = table(["Cohort", "Eligible patients", "Training / test", "Features", "Features / training patient", "Background"], [
        [NAMES[name], d["n_total"], f'{d["n_train"]} / {d["n_test"]}', f'{d["d"]:,}',
         f'{d["d_per_training_patient"]:.1f}', d["primary_background_size"]] for name, d in designs.items()])
    main = table(["Cohort", "Tolerance", "Nodes", "Median API (s)", "Min--max (s)", "Node selection (s)",
                  "Max abs. error", "Relative L2 error", "Observed abs. tolerance met"], [
        [NAMES[r.dataset], f'{r.eps:.0e}', r.m_q, f'{r.median_seconds:.3f}',
         f'{r.min_seconds:.3f}--{r.max_seconds:.3f}', f'{r.median_budget_seconds:.3f}',
         f'{r.max_absolute_error:.3g}', f'{r.relative_l2_error:.3g}', r.observed_absolute_tolerance_met]
        for r in first.itertuples()])
    comparison = []
    for name in DATASETS:
        r = selected.loc[name]
        cpu = timings[timings.dataset.eq(name) & timings.backend.eq("prefix_scan_numpy")].iloc[0]
        comparison.append([NAMES[name], f'{r.median_seconds:.3f}', f'{cpu.median_seconds:.3f}',
                           f'{cpu.median_seconds / r.median_seconds:.2f}x',
                           f'{cpu.max_absolute_error:.3g}', (designs[name]["d"] + 1) // 2])
    compare_table = table(["Cohort", "GPU API (s)", "CPU prefix API (s)", "CPU / GPU time", "CPU max abs. error", "Sufficient exact nodes"], comparison)
    stability = table(["Cohort", "Selection", "B", "Mean hazard / full", "Relative L2 difference", "Top-20 overlap"], [
        [NAMES[r.dataset], r.selection, r.background_size, f'{r.mean_hazard_ratio_to_full:.3g}',
         f'{r.relative_l2_error:.3g}', f'{100*r.top20_overlap:.0f}%']
        for r in sensitivity[sensitivity.patient.eq(0) & sensitivity.selection.isin(
            ["historical_4", "nested_100", "nested_200", "full_training"])].itertuples()])
    accuracy_rows = []
    for (name, eps), group in gpu.groupby(["dataset", "eps"], sort=False):
        accuracy_rows.append([NAMES[name], f'{eps:.0e}', f'{group.m_q.min()}--{group.m_q.max()}',
                              f'{group.median_seconds.min():.3f}--{group.median_seconds.max():.3f}',
                              f'{group.max_absolute_error.max():.3g}', f'{group.relative_l2_error.max():.3g}',
                              f'{int(group.observed_absolute_tolerance_met.sum())}/3'])
    accuracy_table = table(["Cohort", "Tolerance", "Nodes across patients", "Median latency range (s)",
                            "Worst max abs. error", "Worst relative L2 error", "Patients meeting abs. tolerance"], accuracy_rows)

    a, b = (selected.loc[name] for name in DATASETS)
    failures = int((~gpu.observed_absolute_tolerance_met).sum())
    rule = json.loads((OUTPUT.parent / "cox_gpu_tolerance" / "tcga_lgg_methylation" / "exact_gpu_timing.json").read_text())
    paragraph = (
        r"To illustrate QuadraSHAP in the small-$n$, extremely large-$d$ regime, we explain Cox models "
        r"for GSE24080 gene expression and TCGA lower-grade glioma methylation, with "
        r"$(n_{\mathrm{train}},d)=(339,54{,}675)$ and $(383,396{,}065)$, respectively. "
        r"All training patients serve as uniformly weighted backgrounds. For the first held-out patient "
        rf"in each cohort, a quadrature tolerance of $10^{{-3}}$ selects only ${int(a.m_q)}$ and ${int(b.m_q)}$ nodes, "
        r"compared with the $27{,}338$ and $198{,}033$ Gauss--Legendre nodes sufficient for exact quadrature. "
        rf"Median warm explanation times on an Apple M4 Pro GPU are ${a.median_seconds:.2f}$ and ${b.median_seconds:.2f}$ seconds, "
        r"including automatic node selection (Table~\ref{tab:cox-background}). Constructing very large "
        r"quadrature rules can itself impose substantial startup latency: generating the $198{,}033$-node "
        rf"rule took approximately ${rule['rule_seconds']:.0f}$ seconds in our earlier benchmark. "
        r"Using substantially fewer nodes therefore also reduces setup costs when a precomputed rule "
        r"is unavailable. The quadrature certificate excludes floating-point error; on these fits, "
        r"single-precision GPU evaluation exceeds the requested absolute tolerance, as reported separately in the table."
        "\n"
    )
    (OUTPUT / "experiment_paragraph.tex").write_text(paragraph)

    text = f'''# Cox experiments with the complete training background

Using every training patient removes the random background subsampling present in the earlier four-patient experiment. A 100-patient pilot was insufficiently stable on these fitted models, so the primary backgrounds now contain **339 and 383 training patients**. This changes the empirical Shapley game: old four-background attributions and timings are not references for the new results.

## Small n, extremely large d

{cohorts}

Here n counts patients and d counts gene-expression probes or methylation features. We retain every feature and reuse the saved dense ridge Cox coefficients, preprocessing, splits and three held-out patients; no model refitting or tuning was introduced. Background rows come exclusively from training data and have equal weights. All backgrounds refer to the model's relative-hazard prediction exp(x @ beta), with its original scale unchanged.

## GPU timings: first held-out patient

{main}

These are local **Apple M4 Pro / JAX-Metal float32** results. Each time is the median of three repeated, synchronized calls to the current `CoxExplainer.explain(..., eps=...)` API. Timing includes factor preparation, automatic node selection, transfers, GPU integration and conversion back to NumPy. Loading data, fitting models and constructing the explainer occur outside the timer. First-call time and individual repetitions are preserved in the CSV files. Node-selection time is included in API time, not added to it. JIT compilation may occur in the separately recorded first call; warmed shapes and quadrature rules are reused.

Three repetitions on an interactive laptop describe the observed runtime, not a throughput guarantee. Small nonmonotonic timing changes across tolerances should not be overinterpreted.

At tolerance 1e-3, using the same full training background:

{compare_table}

The CPU baseline is the current NumPy prefix implementation in float64 with four BLAS threads. The ratio compares implementations and precision as well as hardware. The independent stable float64 validator below is a separate reference calculation, not this CPU timing baseline.

The earlier construction of the 198,033-node Gauss--Legendre rule took {rule['rule_seconds']:.2f} s on CPU. This is about 200,000 **nodes** for 396,065 features, not 200,000 features. Rule construction is independent of the background size, so that historical setup measurement is still informative here. It adds substantial startup cost when the rule is not already available, but a precomputed rule can be cached and reused across patients. No full-size rule was regenerated in this rerun.

The sufficient exact node counts are ceil(d/2). **Full exact-degree quadrature was not rerun with the enlarged backgrounds**, so no new exact runtime or speedup versus exact computation is claimed. The older 1.2 s / 149 s integrations and 8 / 52 ms approximations used four background patients and must not be attached to this experiment.

## Accuracy and finite precision

The requested tolerance is an absolute per-feature quadrature error bound in exact arithmetic. It excludes rounding error and uncertainty about the background distribution. Across 24 GPU configurations, **{failures}** exceeded their requested absolute tolerance when compared with the independent float64 reference. The new background includes some very large fitted hazards, so small relative rounding errors can become appreciable absolute errors. Increasing the number of nodes does not by itself resolve this issue.

{accuracy_table}

The reference uses an independent, stable float64 implementation based on log1p/expm1, with 64 quadrature nodes and a certified quadrature bound below 1e-10 for each of the six patient/cohort cases. It is a tightly bounded approximation, **not exhaustive Shapley enumeration or a full exact-degree rule**. For patient 0, an additional 96-node calculation differed by at most {refs['gse24080']['crosscheck_max_absolute_difference']:.3g} for GSE24080 and {refs['tcga_lgg_methylation']['crosscheck_max_absolute_difference']:.3g} for TCGA LGG. These checks support its use as a numerical reference; they do not certify all float64 rounding. The validator also passes a tiny exhaustive-coalition test and a large-hazard comparison against the public float64 implementation.

Relative L2 error is the norm of the attribution error divided by the norm of the reference attribution vector. Maximum absolute error is the largest error over features. Neither is a prediction-performance metric.

## Why use the whole training background?

The background represents the empirical population over which missing features are averaged. A larger count alone does not ensure that a random subset adequately represents this fitted model. Its predicted training hazards are concentrated in a few rows.

For the first held-out patient:

{stability}

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
'''
    (OUTPUT / "README.md").write_text(text)
    return OUTPUT / "README.md"


if __name__ == "__main__":
    print(build_report())
