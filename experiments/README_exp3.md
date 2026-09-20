# Experiment 3 v2: informative-feature recovery at a computational budget

This study asks whether certified QuadraSHAP matches the informative-feature
recovery of reference Shapley values at lower computational cost. It separates
explanation accuracy from recovery of the data generator's informative set.
Neither exact Shapley values nor a small quadrature error guarantee perfect
generating-feature recovery.

## Run on this Mac

The tested interpreter is `/usr/local/bin/python3`. From any directory, run:

```sh
/usr/local/bin/python3 /Users/Majid/surfdrive/Research/ExplainableAI/QuadraSHAP/experiments/exp3_synthetic_recovery.py --profile full
```

This is a plain local numerical job: no AI/API requests, network downloads,
GPU, or active assistant supervision. Keep the Mac awake and connected to
power. It prints progress and writes each job immediately. Do not run other
compute-intensive benchmarks simultaneously if you want comparable timings.
Four numerical-library threads are used by default.

The agreed full run uses five fits with six explained inputs each (30 total).
Allow roughly 1-2 hours on the tested Mac, estimated from the pilot rather
than a measured full run. Reference construction and exact-rule verification
account for about 56 minutes of that estimate.

Run the identical command again to resume. Completed jobs are verified and
skipped, including the fitted models. Failed/time-limited jobs remain recorded;
add `--retry-failed` to retry only those jobs. A nonzero exit status means the
collection or numerical checks did not finish successfully. Do not change the
collection script, numerical package versions, core library, or configuration
mid-run: resume checks reject mismatches to avoid mixing experiments. Multiple
collectors cannot use the same output directory concurrently.

Default output:

`experiments/results/exp3_recovery_v2/full/`

Other profiles are separate:

```sh
python3 experiments/exp3_synthetic_recovery.py --quick
python3 experiments/exp3_synthetic_recovery.py --profile pilot
```

`quick` is a d=20 pipeline test with one seed and three explained inputs.
`pilot` is a d=1000 feasibility run with seed 19 and three explained inputs.
Neither is publication evidence. Production uses different seeds, 0-4.
The script defaults to `pilot` if no profile is given, so an unqualified call
does not start the long publication-size sweep.

The existing pilot results and source snapshots are retained unchanged.
Because the collection source changed when reducing the full profile to 30
inputs, use `--profile pilot --replot` to review that saved pilot; rerunning
its collection with the current source requires a fresh `--output` directory.

To redraw from existing vectors without fitting or explaining anything:

```sh
python3 experiments/exp3_synthetic_recovery.py --profile full --replot
```

On a different machine, use its Python environment with NumPy, SciPy,
scikit-learn, SHAP, matplotlib, and threadpoolctl installed, plus this repository's
source. Installed versions and numerical source hashes are recorded. There is
no silent fallback to a different baseline implementation. Use a new output
directory rather than mixing runtimes collected on different machines.

## Fixed production protocol

- **One dimension:** d=1000, with exactly 300 informative features. All raw
  features are independent standard normal; informative coordinates are not
  given larger variances or preferential preprocessing.
- **Target:** a linear signal plus two modest nonlinear components and noise.
  Generating coefficients have independent random signs and magnitudes drawn
  uniformly from [0.5, 1.5] on the informative support, then are normalized to
  Euclidean norm one. The nonlinear terms are
  `0.2*sin(X[a]*X[b]) + 0.15*(X[c]**2 - 1)` for three informative coordinates.
  Independent Gaussian noise has standard deviation 0.1. No normalization
  uses validation or test targets.
- **Five independent fits:** seeds 0-4. Each has independent streams for 2000
  training, 500 validation, and 1000 test inputs. Input means/scales and target
  centering are learned on training data only. All fits are retained.
- **Model:** RBF kernel ridge regression. Select gamma from {0.1/d, 1/d} and
  alpha from {0.01, 0.1}, using validation R2 only. No informative-mask or test
  information is used to select hyperparameters. The selected training fit is
  retained without refitting on validation data. Report held-out R2; a value
  below 0.5 is flagged, not used to discard the fit.
- **Explained inputs:** the first six test inputs per fit, unfiltered. There are
  30 explained inputs overall, grouped into five training repetitions. This
  reduces computation while retaining all five fits; global recovery scores
  based on six inputs may be noisier than scores based on 20 inputs.
- **Value function:** empirical interventional prediction, averaging over the
  first five training inputs. Every estimator sees the identical game, output
  scale, inputs, and background. This study does not compare background choices.
- **QuadraSHAP:** the certified budgets for absolute tolerances
  {1e-1, 1e-2, 1e-3, 1e-6}, plus the 500-node polynomial-exactness rule. This
  uses the package's NumPy prefix-suffix evaluator with component/node blocking.
- **Baselines:** the official installed SHAP package's KernelExplainer with
  sample budgets {2128, 4128, 8128}, and PermutationExplainer with {1, 4, 16}
  forward/reverse permutation pairs. Each condition has three reproducible
  sampling repetitions. KernelSHAP explicitly uses `l1_reg=0`, not its default
  ten-feature selection. No sampler is given the informative support.

The baselines receive a fast, batched coalition evaluator for the fitted RBF
expansion. Coalition indicators are its inputs; a zero-indicator row denotes
the same *empirical interventional game*, not zero replacement in the original
feature space. Ten random coalitions per input are checked against directly
masking the original features and evaluating the fitted model. All d original
features remain players.

## Independent reference and numerical checks

For every input, the reference uses SciPy's independently generated 500-node
Gauss-Legendre rule and shared log-products, without the package's quadrature
implementation or budget calculation. A second 503-node evaluation must agree
within 1e-9 absolute error. Game/prediction agreement and the efficiency
residual must also be within 1e-9. The package's 500-node answer is independently
checked against this reference.

This is a cross-validated floating-point reference, not arbitrary-precision
ground truth. The mathematical certificate excludes rounding and the error of
using a finite background to represent a population. Every certified run's
measured error is checked against its requested tolerance; failures are saved.

The standalone tests compare all the way through to brute-force coalition
enumeration on d=6, test KernelSHAP at exhaustive coalition budget, seeded
sampling reproducibility, data-stream independence, tie handling, and unique
checkpoint paths:

```sh
python3 tests/test_exp3_recovery_protocol.py
```

## What is measured

**Attribution fidelity:** per input, maximum absolute feature error and relative
L2 error against the independent fitted-model reference. The figure averages
the per-input maximum errors; it does not substitute an efficiency residual.

**Global feature recovery:** for each fit/method/budget/sampling repetition,
compute one feature score by averaging absolute attributions over the same six
test inputs. Compare these scores to the 300 informative labels using average
precision (primary), precision among the top 300 features, and AUROC. AP/AUROC
use tie-aware sklearn metrics. Top-300 ties use a seeded, label-independent
ordering. Precision@300 and recall@300 coincide, so only one is reported.

Average sampling repetitions within each fit first, then summarize the five
independent fits. Sampling repetitions and the six inputs are not presented as
additional independent training repetitions. Plot error bars span the minimum
and maximum fit-level summaries, not confidence intervals.

**Runtime:** CPU wall time, including estimator-specific game construction,
bound/budget selection for certified QuadraSHAP, sampling/solving for SHAP,
and quadrature. Model training, reference construction, Python startup/imports,
and a small four-feature SHAP compiler warmup are reported/excluded separately.
The warmup contains no experimental inputs or informative labels. Each point
is a fresh estimator run, not amortized across budgets. Nodes and coalition
evaluations are stored separately and are not equated to one unit of work.

Each worker has a 180-second wall-time cap (including startup); reference and
exact-rule jobs have 1800 seconds. A timeout is not assigned a fictitious
recovery score or treated as a successfully reached accuracy target.

Incomplete groups are not silently averaged over a favorable subset: a recovery
point requires every explained input and sampling repetition from every
planned fit. The failure table retains all missing, failed and timed-out jobs.

## Saved artifacts

- `config.json`, `provenance.json`, `source_snapshot/`: immutable collection
  configuration, versions, thread settings and source hashes/copies.
- `seed_*/fit.npz` and `fit.json`: generating support and coefficients, all data
  splits, transformations, fitted coefficients, predictions, model checks,
  hyperparameter candidates and fit hashes. No pickle is required.
- `seed_*/jobs/*`: each attribution vector, run metadata, checksums, and log.
- `per_instance.csv`, `recovery_per_fit_repeat.csv`, `seed_means.csv`,
  `summary.csv`, `failures.csv`: regenerable analysis at each aggregation level.
- `completion.json`, `REPORT.md`: completeness, numerical checks, and results.
- `figures/recovery_tradeoff.pdf`: the two-panel publication-layout figure;
  its PNG preview stays under `figures/backup/`.
- `analysis_provenance.json`: identifies the plotting/analysis revision.

The previous Experiment 3 script is preserved at
`archived_experiments/exp3_before_recovery_revision_20260915.py`. Its old
300-point, ten-informative-feature, enlarged-feature-scale design is not mixed
with this study. Shared helpers, the sentiment experiments, existing results,
and the manuscript are not modified by this run.
