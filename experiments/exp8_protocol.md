# Experiment 8: synthetic multi-model node budgets

## Question and interpretation

Measure how quadrature effort changes with dimension and factor complexity, and
whether a priori certification at absolute max-attribution error 1e-6 uses
substantially fewer nodes than ceil(d/2). This is an empirical study, not a claim
that d alone determines the budget or that all multiplicative models admit a
uniform sublinear budget. All results, including exceptions, are retained.

Protocol frozen before production seeds 0-4. A feasibility pilot at seed 19 used
d=50 and 1000, three instances for non-RBF models and two for RBF. It identified
an unconverged Poisson fit; the production solver was changed from L-BFGS to
Newton-Cholesky. No signal strengths, test points, or reported dimensions were
selected based on obtaining the requested trend. Pilot results remain separate.

## Data, models and repetitions

- Dimensions: 50, 100, 250, 500, 1000. Exactly 30% generative informative features.
- Five independent training/test seeds; first 20 held-out instances per fit,
  without filtering. Predictive evaluation uses a held-out 500-point test block;
  its first 20 points also serve as explanation instances. No hyperparameter
  tuning uses this test block.
- Independent standard-normal coordinates; informative coefficients have random
  signs. In the main design each nonzero coefficient has magnitude 0.20.
- Poisson regression: counts with mean exp(X beta), ridge alpha=0.1.
- Logistic regression: Bernoulli(sigmoid(X beta)), L2 C=0.1. Explain ODDS,
  not probability or log-odds.
- Gaussian Naive Bayes: balanced classes, independent unit-variance coordinates,
  class means +/-beta/2. Explain posterior ODDS.
- RBF KRR: unit-scale linear signal plus Gaussian noise sd=0.2; gamma=1/d,
  alpha=0.1. This intentionally includes an easy, stable-bandwidth control.
- Training size 4000 for single-component models and 384 for KRR (its component
  count equals its training size). This is not a predictive-performance contest.
- Shared baseline x_b=0, the population mean in each design. Thus we do not
  compare neutral-kernel games to baseline games without disclosure.
- At d=1000, repeat the non-RBF analysis using an empirical interventional
  background consisting of the first eight training rows, fixed before errors
  are calculated. The reference explains exactly this finite-background game.
- Fixed-total-signal control: coefficient magnitude 0.20 sqrt(100/d), with all
  other choices unchanged. This separates accumulating per-feature signal from
  a dimension-only interpretation; parameter-estimation noise can still grow.

## Error, certificate and numerical validation

- Main error is max_i |phi_hat_i - phi_reference_i| on each model's native
  multiplicative output scale. Same absolute epsilon is not the same relative
  accuracy across output scales. Also retain attribution magnitudes.
- Epsilons: 1e-3, 1e-5 (literal 10e-6), 1e-6 (main), 1e-9.
- Use the repository's A_max B(m, Lambda_max) certificate and prefix-suffix
  evaluator. Lambda_max sums all coordinates (slightly looser than leave-i-out).
- Independent references: positive Bernstein coefficient recursion and reverse
  differentiation for single-component / eight-background games; independently
  generated SciPy Gauss-Legendre exactness-threshold rule for KRR. Every
  reference is cross-checked against SciPy at ceil(d/2)+3 nodes. The package's
  exact-threshold result is also retained, not artificially assigned zero error.
- Full integer sweep from 1 to min(ceil(d/2), certified_budget(1e-9)+5), plus the
  exactness threshold. Empirical budget is the hindsight first passing integer
  in the contiguous sweep. It is not an adaptive algorithm or a certificate.
- Keep strict target_met and a separate numerical-resolution diagnostic. A
  reference is flagged unresolved if cross-reference discrepancy or per-feature
  efficiency discrepancy exceeds epsilon/100. This is a diagnostic, not a
  rigorous floating-point bound. Do not add it to the theoretical certificate
  to claim universal bound coverage. Audit difficult transitions at higher
  precision. Retain all numerical-floor points in raw data.
- Show medians and 10th-90th percentile instance variation, labeled as spread,
  NOT confidence intervals. Use seed-level summaries and bootstrap intervals
  for differences/growth estimates to avoid treating instances from one fit as
  independent training repetitions.

## Numerical refinement added after the production run began

Every case flagged at the main 1e-6 target is recomputed using an entire
50-decimal-digit exact-degree Gauss-Legendre reference vector, with both nodes
and weights refined to that precision. Every saved float64 attribution vector
is compared to this decimal reference directly. Original records remain
unchanged; `validated_records.*` and per-instance `*_hp.json` retain the
corrections. No instance is removed. A separate selected-coordinate audit uses
independent 50-digit Bernstein integration to check the reference and the
quadrature-only certificate. This extension changes reference accuracy only,
not models, instances, hyperparameters, or the tested quadrature budgets.

## Stored and reproducible

Save configuration, package versions, all source hashes, training settings,
warnings and prediction quality; fitted parameters, held-out points, informative
mask, baseline/background; every factor table, attribution vector at every
tested m, both references, errors, bounds and timings. Separate collection from
plotting. Results contain no picked best seed and no discarded outliers.

Figures are diagnostic scientific evidence: do not edit the manuscript's
existing claims automatically. In particular, the asymptotic budget heuristic
is O(sqrt(Lambda log(A_max/epsilon))) only in its stated regime. If log A_max
itself scales linearly with d, that expression need not be sublinear in d.
