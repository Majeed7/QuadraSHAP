# QuadraSHAP paper experiments

Seven self-contained experiments. Each writes raw records (`*.csv`), `\input`-ready booktabs
tables (`*.tex`), figures (`*.pdf`, when matplotlib is installed) and a `meta.json` recording
versions, seeds, hardware and timings, under `experiments/results/<experiment>/`.

```bash
cd experiments
python run_all.py                      # everything (about 30 min on a laptop)
python run_all.py --quick              # smoke test, seconds
python run_all.py --only 1 2           # a subset
python exp3_synthetic_recovery.py --help   # every experiment is also a standalone script
```

Requirements: `numpy`, `scipy`, `scikit-learn` (already dependencies of the package). Optional:
`shap` (the SHAP-family baselines fall back to the reference implementations in `baselines.py`
otherwise, and every table records which ran), `matplotlib` (figures; the plotting data is in the
CSVs regardless, so pgfplots can be used instead), `datasets` (experiment 4 on a real corpus;
`--dataset synthetic` needs no network), `joblib` (model cache).

| # | Script | Question | Main outputs |
|---|---|---|---|
| 1 | `exp1_certified_bound.py` | Does the a priori node budget certify what it claims, and how tight is it? | `certificate_summary.tex`, `tightness.tex`, `decay.pdf` |
| 2 | `exp2_ranking_vs_eps.py` | Does the certified tolerance preserve the ranking of the Shapley values? | `ranking_summary.tex`, `ranking.pdf` |
| 3 | `exp3_synthetic_recovery.py` | 300 samples, 1000 features: accuracy, feature recovery and cost against the SHAP family and PKeX-Shapley | `accuracy.tex`, `recovery.tex`, `exact_methods.tex`, `tradeoff.pdf` |
| 4 | `exp4_sentiment_deletion.py` | Sentiment analysis: cost, and how much the sentiment moves when each method's top words are deleted | `timing.tex`, `deletion.tex`, `top_words.tex`, `deletion.pdf` |
| 5 | `exp5_backend_and_stability.py` | The two evaluators of the same rule: which is faster, and which is stable in which regime? | `timing.tex`, `stability.tex`, `backends.pdf` |
| 6 | `exp6_observed_vs_certified.py` | Observed against certified error, with a separately trained model and figure at d = 100, 500, 1000 | `error_d100.pdf`, `error_d500.pdf`, `error_d1000.pdf`, `summary.tex` |
| 7 | `exp7_node_growth.py` | Does the required node count grow moderately as the feature count increases? | `node_growth.pdf`, `summary.tex` |

## What each experiment establishes

**1 — the certificate.** For every (model, instance, tolerance) the node budget and its certified
error are computed from the factor tables *before* any quadrature runs; the quadrature is then run
at that budget and at the exactness threshold, and the observed error is compared with both the
certificate and the tolerance. The summary table reports how often the certificate held (it must be
every time), how many nodes it asked for against ⌈d/2⌉, how conservative it is, and what budget
selection costs. The figure shows observed error and certified bound against the number of nodes for
each setting, which is the picture that ties the appendix to the experiments.

**2 — ranking fidelity.** The certificate bounds an *absolute* error, so what matters for a ranking
is the tolerance relative to the size of the attributions. The same model is fitted to targets of
std 1, 10 and 100 (so the attributions scale with it) and every tolerance is evaluated by Kendall
tau, top-k overlap, sign agreement and whether the full ranking is unchanged. Plotted against
`eps / max|phi|`, all settings collapse onto one curve, which is what justifies the default
`eps = 1e-3` for a model whose output varies by order one.

**3 — synthetic recovery and cost.** One RBF kernel-ridge model on 300×1000 data with ten
informative features. Two references: the exact Shapley values of the fitted model (QuadraSHAP at
⌈d/2⌉ nodes), against which every estimator is scored, and the generative informative set, for
precision@k / recall@k / AUROC. All methods explain the *same* value function (empirical
interventional over a shared background), so the comparison is about the estimator. A separate block
compares the two exact methods for the neutral-factor value function, PKeX-Shapley and QuadraSHAP.

**4 — sentiment analysis.** TF-IDF + RBF-SVC trained once and cached under `models/`. Per instance
every method is timed, then its top-k most positive words are deleted from the document (their
TF-IDF entries zeroed) and the fall in the sentiment score is recorded — the deletion / AOPC
protocol. A random ranking is the floor; the fraction of reviews whose predicted label flips after
five deletions is reported alongside.

**5 — the two evaluators.** The same quadrature rule is evaluated either by one shared product per
node in log-space (working set `O(m_q n)`, two transcendentals per factor, factor magnitudes clamped
at `1e-12`) or by the division-free prefix–suffix scan of the appendix (no transcendentals and no
clamping, working set `O(m_q n d)`). Part A times both over a `(d, m_q)` grid with the automatic
blocking on and off; the scan is 1.5–3.4x faster once one sweep of its working set is blocked to fit
the last-level cache, and slower than log-space when it is not. Part B scores both against a
`float128` prefix–suffix reference on six regimes: the scan loses only when a *partial* product
leaves the float64 range while the answer does not (mixed-magnitude GLM factors), and log-space loses
wherever a factor falls below its clamp — by twenty-nine orders of magnitude when a few factors are
`1e-18`, and by 15% when a factor vanishes at a node. Underflow of the shared product is by itself
harmless: it happens at the upper nodes, which contribute nothing to the integral.

**6 — the certificate over a population.** Where experiment 1 reports the certificate per setting,
this one plots it once per feature count. Independent RBF kernel-ridge models are trained with the
same fixed recipe at `d = 100, 500, 1000`, with 30% informative features (30/150/300) and using
60/40/25 held-out instances respectively. Each
figure varies only the quadrature node count `m_q`: solid is the observed error
`max_i |phi_i(m_q) - phi_i(exact)|`, dashed is the a priori certified error, and the bands are the
5th--95th percentiles over instances. The figures deliberately contain no secondary difficulty
grouping or calibration.

**7 — node growth with feature count.** Independent RBF kernel-ridge models are trained at
`d = 50, 100, 250, 500, 1000` with one fixed raw bandwidth, 30% informative features, and 30 held-out
instances per dimension. At absolute error tolerance `1e-6`, the figure compares the a priori
certified node budget with the smallest node count that reaches the tolerance in hindsight. This
directly tests whether the required quadrature budget grows with dimension and how fast it grows.

**Why the observed curve stops falling** (and `check_rule_precision.py`, which proves it). The
plateau at `~1e-13` is *not* a failure of the certificate. Two things put a floor under any
measurement of the error in float64:

* `phi_i` is a sum whose terms have absolute values summing to `A_i <= A_max` — the same scale the
  certificate is stated on — so its absolute rounding error is of order `u * A_max` with
  `u = 2^-53 = 1.11e-16`. That line is drawn in each panel.
* More importantly, `numpy.polynomial.legendre.leggauss` returns the nodes and weights themselves in
  float64, so the *rule* carries an `O(u)` error that does not decrease with `m_q` at all.

`check_rule_precision.py` recomputes the Gauss–Legendre rule to 50 decimal digits (Newton refinement
on the Legendre recurrence, in `decimal`) and evaluates the integrand there. At `d = 60`,
`Lambda ~ 25`, `m_q = 21` the true quadrature error is `4e-35` against a certified `4e-22` — the
certificate holds with thirteen orders of magnitude to spare, while the same difference computed with
float64 nodes reads `1e-15`. Everything below `~1e-13` in the figure is therefore arithmetic, never
the method. Two consequences worth stating in the paper: a tolerance below `~1e-13 * A_max` cannot be
delivered *or even measured* in double precision by any method, and the observed-error curve of any
such experiment is floor-limited by the accuracy of the quadrature rule, not of the attributions.

The floor is measured per instance rather than assumed: the rule is exact at and above `ceil(d/2)`
nodes, so `ceil(d/2)` and `ceil(d/2)+1` are the same number in exact arithmetic and differ only by
floating-point error. Points below it are shaded and excluded from the slack statistics. (Assuming a
fixed multiple of `eps * A_max` instead produced 31 spurious "violations" at `d=1000` that were pure
noise.) `--replot` redraws the figure from `population.csv` without re-running anything.

## Caveats to state in the paper

* **PKeX-Shapley** is a faithful reimplementation of the published `O(nd²)` weighted
  elementary-symmetric-polynomial recursion (`baselines.pkex_shapley`), not the authors' code, so
  its timings characterise the algorithm rather than their implementation. It agrees with
  QuadraSHAP to machine precision for wide kernels; for narrow kernels the symmetric polynomials
  overflow (`max_abs_e` in the diagnostics) and the recursion loses all accuracy, which is recorded.
* **Sampling baselines** use the `shap` package when it is installed. Without it the reference
  implementations in `baselines.py` are used (validated against brute-force enumeration in the smoke
  tests); the `implementation` column of every CSV and the note under every table says which ran.
* **The exact reference** is QuadraSHAP at the exactness threshold, which is exact in exact
  arithmetic. Where an independent check is available (brute-force enumeration at small `d`,
  PKeX-Shapley at large `d`) it agrees to machine precision.
* **The default evaluator** is the prefix–suffix scan, and experiment 5 is the justification: it is
  both the faster of the two under the automatic blocking and the only one that is exact when a
  factor vanishes or is very small. The log-space evaluator remains the right choice when a partial
  product can leave the float64 range — factors far from one on both sides, as in a GLM with large
  linear predictors — and when memory is too tight to block.
* Timings are single-threaded NumPy wall-clock on the machine recorded in `meta.json`. The automatic
  block plan is cache-aware, so the timing table is machine-dependent by design; `meta.json` records
  the machine and `blocks.last_level_cache_bytes()` the cache it was planned against.
