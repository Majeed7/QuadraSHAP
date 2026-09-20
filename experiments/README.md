# QuadraSHAP paper experiments

Eight self-contained experiments (experiment 8 is the multi-model budget study kept in its own scripts). Each writes raw records (`*.csv`), `\input`-ready booktabs
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
| 3 | `exp3_synthetic_recovery.py` | Equal-scale synthetic data: fitted-model attribution fidelity and global informative-feature recovery versus computational cost | See [README_exp3.md](README_exp3.md); resumable pilot/full profiles, saved vectors, `recovery_tradeoff.pdf` |
| 4 | `exp4_sentiment_deletion.py` | Sentiment analysis: cost, and how much the sentiment moves when each method's top words are deleted | `timing.tex`, `deletion.tex`, `top_words.tex`, `deletion.pdf` |
| 5 | `exp5_backend_and_stability.py` | The two evaluators of the same rule: which is faster, and which is stable in which regime? | `timing.tex`, `stability.tex`, `backends.pdf` |
| 6 | `exp6_certificate_population.py` | Observed against certified error over a population of instances at d = 100, 500, 1000 | `certificate_population.pdf`, `coverage.tex` |
| 7 | `exp7_text_methods.py` | Text classification on four corpora, words masked to zero (d = 15-100): error vs cost against KernelSHAP, SamplingSHAP, Permutation SHAP and LIME | `error_vs_time.pdf`, `cost_bars.pdf`, `cost_vs_d.pdf`, `top5_vs_evals.pdf` |
| 9 | `exp9_text_interventional.py` | The interventional value function over the vocabulary (d = 400-1100 players, 30k-90k games): wall-clock to a certified accuracy against the same baselines | `time_to_target.pdf`, `error_vs_time.pdf`, `summary.tex` |

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

**3 — synthetic recovery and cost (revised).** The production profile uses five RBF-KRR fits
with 2000 training points, d=1000, and 30% informative features of the same input scale as noise
features. Validation-only tuning, independent reference checks, six explained inputs per fit (30 total),
and repeated official SHAP baselines separate attribution fidelity from global feature recovery.
Run `python exp3_synthetic_recovery.py --profile full` explicitly for the long collection;
the default is a separate pilot. Results are checkpointed under `results/exp3_recovery_v2/`.
See [README_exp3.md](README_exp3.md) for the full protocol, restart command, timing conventions,
and saved-data plotting. The earlier script is preserved in `archived_experiments/` at the
repository root; its old data design and conclusions are not mixed with the revised study.

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
this one plots it. Three panels, `d = 100, 500, 1000` (RBF kernel ridge, neutral-factor value
function, 60/40/25 instances per curve), each with two kernel bandwidths so that the total relative
variation is `Lambda ~ 7` and `Lambda ~ 40-60` inside the same panel. Solid is the observed error
`max_i |phi_i(m_q) - phi_i(exact)|`, dashed the a priori certified error; both are medians over the
population with a 5–95 percentile band. The certified curve is above the observed one at every one of
the 2000 points, the median slack is 2.3–3.0 orders of magnitude, and the two fall at the same rate.

**What the panels are for.** The budget is set by `Lambda` and by `eps / A_max`, *not* by the
dimension: at `Lambda ~ 7` the budget is `m* = 5-6` whether `d` is 100 or 1000, while the exactness
threshold `ceil(d/2)` grows from 50 to 500 — a 10x to 100x saving that is the point of the appendix.
Reading down a panel instead shows what does move the budget: at `Lambda ~ 60` it is `m* = 14`. A
figure with one bandwidth per panel makes the budget look suspiciously constant across `d`; two
bandwidths per panel show why it is.

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

**7 — text classification against the SHAP family.** On a TF-IDF + RBF-SVM the exact Shapley values
are available, so this is an error-versus-cost benchmark against ground truth rather than a
comparison of opinions. Four corpora spanning the document lengths that matter — SST-2 (`d ~ 15`),
Rotten Tomatoes (`d ~ 14`), AG News (`d ~ 27`) and IMDB (`d ~ 100`), where `d` is the number of
distinct words in the document and therefore the number of players in the game.

*Everything explains the same game.* Masking a word sets its TF-IDF entry to the baseline value 0 —
the baseline value function of Section 4, and exactly what the deletion protocol does. Words outside
a document's support are never masked and fold into a constant per support vector, leaving a weighted
sum of product games over the support. The construction is checked per document against the model's
own `decision_function` (agreement ~1e-13) and the exact values satisfy efficiency to ~1e-14; on
small games the whole chain was validated against brute-force enumeration over all 2^d coalitions
(agreement 2.5e-16).

*Cost is measured in two currencies.* Wall-clock seconds, and the number of value-function
evaluations — one evaluation costs `n_sv * d` factor operations and so does one quadrature node, so
QuadraSHAP at `m_q` nodes is charged `m_q` evaluations and every method lands on one axis. The
sampling baselines are given the same fast vectorised evaluator QuadraSHAP uses (one matrix product
per batch of coalitions), which is much quicker than calling `decision_function` on masked sparse
vectors: the comparison is deliberately generous to them.

*Result.* Evaluations needed for a relative error of 1e-2 against the exact values, median over 25
documents and 3 seeds:

| corpus | d | KernelSHAP | Permutation | Sampling | LIME | QuadraSHAP | QuadraSHAP exact |
|---|---|---|---|---|---|---|---|
| sst2 | 15 | 514 | 1088 | >8192 | >8192 | **2** | 8 (4 ms) |
| mr | 14 | 1026 | 540 | >8192 | >8192 | **2** | 7 (4 ms) |
| agnews | 27 | 4098 | 2112 | >8192 | >8192 | **2** | 14 (13 ms) |
| imdb | 100 | >8192 | 4040 | >8192 | >8192 | **2** | 50 (0.53 s) |

Two quadrature nodes already beat a KernelSHAP run of several thousand coalitions, and the exactness
threshold — machine precision, not an approximation — costs 7 to 50. The gap widens with `d`
(`error_vs_d.pdf`): the samplers degrade as documents get longer while QuadraSHAP stays exact.
SamplingSHAP and LIME never reach 1e-2 — LIME because it is biased, not merely noisy: it solves a
different problem and its error plateaus instead of decaying.

*Everything is stored for re-plotting.* `records.csv` has one row per (dataset, document, method,
seed, budget) with five metrics, the evaluation count and the seconds; `instances.csv` has each
document, its `d`, its top words by exact attribution, `Lambda` and `A_max`; `exact_phi.npz` has the
exact attributions themselves. `python exp7_text_methods.py --replot` redraws every figure from
those files without running any quadrature. The corpora live in `experiments/data/*.csv.gz` (4 MB,
built by `text_datasets.py --build`, which is the only step that needs network access).

**9 — the interventional value function over the vocabulary.** Experiment 7 masked words to zero, so a
word absent from the document was a dummy player and the game had only the document's own words.
Here an absent word takes the TF-IDF value it has in each of a set of real background documents
(60 / 60 / 40 / 20 for SST-2 / MR / AG News / IMDB), averaged — the empirical interventional value
function of Proposition 5, and the standard setting of `shap` with a background set. Every word in
the document *or in any background document* is then a genuine player, nobody can prune, and the
game has d = 420–1100 players and 29k–89k product games (n_sv × n_b) on a 5000-word vocabulary.

*Cost is wall-clock only.* Experiment 7 originally charged one quadrature node as one value-function
evaluation on the grounds that both are O(n·d); that equivalence does not hold — a node is several
passes over the factor table and returns all d partial products, an evaluation is one pass and one
scalar, and measured end to end the ratio depends on d and on the implementation — so it was
withdrawn. Seconds need no assumption. The samplers use the batched matrix-product evaluator and run
under a 60 s cap per document; where they do not reach the target inside the cap, the time they would
need is extrapolated from their last checkpoint at the Monte-Carlo rate t^-1/2, the most favourable
assumption for a sampler, and drawn hatched.

*Reference.* QuadraSHAP at the node count whose a priori certificate is 1e-14·A_max (7–8 nodes here),
checked on every document against m_ref+8 nodes (agreement 3e-15 – 1e-14) and once per corpus
against the full exactness-threshold rule ⌈d/2⌉ = 211–556 nodes (agreement 5e-15 – 1.5e-14, at
145–237 s). In float64 the full rule is not more accurate than the certified one — both sit on the
rounding floor — so the 2500-node computation the exactness theorem would suggest buys nothing.

*Result.* Λ ≈ 2.2 on every corpus, so the certified budget for ε = 10⁻³ is **3 nodes**, taking ~3 s
(1.7 s for the rule, 1.3 s for the a priori summary) and achieving a relative error of ~5×10⁻⁷;
ε = 10⁻⁶ is 4–5 nodes and ~3.5 s at ~10⁻⁹. Median wall-clock to 10⁻³ relative error:

| corpus | d | games | QuadraSHAP | KernelSHAP | Permutation | Sampling | LIME |
|---|---|---|---|---|---|---|---|
| sst2 | 422 | 89k | **2.9 s** | 33 min | 33 min | 16 h | no convergence |
| mr | 451 | 89k | **3.4 s** | 26 min | 18 min | 10 h | no convergence |
| agnews | 715 | 54k | **3.3 s** | 46 min | 13 min | 2 d | no convergence |
| imdb | 1102 | 29k | **2.7 s** | 11 min | 85 s (measured) | 6 h | no convergence |

Thirty to a thousand times at 10⁻³; at the 10⁻⁶ QuadraSHAP delivers for one more node, the samplers'
t^-1/2 rate puts them at months to years. `--replot` redraws everything from `records.csv`.

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
* **The text baselines are reimplementations** (`text_attribution.py`), not the `shap` package:
  KernelSHAP as coalitions drawn from the Shapley kernel plus constrained least squares,
  SamplingSHAP as per-feature Monte-Carlo differences, Permutation SHAP with antithetic sweeps, LIME
  with the standard exponential kernel. All four were validated against brute-force Shapley values on
  small games. They characterise the algorithms rather than a particular implementation, and they run
  on the fast product-structured evaluator, which flatters them.
* Timings are single-threaded NumPy wall-clock on the machine recorded in `meta.json`. The automatic
  block plan is cache-aware, so the timing table is machine-dependent by design; `meta.json` records
  the machine and `blocks.last_level_cache_bytes()` the cache it was planned against.
