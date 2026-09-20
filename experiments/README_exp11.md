# Four additional text-classifier benchmarks

This extends the independent full-IMDB benchmark in `README_exp10.md` to the
other four text datasets named in the manuscript: Rotten Tomatoes, SST-2,
SMS spam, and emotion. The results are **not** the manuscript's existing
50-instance, Optuna-tuned SVC benchmark. Here each corpus has one fixed
RBF-SVC protocol (`C=2`, `gamma="scale"`) and a fixed 20-instance, stratified
held-out sample. Models are trained separately on complete training splits;
their training time is excluded from explanation timings.

## Corpus and score being explained

| Dataset | Training / evaluation documents | Binary or multiclass score | Notes |
|---|---:|---|---|
| Rotten Tomatoes | 8,530 / 1,066 | Binary SVC decision score | Official train/test |
| SST-2 | 67,349 / 872 | Binary SVC decision score | Official train/validation; public test labels are hidden |
| SMS spam | 4,459 / 1,115 | Binary SVC decision score | Full 5,574-message official source, stratified 80/20 split, seed 20260916 |
| Emotion | 16,000 / 2,000 | Predicted-class one-vs-rest SVC decision score | Six-class model; explain the corresponding binary RBF expansion |

Held-out classifier accuracies are 75.98% (Rotten Tomatoes), 78.10%
(SST-2 validation), 98.21% (SMS), and 89.75% (emotion). Their fitted
binary SVC expansions have 7,868, 31,182, and 2,344 support vectors,
respectively, for Rotten Tomatoes, SST-2 and SMS. Emotion uses six
one-vs-rest expansions with 4,121-10,465 support vectors depending on the
target class. These numbers describe the fixed models; they are not
accuracy results for the attribution methods.

All text is used **whole**, without a word-count or character-count filter.
The shared representation is lower-case English-stopword TF-IDF, unigrams
and bigrams, `max_df=0.95`, sublinear TF, capped at 5,000 terms. `min_df=3`
gives 5,000 terms for Rotten Tomatoes, SST-2 and emotion. On the full SMS
training set it gives only 3,692 terms; **SMS uses `min_df=2`** so that the
comparison really has `d=5,000`. This is a necessary exception to the
manuscript's blanket `min_df=3, d=5,000` statement. The manuscript lists
5,572 SMS messages, whereas the pinned official Hugging Face source used here
has 5,574; this experiment's split is therefore 4,459/1,115, not
4,457/1,115. These differences must be acknowledged if incorporating the
results into the manuscript.

For each selected text, the explained game is

`v_x(S) = sum_r alpha_r prod_{j in S} exp[-gamma (x_j-z_rj)^2]`.

The SVC intercept is excluded because it contributes zero Shapley value.
The same game, model expansion, and 5,000-dimensional numeric input go to
all three methods. Emotion's predicted class is saved for each instance;
the one-vs-rest component for that class is used consistently by CPU, Metal
and PKeX. The model score is **not** a multiclass probability.

## Explanation protocol

- Requested *absolute* max-coordinate tolerances: `1e-3`, `1e-5`, `1e-10`,
  `1e-16`. Certified node counts are selected per instance from a float64
  summary. The certificate covers quadrature error in exact arithmetic.
- QuadraSHAP CPU runs a float64 NumPy prefix scan. QuadraSHAP Metal runs the
  JAX prefix scan in float32 on the Apple M5 Max GPU. The requested `1e-16`
  does **not** imply float32 or float64 machine-accurate output.
- The CPU float64 `1e-16` result for the *same instance* is the stipulated
  reference for observed attribution errors. Its self-error is zero by
  definition; this is not an arbitrary-precision ground truth. We separately
  save the efficiency residual `|sum(phi) - (f_kernel(x)-sum(alpha))|` as a
  necessary, not sufficient, numerical stability diagnostic.
- The PKeX baseline is the user's repository
  [`RKHS-ExactSHAP`](https://github.com/Majeed7/RKHS-ExactSHAP),
  `ESPComputer(method="quadratic", use_scaling=True)`, run in
  one-support-vector chunks. PKeX is algebraically exact, so its
  *exact-arithmetic approximation* bound is zero. If it times out, the
  observed error remains unavailable rather than being set to zero.
- Each method-instance job has a 300-second wall-time cap, including process
  startup. CPU and Metal are run alone for fair timings; PKeX's independent
  CPU-only jobs were scheduled four at a time per dataset. The SMS and
  emotion batches overlapped, for up to eight PKeX jobs on the 18-core
  machine; each dataset also had one **isolated** pilot, which timed out.
  The stated per-tolerance
  QuadraSHAP time includes factor/bound summarization plus the median of two
  post-warm-up evaluations, excluding model training and one-time JIT
  compilation. Raw repeats, warm-up time, errors, certificates, diagnostics,
  statuses and machine metadata remain in the per-dataset result directory.

## Reproduction

From the QuadraSHAP repository root, for each `DATASET` in
`rotten_tomatoes sst2 sms_spam emotion`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp10_pkex_metal_imdb.py prepare --dataset DATASET --n-instances 20
ENABLE_PJRT_COMPATIBILITY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp10_pkex_metal_imdb.py run --dataset DATASET \
  --n-instances 20 --timeout 300 --chunk 64 --node-block 2 --repeats 2
```

The runner defaults to all three methods. It uses cached source files and
models when available. Per-dataset output is in
`experiments/results/exp11_pkex_metal_DATASET/`; `records.csv` is the
instance-level analysis table, while `summary.csv` contains aggregates.
`experiments/exp11_plot.py` regenerates the combined paper figures from the
saved records without rerunning any experiment.

The three PDF figures have distinct interpretations:

1. `text_certified_observed.pdf` places the maximum exact-arithmetic
   quadrature certificate and the worst **observed** coordinate error against
   the requested tolerance. It omits the CPU `1e-16` self-error from the log
   plot, because that value is zero by definition, not measured exactness.
   The annotation in each panel lists the node counts in x-axis order.
2. `text_runtime.pdf` reports the median end-to-end time per text at each
   tolerance. The 300-second line is a job cap, not a measured PKeX time;
   each panel displays its baseline completion count.
3. `text_numerical_stability.pdf` reports the worst efficiency residual over
   20 texts. A small residual is necessary but does not by itself prove that
   every coordinate is accurate.

`all_five_text_summary.csv` combines the four new experiments with the
previous full-IMDB benchmark for replotting or paper tables.

## Results

All 80 CPU and 80 Metal method-instance jobs completed. The table gives
per-dataset results over the fixed 20-text sample. `m_q` is the **median**
certified node budget at `1e-5`; errors are the **worst** max-coordinate error
over 20 texts against each text's own CPU `1e-16` reference; times are the
**median** per-text end-to-end explanation time at `1e-5` (seconds).

| Dataset | Median `m_q` at `1e-5` | CPU worst error at `1e-10` | Metal worst error at `1e-5` | CPU median time | Metal median time |
|---|---:|---:|---:|---:|---:|
| Rotten Tomatoes | 5 | 4.23e-13 | 1.74e-6 | 2.75 | 2.44 |
| SST-2 | 6 | 2.85e-12 | 6.15e-6 | 17.9 | 11.5 |
| SMS spam | 5 | 2.49e-14 | 7.55e-7 | 1.04 | 0.708 |
| Emotion | 5 | 3.54e-13 | 1.42e-6 | 4.21 | 2.91 |

At the same `d=5,000`, the typical `1e-5` budget is only five nodes for
three classifiers and six for SST-2. SST-2 also has the largest median
coefficient-amplitude summary (`A_max` approximately `6.97e3`, versus
`1.67e3` for Rotten Tomatoes, `8.66e2` for emotion and `1.36e2` for SMS).
These are descriptive model differences, **not** an experiment isolating a
causal effect of `A_max` or feature count: all four datasets have the same
nominal dimension and many other quantities change with the fitted model.

Every CPU and Metal case meets the requested `1e-3` and `1e-5` *observed*
max-coordinate error against this reference. CPU also meets `1e-10` on
all 80 texts. Metal does **not** meet `1e-10` or `1e-16` on any of the
80 texts; its float32 observed errors level off around `1e-6`. The
CPU `1e-16` rows are self-comparisons, not evidence of realized
`1e-16` accuracy. Their worst efficiency residuals range from `9.04e-14`
(SMS) to `4.74e-11` (SST-2).

Each dataset's *isolated* PKeX pilot timed out at 300 seconds without an
attribution. The remaining independent jobs also timed out: **0/20 PKeX
completions on each dataset**, versus 20/20 for CPU and 20/20 for Metal.
A timeout is a censored runtime
(`>300 s`), not a 300-second runtime estimate, and its observed error is
missing rather than zero.
