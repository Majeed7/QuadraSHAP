# KernelSHAP and SamplingSHAP with 30 text backgrounds

This experiment supplements the four 5,000-feature text classifiers in
`README_exp11.md`. The subsequent matched-game accuracy comparison is in
`README_exp13.md`; the raw KernelSHAP and SamplingSHAP vectors here are
reused there without rerunning the baselines. This experiment uses the same
fitted RBF-SVCs, the same 20 fixed held-out texts per dataset, and the same
predicted-class one-vs-rest decision score
for emotion. No classifier is retrained.

For each dataset, `exp12_shap_background_baselines.py prepare` stores one
fixed, seed-20260916, stratified sample of **30 training texts**. The binary
backgrounds contain 15 texts per class; emotion contains five per class.
KernelSHAP and SamplingSHAP use this identical transformed TF-IDF background
and the same model score for every instance. Both receive `nsamples=1000`
through SHAP 0.51.0, with the library's default KernelSHAP
`l1_reg="num_features(10)"`. The code saves the actual number of model rows
evaluated, since `nsamples=1000` is **not** an equal number of model
predictions: KernelSHAP applies sampled masks to all 30 backgrounds, whereas
SamplingSHAP samples backgrounds within its permutation estimator. The
complete runs used 30,032 and 2,032 model rows respectively, including setup
and endpoint checks. Each method-instance subprocess has a 300-second wall cap.

## Critical value-function distinction

These two SHAP methods explain the usual **30-background interventional**
game: missing TF-IDF coordinates are replaced by those from a background
text, with the resulting model predictions averaged over the background.
Exp11's certified QuadraSHAP and PKeX results explain a **neutral-factor**
product-kernel game, in which missing features remove a kernel factor.
These are different mathematical targets. Therefore:

- Do not compare the KernelSHAP or SamplingSHAP attribution vector to
  exp11's CPU `1e-16` vector and call the discrepancy an observed error.
- Neither stochastic/finite-sample method supplies the exp11 quadrature
  certificate. Its `certified_bound` and observed error to an exact
  interventional reference remain unavailable.
- Efficiency residual relative to the same background game, finite outputs,
  and *disagreement* between KernelSHAP and SamplingSHAP are recorded. A
  small efficiency residual is necessary but not sufficient for accuracy;
  disagreement is not a ground-truth error.

To make a common error plot, all methods would need to be rerun on the same
30-background interventional game and a sufficiently accurate reference
would need to be computed. At 5,000 features and up to 31,182 support
vectors, doing this for all 20 texts is a separate, much larger experiment.

## Saved outputs and reproduction

`results/exp12_shap_background_DATASET/` contains `background.npz`,
`prepare_meta.json`, `run_meta.json`, `records.csv` and per-instance
attribution vectors, timing, model-call counts and process logs under `raw/`.
The class-balanced background indices are saved, as are source and model
hashes. The run is deterministic given the saved backgrounds and per-text
NumPy seed, subject to the installed SHAP and scikit-learn versions.

From the repository root, for each `DATASET` in
`rotten_tomatoes sst2 sms_spam emotion`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp12_shap_background_baselines.py prepare --dataset DATASET
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/exp12_shap_background_baselines.py run --dataset DATASET \
  --methods kernel sampling --n-instances 20 --timeout 300
```

## Results

All **160 method–text runs completed**, returned finite attribution vectors,
and stayed below the 300-second cap. The table reports the median elapsed
time over the same 20 held-out texts in each dataset. Times include explainer
initialization and attribution evaluation, but exclude model training and the
one final score call used only to compute the efficiency diagnostic. The
recorded model-row count does include that final one-row call.

| Dataset | KernelSHAP median time | SamplingSHAP median time | Median model rows, Kernel / Sampling | Median max-coordinate disagreement |
|---|---:|---:|---:|---:|
| Rotten Tomatoes | 9.07 s | 0.85 s | 30,032 / 2,032 | 0.196 |
| SST-2 | 19.72 s | 1.63 s | 30,032 / 2,032 | 0.134 |
| SMS spam | 3.22 s | 0.41 s | 30,032 / 2,032 | 0.153 |
| Emotion (OVR) | 10.81 s | 1.00 s | 30,032 / 2,032 | 0.167 |

The disagreement column is the median, across texts, of
`max_j |phi_j^KernelSHAP - phi_j^SamplingSHAP|`. **It is not an observed
error** for either estimator. KernelSHAP's default regularization selects
10 nonzero coordinates; SamplingSHAP returned a median of 34.5–49.5
nonzero coordinates, depending on the corpus. Thus differences reflect both
estimation and the default feature-selection behavior. The largest absolute
efficiency residual over all four datasets is `4.44e-16` for KernelSHAP
and `3.61e-7` for SamplingSHAP. This checks the sum constraint only; it
does not establish attribution accuracy.

`results/exp12_shap_background_figures/text_shap_background_runtime.pdf`
is the paper-sized, paired 20-text runtime figure. The same directory holds
`method_summary.csv` and `pairwise_disagreement.csv`, so the plot can be
restyled without rerunning explainers. Interpret this figure as the cost of
two standard **interventional** estimators at the specified sampling
settings, not as an equal-error or same-value-function comparison with
QuadraSHAP. No certified bound or observed error to an exact interventional
reference is claimed for these two baselines.

## Full-length IMDB pilot (not pooled with the four-dataset result)

I also prepared 30 stratified **full-length** IMDB training reviews and
attempted the first of the 20 fixed IMDB test reviews at the same settings.
KernelSHAP completed in 183.19 seconds; its vector and diagnostics are saved
under `results/exp12_shap_background_imdb/`. SamplingSHAP could not return
an attribution vector with `nsamples=1000`: 1,469 TF-IDF features vary
between that review and the 30-background set, while SHAP 0.51.0 allocates
samples in pairs and therefore needs at least 2,938 samples to cover them
once. The explicit failure status and log are retained. The IMDB pilot is
**not** counted in the 160 completed four-dataset runs or the runtime figure;
running more IMDB texts at an altered sampling budget would be a different
protocol.
