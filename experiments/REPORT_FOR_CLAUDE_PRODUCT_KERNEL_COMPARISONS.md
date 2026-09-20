# Product-kernel experiments: manuscript handoff

This report supplies two separate paper sections. The main-text comparison is **QuadraSHAP versus PKeX-Shapley on the same neutral-factor game**. The appendix comparison is **QuadraSHAP versus KernelSHAP and SamplingSHAP on the same 30-background interventional game**. These games have different Shapley values. Do not place their attribution errors in one ranking or interpret the difference between a neutral-game vector and an interventional-game vector as error.

All figures below are existing vector PDFs. All times are seconds per explained input, exclude model training, and are medians over the indicated held-out inputs unless stated otherwise. “Observed error” means the maximum absolute difference over attribution coordinates from a separate float64 CPU QuadraSHAP reference at requested quadrature tolerance $10^{-16}$; it is an empirical comparison, not an arbitrary-precision proof of exactness. “Certified bound” is the a-priori **quadrature** bound in exact arithmetic. It does not cover float32 or float64 rounding, and, for the interventional experiment, it does not cover the finite background's error relative to a population distribution.

## Section 1 for the main paper: product-kernel comparison with PKeX-Shapley

### Paper-ready account

We compare QuadraSHAP with the quadratic, scaling-enabled ESP implementation of PKeX-Shapley from the authors' [RKHS-ExactSHAP repository](https://github.com/Majeed7/RKHS-ExactSHAP), commit `f74de21548ae23fabd467d7dcae3b3fe5ea76428`. Both methods explain the same neutral-factor product-kernel game,

$$v_x(S)=\sum_r\alpha_r\prod_{j\in S}\exp[-\gamma(x_j-z_{rj})^2].$$

An absent feature removes its kernel factor; there is no replacement background set. For an RBF SVC, the intercept has zero Shapley contribution and is omitted. PKeX is algebraically exact in exact arithmetic. QuadraSHAP chooses a certified quadrature budget $m_q$ for each input and requested maximum-coordinate tolerance $\varepsilon\in\{10^{-3},10^{-5},10^{-10},10^{-16}\}$. The Metal prefix scan uses float32; factor construction, certificate selection, and the final weighted reduction use float64 on CPU. Hence $10^{-16}$ is a request to the quadrature rule, **not** a claim of $10^{-16}$ end-to-end Metal accuracy.

For the synthetic study, we generated 1,000 regression samples independently at each $d\in\{50,500,1000,2000,5000\}$ using `make_regression` with $\lfloor d/4\rfloor$ informative features, noise 0.1, and seed 42. We used a shuffled 80/20 train/test split (seed 42), fit feature and target standardization on the training set, and trained RBF kernel ridge regression with $\gamma=1/d$ and ridge parameter 0.1. Fifty held-out inputs per $d$ were fixed with seed 20260917. QuadraSHAP completed all 50 inputs at each of four tolerances for every $d$. PKeX completed all 50 inputs at each **attempted** dimension, $d=50,500,1000$, with a 300-second per-input cap. It was intentionally **not run** at $d=2000$ or $5000$ in this new study, so no timeout claim should be made for those cells.

At $\varepsilon=10^{-5}$, QuadraSHAP's median $m_q$ was 5, 5, 4, 4, and 4 as $d$ increased; median times were 0.0063, 0.0183, 0.0241, 0.0359, and 0.0945 seconds. PKeX's median times at the first three dimensions were 0.171, 17.50, and 79.35 seconds. PKeX agreed with the float64 reference to approximately $10^{-15}$, whereas Metal's median maximum-coordinate differences were $9.03\times10^{-8}$ to $4.05\times10^{-7}$ at $\varepsilon=10^{-5}$. Tightening the quadrature request to $10^{-16}$ raised the median budget only to 10, 9, 9, 8, and 8 nodes; the observed Metal errors then plateaued at $6.90\times10^{-8}$ to $3.85\times10^{-7}$. This is the float32 precision limit, not a failure of the exact-arithmetic certificate. The maximum change in the float64 reference after adding two nodes was $1.78\times10^{-15}$. The particular $\gamma=1/d$ RBF family controls total kernel variation as $d$ grows, so the nearly flat node counts in this study are **not** evidence that $m_q$ is independent of dimension for all product models. [Synthetic runtime and nodes](results/exp15_synthetic_pkex_metal_precision/summary/synthetic_runtime_nodes.pdf) and [synthetic observed error](results/exp15_synthetic_pkex_metal_precision/summary/synthetic_accuracy.pdf) show the full sweep.

For text classification, we trained one fixed-parameter RBF SVC per dataset on full training splits and explained 20 fixed, complete held-out texts per dataset. The datasets were IMDB, Rotten Tomatoes (RT), SST-2, SMS spam, and emotion. Each input has 5,000 TF-IDF features, with unigrams/bigrams, lower-casing, English stop-word removal, `max_df=0.95`, and sublinear TF. `min_df=3` was used except for SMS, where `min_df=2` was needed to reach 5,000 terms. The model had `C=2` and `gamma="scale"`; emotion used the predicted class's one-vs-rest decision score. Held-out classifier accuracies were 88.03% (IMDB), 75.98% (RT), 78.10% (SST-2 validation), 98.21% (SMS), and 89.75% (emotion). They describe the models, not explainer accuracy. No review-length filter or truncation was applied: IMDB test reviews had 118–582 words (median 216). The explanation experiment used 20 inputs per corpus, not the manuscript's older 50-input, Optuna-tuned protocol.

All 100 text inputs completed for QuadraSHAP on both CPU and Metal at all four tolerances. At $\varepsilon=10^{-5}$, the median certified budget was five nodes for IMDB, RT, SMS, and emotion and six for SST-2. Median Metal time ranged from 0.708 seconds (SMS) to 11.50 seconds (SST-2), and the **worst** maximum-coordinate Metal difference per dataset from its own float64 reference ranged from $7.55\times10^{-7}$ to $6.15\times10^{-6}$. At $10^{-10}$, all CPU float64 results met the requested observed error against the stipulated reference; Metal did not, because its float32 error remained near $10^{-6}$. PKeX returned no vector within the 300-second cap for any of the 20 inputs in any of the five corpora: 0/100 completions. Its runtime is therefore **greater than 300 seconds per attempted text**, not “300 seconds,” and its observed error is unavailable. One PKeX pilot per dataset ran in isolation and also timed out; remaining independent jobs were scheduled concurrently. For the four non-IMDB corpora, see [text certificates and errors](results/exp11_text_classifiers_figures/text_certified_observed.pdf), [text runtime](results/exp11_text_classifiers_figures/text_runtime.pdf), and [text numerical stability](results/exp11_text_classifiers_figures/text_numerical_stability.pdf). IMDB has its own [accuracy/runtime](results/exp10_pkex_metal_imdb/figures/imdb_accuracy_runtime.pdf) and [stability](results/exp10_pkex_metal_imdb/figures/imdb_numerical_stability.pdf) figures.

### Complete synthetic summaries

Each QuadraSHAP row summarizes the same 50 inputs. “Bound” is the largest certified quadrature bound among them; “error” is the median or largest *observed* maximum-coordinate error. The certificate and observed error are different quantities.

| $d$ | $\varepsilon$ | Median $m_q$ | Largest bound | Median time (s) | Median error | Worst error |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | $10^{-3}$ | 4 | 8.55e-4 | 0.0048 | 1.58e-7 | 7.32e-7 |
| 50 | $10^{-5}$ | 5 | 7.95e-6 | 0.0063 | 9.03e-8 | 2.22e-7 |
| 50 | $10^{-10}$ | 7 | 9.53e-11 | 0.0080 | 7.66e-8 | 1.56e-7 |
| 50 | $10^{-16}$ | 10 | 9.71e-17 | 0.0096 | 6.90e-8 | 1.65e-7 |
| 500 | $10^{-3}$ | 4 | 4.19e-5 | 0.0157 | 2.38e-7 | 3.59e-7 |
| 500 | $10^{-5}$ | 5 | 6.18e-6 | 0.0183 | 1.93e-7 | 2.67e-7 |
| 500 | $10^{-10}$ | 7 | 5.37e-11 | 0.0178 | 1.85e-7 | 2.86e-7 |
| 500 | $10^{-16}$ | 9 | 3.56e-18 | 0.0188 | 1.75e-7 | 2.82e-7 |
| 1,000 | $10^{-3}$ | 4 | 1.39e-5 | 0.0256 | 2.87e-7 | 4.28e-7 |
| 1,000 | $10^{-5}$ | 4 | 9.84e-6 | 0.0241 | 2.72e-7 | 4.12e-7 |
| 1,000 | $10^{-10}$ | 6 | 9.94e-11 | 0.0249 | 2.44e-7 | 3.65e-7 |
| 1,000 | $10^{-16}$ | 9 | 6.96e-19 | 0.0285 | 2.27e-7 | 3.97e-7 |
| 2,000 | $10^{-3}$ | 3 | 9.86e-4 | 0.0392 | 2.88e-6 | 4.45e-6 |
| 2,000 | $10^{-5}$ | 4 | 6.29e-6 | 0.0359 | 3.35e-7 | 5.56e-7 |
| 2,000 | $10^{-10}$ | 6 | 6.33e-11 | 0.0394 | 3.05e-7 | 4.68e-7 |
| 2,000 | $10^{-16}$ | 8 | 9.91e-17 | 0.0430 | 2.96e-7 | 4.39e-7 |
| 5,000 | $10^{-3}$ | 3 | 4.27e-4 | 0.0976 | 1.57e-6 | 2.43e-6 |
| 5,000 | $10^{-5}$ | 4 | 2.41e-6 | 0.0945 | 4.05e-7 | 6.25e-7 |
| 5,000 | $10^{-10}$ | 6 | 1.96e-11 | 0.1048 | 3.92e-7 | 5.32e-7 |
| 5,000 | $10^{-16}$ | 8 | 4.12e-17 | 0.1153 | 3.85e-7 | 5.72e-7 |

| PKeX $d$ | Completed | Median time (s) | Median error | Worst error |
|---:|---:|---:|---:|---:|
| 50 | 50/50 | 0.1710 | 1.11e-15 | 2.66e-15 |
| 500 | 50/50 | 17.4986 | 7.49e-16 | 1.33e-15 |
| 1,000 | 50/50 | 79.3505 | 4.02e-16 | 7.77e-16 |
| 2,000 | Not attempted | — | — | — |
| 5,000 | Not attempted | — | — | — |

PKeX has zero *approximation* error in exact arithmetic, but its recorded nonzero observed difference includes floating-point effects and error in the reference. At $d=5,000$, $\varepsilon=10^{-16}$, Metal's median efficiency residual was $1.03\times10^{-5}$ despite its median per-coordinate error of $3.85\times10^{-7}$; summing 5,000 float32-influenced attributions can amplify residual error.

### Complete neutral-factor text summaries

Each row uses the same 20 held-out texts in that dataset. “Largest bound” is over those 20; CPU/Metal errors are the **worst** observed maximum-coordinate errors, whereas times are medians. The CPU $10^{-16}$ error is zero by definition because that run is the reference, not because exactness was proven.

| Dataset | $\varepsilon$ | Median $m_q$ | Largest bound | CPU time (s) | CPU worst error | Metal time (s) | Metal worst error |
|---|---:|---:|---:|---:|---:|---:|---:|
| IMDB | $10^{-3}$ | 4 | 9.89e-4 | 4.550 | 1.66e-7 | 3.182 | 2.16e-6 |
| IMDB | $10^{-5}$ | 5 | 3.02e-6 | 5.257 | 4.27e-10 | 3.847 | 2.18e-6 |
| IMDB | $10^{-10}$ | 7 | 9.05e-12 | 6.638 | 1.91e-13 | 4.395 | 2.18e-6 |
| IMDB | $10^{-16}$ | 9 | 8.59e-18 | 7.979 | 0 (reference) | 4.818 | 2.19e-6 |
| RT | $10^{-3}$ | 4 | 9.23e-4 | 2.360 | 1.19e-7 | 2.067 | 1.65e-6 |
| RT | $10^{-5}$ | 5 | 2.45e-6 | 2.746 | 1.32e-10 | 2.441 | 1.74e-6 |
| RT | $10^{-10}$ | 7 | 6.02e-12 | 3.443 | 4.23e-13 | 2.726 | 1.95e-6 |
| RT | $10^{-16}$ | 9 | 4.87e-18 | 4.137 | 0 (reference) | 2.995 | 1.23e-6 |
| SST-2 | $10^{-3}$ | 5 | 8.51e-4 | 16.220 | 6.28e-8 | 11.490 | 3.61e-6 |
| SST-2 | $10^{-5}$ | 6 | 1.18e-6 | 17.875 | 1.92e-11 | 11.502 | 6.15e-6 |
| SST-2 | $10^{-10}$ | 7 | 5.17e-11 | 20.109 | 2.85e-12 | 12.645 | 2.76e-6 |
| SST-2 | $10^{-16}$ | 9 | 5.23e-17 | 23.807 | 0 (reference) | 14.333 | 2.71e-6 |
| SMS | $10^{-3}$ | 4 | 1.02e-4 | 0.896 | 2.77e-7 | 0.624 | 5.66e-7 |
| SMS | $10^{-5}$ | 5 | 2.66e-7 | 1.043 | 3.86e-10 | 0.708 | 7.55e-7 |
| SMS | $10^{-10}$ | 7 | 6.13e-13 | 1.297 | 2.49e-14 | 0.799 | 8.22e-7 |
| SMS | $10^{-16}$ | 9 | 5.46e-19 | 1.542 | 0 (reference) | 0.874 | 1.03e-6 |
| Emotion | $10^{-3}$ | 4 | 7.81e-4 | 3.601 | 6.97e-8 | 2.541 | 1.35e-6 |
| Emotion | $10^{-5}$ | 5 | 1.86e-6 | 4.208 | 9.21e-11 | 2.905 | 1.42e-6 |
| Emotion | $10^{-10}$ | 7 | 4.01e-12 | 5.200 | 3.54e-13 | 3.261 | 1.58e-6 |
| Emotion | $10^{-16}$ | 9 | 3.37e-18 | 6.074 | 0 (reference) | 3.688 | 3.49e-6 |

PKeX completion counts were 0/20 for **each** of IMDB, RT, SST-2, SMS, and emotion, with a 300-second cap per input. All QuadraSHAP CPU and Metal rows were 20/20 complete. The worst CPU $10^{-16}$ efficiency residual across these datasets was $4.74\times10^{-11}$ (SST-2); even the reference is not a mathematical exact value.

### Suggested main-text figure captions

- **Synthetic runtime and node budget:** “RBF-KRR neutral-factor benchmark over 50 held-out inputs per feature count. Left: median wall-clock time with interquartile range; PKeX-Shapley was run only through $d=1000$. Right: median certified quadrature nodes at four requested tolerances. Model bandwidth scales as $\gamma=1/d$.” Figure file: [synthetic_runtime_nodes.pdf](results/exp15_synthetic_pkex_metal_precision/summary/synthetic_runtime_nodes.pdf).
- **Synthetic numerical agreement:** “Median maximum-coordinate difference from the separate float64 CPU $10^{-16}$ reference for the same neutral-factor game. PKeX is algebraically exact; QuadraSHAP's Metal scan is float32, so tighter quadrature requests eventually reach a rounding floor.” Figure file: [synthetic_accuracy.pdf](results/exp15_synthetic_pkex_metal_precision/summary/synthetic_accuracy.pdf).
- **Text benchmark:** The four-corpus [text_runtime.pdf](results/exp11_text_classifiers_figures/text_runtime.pdf) makes the 300-second PKeX cutoff visible. [text_certified_observed.pdf](results/exp11_text_classifiers_figures/text_certified_observed.pdf) separates exact-arithmetic certificates from CPU/Metal observed errors. IMDB is reported in separate PDFs linked above.

## Section 2 for the appendix: matched interventional comparison

### Paper-ready account

To compare with standard model-agnostic SHAP estimators, we changed the estimand to the finite-background interventional game

$$v_x^{B}(S)=\frac{1}{30}\sum_{b\in B}f(x_S,b_{\bar S}),$$

where $B$ is one **fixed, uniformly weighted, class-stratified sample of 30 transformed training texts per dataset** (15 per class in each binary corpus, five per class for emotion). The four included datasets are RT, SST-2, SMS spam, and emotion. For each dataset, QuadraSHAP, SHAP 0.51.0's KernelSHAP, and SHAP 0.51.0's SamplingSHAP use the **same fitted 5,000-feature RBF SVC, same score, same 20 held-out texts, and exactly the same stored 30-background matrix**. Both SHAP baselines use the identity link and receive `nsamples=1000`; KernelSHAP retains the installed default `l1_reg="num_features(10)"`. This is an equal *nominal sample parameter*, not an equal number of model evaluations. The 30-background set is reused across methods and texts within each dataset, but different datasets have different sets. PKeX has **no** background set in the neutral-factor section and is not part of this comparison.

The observed error for every method is $\max_j|\widehat\phi_j-\phi_j^{\rm ref}|$ against a separate float64 CPU $10^{-16}$ QuadraSHAP vector for **that same 30-background game and input**. The reference was checked against two additional quadrature nodes; the largest change across the 80 inputs was $4.87\times10^{-14}$. The reference remains numerical rather than an exact symbolic solution. QuadraSHAP Metal used the same four tolerances as in the main section. It completed all 80 inputs at each tolerance; both SHAP baselines completed all 80 inputs at their 1,000-sample setting, so the comparison contains 320 QuadraSHAP and 160 baseline vectors. Times exclude model training and reference computation; QuadraSHAP includes its per-input factor/certificate pass plus Metal evaluation, and SHAP baseline time includes explainer construction and explanation. The $10^{-16}$ Metal request is not a realized $10^{-16}$ attribution guarantee.

At $\varepsilon=10^{-5}$, QuadraSHAP's median maximum-coordinate error was $2.48\times10^{-8}$ (RT), $3.62\times10^{-8}$ (SST-2), $2.95\times10^{-8}$ (SMS), and $3.66\times10^{-8}$ (emotion), with median times 0.286, 0.675, 0.208, and 0.345 seconds respectively. KernelSHAP's median errors were 0.0966, 0.0475, 0.0696, and 0.0736, with times 9.07, 19.72, 3.22, and 10.81 seconds. SamplingSHAP's median errors were 0.129, 0.140, 0.147, and 0.122, with times 0.850, 1.63, 0.408, and 1.00 seconds. For these 80 paired inputs, KernelSHAP had smaller maximum-coordinate error than SamplingSHAP on 61/80 inputs, although its default ten-feature selection also changes the estimators' behavior. [The matched comparison figure](results/exp14_interventional_metal_precision/summary/interventional_comparison.pdf) displays each input as a faint dot, the median as a diamond, and the interquartile range as a short vertical stroke. [The precision-limit figure](results/exp14_interventional_metal_precision/summary/interventional_precision_limit.pdf) shows that the certified quadrature bound falls as requested $\varepsilon$ tightens, while Metal's observed error stays near $10^{-8}$ once floating-point effects dominate. The certificate does not bound background sampling error relative to a population interventional game.

### Complete matched interventional summaries

Every row represents 20 paired inputs. “Bound” is the largest certified exact-arithmetic quadrature bound among them. KernelSHAP and SamplingSHAP have no deterministic certified bound here, so their entries are **not applicable**, not zero. Errors are median and worst maximum-coordinate differences from the same reference; times are medians.

| Dataset | Method | Median $m_q$ | Median time (s) | Median error | Worst error | Largest bound |
|---|---|---:|---:|---:|---:|---:|
| RT | Quadra $10^{-3}$ | 4 | 0.271 | 2.62e-8 | 1.12e-7 | 9.85e-4 |
| RT | Quadra $10^{-5}$ | 5 | 0.286 | 2.48e-8 | 1.12e-7 | 5.34e-6 |
| RT | Quadra $10^{-10}$ | 7 | 0.314 | 2.41e-8 | 5.36e-8 | 2.66e-11 |
| RT | Quadra $10^{-16}$ | 9 | 0.359 | 2.85e-8 | 4.69e-8 | 4.59e-17 |
| RT | KernelSHAP | — | 9.07 | 0.0966 | 0.442 | N/A |
| RT | SamplingSHAP | — | 0.850 | 0.129 | 0.257 | N/A |
| SST-2 | Quadra $10^{-3}$ | 5 | 0.684 | 3.60e-8 | 7.44e-8 | 3.75e-5 |
| SST-2 | Quadra $10^{-5}$ | 6 | 0.675 | 3.62e-8 | 6.21e-8 | 1.03e-7 |
| SST-2 | Quadra $10^{-10}$ | 8 | 0.712 | 4.03e-8 | 9.16e-8 | 2.39e-11 |
| SST-2 | Quadra $10^{-16}$ | 10 | 0.757 | 3.88e-8 | 8.05e-8 | 5.29e-19 |
| SST-2 | KernelSHAP | — | 19.72 | 0.0475 | 0.311 | N/A |
| SST-2 | SamplingSHAP | — | 1.63 | 0.140 | 0.301 | N/A |
| SMS | Quadra $10^{-3}$ | 4 | 0.166 | 2.37e-8 | 6.85e-8 | 1.07e-4 |
| SMS | Quadra $10^{-5}$ | 5 | 0.208 | 2.95e-8 | 7.79e-8 | 3.75e-7 |
| SMS | Quadra $10^{-10}$ | 7 | 0.228 | 3.06e-8 | 7.29e-8 | 2.08e-12 |
| SMS | Quadra $10^{-16}$ | 9 | 0.249 | 2.65e-8 | 1.32e-7 | 3.68e-18 |
| SMS | KernelSHAP | — | 3.22 | 0.0696 | 0.268 | N/A |
| SMS | SamplingSHAP | — | 0.408 | 0.147 | 0.492 | N/A |
| Emotion | Quadra $10^{-3}$ | 4 | 0.340 | 4.04e-8 | 1.13e-7 | 9.73e-4 |
| Emotion | Quadra $10^{-5}$ | 5 | 0.345 | 3.66e-8 | 1.09e-7 | 3.51e-6 |
| Emotion | Quadra $10^{-10}$ | 7 | 0.386 | 4.37e-8 | 1.96e-7 | 1.49e-11 |
| Emotion | Quadra $10^{-16}$ | 9 | 0.405 | 3.10e-8 | 1.28e-7 | 2.29e-17 |
| Emotion | KernelSHAP | — | 10.81 | 0.0736 | 0.278 | N/A |
| Emotion | SamplingSHAP | — | 1.00 | 0.122 | 0.377 | N/A |

### Why SamplingSHAP is faster here, despite the same background

The two SHAP baselines have the **same 30 background rows**, but use them differently. KernelSHAP constructs coalition masks and, for each mask, averages model predictions over **all 30** background rows before fitting its weighted coalition regression. With `nsamples=1000`, the logged total was **30,032 model-evaluation rows** per text (including setup and diagnostic endpoint calls). SamplingSHAP estimates feature marginal contributions by sampling a permutation and **one randomly selected row from the same 30-row background pool** for each paired on/off evaluation. It performed **2,032 model-evaluation rows** per text, including the same kinds of ancillary calls, about 14.8 times fewer rows. Its final efficiency correction is lightweight; it does not perform KernelSHAP's large weighted coalition regression. The observed median time ratio, KernelSHAP divided by SamplingSHAP, was approximately 10.7 (RT), 12.1 (SST-2), 7.9 (SMS), and 10.8 (emotion). This is an empirical explanation for **this implementation and sample budget**, not a universal complexity ranking.

The logged calls make the contrast concrete: KernelSHAP used four model calls per text, one of them a large masked batch. SamplingSHAP made many smaller calls (107–379 across these datasets and inputs) but still evaluated far fewer rows in total. Both estimators target the same finite-background interventional value function in expectation. Their finite-sample numerical behavior differs: KernelSHAP's installed default `l1_reg="num_features(10)"` produced exactly ten nonzero attributions per text, while SamplingSHAP produced median nonzero counts of 37.5 (RT), 34.5 (SST-2), 49.5 (SMS), and 41.0 (emotion). Therefore the 1,000-sample setting does **not** equalize accuracy, model-evaluation count, or sparsity. KernelSHAP was more accurate on 61/80 inputs under the recorded defaults, and neither baseline supplies a deterministic error certificate at this budget. Both methods met the efficiency sum approximately (the largest absolute residual across the four datasets was $4.44\times10^{-16}$ for KernelSHAP and $3.61\times10^{-7}$ for SamplingSHAP), but efficiency alone does not establish coordinatewise accuracy.

Full-length IMDB was **not** silently shortened. At the same `nsamples=1000` and 30-background protocol, an IMDB pilot had 1,469 varying TF-IDF coordinates; SHAP 0.51.0's paired allocation needs at least 2,938 samples to give each such coordinate one pair, so SamplingSHAP did not return a vector. KernelSHAP's one completed pilot took 183.19 seconds. IMDB is therefore omitted from the fully paired four-dataset interventional accuracy figure; its failed pilot is neither counted as a completed baseline nor assigned an error value.

### Suggested appendix figure captions

- **Matched interventional comparison:** “Per-text maximum-coordinate attribution error relative to a separate float64 CPU reference (top) and time per explanation (bottom), on identical 30-background interventional games. Twenty held-out texts are shown for each of four 5,000-feature RBF SVCs. The four QuadraSHAP columns are requested *quadrature* tolerances; KernelSHAP and SamplingSHAP each use `nsamples=1000`. Faint dots are inputs, diamonds medians, and strokes interquartile ranges.” Figure file: [interventional_comparison.pdf](results/exp14_interventional_metal_precision/summary/interventional_comparison.pdf).
- **Precision limit:** “Median certified exact-arithmetic quadrature bound and median observed Metal maximum-coordinate error as the requested tolerance decreases. The bound decreases, while float32-influenced output reaches a numerical floor. Neither curve measures error from representing the background population by 30 texts.” Figure file: [interventional_precision_limit.pdf](results/exp14_interventional_metal_precision/summary/interventional_precision_limit.pdf).

## Integration and reproducibility notes for Claude

1. The current manuscript's product-kernel subsection and `tab:kernel_bench_combined` describe an **older, different benchmark**: 50 text explanations, Optuna-tuned SVCs, a fixed `max m_q=400`, and different synthetic times. The new 20-text fixed-SVC precision experiment and its medians **must replace or be explicitly distinguished from** those claims and that table, not be appended as though they were the same run. The new synthetic study follows the manuscript's $d$, sample count, informative-feature fraction, noise, and seed, but its 80/20 split, standardization, $\gamma=1/d$, and ridge strength are newly specified. Its PKeX results at 2,000/5,000 features are *not attempted*, not timeouts. The manuscript's shared dataset appendix currently says SMS has 4,457/1,115 records and `min_df=3` for all five corpora; this new experiment used a pinned 5,574-message SMS source, a 4,459/1,115 split, and `min_df=2` for SMS. Update that text if these new results are inserted.
2. The neutral-factor and interventional sections should not share one attribution-error table. They can share a discussion of product-kernel computation, but their baselines $v_x(\emptyset)$ and Shapley vectors differ. PKeX is directly comparable only in the neutral-factor section. KernelSHAP/SamplingSHAP are directly comparable only in the matched-background appendix section.
3. The synthetic benchmark varies **held-out inputs**, not training seeds or model fits: one fitted KRR per dimension, 50 inputs each. Text likewise has one fitted SVC per corpus and 20 inputs each. Do not claim uncertainty across retrained models. Interquartile ranges in the figures are across inputs. The text PKeX jobs were censored at 300 seconds; a cutoff line is not a measured completed runtime.
4. The manuscript already has BibTeX keys `pkex_shapley` and `shap`; use those for PKeX and SHAP. Figure paths in this report are relative to `experiments/`; copy the desired PDFs to the manuscript's figure folder and assign normal LaTeX labels. Suggested labels: `fig:synthetic-runtime-nodes`, `fig:synthetic-accuracy`, `fig:text-runtime`, `fig:text-certified-observed`, `fig:interventional-comparison`, and `fig:interventional-precision-limit`.
5. Raw vectors and diagnostics remain saved. Synthetic: `results/exp15_synthetic_pkex_metal_precision/summary/all_instances.csv`, `metal_summary.csv`, `pkex_summary.csv`, plus `d*/raw/`. Neutral text: `results/exp11_text_classifiers_figures/all_five_text_summary.csv`, `results/exp10_pkex_metal_imdb/records.csv`, and `results/exp11_pkex_metal_*/records.csv`. Interventional: `results/exp14_interventional_metal_precision/summary/all_instances.csv`, `method_summary.csv`, and per-dataset `raw/` vectors; SHAP background matrices and raw baseline vectors are under `results/exp12_shap_background_*/`. These retained records permit a figure-layout change without rerunning explainers.

