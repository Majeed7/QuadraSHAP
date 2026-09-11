# GPU benchmark: QuadraSHAP vs Quadrature-TreeSHAP vs GPUTreeSHAP

Date: 2026-09-02

## Bottom line

On this benchmark, the CUDA implementation in the QuadraSHAP repository is:

- faster than XGBoost 3.4.1's GPU Quadrature-TreeSHAP in 17 of 18 model/batch combinations;
- faster than SHAP's GPUTreeSHAP in all 18 combinations;
- 5.46x faster than Quadrature-TreeSHAP and 5.13x faster than GPUTreeSHAP by geometric mean of the 18 relative timings.

The Quadrature-TreeSHAP comparison does **not** establish a better asymptotic algorithm. Both methods use the same quadrature family. Most of QuadraSHAP's observed advantage comes from GPU organization and explanation-call lifecycle, with an additional genuine constant-factor benefit from its adaptive exact quadrature order on shallow trees.

The comparison with GPUTreeSHAP is different: the advantage is both algorithmic and implementation-related. The quadrature formulation avoids the quadratic-in-path-depth work of the original TreeSHAP recurrence, while QuadraSHAP also avoids repeated public-call preprocessing performed by SHAP's released GPU wrapper.

## Primary timing results

Times are median milliseconds for one complete public explanation call. For every model, each displayed value is the median of the three seed-level medians; every seed-level median contains nine synchronized repetitions after three warmups.

`QTS / ours` and `GPU TreeSHAP / ours` are relative times. A value above 1 means QuadraSHAP is faster.

| Model | Batch | QuadraSHAP CUDA exact (ms) | XGBoost Quadrature-TreeSHAP (ms) | SHAP GPUTreeSHAP (ms) | QTS / ours | GPU TreeSHAP / ours |
|---|---:|---:|---:|---:|---:|---:|
| 1 tree, 16 features, 256 leaves | 1 | 0.410 | 1.327 | 2.407 | 3.23x | 5.87x |
| 1 tree, 16 features, 256 leaves | 10 | 0.473 | 2.211 | 2.450 | 4.67x | 5.18x |
| 1 tree, 16 features, 256 leaves | 64 | 0.780 | 1.107 | 2.557 | 1.42x | 3.28x |
| 1 tree, 16 features, 4,096 leaves | 1 | 1.410 | 18.250 | 7.138 | 12.95x | 5.06x |
| 1 tree, 16 features, 4,096 leaves | 10 | 1.564 | 30.620 | 7.259 | 19.58x | 4.64x |
| 1 tree, 16 features, 4,096 leaves | 64 | 3.055 | 14.653 | 7.563 | 4.80x | 2.48x |
| 1 tree, 16 features, 16,384 leaves | 1 | 1.742 | 68.251 | 22.035 | 39.19x | 12.65x |
| 1 tree, 16 features, 16,384 leaves | 10 | 1.952 | 121.789 | 22.307 | 62.38x | 11.43x |
| 1 tree, 16 features, 16,384 leaves | 64 | 3.772 | 54.777 | 23.780 | 14.52x | 6.30x |
| 1 tree, 100 features, 4,096 leaves | 1 | 1.395 | 17.898 | 7.527 | 12.83x | 5.40x |
| 1 tree, 100 features, 4,096 leaves | 10 | 1.601 | 30.857 | 7.641 | 19.27x | 4.77x |
| 1 tree, 100 features, 4,096 leaves | 64 | 3.156 | 14.436 | 8.008 | 4.57x | 2.54x |
| 32 trees, 16 features, 128 leaves/tree | 1 | 0.715 | 1.579 | 5.343 | 2.21x | 7.47x |
| 32 trees, 16 features, 128 leaves/tree | 10 | 0.783 | 2.461 | 5.450 | 3.14x | 6.96x |
| 32 trees, 16 features, 128 leaves/tree | 64 | 1.426 | 1.460 | 5.686 | 1.02x | 3.99x |
| 32 trees, 100 features, 128 leaves/tree | 1 | 1.028 | 1.597 | 5.847 | 1.55x | 5.69x |
| 32 trees, 100 features, 128 leaves/tree | 10 | 1.121 | 2.502 | 5.890 | 2.23x | 5.26x |
| 32 trees, 100 features, 128 leaves/tree | 64 | 2.119 | 1.544 | 6.102 | **0.73x** | 2.88x |

The last row is the sole Quadrature-TreeSHAP win: XGBoost is 1.37x faster than QuadraSHAP there. The 16-feature forest at batch 64 is effectively a tie. XGBoost Quadrature-TreeSHAP and SHAP GPUTreeSHAP each win 9 of their 18 pairwise comparisons, so neither is uniformly faster than the other in this suite.

## What was compared

The primary methods were:

1. **QuadraSHAP CUDA exact:** `TreeExplainer(model, tree_solver="quadrature_tree", device="cuda")` from the user's repository at revision `3aac1099ac26a59a42acca8f12fdfe6b320cafff`.
2. **Quadrature-TreeSHAP:** XGBoost 3.4.1's CUDA `Booster.predict(..., pred_contribs=True)`. The installed binary contains the new Quadrature-TreeSHAP kernels, and the matching XGBoost 3.4.1 source revision inspected was `6fe8c547bdd21c73e4555d85b087d9260595d30d`.
3. **GPUTreeSHAP:** `shap.GPUTreeExplainer(..., feature_perturbation="tree_path_dependent")` built from the official SHAP v0.52.0 commit `8461059bd4e5db2d5d401472ef871c5d411984fe`.

The SHAP source needed only build-compatibility changes for CUDA 13.3 and the RTX A6000: target compute capability 8.6, use C++17, and explicitly include `thrust/pair.h`. No GPUTreeSHAP algorithm code was changed. The exact patch is included with the results and applies cleanly to the pinned source revision. The locally installed package reports `0.52.1.dev0` because the source tree is dirty after this patch; the pinned algorithm source is v0.52.0.

All three methods received the same fitted `RandomForestRegressor` trees, including topology, split data, leaf values, and covers. The XGBoost bridge's prediction discrepancy was at most `9.35e-7`. Models had either 16 or 100 features; one tree with 256, 4,096, or 16,384 leaves; or 32 trees with 128 leaves each. There were three independent seeds and batches of 1, 10, and 64 rows.

Trees were capped at depth 24 because released SHAP GPUTreeSHAP uses a path representation that cannot safely cover the deeper original models. Every requested leaf count was reached after applying the cap. This gives all three methods complete coverage of the same 18 cells.

## Fairness and the meaning of caching

The primary benchmark did not cache explanations or reuse an input container:

- every timed call received a new contiguous host-array copy;
- XGBoost constructed a new `DMatrix` inside every primary timed call;
- prepared model/explainer objects were reused, and initial construction/JIT compilation was outside timing;
- GPU synchronization was performed around each measurement;
- worker processes were restricted to one numeric CPU thread.

The separately labeled `xgb_qts_cached_dmatrix` result means only that the same already-constructed XGBoost `DMatrix` was reused. It does **not** mean an explanation result was cached. Reusing the `DMatrix` improved QTS time by about 4.3% geometrically across the suite and did not remove the large gap on the large single trees. Therefore input-container construction is not the main explanation for the result.

This is an end-to-end public-call benchmark, not a kernel-only benchmark. Work that the released public method repeats on every explanation call is intentionally included.

## Why QuadraSHAP is faster than Quadrature-TreeSHAP here

### 1. The main difference is implementation-level algorithm engineering

XGBoost 3.4.1 calls `PrepareGpuQuadratureModel(...)` from inside every `ShapValues(...)` call. That preparation scans the trees, calculates root means, assigns trees to depth buckets, constructs compressed host arrays, and transfers the compressed data to device vectors. Reusing a `DMatrix` does not cache this prepared model.

QuadraSHAP prepares and uploads its model metadata when the explainer is built and keeps it resident on the GPU. The timed explanation call therefore starts from an already-prepared device representation.

Their GPU execution organizations also differ:

- XGBoost assigns eight warp lanes to its eight quadrature points, performs a DFS for each row/tree task, and accumulates edge contributions with `atomicAdd`.
- QuadraSHAP's small-batch compact path computes descendant state bottom-up once, then performs a depth/feature-ordered top-down contraction. Parents are grouped by feature so there is one writer for each depth/feature/sample result, avoiding atomics.
- QuadraSHAP stores state for internal nodes only on this path and uses persistent precomputed factors and execution ordering.

These are substantial algorithm-engineering differences, but they do not change the fundamental quadrature complexity class.

### 2. Adaptive quadrature provides a real constant-factor saving on shallow trees

XGBoost fixes the quadrature rule at eight points. QuadraSHAP chooses the exact order from each tree's effective path degree. On the shallow 16-feature forests, the exact solver used five points in the validation models, and forcing QuadraSHAP to eight points increased median time from 0.715 to 1.093 ms at batch 1 and from 1.426 to 2.257 ms at batch 64.

This is a genuine reduction in arithmetic work. It helps explain the shallow-forest results, but it cannot explain the full benchmark gap.

### 3. The controls show that implementation dominates the large-tree gap

For the 16-feature 4,096- and 16,384-leaf single trees, QuadraSHAP's exact order was already eight, and its exact and forced-eight timings were essentially identical. It was nevertheless 4.80x to 62.38x faster than XGBoost QTS.

For the 100-feature, 4,096-leaf single tree, the exact solver used more work than the forced-eight diagnostic: for example, 1.395 versus 1.024 ms at batch 1. Even then, exact QuadraSHAP was 12.83x faster than QTS. This is strong evidence that the large-tree advantage is not caused merely by using fewer quadrature nodes.

The one crossover at 32 trees, 100 features, and batch 64 also matters: XGBoost's warp-per-tree/row mapping amortizes well enough there to be 1.37x faster. The result is therefore shape-dependent, not a universal dominance claim.

### 4. There is no new asymptotic superiority over exact Quadrature-TreeSHAP

Gauss-Legendre quadrature is exactly integrated when the rule is sufficiently high for the polynomial degree. An exact first-order quadrature TreeSHAP method therefore needs an order tied to effective path degree, commonly written `ceil(d/2)`. If both implementations use the same exact order, they share the same leading theoretical dependence.

XGBoost's fixed-eight implementation treats eight nodes as a practical precision choice. That gives constant quadrature work, but it is not an all-depth exactness theorem. QuadraSHAP's adaptive rule preserves exactness by increasing the order where required. Thus the defensible statement is:

> QuadraSHAP is faster here mainly because of model-resident preprocessing, a compact levelwise recurrence, an atomics-free reduction, and better small-batch organization; adaptive quadrature adds a real constant-factor advantage on shallow trees. It is not asymptotically better than an exact implementation of the same quadrature formulation.

## Why QuadraSHAP is faster than SHAP GPUTreeSHAP here

This comparison has both theoretical and implementation components.

The original TreeSHAP/GPUTreeSHAP recurrence retains the quadratic path-depth term. The Quadrature-TreeSHAP paper describes first-order TreeSHAP/GPUTreeSHAP work as `O(M T L D^2)` and exact quadrature work as `O(M T N ceil(d/2))`, with fixed quadrature becoming practical `O(M T N)`. The symbols denote rows, trees, leaves/nodes, and path depth according to that paper's notation. Consequently the quadrature formulation removes an important depth factor.

The released SHAP wrapper also repeats work on every call. Its path-dependent GPU function recursively expands root-to-leaf paths on the host, copies the input into new device vectors, allocates output vectors, copies the paths to the GPU, deduplicates and bin-packs them, computes SHAP values, transposes the result, and copies it back. QuadraSHAP instead retains a prepared tree representation and precomputed execution metadata across calls.

In this suite, GPUTreeSHAP was 2.48x to 12.65x slower than QuadraSHAP and never won a cell.

## Numerical agreement

An untimed validation used 10 rows from every seed/model pair and compared attributions directly with QuadraSHAP CUDA exact:

| Check | Maximum observed error |
|---|---:|
| QuadraSHAP forced-eight vs exact, feature attribution | `4.10e-14` |
| XGBoost QTS vs QuadraSHAP exact, feature attribution | `1.79e-6` |
| SHAP GPUTreeSHAP vs QuadraSHAP exact, feature attribution | `7.26e-6` |
| QuadraSHAP exact additivity | `1.42e-14` |
| SHAP GPUTreeSHAP additivity | `6.75e-5` |

Across the timed runs, worst additivity errors were `1.78e-14` for QuadraSHAP exact, `4.08e-6` for XGBoost QTS, and `9.96e-5` for SHAP GPUTreeSHAP. These results show agreement at the expected single/mixed-precision scale for the two external GPU implementations. They are validation of this suite, not a general numerical-accuracy theorem.

## Important limitations

- One RTX A6000 and one software stack were tested.
- The models are synthetic scikit-learn regression forests, not a representative application corpus.
- Batch sizes stop at 64. QuadraSHAP's treelet path starts at 1,536 rows, so none of these timings exercises it; all reported QuadraSHAP timings use the compact small-batch path.
- The maximum depth is 24 to accommodate SHAP GPUTreeSHAP.
- Timings compare released public calls, not isolated kernels with identically preprocessed device inputs.
- Three seeds and nine repeats per seed are enough to expose a large engineering gap, but not enough for a broad hardware-generalization claim.

## Artifacts

- `gpu_three_way_benchmark.py`: benchmark harness
- `gpu_three_way_results.json`: full metadata and every raw repetition
- `gpu_three_way_results.csv`: flattened timing summaries
- `validate_gpu_three_way.py`: untimed direct-agreement validator
- `validation.json`: per-model correctness results
- `shap-v0.52.0-sm86.patch`: build-only compatibility patch for the pinned SHAP source

## Primary references

- Quadrature-TreeSHAP paper: https://arxiv.org/html/2605.04497
- GPUTreeSHAP paper: https://arxiv.org/abs/2010.13972
- SHAP source: https://github.com/shap/shap/tree/8461059bd4e5db2d5d401472ef871c5d411984fe
- XGBoost GPU documentation: https://github.com/dmlc/xgboost/blob/master/doc/gpu/index.rst

No paper or manuscript file was modified during this benchmark.
