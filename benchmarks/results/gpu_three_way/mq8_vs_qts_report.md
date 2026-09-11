# Equal-eight-node GPU benchmark

Date: 2026-09-03

## Comparison

This focused benchmark compares:

1. QuadraSHAP CUDA with `m_q=8` forced explicitly.
2. XGBoost 3.4.1 GPU Quadrature-TreeSHAP, whose implementation fixes the quadrature count at eight.

Both methods therefore perform eight quadrature evaluations. Each method uses its native quadrature rule: QuadraSHAP uses its standard eight-point Gauss--Legendre rule on `[0,1]`, while XGBoost uses the endpoint-transformed eight-point rule shipped in Quadrature-TreeSHAP. The node counts are matched, but the abscissae and weights are not identical.

## Results

Times are median milliseconds for a complete public explanation call. Each table entry is the median of three independently trained model seeds. Each seed-level result uses five warmups followed by 25 synchronized timed repetitions.

`QTS / Quadra` is the ratio of median times. Values greater than one favor QuadraSHAP.

| Model | Batch | QuadraSHAP `m_q=8` (ms) | XGBoost QTS, 8 points (ms) | QTS / Quadra |
|---|---:|---:|---:|---:|
| 1 tree, 16 features, 256 leaves | 1 | 0.487 | 1.324 | 2.72x |
| 1 tree, 16 features, 256 leaves | 10 | 0.561 | 2.204 | 3.93x |
| 1 tree, 16 features, 256 leaves | 64 | 0.963 | 1.110 | 1.15x |
| 1 tree, 16 features, 4,096 leaves | 1 | 1.408 | 17.217 | 12.23x |
| 1 tree, 16 features, 4,096 leaves | 10 | 1.572 | 30.756 | 19.56x |
| 1 tree, 16 features, 4,096 leaves | 64 | 3.046 | 14.139 | 4.64x |
| 1 tree, 16 features, 16,384 leaves | 1 | 1.743 | 68.582 | 39.34x |
| 1 tree, 16 features, 16,384 leaves | 10 | 1.948 | 122.427 | 62.83x |
| 1 tree, 16 features, 16,384 leaves | 64 | 3.798 | 54.800 | 14.43x |
| 1 tree, 100 features, 4,096 leaves | 1 | 1.021 | 16.889 | 16.55x |
| 1 tree, 100 features, 4,096 leaves | 10 | 1.162 | 30.055 | 25.88x |
| 1 tree, 100 features, 4,096 leaves | 64 | 2.220 | 13.601 | 6.13x |
| 32 trees, 16 features, 128 leaves/tree | 1 | 1.087 | 1.577 | 1.45x |
| 32 trees, 16 features, 128 leaves/tree | 10 | 1.176 | 2.460 | 2.09x |
| 32 trees, 16 features, 128 leaves/tree | 64 | 2.243 | 1.444 | **0.64x** |
| 32 trees, 100 features, 128 leaves/tree | 1 | 1.167 | 1.593 | 1.36x |
| 32 trees, 100 features, 128 leaves/tree | 10 | 1.266 | 2.490 | 1.97x |
| 32 trees, 100 features, 128 leaves/tree | 64 | 2.426 | 1.525 | **0.63x** |

Summary:

- QuadraSHAP wins 16 of 18 model/batch cells.
- Geometric mean of the 18 QTS/Quadra ratios: **5.03x**.
- Median QTS/Quadra ratio: **4.29x**.
- Largest QuadraSHAP advantage: **62.83x**, for the 16,384-leaf single tree at batch 10.
- XGBoost QTS is **1.55x** faster for the 16-feature forest at batch 64 and **1.59x** faster for the 100-feature forest at batch 64.

## Protocol

- GPU: NVIDIA RTX A6000, compute capability 8.6.
- QuadraSHAP revision: `3aac1099ac26a59a42acca8f12fdfe6b320cafff`.
- XGBoost: 3.4.1.
- Six model configurations, three seeds (`42`, `43`, `44`), and batches `1`, `10`, and `64`.
- The same fitted scikit-learn trees, values, covers, and input rows were supplied to both implementations.
- Every timed call received a fresh contiguous host-array copy.
- XGBoost constructed a fresh `DMatrix` inside every timed primary call.
- Explanations were never cached.
- Explainer/model construction and initial CUDA compilation were outside the timed region.
- GPU synchronization followed every call.
- Numeric CPU libraries were restricted to one thread.
- Trees were capped at depth 24, and every requested leaf count was reached.

This is an end-to-end public-API comparison. It equalizes the quadrature count, but it does not remove implementation-level lifecycle work that the APIs perform internally. In particular, XGBoost's public QTS path prepares and transfers its compressed GPU tree representation during every explanation call, whereas QuadraSHAP retains prepared model data on the device.

## Numerical checks

- Worst timed-run additivity residual for QuadraSHAP `m_q=8`: `7.07e-13`.
- Worst timed-run additivity residual for XGBoost QTS: `4.08e-6`.
- In the separate direct validator, QuadraSHAP `m_q=8` differed from its structure-dependent exact result by at most `4.10e-14` over this suite.
- XGBoost QTS differed from the same exact reference by at most `1.79e-6`.

Thus the speed comparison is not explained by an observed loss of numerical accuracy from forcing QuadraSHAP to eight nodes on these models.

## Interpretation

Matching the quadrature count removes the clearest methodological workload difference. The remaining gap is primarily attributable to the GPU execution and public-call implementations: persistent preprocessing, levelwise compact state propagation, and atomics-free feature reduction in QuadraSHAP versus per-call model preparation, DFS-oriented row/tree scheduling, and atomic output accumulation in XGBoost QTS.

The two batch-64 forest wins for XGBoost show that its row/tree scheduling can be more effective when many small trees and enough rows are available. The result is therefore workload-dependent, not universal dominance.

No manuscript or paper file was modified.
