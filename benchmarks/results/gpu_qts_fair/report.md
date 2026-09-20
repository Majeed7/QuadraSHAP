# Fresh-input CUDA benchmark: QuadraSHAP vs. QuadratureTreeSHAP

Run date: 2026-09-02

## Implementations

- **QuadraSHAP:** exact repository `https://github.com/Majeed7/QuadraSHAP`,
  revision `3aac1099ac26a59a42acca8f12fdfe6b320cafff`.
- **QuadratureTreeSHAP:** XGBoost 3.4.1 GPU
  `Booster.predict(pred_contribs=True)`.

The run used an NVIDIA RTX A6000 (48 GiB, compute capability 8.6), CuPy
14.2.0, Python 3.12.14, NumPy 2.5.2, and scikit-learn 1.9.0.

## Fair fresh-input protocol

- Both implementations receive the same fitted scikit-learn forest topology,
  path covers, and test rows.
- Every timed call starts with a new contiguous host copy of the input batch.
- QuadraSHAP's public `shap_values` call performs its normal host-to-device
  handling on that fresh input.
- XGBoost constructs a new `DMatrix` inside every timed call. No DMatrix is
  reused and no prediction/explanation result is cached.
- Both sides reuse a prepared model. Model conversion/construction, CUDA JIT,
  and four warm-up calls are excluded symmetrically.
- Each cell uses three independently generated tree seeds. Each per-seed
  result is the median of 15 synchronized repetitions; the reported value is
  the median across seeds.
- QuadraSHAP uses float64. The bridge required by XGBoost stores thresholds
  and leaf values as float32, while preserving topology and covers.

This is a fair public-API, fresh-input comparison. It is intentionally not a
pure CUDA-kernel comparison: constructing XGBoost's required `DMatrix` is part
of its measured fresh-input path.

## Results

All values are milliseconds per complete explanation call. A speedup above
1 means QuadraSHAP is faster.

| Model shape | Batch | Quadra exact | Quadra m_q=8 | QTS fresh DMatrix | Exact speedup | m_q=8 speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1 tree, 16 features, 256 leaves | 1 | 0.391 | 0.461 | 21.126 | 53.97x | 45.78x |
| 1 tree, 16 features, 256 leaves | 64 | 0.751 | 0.926 | 26.700 | 35.56x | 28.83x |
| 1 tree, 16 features, 256 leaves | 1024 | 2.486 | 3.163 | 31.640 | 12.73x | 10.00x |
| 1 tree, 16 features, 4,096 leaves | 1 | 1.526 | 1.524 | 33.506 | 21.95x | 21.98x |
| 1 tree, 16 features, 4,096 leaves | 64 | 3.384 | 3.385 | 35.954 | 10.62x | 10.62x |
| 1 tree, 16 features, 4,096 leaves | 1024 | 9.530 | 9.248 | 41.678 | 4.37x | 4.51x |
| 1 tree, 16 features, 16,384 leaves | 1 | 2.246 | 2.248 | 84.879 | 37.79x | 37.76x |
| 1 tree, 16 features, 16,384 leaves | 64 | 5.033 | 5.029 | 78.633 | 15.62x | 15.64x |
| 1 tree, 16 features, 16,384 leaves | 1024 | 18.697 | 18.724 | 82.096 | 4.39x | 4.38x |
| 1 tree, 100 features, 4,096 leaves | 1 | 1.837 | 1.189 | 33.185 | 18.07x | 27.91x |
| 1 tree, 100 features, 4,096 leaves | 64 | 4.227 | 2.575 | 37.648 | 8.91x | 14.62x |
| 1 tree, 100 features, 4,096 leaves | 1024 | 16.605 | 10.078 | 42.402 | 2.55x | 4.21x |
| 32 trees, 16 features, 128 leaves/tree | 1 | 0.746 | 1.081 | 21.121 | 28.31x | 19.54x |
| 32 trees, 16 features, 128 leaves/tree | 64 | 1.520 | 2.271 | 26.746 | 17.59x | 11.78x |
| 32 trees, 16 features, 128 leaves/tree | 1024 | 4.405 | 6.572 | 32.080 | 7.28x | 4.88x |
| 32 trees, 100 features, 128 leaves/tree | 1 | 1.091 | 1.187 | 21.119 | 19.36x | 17.79x |
| 32 trees, 100 features, 128 leaves/tree | 64 | 2.277 | 2.463 | 26.704 | 11.73x | 10.84x |
| 32 trees, 100 features, 128 leaves/tree | 1024 | 7.850 | 8.304 | 32.886 | 4.19x | 3.96x |

## Numerical checks

| Check | Worst observed value |
|---|---:|
| QuadraSHAP adaptive-exact additivity residual | 2.14e-14 |
| QuadraSHAP m_q=8 additivity residual | 1.87e-11 |
| QuadraSHAP m_q=8 attribution difference from adaptive exact | 6.00e-13 |
| XGBoost QTS additivity residual | 7.87e-6 |
| XGBoost QTS attribution difference from QuadraSHAP exact | 4.16e-6 |
| XGBoost bridge prediction difference from the sklearn model | 1.55e-6 |

The XGBoost differences combine its fixed eight-point rule with float32 model
conversion and output arithmetic, so they are not pure quadrature errors.

## Interpretation

1. With fresh inputs and no DMatrix reuse, QuadraSHAP wins every tested cell.
2. Adaptive-exact QuadraSHAP is 2.55x--53.97x faster; matched-eight-point
   QuadraSHAP is 3.96x--45.78x faster.
3. The advantage remains at batch 1024, where input-construction overhead is
   better amortized: 2.55x--12.73x for adaptive exact QuadraSHAP.
4. The largest single-tree case shows a 4.39x--37.79x advantage over the
   tested batch range.
5. Adaptive per-tree quadrature saves work on shallow forests. For the
   100-feature tree it instead spends extra work to retain the exactness
   guarantee, but remains faster than fresh-input QTS.

These measurements support a strong fresh-input/end-to-end claim. They must
not be presented as proof that QuadraSHAP's CUDA kernels alone are up to 54x
faster, because XGBoost's required fresh `DMatrix` construction is included.
The cached-DMatrix experiment is the complementary kernel-oriented view, but
it is not part of this report's headline comparison.
