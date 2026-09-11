# Fresh-input benchmark: QuadraSHAP, Quadrature TreeSHAP, and TreeGrad-Shap

Run date: 2026-09-02

## Executive result

On this controlled synthetic suite, CUDA QuadraSHAP was faster than XGBoost's
GPU Quadrature TreeSHAP path in 17 of 18 configuration/batch cells.  The one
exception was the wide, shallow 32-tree ensemble at batch 64, where Quadrature
TreeSHAP took 1.554 ms and QuadraSHAP took 2.122 ms (Quadrature TreeSHAP was
1.37x faster).  The 16-feature version of that ensemble at batch 64 was close
to a tie: 1.428 ms versus 1.473 ms in QuadraSHAP's favor.

CUDA QuadraSHAP was also faster than the official TreeGrad-Shap implementation
in all 18 cells, by 9.85x to 3578.68x.  This is not an algorithm-only comparison:
the official TreeGrad implementation is CPU-only, written as a per-row
NumPy/Python routine, whereas the other two implementations ran on an NVIDIA
RTX A6000 GPU.  These numbers therefore compare the available implementations
as systems, not identical GPU kernels.

## Median end-to-end explanation time

Times are milliseconds for the complete batch.  Each value is the median
across three model seeds; the value for each seed is itself the median of five
synchronized repetitions after one warm-up.

| Model | Batch | QuadraSHAP CUDA exact | Quadrature TreeSHAP GPU | TreeGrad-Shap CPU | QTS / Quadra | TreeGrad / Quadra |
|---|---:|---:|---:|---:|---:|---:|
| 1 tree, 16 features, 256 leaves | 1 | 0.420 | 1.347 | 4.132 | 3.21x | 9.85x |
| 1 tree, 16 features, 256 leaves | 10 | 0.481 | 2.238 | 41.957 | 4.65x | 87.25x |
| 1 tree, 16 features, 256 leaves | 64 | 0.788 | 1.132 | 268.006 | 1.44x | 340.20x |
| 1 tree, 16 features, 4,096 leaves | 1 | 1.399 | 18.453 | 68.949 | 13.19x | 49.28x |
| 1 tree, 16 features, 4,096 leaves | 10 | 1.567 | 32.534 | 691.487 | 20.77x | 441.35x |
| 1 tree, 16 features, 4,096 leaves | 64 | 3.011 | 14.702 | 4,431.515 | 4.88x | 1,471.98x |
| 1 tree, 16 features, 16,384 leaves | 1 | 2.296 | 69.045 | 278.537 | 30.07x | 121.32x |
| 1 tree, 16 features, 16,384 leaves | 10 | 2.649 | 122.350 | 2,816.034 | 46.19x | 1,063.06x |
| 1 tree, 16 features, 16,384 leaves | 64 | 5.065 | 57.229 | 18,126.543 | 11.30x | 3,578.68x |
| 1 tree, 100 features, 4,096 leaves | 1 | 1.913 | 17.988 | 62.277 | 9.40x | 32.55x |
| 1 tree, 100 features, 4,096 leaves | 10 | 2.167 | 31.996 | 625.411 | 14.77x | 288.62x |
| 1 tree, 100 features, 4,096 leaves | 64 | 4.386 | 14.493 | 4,032.519 | 3.30x | 919.49x |
| 32 trees, 16 features, 128 leaves/tree | 1 | 0.724 | 1.601 | 67.059 | 2.21x | 92.66x |
| 32 trees, 16 features, 128 leaves/tree | 10 | 0.787 | 2.481 | 680.909 | 3.15x | 865.48x |
| 32 trees, 16 features, 128 leaves/tree | 64 | 1.428 | 1.473 | 4,366.915 | 1.03x | 3,058.71x |
| 32 trees, 100 features, 128 leaves/tree | 1 | 1.032 | 1.621 | 64.638 | 1.57x | 62.64x |
| 32 trees, 100 features, 128 leaves/tree | 10 | 1.122 | 2.509 | 651.761 | 2.24x | 580.93x |
| 32 trees, 100 features, 128 leaves/tree | 64 | 2.122 | **1.554** | 4,194.572 | **0.73x** | 1,976.55x |

The ratio columns are competitor time divided by QuadraSHAP time.  A ratio
greater than one favors QuadraSHAP; a ratio below one favors the competitor.

## Correctness check

An independent attribution-level validation used ten rows from every model and
seed (18 trained model variants):

| Check | Maximum observed error |
|---|---:|
| TreeGrad phi versus exact QuadraSHAP phi | 8.88e-15 |
| TreeGrad additivity | 2.13e-14 |
| Quadrature TreeSHAP phi versus exact QuadraSHAP phi | 1.30e-06 |
| Quadrature TreeSHAP mean absolute phi difference | 1.65e-07 |
| Quadrature TreeSHAP bias difference | 1.11e-07 |

TreeGrad-Shap and QuadraSHAP therefore compute the same path-dependent Shapley
attributions to floating-point precision under the aligned cover semantics.
The small XGBoost differences include conversion of the scikit-learn model to
XGBoost's float32 representation as well as float32 GPU arithmetic; the largest
observed prediction difference introduced by the bridge was 1.06e-06.

## Fairness protocol

- Exact repositories were pinned to QuadraSHAP revision
  `3aac1099ac26a59a42acca8f12fdfe6b320cafff` and TreeGrad revision
  `7ba2b5a12591fe0b65f0061b4788f2ee2ef1d12c`.
- QuadraSHAP and Quadrature TreeSHAP ran on one NVIDIA RTX A6000.  The official
  TreeGrad code ran on an AMD EPYC 9334 CPU because it has no GPU backend.
- The XGBoost 3.4.1 shared library was inspected to verify that this build
  contains the fixed-point `QuadratureShap...Kernel` CUDA implementation and
  identifies its exact contribution path as QuadratureTreeSHAP.  Thus the
  measured CUDA `pred_contribs=True` path is not the superseded GPUTreeSHAP
  implementation.
- Every timed call received a new host-array copy.  The XGBoost path also
  constructed a fresh `DMatrix` inside every timed call.  No SHAP outputs or
  input matrices were cached.
- Prepared explainers/models were reused; model training, model conversion,
  explainer construction, and first-use compilation were excluded.  This
  isolates repeated explanation latency while still charging input preparation.
- All timed worker processes were pinned to one numerical-library host thread.
- The same trained scikit-learn tree topology, thresholds, leaf values, and
  node covers were supplied to all methods.  Training used `bootstrap=False`,
  making TreeGrad's `n_node_samples` covers match the weighted covers consumed
  by the other implementations.
- For the 32-tree forests, the unmodified official TreeGrad routine was applied
  to every estimator and the results were averaged.  This is exact by Shapley
  linearity and matches random-forest prediction semantics.
- The suite used three seeds (42, 43, 44), batches 1, 10, and 64, one warm-up,
  and five timed repetitions.  All 162 seed-level method runs completed.

Software: Python 3.12.14, NumPy 2.5.2, CuPy 14.2.0, scikit-learn 1.9.0, and
XGBoost 3.4.1.

## Interpretation and limits

1. The strongest supported performance claim is: **on this hardware and
   controlled suite, the current CUDA QuadraSHAP implementation has lower
   fresh-input latency than the XGBoost GPU Quadrature TreeSHAP path in 17 of
   18 tested cells.**
2. QuadraSHAP's advantage grows strongly with the leaf count for the single-tree
   tests.  This is the most convincing part of the result: at 16,384 leaves it
   is 11.30x to 46.19x faster than the tested Quadrature TreeSHAP path.
3. Quadrature TreeSHAP catches up on shallow ensembles as batch size grows and
   wins the wide 100-feature ensemble at batch 64.  A paper should retain this
   crossover rather than summarize the result as an unconditional win.
4. The TreeGrad comparison establishes the performance of the official released
   implementation.  It does not establish a 9.85x--3578.68x intrinsic
   algorithmic advantage because the devices, batching strategies, languages,
   and implementation maturity differ.
5. The benchmark is synthetic and controlled.  Before using the numbers as the
   main empirical claim in a paper, repeat the protocol on representative real
   trained models, add larger batches where relevant, and report variability
   across machines or at least repeat sessions on the same machine.

## Artifacts

- `three_way_results.json`: complete metadata, protocol, per-seed timings, and
  aggregate results.
- `three_way_results.csv`: compact aggregate table.
- `validation.json`: per-model attribution agreement and global maxima.
- `three_way_benchmark.py`: reproducible timing harness.
- `validate_three_way.py`: independent attribution comparison.

The manuscript files were not modified.
