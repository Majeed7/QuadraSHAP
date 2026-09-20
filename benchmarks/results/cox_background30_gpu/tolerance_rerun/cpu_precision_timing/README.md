# CPU/GPU precision and latency comparison

CPU FP32 and FP64 measurements are new. All GPU results are reused from the matched saved experiments; no GPU timing was rerun. Model, inputs, backgrounds, patients, node counts, numerical sources and package versions are matched. CPU exact-degree experiments were not measured and are blank below.

**The error reference has changed from the earlier main table.** All Delta columns here compare with the same independent 64-node FP64 reference, not GPU degree-exact output. The reference has analytical quadrature bounds below 1e-10 and a 96-node cross-check for the first patient in each cohort; it is not a full degree-exact FP64 computation.

| Dataset | Method | Nodes | C_max | GPU FP32 time (s) | Delta_GPU | GPU pass | CPU FP64 time (s) | Delta_CPU | CPU pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSE24080 | exact | 27,338 | 0 | 1.172 ± 0.035 | 0.00187 | --- | — | — | — |
| GSE24080 | 1e-1 | 44–53 | 0.0769 | 0.090 ± 0.020 | 0.00203 | 5/5 | 0.156 ± 0.013 | 1.21e-11 | 5/5 |
| GSE24080 | 1e-2 | 45–54 | 0.00709 | 0.085 ± 0.007 | 0.00215 | 5/5 | 0.164 ± 0.012 | 1.28e-11 | 5/5 |
| GSE24080 | 1e-3 | 46–54 | 0.000882 | 0.086 ± 0.006 | 0.00215 | 1/5 | 0.162 ± 0.010 | 1.28e-11 | 5/5 |
| TCGA LGG | exact | 198,033 | 0 | 66.377 ± 1.196 | 0.0332 | --- | — | — | — |
| TCGA LGG | 1e-1 | 37–49 | 0.0701 | 0.320 ± 0.028 | 0.0305 | 5/5 | 0.992 ± 0.076 | 1.4e-10 | 5/5 |
| TCGA LGG | 1e-2 | 38–50 | 0.00651 | 0.312 ± 0.006 | 0.0352 | 3/5 | 1.009 ± 0.097 | 1.31e-10 | 5/5 |
| TCGA LGG | 1e-3 | 39–51 | 0.00058 | 0.310 ± 0.011 | 0.0391 | 0/5 | 1.035 ± 0.100 | 1.59e-10 | 5/5 |

Times are mean ± sample standard deviation across five distinct patients, with one measured explanation per patient. Errors are maxima over all five patients and all features. Pass counts use the unrounded per-patient maximum error. All analytical quadrature bounds satisfy their requested tolerance, independently of the measured floating-point error.

## CPU FP32 control

| Dataset | Tolerance | CPU FP32 time (s) | Delta_CPU32 | Pass | Max CPU32–GPU32 gap |
|---|---|---:|---:|---:|---:|
| GSE24080 | 1e-1 | 0.143 ± 0.091 | 0.00199 | 5/5 | 0.000179 |
| GSE24080 | 1e-2 | 0.103 ± 0.006 | 0.00215 | 5/5 | 9.78e-05 |
| GSE24080 | 1e-3 | 0.102 ± 0.004 | 0.00215 | 1/5 | 9.9e-05 |
| TCGA LGG | 1e-1 | 0.619 ± 0.033 | 0.0306 | 5/5 | 0.000262 |
| TCGA LGG | 1e-2 | 0.629 ± 0.043 | 0.0356 | 3/5 | 0.000717 |
| TCGA LGG | 1e-3 | 0.643 ± 0.048 | 0.0398 | 0/5 | 0.000652 |

Against the common reference: GPU FP32 passes 19/30, CPU FP32 passes 19/30, and CPU FP64 passes 30/30. Their worst approximate errors are 0.0391, 0.0398, and 1.59e-10, respectively.

CPU FP32 versus CPU FP64 holds hardware and formula fixed while changing work precision. CPU FP32 versus GPU FP32 exposes backend-dependent numerical effects as well. The controls are empirical evidence about numerical precision, not a floating-point error certificate. Both FP32 implementations accumulate blocks on the host in FP64.

## Timing scope and interpretation

Each distinct dataset/node-count configuration receives one untimed warm-up. Measured API wall times include automatic node selection, transfers, and host accumulation; they exclude model fitting, compilation/warm-up, and cached rule construction. Memory planning uses 8 GiB with one background row per block. GPU results and CPU results were recorded in separate runs, so ratios are not contemporaneous paired benchmarks. The CPU FP64/GPU FP32 ratio additionally changes precision and must not be described as a matched-precision GPU speedup. The summary CSV also provides the same-work-precision CPU FP32/GPU FP32 ratio.

The quadrature certificate applies to the fixed fitted relative-hazard model and 30-row empirical background game. It excludes rounding, fitted-model uncertainty, and error from approximating a population background. CPU FP64 accuracy must not be assigned to GPU FP32 timings. The five-patient standard deviations do not establish repeated-run variability or statistical significance.

## Outputs

- `cpu_gpu_table.tex`: main comparison with a common error reference.
- `cpu_fp32_control_table.tex`: additional same-device precision control.
- `cpu_gpu_analysis.tex`: interpretation and reproducibility details.
- `table.csv`: full-precision aggregate values, bounds, pass counts, and latency ratios.
- `preview.tex`: standalone LaTeX preview.
