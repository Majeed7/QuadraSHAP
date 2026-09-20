# Same-backend degree-exact Cox comparison

The requested reference convention is used throughout the main table: **Delta_GPU compares GPU FP32 approximation only with GPU FP32 degree-exact output; Delta_CPU compares CPU FP64 approximation only with CPU FP64 degree-exact output.** No smaller-node proxy is used. Exact-row discrepancies are zero by construction, not evidence of numerically exact values.

All ten full CPU degree-exact runs are complete. GPU exact/approximate and CPU approximate timings reuse the matched saved runs. Every entry aggregates one measured call for each of five distinct patients, using the same 30 training-background samples.

| Dataset | Method | Nodes | C_max | GPU FP32 time (s) | Delta_GPU | GPU pass | CPU FP64 time (s) | Delta_CPU | CPU pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSE24080 | exact | 27,338 | 0 | 1.172 ± 0.035 | 0 | --- | 69.277 ± 2.833 | 0 | --- |
| GSE24080 | 1e-1 | 44–53 | 0.0769 | 0.090 ± 0.020 | 0.00034 | 5/5 | 0.156 ± 0.013 | 2.36e-09 | 5/5 |
| GSE24080 | 1e-2 | 45–54 | 0.00709 | 0.085 ± 0.007 | 0.000371 | 5/5 | 0.164 ± 0.012 | 2.36e-09 | 5/5 |
| GSE24080 | 1e-3 | 46–54 | 0.000882 | 0.086 ± 0.006 | 0.00131 | 4/5 | 0.162 ± 0.010 | 2.36e-09 | 5/5 |
| TCGA LGG | exact | 198,033 | 0 | 66.377 ± 1.196 | 0 | --- | 3656.562 ± 12.642 | 0 | --- |
| TCGA LGG | 1e-1 | 37–49 | 0.0701 | 0.320 ± 0.028 | 0.00447 | 5/5 | 0.992 ± 0.076 | 2.44e-07 | 5/5 |
| TCGA LGG | 1e-2 | 38–50 | 0.00651 | 0.312 ± 0.006 | 0.0038 | 5/5 | 1.009 ± 0.097 | 2.44e-07 | 5/5 |
| TCGA LGG | 1e-3 | 39–51 | 0.00058 | 0.310 ± 0.011 | 0.00595 | 2/5 | 1.035 ± 0.100 | 2.44e-07 | 5/5 |

All analytical quadrature bounds are below the requested tolerance. Against their respective degree-exact references, GPU FP32 passes 26/30 cases and CPU FP64 passes 30/30; the worst CPU discrepancy is 2.44e-07. Pass counts use unrounded per-patient values.

## Interpretation

C_max is the maximum analytical absolute quadrature bound across patients, assuming exact arithmetic. Observed discrepancies include numerical errors in both the approximation and its degree-exact reference. A same-backend pass is therefore a consistency check, not an absolute floating-point certificate. The fixed fitted model and empirical background define the game; population-background approximation and fitted-model uncertainty are outside the certificate. CPU FP64 errors must not be paired with GPU FP32 runtimes.

Times are mean ± sample standard deviation across patients. Rule construction and warm-up are excluded. Approximate API measurements include automatic node selection. Exact CPU timing is accumulated active checkpointed integration, including factor preparation, planning, transfers, synchronized core evaluation and host accumulation. Checkpoint I/O and one kernel-shape warm-up plus three pilot calls per dataset are excluded; this is not a full-explanation warm-up. Resumed-run downtime and uncheckpointed work lost to interruption are excluded. The checkpoint loop preserves public-API factor/block order and passed a bitwise small-problem check. Detailed provenance is in `provenance.json`. GPU/CPU and exact/approximate measurements were recorded in separate runs. Comparing GPU FP32 with CPU FP64 also changes precision, so their latency ratio is not a matched-precision GPU speedup.

## Outputs

- `cox_device_exact_table.tex`: table with each backend's own degree-exact reference.
- `cox_main_text.tex` and `cox_appendix.tex`: concise discussion and detailed experimental settings.
- `table.csv`: unrounded aggregate statistics.
- `case_errors.csv`: per-patient discrepancies calculated directly from saved attributions.
- `independent_reference_audit.csv`: additional numerical diagnostic, explicitly separate from the main table's Delta definitions.
- `preview.tex`: standalone LaTeX preview.
