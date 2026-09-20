# FP64 accuracy diagnostic at the saved 1e-6 node counts

All ten tested cases passed the requested 1e-6 absolute error threshold against
an independent, tightly converged FP64 64-node reference. Worst errors across
five patients and all features were 1.574563e-11 (GSE24080) and 1.277840e-10
(TCGA LGG methylation). The previous GPU FP32 errors against the same reference
were 2.267432e-3 and 2.242927e-2, respectively.

This diagnostic used CPU JAX with x64 enabled, the same parallel log-space
kernel formula used on GPU, the same saved 30 backgrounds, five patients,
model coefficients, and node counts. The CPU dispatch was changed only on the
local explainer instance; numerical library source files were not edited.
Actual kernel output dtype was checked as float64. These are CPU accuracy
results, not FP64 GPU timings or a new degree-exact FP64 computation. Only
the strict 1e-6 setting was rerun here.

Apple JAX-Metal does not support float64. The current library also explicitly
forces float32 on all non-CPU backends, so that dtype policy would need revision
before testing FP64 on a compatible GPU. CPU success does not substitute for
GPU validation or establish a formal bound on floating-point rounding.

Reproduction (preserve this output directory before rerunning):

```sh
PYTHONPATH=src:. JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv-metal/bin/python -u -m benchmarks.cox_fp64_precision_check
```
