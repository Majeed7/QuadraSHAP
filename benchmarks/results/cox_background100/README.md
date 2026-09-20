# Cox explanations with 100 backgrounds

This study tests whether the approximation-versus-exact discrepancies change
when the fixed empirical background is enlarged from 30 to 100 training patients.
The first 100 entries of the saved seed-42 permutation are used, so the original
30 backgrounds are an exact prefix. Both cohorts retain the same fitted model,
preprocessing and five held-out patients as the B30 study. Models are not refitted.

There are 40 measured cases per device: two datasets, five patients, and exact
quadrature plus tolerances 0.1, 0.01 and 0.001. GPU kernels use JAX-Metal FP32;
CPU kernels use JAX FP64 with the same parallel formula. Both accumulate returned
blocks on the host in FP64. Each approximation is compared with its own patient's
new full-degree exact result on the same device and precision. No B30 attribution
or smaller-node reference is substituted for these new references.

The numerical source hashes, inputs, patient IDs, precision, quadrature rules,
memory plan and package versions are recorded and checked. Degree-exact rules
are cached from the previous study; their historical construction costs are
reported separately. Approximate timings include automatic node selection.
All timings exclude warm-up and rule construction. Exact timings measure
accumulated active integration, excluding checkpoint I/O, initial output
allocation, resume loading, inactive downtime and lost uncheckpointed work.

## Running and resuming

From the repository root, preparation is idempotent:

```sh
PYTHONPATH=src:. .venv-metal/bin/python -m benchmarks.cox_background100_inputs
```

The supervisor runs GPU followed by CPU, with an OS lock preventing duplicates:

```sh
PYTHONPATH=src:. .venv-metal/bin/python -u -m benchmarks.cox_background100_run all >> benchmarks/results/cox_background100/run.log 2>&1
```

JAX-Metal must have access to the Apple GPU outside the execution sandbox.
The supervisor configures each child device and precision, and prevents idle
sleep while it is alive. Re-running the same command resumes exact checkpoints
and skips completed case records only when their hashes and settings match.
Do not change numerical or benchmark sources while a study is running.

`run_status.json` describes the supervisor. `gpu/status.json` and
`cpu/status.json` show device progress; each must finish with 40 cases.
The per-device `cases/` records and `attributions/` files are authoritative,
and `summary.csv` provides a convenient aggregate index. Exact progress is
saved beneath each device's `checkpoints/` directory.

After all 80 cases finish, run `benchmarks.cox_background100_report`. It validates
the saved arrays and writes `table.csv`, `case_errors.csv`, `results.txt` and
`report_validation.json`. The requested plain-text email is sent only after
that validation; `email_request.json` records the requested delivery and
`email_delivery.json` will record a successful send. Increasing the number of
backgrounds changes the Shapley game and does not guarantee less FP32 rounding.
