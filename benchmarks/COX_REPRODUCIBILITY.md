# Reproducing the Cox survival experiments

The easiest reliable workflow is to download the two public datasets from their
original providers, prepare them locally, and keep the large matrices out of Git.
This avoids a roughly 1.4 GiB repository download while preserving the exact
source URLs, deterministic preparation code, patient splits, background choices,
and validation checks used by the experiments.

Here, **reproduction** means rebuilding the same prepared cohorts and rerunning
the same model-fitting and explanation procedures. Wall-clock timings are
hardware- and system-load-dependent, so another machine should reproduce the
experimental design and numerical checks, not necessarily the recorded seconds.

## What each experiment does

An empirical **background** is the set of training patients used to average over
missing features in the interventional Shapley game. Changing that set changes
the explanation target; it is not merely a numerical approximation setting.

| Stage | Entry point | Data/background | Hardware |
|---|---|---|---|
| Fast correctness check | `run_cox_example.py` | Synthetic, 20 backgrounds | CPU |
| Original survival benchmark | `tutorials/cox_survival.ipynb` | Both real cohorts, 4 backgrounds | CPU |
| Full exact glioma follow-up | `tutorials/cox_survival_exact_glioma.ipynb` | TCGA LGG, 4 backgrounds | CPU; long run |
| GPU tolerance benchmark | `tutorials/cox_gpu_tolerance.ipynb` | Both real cohorts, 4 backgrounds | Apple Silicon + JAX-Metal |
| Full-background study | `tutorials/cox_background_sensitivity.ipynb` | 339 and 383 training backgrounds | CPU + Apple Silicon |

The later 30-background studies are documented in
[`results/cox_background30_gpu/README.md`](results/cox_background30_gpu/README.md).
The saved 100-background study is **not complete**: its GPU stage finished, but
the CPU stage was stopped after 36 of 40 cases. Do not report it as a completed
80-case comparison.

## 1. Check out Alan's branch

From a new clone:

```sh
git clone https://github.com/Majeed7/QuadraSHAP.git
cd QuadraSHAP
git switch --track origin/alan
```

From an existing clone:

```sh
git fetch origin
git switch alan
git pull --ff-only origin alan
```

Run every command below from the repository root.

## 2. Create the CPU environment

The pinned file reproduces the package versions recorded for the CPU experiments.
Python 3.13 was used for the saved run. [`uv`](https://docs.astral.sh/uv/) is the
recommended environment manager.

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
  -r benchmarks/requirements-cox-cpu-reproduction.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

The project is installed with `--no-deps` because the exact dependency versions
were installed by the preceding command.

First run the small synthetic example. It does not use either downloaded cohort;
its purpose is to verify the Cox adapter, quadrature calculation, exhaustive
coalition check, and Shapley additivity identity on a tractable eight-feature model.

```sh
PYTHONPATH=src:. .venv/bin/python run_cox_example.py
```

Then run the focused tests:

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_cox.py \
  tests/test_cox_float64_reference.py \
  tests/test_large_quadrature_rule.py \
  tests/test_experiment_data.py
```

## 3. Download and prepare the two cohorts

The download script uses these public sources:

- **GSE24080 multiple myeloma:** the processed series matrix and clinical
  workbook from [NCBI GEO GSE24080](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080).
- **TCGA lower-grade glioma methylation:** the public
  [UCSC Xena HumanMethylation450 matrix](https://xenabrowser.net/datapages/?dataset=TCGA.LGG.sampleMap%2FHumanMethylation450&host=https%3A%2F%2Ftcga.xenahubs.net)
  and the TCGA Clinical Data Resource table from the
  [GDC Pan-Cancer Atlas files](https://gdc.cancer.gov/about-data/publications/pancanatlas).

The exact file URLs are versioned in `benchmarks/fetch_experiment_data.py` and
`data/experiments/manifest.json`. Downloads are resumable. Each downloaded file
gets a local receipt containing its URL, byte count, and SHA-256 digest.

```sh
.venv/bin/python benchmarks/fetch_experiment_data.py \
  --datasets gse24080 tcga_lgg_methylation --workers 2

.venv/bin/python benchmarks/prepare_experiment_data.py \
  --datasets gse24080 tcga_lgg_methylation

.venv/bin/python benchmarks/validate_experiment_data.py \
  --datasets gse24080 tcga_lgg_methylation
```

The expected prepared cohorts are:

| Dataset | Patients | Predictors | OS events | Prepared matrix |
|---|---:|---:|---:|---|
| `gse24080` | 553 | 54,675 | 168 | `data/experiments/processed/gse24080/X.npy` |
| `tcga_lgg_methylation` | 511 | 396,065 | 125 | `data/experiments/processed/tcga_lgg_methylation/X.npy` |

The two raw downloads occupy about 525 MiB and the prepared files about 903 MiB.
Reserve additional space for environments, fitted models, exact rules, and saved
attributions. The validator writes
`data/experiments/validation_gse24080_tcga_lgg_methylation.json` and checks raw
hashes, array dimensions, event coding, missing-value counts, identifiers, and
selected values against the raw sources.

Detailed cohort construction decisions are in
[`data/experiments/README.md`](../data/experiments/README.md).

## 4. Run the main CPU benchmark

This notebook fits the ridge-regularized Cox models, keeps preprocessing fitted
only on training patients, explains three held-out patients per cohort, and uses
four training patients as the empirical background. The fixed random seed is 42.

```sh
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/quadrashap-cox-matplotlib \
  .venv/bin/python benchmarks/run_cox_notebook.py \
  --notebook tutorials/cox_survival.ipynb --timeout 3600
```

The notebook writes fitted-model state, inputs, timings, comparisons, and figures
under `benchmarks/results/cox_survival/`. It runs the full exact-degree rule for
GSE24080 and compares smaller rules against it. For TCGA LGG, the main notebook
uses the 128-node result as its initial numerical reference; the separate
follow-up below computes the full exact-degree rule.

Useful checks after the run:

```sh
cat benchmarks/results/cox_survival/run_summary.json
test -f benchmarks/results/cox_survival/gse24080/exact_attributions_patient_0.npy
test -f benchmarks/results/cox_survival/tcga_lgg_methylation/model_and_preprocessing.npz
```

## 5. Optional: full exact TCGA LGG reference

An **exact-degree rule** uses enough Gauss--Legendre nodes to integrate the
finite product-game polynomial exactly in real arithmetic. It does not eliminate
floating-point rounding. TCGA LGG has 396,065 active features, so the sufficient
rule contains 198,033 nodes and is intentionally expensive.

```sh
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/quadrashap-cox-matplotlib \
  .venv/bin/python benchmarks/run_cox_notebook.py \
  --notebook tutorials/cox_survival_exact_glioma.ipynb --timeout 7200
```

The recorded M4 Pro run took about 42 minutes in total, including roughly nine
minutes to construct the rule. Treat that duration only as a planning estimate.
The run saves the rule, exact attributions, timing, provenance, and comparisons
under `benchmarks/results/cox_survival/tcga_lgg_methylation/`. It refuses to
overwrite a complete local exact result.

## 6. Optional: Apple Silicon GPU benchmark

JAX-Metal is the experimental JAX backend that sends JAX operations to an Apple
GPU. These commands reproduce the tested Python 3.12 environment. They are not a
CUDA/NVIDIA recipe, and the smoke test deliberately fails if JAX silently falls
back to the CPU.

```sh
uv venv --python 3.12 .venv-metal
uv pip install --python .venv-metal/bin/python \
  -r benchmarks/requirements-cox-metal.txt
uv pip install --python .venv-metal/bin/python --no-deps -e .

PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  .venv-metal/bin/python benchmarks/jax_metal_smoke.py --case primitive

PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  .venv-metal/bin/python benchmarks/jax_metal_smoke.py --case logspace_jax
```

The full GPU notebook requires the CPU model/input artifacts and both patient-0
exact references from Steps 4--5. Missing GPU-side fit artifacts are regenerated
automatically with the CPU `.venv` and checked against those CPU artifacts.

```sh
PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  MPLCONFIGDIR=/tmp/quadrashap-mpl \
  .venv-metal/bin/python benchmarks/run_cox_notebook.py \
  --notebook tutorials/cox_gpu_tolerance.ipynb --timeout 7200
```

The notebook uses four background patients and compares tolerance-selected rules,
fixed node counts, and full exact-degree GPU integrations. JAX-Metal uses float32
arithmetic here, while accumulated node blocks and the CPU references use
float64. Therefore, a requested quadrature tolerance is not a guarantee on the
total observed floating-point error.

## 7. Optional: full-training-background study

This study changes the empirical Shapley game from four background patients to
all training patients: 339 for GSE24080 and 383 for TCGA LGG. It requires the
fit artifacts created by the GPU workflow, but preparation and reference
construction themselves run on the CPU.

```sh
PYTHONPATH=src:. JAX_PLATFORMS=cpu \
  .venv/bin/python -m benchmarks.cox_background_experiment prepare

PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  .venv-metal/bin/python -m benchmarks.cox_background_experiment stability

PYTHONPATH=src:. JAX_PLATFORMS=cpu \
  .venv/bin/python -m benchmarks.cox_background_experiment references

PYTHONPATH=src:. JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
  .venv-metal/bin/python benchmarks/run_cox_notebook.py \
  --notebook tutorials/cox_background_sensitivity.ipynb --timeout 7200
```

The independent float64 reference is a tightly bounded 64-node approximation,
cross-checked at 96 nodes for patient 0. It is not exhaustive coalition
enumeration or a full 198,033-node calculation. See
[`results/cox_background_sensitivity/README.md`](results/cox_background_sensitivity/README.md)
for the estimand, timing scope, and saved-file inventory.

## Interpreting a successful reproduction

- Patient counts, feature counts, split membership, fixed patient IDs, and fixed
  background selections should match the saved metadata.
- Numerical results should be finite and satisfy the recorded validation checks.
- CPU and GPU wall times should be reported as new measurements on the friend's
  hardware, not substituted into claims about the original M4 Pro run.
- Four-, 30-, 100-, and full-training-background results explain different
  empirical games. Differences between them are background sensitivity, not
  quadrature error.
- Exact-degree quadrature removes integration truncation error in real arithmetic;
  it does not remove floating-point rounding or uncertainty about whether the
  empirical background represents a wider patient population.

## Troubleshooting

- Run commands from the repository root and keep `PYTHONPATH=src:.` on benchmark
  commands that use module execution.
- Re-run the fetch command after an interrupted download; `.part` files resume.
- Preparation retains completed outputs. Use `--force` only when intentionally
  rebuilding from unchanged raw files.
- If a Metal smoke test reports a CPU device, stop: those are not GPU timings.
- Large `.npy` and `.npz` artifacts are intentionally ignored by Git. Small
  tables and reports can still appear as modified files after a reproduction run.
- Compare package versions with `benchmarks/results/cox_survival/environment.json`
  and `benchmarks/results/cox_gpu_tolerance/environment.json` before interpreting
  numerical or timing differences.
