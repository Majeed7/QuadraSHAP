# QuadraSHAP: Exact Shapley Values in Logarithmic Time for Trees and Kernels

<p align="center">
  <img src="data/quadraSHAP_logo.png" alt="alt text" width="400">
</p>

This repository provides the official implementation accompanying the paper:

> **QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature**

[//]: # (If you use our algorithm in your research we would appreciate a citation to the following paper:)

[//]: # (```)

[//]: # (@article{mohammadi2026quadrashap,)

[//]: # (  title = {QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss--Legendre Quadrature},)

[//]: # (  author = {Mohammadi, Majid and Reznikov, Grigory and Sinitcyn, Pavel and Muandet, Krikamol and Chau, Siu Lun},)

[//]: # (  journal = {arXiv preprint arXiv:2605.05870},)

[//]: # (  year = {2026})

[//]: # (})

[//]: # (```)

QuadraSHAP reformulates Shapley-value computation for product games as a Gauss-Legendre quadrature problem, yielding estimates that are both numerically stable and scalable to high-dimensional settings. The library covers three concrete application domains:

- **`TreeExplainer`**: TreeSHAP-style explanations for scikit-learn tree models, with interchangeable numerical backends.
- **Product-kernel explainers**: local Shapley values for models whose prediction function factorizes across features, such as RBF kernel methods.
- **`CoxPHExplainer`**: relative-hazard explanations for fitted Cox models, using whole empirical background rows and memory-bounded quadrature.

The repository is organized as a research artifact: library code lives under `src/`, correctness tests under `tests/`, and benchmark scripts with precomputed outputs under `benchmarks/`.

## Repository Structure

| Path | Description |
|---|---|
| `src/quadrashap/` | Package source code |
| `src/quadrashap/treeshap/` | Tree-model explainers and numerical backends |
| `src/quadrashap/kernels/` | Explainers for product-form kernel models |
| `csrc/` | Optional C++ extension for the quadrature-tree backend |
| `tests/` | Correctness and regression tests |
| `benchmarks/` | Scripts for runtime and approximation experiments |
| `benchmarks/results/` | Saved benchmark outputs and figures |
| `model/` | Cached models used by the text-classification benchmarks |

## Installation

The package requires Python `>=3.11`.

**Using `uv` (recommended):**

```bash
uv sync --extra jax --group testing
```

**Using `pip`:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[jax]
pip install pytest pytest-benchmark scikit-learn shap
```

> **Notes**
> - The build system attempts to compile the optional C++ extension if a compatible compiler is detected. If compilation fails, installation falls back gracefully to a pure-Python build.
> - JAX is optional for some backends, but `jax` and `jaxlib` are currently declared as core dependencies in `pyproject.toml`.

## Quick Start

### Tree models

`quadrashap.TreeExplainer` follows the familiar SHAP-style interface for supported scikit-learn tree models.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from quadrashap import TreeExplainer

X, y = make_regression(n_samples=300, n_features=6, random_state=0)
model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0).fit(X, y)

explainer = TreeExplainer(model, tree_solver="product_games")
phi = explainer.shap_values(X[:10])

print(phi.shape)          # (10, 6)
print(explainer.expected_value)
```

**Available tree backends (`tree_solver`):**

| Value | Description |
|---|---|
| `"product_games"` | TreeSHAP via product-game factorization |
| `"quadrature_tree"` | Direct quadrature-tree backend |

**Useful options:**

| Option | Values |
|---|---|
| `backend_method` | `"numpy_prefix_scan"`, `"numpy_logspace"`, `"jax_prefix_scan"`, `"jax_logspace"` |
| `m_q` | Number of quadrature nodes (integer) |
| `use_cpp` | `True` / `False` (quadrature-tree backend only) |

**Current limitations:**

- Only `model_output="raw"` is supported.
- Only `feature_perturbation="tree_path_dependent"` is implemented.
- The frontend currently targets scikit-learn tree estimators.

### Product-kernel models

For kernel methods with factorized feature kernels, use `RBFLocalExplainer` or `ProductKernelLocalExplainer`.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge

from quadrashap.kernels.explainer import RBFLocalExplainer

X, y = make_regression(n_samples=200, n_features=5, random_state=0)
model = KernelRidge(kernel="rbf", gamma=0.5, alpha=1.0).fit(X, y)

explainer = RBFLocalExplainer(model)
phi = explainer.explain(X[0], method="logspace_numpy")

print(phi.shape)          # (5,)
```

**Supported kernel backends (`method`):** `logspace_numpy`, `logspace_jax`, `prefix_scan_numpy`, `prefix_scan_jax`.

### Cox relative hazard

To explain a fitted Cox model on its multiplicative output scale, provide its
coefficient vector and processed training-background rows. Use the same feature
order and preprocessing as the fitted model:

```python
from quadrashap import CoxPHExplainer

explainer = CoxPHExplainer(model.coef_, X_train[:4])
explanation = explainer.explain(X_test[0], m_q=16)
print(explanation.values)
print(explanation.base_value, explanation.prediction)
# m_q=None uses the sufficient exact rule, ceil(nonzero_coefficients / 2).
```

This attributes `exp(x @ model.coef_)` using whole empirical background rows.
The positive-factor backend evaluates the product in logarithms and processes
blocks of nodes to bound memory. SciPy supplies the quadrature rule. See the
[executed Cox notebook](tutorials/cox_survival.ipynb) for model fitting,
training-only preprocessing, measured runtimes, and exact-versus-approximate
comparisons. Install its dependencies with
`uv pip install --python .venv/bin/python -r benchmarks/requirements-cox.txt`.

## Tutorials

The [`tutorials/`](tutorials/) directory contains executable Jupyter
notebooks that derive the method from the paper, connect the mathematics to
the implementation, and include naive exact baselines, correctness checks,
quadrature-convergence examples, and measured timing comparisons:

- [Tree-model tutorial](tutorials/tree_models.ipynb) — path-dependent TreeSHAP
  for a scikit-learn decision tree versus exhaustive coalition enumeration.
- [Product-kernel tutorial](tutorials/kernel_methods.ipynb) — local Shapley
  values for RBF Kernel Ridge versus exhaustive product-game enumeration.
- [Cox survival experiment](tutorials/cox_survival.ipynb) — dense ridge Cox
  models on myeloma and glioma data, with measured approximate attribution and
  a full exact calculation on the myeloma model.
- [Full exact glioma follow-up](tutorials/cox_survival_exact_glioma.ipynb) —
  the 198,033-node exact calculation on the saved 396,065-feature model, using
  the same explained patient and background as the approximations.

See the [tutorial guide](tutorials/README.md) for installation and launch
instructions.

## Running Tests

```bash
pytest tests
```

The test suite verifies:

- agreement with naive Shapley implementations on small problems;
- frontend conversion from scikit-learn trees to the internal unified format;
- end-to-end agreement with `shap.TreeExplainer` on supported tree models;
- optional C++ extension behavior.

## Reproducing Experiments

All benchmark scripts are run from the repository root.

### 1. Quadrature-node convergence for kernel explainers

Generate raw convergence data:

```bash
python benchmarks/bench_mq_sweep.py
```

Aggregate and plot results:

```bash
python benchmarks/plot_mq_results.py
```

Outputs are written to `benchmarks/results/mq/`.

### 2. TreeSHAP runtime benchmark

```bash
python benchmarks/treeshap_bench.py
```

Compares several TreeSHAP implementations across varying tree sizes. Results are saved to `benchmarks/treeshap_bench_results.json`.

### 3. Text-classification benchmark

```bash
python benchmarks/text_classification_benchmark.py
```

Evaluates tree and kernel explainers on TF-IDF text-classification setups. Outputs are written to `benchmarks/results/text_clf/`.

> Additional dependencies may be required: `datasets`, `pandas`, `matplotlib`, `scipy`, `joblib`, and optionally `optuna`.

## Precomputed Results

Saved benchmark artifacts are included for inspection without rerunning experiments:

- `benchmarks/results/mq/` — convergence CSVs and figures for the quadrature-node sweep
- `benchmarks/results/text/` — tables and plots from the text-classification benchmark

## Implementation Notes

- The package uses `scikit-build-core` and `pybind11` for the optional C++ extension.
- Tree explanations are computed via an internal unified tree representation converted from scikit-learn models.
- Kernel explainers use Gauss-Legendre quadrature with a configurable number of nodes `m_q`; when unset, a default is chosen based on the feature dimension.
