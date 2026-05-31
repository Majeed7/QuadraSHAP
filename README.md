# QuadraSHAP

Anonymous code repository for a NeurIPS submission.

This repository implements Shapley-value computation for two related settings:

- `TreeExplainer`: TreeSHAP-style explanations for scikit-learn tree models, with interchangeable backends.
- Product-kernel explainers: local Shapley values for models whose prediction function factorizes across features, such as RBF kernel methods.

The code is organized as a research artifact first: it contains the library code under `src/`, tests under `tests/`, and benchmark scripts plus precomputed outputs under `benchmarks/`.

## Repository Structure

- `src/quadrashap/`: package source code
- `src/quadrashap/treeshap/`: tree-model explainers and backends
- `src/quadrashap/kernels/`: explainers for product-form kernel models
- `csrc/`: optional C++ extension used by the quadrature-tree backend
- `tests/`: correctness and regression tests
- `benchmarks/`: scripts for runtime and approximation experiments
- `benchmarks/results/`: saved benchmark outputs and figures
- `model/`: cached benchmark models for text-classification experiments

## Installation

The package targets Python `>=3.11`.

Using `uv`:

```bash
uv sync --extra jax --group testing
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[jax]
pip install pytest pytest-benchmark scikit-learn shap
```

Notes:

- The build system will try to compile the optional C++ extension if a compatible compiler is available.
- If compilation fails, installation falls back to a pure-Python build.
- JAX is optional for some backends, but the package currently declares `jax` and `jaxlib` as core dependencies in `pyproject.toml`.

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

print(phi.shape)              # (10, 6)
print(explainer.expected_value)
```

Available tree backends:

- `tree_solver="product_games"`: TreeSHAP via product-game factorization
- `tree_solver="quadrature_tree"`: direct quadrature-tree backend

Useful options:

- `backend_method="numpy_prefix_scan"`
- `backend_method="numpy_logspace"`
- `backend_method="jax_prefix_scan"`
- `backend_method="jax_logspace"`
- `m_q=<int>` to control the number of quadrature nodes
- `use_cpp=True/False` for the quadrature-tree backend

Current limitations:

- Only `model_output="raw"` is supported.
- Only `feature_perturbation="tree_path_dependent"` is implemented.
- The frontend currently targets supported scikit-learn tree estimators.

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

print(phi.shape)              # (5,)
```

Supported kernel backends:

- `logspace_numpy`
- `logspace_jax`
- `prefix_scan_numpy`
- `prefix_scan_jax`

## Running Tests

Run the main test suite with:

```bash
pytest tests
```

The tests cover:

- agreement with naive Shapley implementations on small problems
- frontend conversion from scikit-learn trees to the internal unified format
- end-to-end agreement with `shap.TreeExplainer` on supported tree models
- optional C++ extension behavior

## Reproducing Benchmarks

The repository contains several standalone benchmark scripts. Run them from the repository root.

### 1. `m_q` convergence for kernel explainers

Generate raw convergence data:

```bash
python benchmarks/bench_mq_sweep.py
```

Plot the aggregated results:

```bash
python benchmarks/plot_mq_results.py
```

Outputs are written under `benchmarks/results/mq/`.

### 2. TreeSHAP runtime benchmark

```bash
python benchmarks/treeshap_bench.py
```

This benchmark compares several TreeSHAP implementations across different tree sizes and writes results to:

- `benchmarks/treeshap_bench_results.json`

### 3. Text-classification benchmark

```bash
python benchmarks/text_classification_benchmark.py
```

This script evaluates tree and kernel explainers on TF-IDF text-classification setups and writes outputs under:

- `benchmarks/results/text_clf/`

The script may require extra packages such as `datasets`, `pandas`, `matplotlib`, `scipy`, `joblib`, and optionally `optuna`.

## Precomputed Results

The repository already includes saved benchmark artifacts, including:

- `benchmarks/results/mq/`: convergence CSVs and figures
- `benchmarks/results/text/`: text benchmark tables and plots

These are useful for inspection without rerunning the full experiments.

## Implementation Notes

- The package uses `scikit-build-core` and `pybind11` for the optional C++ extension.
- Tree explanations are computed through an internal unified tree representation converted from scikit-learn models.
- The kernel explainers use Gauss-Legendre quadrature with configurable `m_q`; if `m_q` is left unset, the implementation uses a default based on the feature dimension.

## Anonymous Submission Notice

This README is intentionally anonymized for double-blind review. It does not include author names, affiliations, acknowledgments, or links that would identify the submission.
