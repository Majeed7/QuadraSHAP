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

QuadraSHAP reformulates Shapley-value computation for product games as a Gauss-Legendre quadrature problem, yielding estimates that are both numerically stable and scalable to high-dimensional settings. The library covers two concrete application domains:

- **`TreeExplainer`**: TreeSHAP-style explanations for scikit-learn tree models, with interchangeable numerical backends.
- **Multiplicative-model explainers** (`RKHSExplainer`, `PoissonExplainer`, `GammaExplainer`, `TweedieExplainer`, `GLMExplainer`, `LogisticExplainer`, `NaiveBayesExplainer`, `CoxExplainer`): exact or certified-accuracy Shapley values for every model whose prediction is a sum of products of per-feature factors, from product-kernel machines to log-link GLMs, odds-scale classifiers and Cox models.

The repository is organized as a research artifact: library code lives under `src/`, correctness tests under `tests/`, and benchmark scripts with precomputed outputs under `benchmarks/`.

## Repository Structure

| Path | Description |
|---|---|
| `src/quadrashap/` | Package source code |
| `src/quadrashap/treeshap/` | Tree-model explainers and numerical backends |
| `src/quadrashap/multiplicative/` | Engine and named explainers for multiplicative models (product kernels, GLMs, odds-scale classifiers, Cox) |
| `src/quadrashap/product_games/` | Quadrature cores, a priori node budget, blockwise evaluation |
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

For the CUDA tree backend, install the GPU extra matching a CUDA 12 runtime:

```bash
uv sync --extra gpu --group testing
```

## Quick Start

### Drop-in replacement for SHAP

```python
# import shap
# explainer = shap.TreeExplainer(model)
from quadrashap import TreeExplainer

explainer = TreeExplainer(model)
phi = explainer.shap_values(X_test)
base_value = explainer.expected_value
```

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
| `device` | `"cpu"` / `"cuda"` (quadrature-tree backend only) |

The CUDA path uses the same exact edge-telescoping algorithm as the native
quadrature-tree solver.  It keeps model data resident, streams samples through
sibling-paired warps, precomputes quadrature factors, and converts the forward
node products into subtree sums in place; no leaf-prefix tensor is allocated.

```python
explainer = TreeExplainer(
    model,
    tree_solver="quadrature_tree",
    device="cuda",
)
phi = explainer.shap_values(X_test)
```

**Current limitations:**

- Only `model_output="raw"` is supported.
- Only `feature_perturbation="tree_path_dependent"` is implemented.
- The frontend currently targets scikit-learn tree estimators.

### Multiplicative models: kernels, GLMs, classifiers on the odds scale, Cox models

Every model with a sum-of-products representation (Definition 2 of the paper) is explained
by the same engine, through a named explainer per family. All of them share one interface:
`explain(x, value_function=..., ...)`, `shap_values(X)`, `expected_value`, `node_budget(x, eps)`.

| Explainer | Models | Attributed quantity (scale) |
|---|---|---|
| `RKHSExplainer` | SVR/SVC, `KernelRidge`, Gaussian processes with an RBF kernel | kernel expansion `sum_r alpha_r k(x, x_r)` |
| `PoissonExplainer`, `GammaExplainer`, `TweedieExplainer` | scikit-learn log-link GLMs | mean response |
| `GLMExplainer` | any log-link GLM/GAM: `(beta, intercept)` or a statsmodels result | mean response |
| `LogisticExplainer` | `LogisticRegression` (binary or a class pair of a multinomial model) | odds |
| `NaiveBayesExplainer` | `GaussianNB`, `BernoulliNB`, `MultinomialNB` | odds |
| `CoxExplainer` | lifelines `CoxPHFitter`, scikit-survival `CoxPHSurvivalAnalysis`, or log hazard ratios | hazard ratio |
| `TreeExplainer` | scikit-learn tree ensembles | prediction (path-dependent value function) |
| `Explainer(model)` | picks one of the above from the fitted estimator | |

Anything else with a product structure is wrapped with `FactorModel(coef, factor_fn, d)` and passed to
`QuadraSHAP`, the engine itself. Each family has a flat, step-through example script in the repository
root: `run_rkhs_example.py`, `run_poisson_example.py`, `run_gamma_example.py`, `run_tweedie_example.py`,
`run_glm_example.py`, `run_logistic_example.py`, `run_naive_bayes_example.py`, `run_cox_example.py`,
`run_tree_example.py` and `run_explainer_example.py`; each verifies the attributions against exhaustive
coalition enumeration through the model itself.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge

from quadrashap import RKHSExplainer

X, y = make_regression(n_samples=200, n_features=50, random_state=0)
model = KernelRidge(kernel="rbf", gamma=0.02, alpha=1.0).fit(X, y)

explainer = RKHSExplainer(model, background=X[:100])

# neutral-factor value function (absent features are dropped from the kernel; the default for kernels)
phi = explainer.explain(X[0])

# empirical interventional value function over the background dataset,
# and the baseline value function for a single reference point
phi_int = explainer.explain(X[0], "interventional")
phi_base = explainer.explain(X[0], "baseline", baseline=X[1])

# the number of Gauss-Legendre nodes is chosen a priori so that every
# attribution is within eps of its exact value (default eps=1e-3)
phi, report = explainer.explain(X[0], return_report=True)
print(report)                    # m_q, Lambda, certified error, exactness threshold, efficiency residual, time
phi_exact = explainer.explain(X[0], m_q="exact")   # ceil(d/2) nodes
```

```python
from sklearn.linear_model import PoissonRegressor
from quadrashap import PoissonExplainer, Explainer

glm = PoissonRegressor().fit(X_counts, y_counts)
phi = PoissonExplainer(glm, background=X_counts[:100]).explain(x)      # on the expected-count scale
phi = Explainer(glm, background=X_counts[:100]).explain(x)             # same, dispatched automatically
```

**Value functions (`value_function`):** `"neutral"` (product kernels only, their default;
the value function of PKeX-Shapley), `"baseline"` (reference point `baseline=`),
`"interventional"` (background dataset `background=`; the default for all other
families; the baseline value function is its single-row special case). All three are weighted sums of product games and
share the same quadrature machinery; the interventional cost grows linearly
with the number of background rows.

**Node selection:** `m_q=None` (default) computes a certified budget from the
factor tables before any quadrature is run (`QuadraSHAP.node_budget`,
`quadrashap.product_games.budget`); `m_q="exact"` uses `ceil(d/2)`; an integer
is used as given. The certificate bounds the quadrature error on the scale of
the largest marginal contribution and costs one pass over the factor tables.

**Blockwise evaluation (`block_size`):** the sum over component-background pairs
and over quadrature nodes is a reduction, so both are processed in blocks that are
accumulated into the attribution vector, exactly and with bounded peak memory.
`block_size="auto"` (default) plans the blocks from the memory currently available
on the machine (or an explicit `memory_budget="512MB"`) and does not block when the
full computation fits; for the NumPy prefix scan it additionally keeps each block
around 16 MB, where cache-resident blocks run 1.5-2.5x faster than the unblocked
computation (`target_block_bytes=None` disables this cap). A positive integer fixes
the number of pairs per block and `block_size=-1` switches blocking off. The plan
used by the last call is available as `explainer.last_block_plan`. Install `psutil` (`pip install -e .[memory]`) for accurate detection of the available memory; otherwise the planner falls back to `sysconf`/`sysctl`.

With `return_report=True` the report also carries `efficiency_residual`, the a posteriori check $|\sum_i\phi_i-(f(x)-v(\varnothing))|$: exactly zero at or above the exactness threshold, nonzero below it, and a necessary condition only (signed per-feature errors can cancel), so it complements rather than replaces the certified bound.

**Backends (`backend`):** `logspace_numpy`, `logspace_jax`, `prefix_scan_numpy`,
`prefix_scan_jax`, or `"auto"`. The previous `RBFLocalExplainer` /
`ProductKernelLocalExplainer` classes remain available with their old
signature (exact quadrature by default); `QuadraSHAP(model)` still works and dispatches to the right adapter.

## Tutorials

The [`tutorials/`](tutorials/) directory contains two executable Jupyter
notebooks that derive the method from the paper, connect the mathematics to
the implementation, and include naive exact baselines, correctness checks,
quadrature-convergence examples, and measured timing comparisons:

- [Tree-model tutorial](tutorials/tree_models.ipynb) — path-dependent TreeSHAP
  for a scikit-learn decision tree versus exhaustive coalition enumeration.
- [Product-kernel tutorial](tutorials/kernel_methods.ipynb) — local Shapley
  values for RBF Kernel Ridge versus exhaustive product-game enumeration.

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

### 0. A priori node budget on a trained kernel ridge model

```bash
python benchmarks/kernel_node_budget_bench.py --d 1000 --n-train 500
python benchmarks/kernel_node_budget_bench.py --value-function interventional --n-background 20
```

Writes `benchmarks/results/kernel_node_budget/{tuned,sweep}_<value_function>.csv`
with the certified budget, the certified bound, the observed error against the
exact rule, the empirically minimal budget and the timings, for a CV-tuned
bandwidth and for a bandwidth sweep.

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

### 4. GPU TreeSHAP benchmark

```bash
uv sync --extra gpu --extra benchmarks --group testing
python benchmarks/gpu_treeshap_bench.py
```

This reproduces the synthetic and text tree comparisons with
QuadraSHAP-GPU and SHAP's CUDA `GPUTreeExplainer`. Outputs are written to
`benchmarks/results/gpu/`. The GPUTreeSHAP baseline requires SHAP to be built
from source with its optional CUDA extension.
The worker mode accepts `--n-samples` and repeats cached inputs as necessary,
which can be used to reproduce the saved batch-scaling experiment (batch
sizes 1 through 32,000).
Large scalar batches use exact FP64 checkpointed treelets: only component-root
state is stored in global memory, while 7- or 15-node connected components are
reconstructed in shared memory. The compact internal-node kernel remains the
lower-latency path below the measured 1,536-row crossover.

## Precomputed Results

Saved benchmark artifacts are included for inspection without rerunning experiments:

- `benchmarks/results/mq/` — convergence CSVs and figures for the quadrature-node sweep
- `benchmarks/results/text/` — tables and plots from the text-classification benchmark
- `benchmarks/results/gpu/` — GPU tables and optimized batch-scaling results

## Implementation Notes

- The package uses `scikit-build-core` and `pybind11` for the optional C++ extension.
- Tree explanations are computed via an internal unified tree representation converted from scikit-learn models.
- CUDA tree explanations use ragged per-tree quadrature checkpoints and reuse
  their feature-partial workspace after every treelet depth.
- Kernel explainers use Gauss-Legendre quadrature with a configurable number of nodes `m_q`; when unset, the number of nodes is chosen a priori from the factor tables so that the attributions are certified to lie within `eps` (default `1e-3`) of their exact values (see `quadrashap.product_games.budget`).
