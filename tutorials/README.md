# QuadraSHAP tutorials

These notebooks introduce QuadraSHAP through reproducible synthetic examples
and a high-dimensional survival experiment:

| Notebook | What it demonstrates |
|---|---|
| [`tree_models.ipynb`](tree_models.ipynb) | Exact path-dependent TreeSHAP for a scikit-learn decision tree, checked against exhaustive coalition enumeration |
| [`kernel_methods.ipynb`](kernel_methods.ipynb) | Exact local Shapley values for an RBF `KernelRidge` model, checked against exhaustive product-game enumeration |
| [`cox_survival.ipynb`](cox_survival.ipynb) | Ridge Cox models on myeloma and glioma data, with measured approximate attribution and a full exact myeloma calculation |
| [`cox_survival_exact_glioma.ipynb`](cox_survival_exact_glioma.ipynb) | Full exact 198,033-node glioma calculation, reusing the first notebook's saved model and identical background |

The tree and kernel notebooks:

- train a small model on synthetic data;
- explain one prediction with QuadraSHAP;
- implement the corresponding naive Shapley calculation from scratch;
- verify attribution agreement and the Shapley additivity identity; and
- time both methods and print the speedup measured on the current machine.

The explanations follow the derivation and notation in:

> Mohammadi, M., Reznikov, G., Sinitcyn, P., Muandet, K., and Chau, S. L.
> **QuadraSHAP: Stable and Scalable Shapley Values for Product Games via
> Gauss-Legendre Quadrature.** arXiv:2605.05870v2, 2026.

In particular, the notebooks unpack the Beta-integral reduction (Proposition
2), Gauss-Legendre exactness and approximation result (Proposition 3), shared
log-space and scan computation (Proposition 4), the product-kernel value
function (Section 4.1), and the optimized tree traversal (Section 4.2 and
Appendix C).

The naive methods enumerate all `2**d` feature coalitions. They are included
only as transparent correctness and timing baselines, so increase the feature
counts cautiously.

## Running the notebooks

From the repository root, install QuadraSHAP and the tutorial dependencies:

```bash
python -m pip install -e .
python -m pip install jupyter scikit-learn
```

Then start Jupyter:

```bash
jupyter lab tutorials/
```

The tree notebook automatically uses the native C++ quadrature backend when
it is installed. Otherwise it uses the pure-Python implementation. The kernel
notebook uses the NumPy log-space backend, so neither notebook requires a GPU.

## Cox survival experiment

The Cox notebook uses the prepared data described in
[`data/experiments/README.md`](../data/experiments/README.md). Install its
dependencies into the existing project environment and execute it with:

```bash
uv pip install --python .venv/bin/python -r benchmarks/requirements-cox.txt
.venv/bin/python benchmarks/run_cox_notebook.py
```

This saves an executed notebook with outputs and cell timings. Models, source
patient roles, coefficients, predictions, attribution arrays, timing repetitions,
and plots are saved in `benchmarks/results/cox_survival/`. Rerunning replaces
these experiment outputs. The exact myeloma run uses all retained features and
four background patients; the larger glioma run uses approximate quadrature.
The notebook explains relative hazard, `exp(x @ beta)`, as specified by the
repository paper. It distinguishes quadrature error from the choice of a small
empirical background and reports whether an error reference is exact.
Large model states, coefficient tables, and full attribution arrays are retained
locally but ignored by Git; rerunning the notebook regenerates them.

The full exact glioma follow-up uses the saved coefficients and explanation
inputs from the first notebook. It does not repeat model fitting:

```bash
.venv/bin/python benchmarks/run_cox_notebook.py --notebook tutorials/cox_survival_exact_glioma.ipynb --timeout 14400
```

It records rule construction and integration separately, saves the full rule
and attributions, and compares the earlier approximations with the exact result.
The follow-up preserves the original comparison against 128 nodes and writes
`comparison_vs_exact.csv` separately. It refuses to overwrite an existing exact
glioma result; preserve or move that result before deliberately repeating the run.
