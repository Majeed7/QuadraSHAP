# High-dimensional Cox manuscript insert

Include `cox_high_dimensional_subsection.tex` and `cox_high_dimensional_table.tex`
in the manuscript. The preamble needs `amsmath`, `booktabs`, and `url`. Merge the
two entries from `cox_high_dimensional_sources.bib` into the manuscript bibliography.
The standalone `preview.tex` uses `hyperref` for clickable references.

The table reports absolute errors on the relative-hazard attribution scale.
`C_max` is the largest saved quadrature certificate across the five patients.
`Delta_GPU` is the maximum over all five patients and every feature of the
absolute difference between approximate and same-patient degree-exact GPU output.
Pass counts use each patient's unrounded maximum absolute discrepancy. The exact
row is a reference, so its comparison error and pass count are marked with dashes;
its zero quadrature bound does not imply zero floating-point error.

All numbers were checked against `../summary.csv`, `../table.csv`, and
`../comparison_to_gpu_exact.csv`; attribution differences were independently
recomputed from `../attributions.npz`. No new GPU runs were needed. The background
and five explained patients are unchanged from the completed benchmark.

The recorded GPU comparison passes are 5/5, 4/5, 0/5 for GSE24080 and
5/5, 2/5, 0/5 for TCGA LGG at tolerances 1e-1, 1e-3, 1e-6. The corresponding
passes against the independent FP64 reference are 5/5, 1/5, 0/5 and 5/5, 0/5, 0/5.
These are different comparisons. Neither a small relative error nor agreement
with the FP32 exact-degree output establishes the requested absolute accuracy.

Dataset characteristics and modeling settings were checked against the saved
processed-data metadata, model fit summaries, preprocessing code, and current
CoxExplainer implementation. The current paper PDF explains Cox attribution on
the relative-hazard scale and exactness at ceil(d/2) nodes. The numeric certificates
reported here are the outputs of the implemented automatic node-budget routine.

Compile the preview from this directory with `latexmk -pdf preview.tex`.
