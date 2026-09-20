# Experiment 8: multi-model node budgets

This experiment is separate from the earlier RBF-only experiments. It retains
both easy controls and harder positive-factor models. See `exp8_protocol.md`
for the frozen design and `results/exp8_multimodel_budget/REPORT.md` for results,
figure captions, numerical qualifications, and suggested manuscript wording.

Collection: `python exp8_multimodel_budget.py --interventional`

Independent analytic checks: `python exp8_stress_and_tests.py --output results/exp8_multimodel_budget`

High-precision audit: `python exp8_precision_audit.py --output results/exp8_multimodel_budget`

Resolve flagged references: `python exp8_refine_references.py --output results/exp8_multimodel_budget`

Final validation: `python exp8_finalize.py --output results/exp8_multimodel_budget`

Replot only, preserving saved summaries:
`python exp8_plot.py --output results/exp8_multimodel_budget --figures-only`

Results are resumable at the completed-fit level and configuration-checked.
Every instance's factor tables, tested node counts, attribution vectors, errors,
certificates and two independent references are in compressed NPZ files.
`model_data.npz` stores fitted parameters, explanation inputs and background.
`config.json` records versions, seeds and source hashes. Figures are exported as
vector PDFs and high-resolution PNGs; `figures/paper_figures.pdf` collects them.
PDFs remain in `figures/`; PNG previews are written to `figures/backup/`.
The four-column node-budget figure uses short model titles, a shared `#d`
label, all four y-axis lines and ticks, and leftmost-only y-axis numbers.
The bold tick labels are 6.5 pt (x) and 7 pt (y) at the native 5.5-inch paper width.

Use Python with NumPy, SciPy, scikit-learn, matplotlib and threadpoolctl. The
experiment calls the explicit NumPy float64 prefix-suffix evaluator, not the
package's automatic JAX backend. The decimal precision audit uses the standard
library. The collection scripts do not modify manuscript source or previous
result files. The publication revision presents the main node-count figure in
Section 5.1 and the protocol and additional tests in three Appendix I
subsections. It preserves the former Appendix I tree
tables as Appendix J. The original production report and source snapshot are
historical records and are retained unchanged. Earlier plot versions are in
`figures/backup/before_appendix_I_20260914/`.
