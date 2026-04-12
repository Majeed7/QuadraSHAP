"""Emit a LaTeX table from treeshap_bench_results.json.

The table has one row group per feature width and, inside each, two rows
per method: the first row shows runtime (ms) and the second shows the
worst-case additivity residual.  The fastest time in each column is bolded.

Run with:
    .venv/bin/python benchmarks/treeshap_bench_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "benchmarks" / "treeshap_bench_results.json"
TABLE_PATH = REPO / "benchmarks" / "treeshap_bench_table.tex"

# Order must match treeshap_bench_plot.py. Labels are LaTeX-safe.
METHOD_ORDER = [
    ("shap",                                r"\textsc{shap}"),
    ("fasttreeshap_v1",                     r"\textsc{FastTreeSHAP v1}"),
    ("fasttreeshap_v2",                     r"\textsc{FastTreeSHAP v2}"),
    ("linear_tree_shap",                    r"\textsc{linear\_tree\_shap}"),
    ("shapiq",                              r"\textsc{shapiq}"),
    ("pg_quadrature_tree_cpp",              r"\textbf{Quadrature} ($m_q{=}\lceil D/2 \rceil$)"),
]


def _fmt_time(t_s: float | None) -> str:
    if t_s is None:
        return "---"
    ms = t_s * 1e3
    if ms < 0.1:
        return f"{ms:.3f}"
    if ms < 1.0:
        return f"{ms:.2f}"
    if ms < 100.0:
        return f"{ms:.1f}"
    return f"{ms:.0f}"


def _fmt_err(e: float | None) -> str:
    if e is None:
        return "---"
    if e == 0.0:
        return r"$0$"
    mant, exp = f"{e:.0e}".split("e")
    return rf"${int(mant)}{{\cdot}}10^{{{int(exp)}}}$"


# Maximum additivity residual for which a method's runtime is still
# eligible to be marked as "best time". Anything above this is numerically
# broken and giving it the speed crown would be misleading.
EXACTNESS_THRESHOLD = 1e-6


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    widths = sorted(int(k) for k in results.keys())
    leaf_sizes = sorted(
        {int(k) for w in widths for k in results[str(w)].keys()}
    )
    leaf_headers = [
        (f"${n//1000}\\text{{k}}$" if n >= 1000 else f"${n}$")
        for n in leaf_sizes
    ]

    n_cols = len(leaf_sizes)
    lines: list[str] = []
    col_spec = "l" + "c" * n_cols
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Single-threaded TreeSHAP runtime (ms, median over 3 "
        r"sklearn \texttt{RandomForestRegressor}s with 10 samples each) "
        r"and worst-case additivity residual $\max_i|\mathbb{E}[f] + "
        r"\sum_j \phi_{ij} - f(x_i)|$. "
        r"Best time in each column is \textbf{bold}. Rows are grouped by "
        r"feature width $d$; columns are total leaves in the ensemble.}"
    )
    lines.append(r"\label{tab:treeshap_bench}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header_row = " & ".join(["Method"] + leaf_headers) + r" \\"
    lines.append(header_row)
    lines.append(r"\midrule")

    for wi, w in enumerate(widths):
        lines.append(
            rf"\multicolumn{{{n_cols + 1}}}{{l}}{{"
            rf"\emph{{$d = {w}$ features}}}} \\"
        )
        # Fastest time per column, restricted to methods with good accuracy.
        best_times: dict[int, float] = {}
        for n in leaf_sizes:
            candidates = []
            for m_key, _ in METHOD_ORDER:
                e = results[str(w)].get(str(n), {}).get(m_key)
                if (
                    e
                    and e.get("status") == "ok"
                    and e.get("time_s") is not None
                    and e.get("error") is not None
                    and e.get("error") <= EXACTNESS_THRESHOLD
                ):
                    candidates.append(e["time_s"])
            if candidates:
                best_times[n] = min(candidates)

        for method_key, label in METHOD_ORDER:
            # Row 1: method name + times
            time_cells = [label]
            for n in leaf_sizes:
                entry = results[str(w)].get(str(n), {}).get(method_key)
                if entry is None or entry.get("status") != "ok":
                    time_cells.append("---")
                    continue
                t_str = _fmt_time(entry.get("time_s"))
                is_best = (
                    entry.get("time_s") is not None
                    and entry["time_s"] == best_times.get(n, float("inf"))
                )
                if is_best:
                    t_str = rf"\textbf{{{t_str}}}"
                time_cells.append(t_str)
            lines.append(" & ".join(time_cells) + r" \\")

            # Row 2: empty label + errors in smaller font
            err_cells = [""]
            for n in leaf_sizes:
                entry = results[str(w)].get(str(n), {}).get(method_key)
                if entry is None or entry.get("status") != "ok":
                    err_cells.append("")
                    continue
                e_str = _fmt_err(entry.get("error"))
                err_cells.append(rf"{{\scriptsize {e_str}}}")
            lines.append(" & ".join(err_cells) + r" \\[2pt]")

        if wi < len(widths) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
