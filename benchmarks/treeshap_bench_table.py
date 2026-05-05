"""Emit a LaTeX table from treeshap_bench_results.json.

The table mirrors the format of ``table4.tex``: feature widths are placed
side-by-side as ``\\multicolumn`` groups, every cell is a two-line
``\\shortstack`` whose top row is the runtime (``mean $\\pm$ std`` over seeds)
and bottom row is the worst-case additivity residual at ``\\tiny`` size.
Numerically broken entries are coloured red; the fastest stable time in
each column is bolded.

Run with:
    .venv/bin/python benchmarks/treeshap_bench_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "benchmarks" / "treeshap_bench_results.json"
TABLE_PATH = REPO / "benchmarks" / "treeshap_bench_table.tex"

# Order must match treeshap_bench_plot.py. Labels mirror table4.tex.
METHOD_ORDER = [
    ("shap",                                r"\textsc{TreeShap}"),
    ("fasttreeshap_v1",                     r"\textsc{FastTreeSHAP v1}"),
    ("fasttreeshap_v2",                     r"\textsc{FastTreeSHAP v2}"),
    ("linear_tree_shap",                    r"\textsc{Linear TreeSHAP}"),
    ("shapiq",                              r"\textsc{shapiq}"),
    ("pg_quadrature_tree_cpp",              r"\textbf{\QuadraSHAP}"),
]

# Maximum additivity residual for which a method's runtime is still
# eligible to be marked as "best time".
EXACTNESS_THRESHOLD = 1e-6
# Above this residual, the entry is rendered in red as a numerical breakdown.
BREAKDOWN_THRESHOLD = 1.0


def _ms_decimals(ms: float) -> int:
    if ms < 0.1:
        return 3
    if ms < 1.0:
        return 2
    if ms < 100.0:
        return 1
    return 0


def _per_seed_std_s(entry: dict) -> float | None:
    """Sample std dev (in seconds) of per-seed time_s values."""
    per_seed = entry.get("per_seed") or []
    times = [
        r["time_s"]
        for r in per_seed
        if r.get("status") == "ok" and r.get("time_s") is not None
    ]
    if len(times) < 2:
        return None
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
    return math.sqrt(var)


def _fmt_err(e: float | None) -> str:
    if e is None:
        return ""
    if e == 0.0:
        return r"$0$"
    mant, exp = f"{e:.0e}".split("e")
    return rf"${int(mant)}{{\cdot}}10^{{{int(exp)}}}$"


def _fmt_cell(entry: dict | None, is_best: bool) -> str:
    """Render one ``\\shortstack`` cell: top = time (mean ± std), bottom = error."""
    if entry is None or entry.get("status") not in ("ok", "timeout"):
        return r"\shortstack{--- \\ {\tiny }}"
    if entry.get("status") == "timeout":
        return r"\shortstack{\timeout \\ {\tiny }}"
    if entry.get("time_s") is None:
        return r"\shortstack{--- \\ {\tiny }}"

    ms = entry["time_s"] * 1e3
    decimals = _ms_decimals(ms)
    mean_str = f"{ms:.{decimals}f}"
    std_s = _per_seed_std_s(entry)
    err = entry.get("error")
    err_str = _fmt_err(err)
    breakdown = err is not None and err > BREAKDOWN_THRESHOLD

    if is_best and not breakdown:
        mean_str = rf"\textbf{{{mean_str}}}"

    if std_s is not None:
        std_ms = std_s * 1e3
        first_line = rf"{mean_str} {{\scriptsize $\pm$ {std_ms:.{decimals}f}}}"
    else:
        first_line = mean_str

    second_line = rf"{{\tiny {err_str}}}"

    if breakdown:
        first_line = rf"\textcolor{{red}}{{{first_line}}}"
        second_line = rf"{{\tiny \textcolor{{red}}{{{err_str}}}}}"

    return rf"\shortstack{{{first_line} \\ {second_line}}}"


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
    n_leaves = len(leaf_sizes)

    # Best stable time per (width, n_leaves).
    best_times: dict[tuple[int, int], float] = {}
    for w in widths:
        for n in leaf_sizes:
            cands = []
            for m_key, _ in METHOD_ORDER:
                e = results[str(w)].get(str(n), {}).get(m_key)
                if (
                    e
                    and e.get("status") == "ok"
                    and e.get("time_s") is not None
                    and e.get("error") is not None
                    and e["error"] <= EXACTNESS_THRESHOLD
                ):
                    cands.append(e["time_s"])
            if cands:
                best_times[(w, n)] = min(cands)

    lines: list[str] = []
    col_spec = "@{}l @{\\hspace{8pt}} *{" + str(n_leaves) + "}{c} @{}"
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize\setlength{\tabcolsep}{2pt}")
    lines.append(
        r"\caption{Single-threaded TreeSHAP runtime "
        r"(ms, mean $\pm$ sample std.\ over 3 seeds, top) and worst-case "
        r"efficiency-axiom violation $\max_i|\mathbb{E}[f] + "
        r"\sum_j \phi_{ij} - f(x_i)|$ (bottom) on synthetic data. "
        r"{\color{red}Red}: numerical breakdown. \textbf{Bold}: fastest stable.}"
    )
    lines.append(r"\label{tab:treeshap_bench}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join([r"\#leaves"] + leaf_headers) + r" \\")
    lines.append(r"\midrule")

    last_idx = len(METHOD_ORDER) - 1
    for wi, w in enumerate(widths):
        lines.append(
            rf"\multicolumn{{{n_leaves + 1}}}{{@{{}}l}}{{\emph{{$d = {w}$}}}} \\"
        )
        for mi, (method_key, label) in enumerate(METHOD_ORDER):
            cells = [label]
            for n in leaf_sizes:
                entry = results[str(w)].get(str(n), {}).get(method_key)
                t_s = entry.get("time_s") if entry else None
                is_best = (
                    t_s is not None
                    and t_s == best_times.get((w, n), float("inf"))
                )
                cells.append(_fmt_cell(entry, is_best))
            is_last_in_group = mi == last_idx
            is_last_overall = is_last_in_group and wi == len(widths) - 1
            if is_last_overall:
                suffix = r" \\"
            elif is_last_in_group:
                suffix = r" \\"
            else:
                suffix = r" \\[6pt]"
            lines.append(" & ".join(cells) + suffix)
        if wi < len(widths) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
