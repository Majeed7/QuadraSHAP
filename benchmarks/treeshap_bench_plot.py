"""Plot the results written by treeshap_bench.py as a 2x2 paper figure.

Columns = {amortized time per leaf, additivity error}, rows = {10 features,
100 features}. Both panels use log-log axes.

The left column plots ``wall_time / total_leaves`` (in microseconds per
leaf) rather than raw wall time. All methods scale roughly linearly in the
total number of leaves, so on a plain wall-time plot the curves run
parallel and the (large!) constant-factor differences are visually
compressed. Normalising by leaf count divides out the shared linear trend
and turns constant-factor speedups into vertical separation.

Run with:
    .venv/bin/python benchmarks/treeshap_bench_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
})

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "benchmarks" / "treeshap_bench_results.json"
FIGURE_PATH = REPO / "benchmarks" / "treeshap_bench_figure.pdf"

# Stable method order + labels used in the legend.
METHOD_ORDER = [
    ("shap",                                "SHAP TreeExplainer"),
    ("fasttreeshap_v1",                     "FastTreeSHAP v1"),
    ("fasttreeshap_v2",                     "FastTreeSHAP v2"),
    ("linear_tree_shap",                    "linear_tree_shap"),
    ("pg_quadrature_tree_cpp",              "Quadrature (exact, $m_q=\\lceil D/2 \\rceil$)"),
    ("pg_quadrature_tree_cpp_mq_d_over_4",  "Quadrature (approx, $m_q=d/4$)"),
]

# A colourblind-friendly palette. The two "ours" variants are highlighted.
METHOD_STYLE = {
    "shap":                                dict(color="#1f77b4", marker="o", linestyle="-"),
    "fasttreeshap_v1":                     dict(color="#2ca02c", marker="s", linestyle="-"),
    "fasttreeshap_v2":                     dict(color="#17becf", marker="D", linestyle="-"),
    "linear_tree_shap":                    dict(color="#9467bd", marker="^", linestyle="-"),
    "pg_quadrature_tree_cpp":              dict(color="#d62728", marker="*", linestyle="-",
                                                markersize=10, linewidth=2.2),
    "pg_quadrature_tree_cpp_mq_d_over_4":  dict(color="#ff7f0e", marker="P", linestyle="--",
                                                markersize=7, linewidth=1.8),
}

TIMEOUT_MARKER = dict(marker="x", s=60, linewidths=1.5)


def _plot_one(ax_time, ax_err, results_for_width: dict):
    """Draw both panels for a single feature width."""
    sizes = sorted(int(k) for k in results_for_width.keys())

    for method_key, label in METHOD_ORDER:
        xs_time, ys_time = [], []
        xs_err, ys_err = [], []
        for n in sizes:
            entry = results_for_width.get(str(n), {}).get(method_key)
            if entry is None or entry.get("status") != "ok":
                continue
            t = entry.get("time_s")
            e = entry.get("error")
            if t is not None and t > 0:
                # Amortised microseconds per leaf — this divides out the
                # shared linear-in-leaves scaling so constant-factor
                # differences between methods are visible.
                xs_time.append(n)
                ys_time.append((t * 1e6) / n)
            if e is not None and e > 0:
                xs_err.append(n)
                ys_err.append(e)

        style = METHOD_STYLE[method_key]
        if xs_time:
            ax_time.plot(xs_time, ys_time, label=label, **style)
        if xs_err:
            ax_err.plot(xs_err, ys_err, label=label, **style)

    for ax in (ax_time, ax_err):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    widths = sorted(int(k) for k in results.keys())
    n_rows = len(widths)

    fig = plt.figure(figsize=(8.5, 3.2 * n_rows + 1.3))
    gs = fig.add_gridspec(
        n_rows, 2,
        left=0.10, right=0.98,
        top=0.93, bottom=0.17,
        hspace=0.35, wspace=0.30,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)]
                     for r in range(n_rows)])

    for row, w in enumerate(widths):
        ax_time = axes[row, 0]
        ax_err = axes[row, 1]
        _plot_one(ax_time, ax_err, results[str(w)])

        ax_time.set_ylabel("time per leaf ($\\mu$s)")
        ax_err.set_ylabel(r"$\max_i |\mathbb{E}[f] + \sum_j \phi_{ij} - f(x_i)|$")
        ax_time.set_title(f"$d = {w}$ features — amortised runtime", pad=4)
        ax_err.set_title(f"$d = {w}$ features — additivity residual", pad=4)
        if row == n_rows - 1:
            ax_time.set_xlabel("total leaves in ensemble")
            ax_err.set_xlabel("total leaves in ensemble")

    # Shared legend at the bottom, outside the axes.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.00),
    )

    fig.suptitle(
        "TreeSHAP scaling on random-forest regressors",
        fontsize=11, y=0.985,
    )

    fig.savefig(FIGURE_PATH)
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
