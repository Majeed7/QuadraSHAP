"""Figure 1 of the paper, redrawn from the saved records of experiment 8.

Reads ``results/exp8_multimodel_budget/validated_records.json`` (nothing is recomputed) and
writes ``node_budget_scaling_wide.pdf`` next to the other exp8 figures.  Same content as the
``scaling()`` figure of ``exp8_plot.py`` -- observed and certified node counts for a maximum
attribution error of 1e-6, medians with 10th--90th percentile bands, the exactness threshold
ceil(d/2) as a dotted line -- with a wider aspect ratio, no shared x-axis label, slightly smaller type, and the PDF cropped
to its content (the figure is placed at the ICLR text width of 5.5 in, so sizes are final).

    python fig1_node_budget_scaling.py [--root results/exp8_multimodel_budget] [--height 1.7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter

OBS, CERT, GRAY = "#0072B2", "#D55E00", "#757575"
DIMENSIONS = [50, 100, 250, 500, 1000]
FAMILIES = ["rbf", "poisson", "logistic", "naive_bayes"]
TITLES = {"rbf": "KRR", "poisson": "Poisson", "logistic": "Logistic", "naive_bayes": "Naive Bayes"}
WIDTH = 5.5                                    # ICLR \linewidth in inches; \includegraphics[width=\linewidth]
STYLE = {"font.family": "DejaVu Sans", "font.weight": "bold", "axes.labelweight": "bold",
         "axes.titleweight": "bold", "mathtext.default": "bf",
         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
         "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 240}
FS = {"title": 7.5, "legend": 7, "ylabel": 8, "ytick": 6.5, "xtick": 6.5}


def quantiles(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return np.quantile(values, [.1, .5, .9]) if values else np.full(3, np.nan)


def budget_curve(ax, rows, key, color, ls="-"):
    qs = np.array([quantiles([r for r in rows if r["d"] == d], key) for d in DIMENSIONS])
    ax.plot(DIMENSIONS, qs[:, 1], color=color, ls=ls, marker="o", ms=2.4)
    ax.fill_between(DIMENSIONS, qs[:, 0], qs[:, 2], color=color, alpha=.13, lw=0)


def make_figure(rows, height):
    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH, height), sharey=True)
    for ax, family in zip(axs, FAMILIES):
        subset = [r for r in rows if r["family"] == family and r["regime"] == "fixed_feature"
                  and r["value_function"] == "baseline" and r["epsilon"] == 1e-6]
        budget_curve(ax, subset, "m_observed", OBS)
        budget_curve(ax, subset, "m_certified", CERT, "--")
        ax.plot(DIMENSIONS, np.ceil(np.array(DIMENSIONS) / 2), color=GRAY, ls=":", lw=1.1)
        ax.set_title(TITLES[family], fontsize=FS["title"], pad=3)
        ax.set_xlim(25, 1040)
        ax.set_xticks([50, 500, 1000])
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.tick_params(axis="x", labelsize=FS["xtick"], pad=2.5)
        ax.tick_params(axis="y", labelsize=FS["ytick"])
        ax.grid(axis="y", color=".90", lw=.5)
        ax.set_yscale("log")
        ax.set_ylim(2, 650)
        ax.yaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 500]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        if ax is not axs[0]:
            ax.tick_params(axis="y", left=True, labelleft=False)
    axs[0].set_ylabel(r"$m_q$", fontsize=FS["ylabel"])
    handles = [Line2D([], [], color=OBS, marker="o", ms=2.6, label="Observed"),
               Line2D([], [], color=CERT, ls="--", marker="o", ms=2.6, label="Certified"),
               Line2D([], [], color=GRAY, ls=":", label=r"Exact: $\lceil d/2\rceil$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0), ncol=3,
               frameon=False, fontsize=FS["legend"], handlelength=2.2, columnspacing=1.6,
               borderaxespad=0, borderpad=0)
    fig.subplots_adjust(left=.09, right=.985, bottom=.17, top=.72, wspace=.13)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/exp8_multimodel_budget")
    ap.add_argument("--height", type=float, default=1.2, help="figure height in inches before cropping (width is fixed at 5.5)")
    ap.add_argument("--name", default="node_budget_scaling_wide")
    args = ap.parse_args()
    root = Path(args.root)
    src = root / "validated_records.json"
    rows = json.loads(src.read_text())
    fig = make_figure(rows, args.height)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    if bb.x0 < -.01 or bb.y0 < -.01 or bb.x1 > w + .01 or bb.y1 > h + .01:
        raise SystemExit(f"text extends outside the page: {bb.bounds}")
    out = root / "figures" / f"{args.name}.pdf"
    # crop to the drawn content: no margin is left around the legend, tick labels or y-label
    save = dict(bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out, **save)
    (root / "figures" / "backup").mkdir(exist_ok=True)
    fig.savefig(root / "figures" / "backup" / f"{args.name}.png", **save)
    print(f"wrote {out}  (canvas {w:.2f} x {h:.2f} in, cropped to content; {len(rows)} records read from {src.name})")


if __name__ == "__main__":
    main()
