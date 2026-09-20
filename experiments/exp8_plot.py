"""Replot experiment 8 from saved results; never retrain or recompute Shapley values."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter
from exp8_multimodel_budget import FAMILIES, write_csv, write_json

LABELS = {"rbf": "RBF kernel ridge", "poisson": "Poisson regression (mean)",
          "logistic": "Logistic regression (odds)", "naive_bayes": "Gaussian Naive Bayes (odds)"}
OBS, CERT, GRAY = "#0072B2", "#D55E00", "#757575"
DIMENSIONS = [50, 100, 250, 500, 1000]
STYLE = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
         "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
         "mathtext.default": "bf",
         "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .6, "lines.linewidth": 1.5, "pdf.fonttype": 42,
         "ps.fonttype": 42, "savefig.dpi": 240, "figure.dpi": 120}
plt.rcParams.update(STYLE)


def select(rows, **kwargs):
    return [r for r in rows if all(r[k] == v for k, v in kwargs.items())]


def quantiles(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return np.quantile(values, [.1, .5, .9]) if values else np.full(3, np.nan)


def budget_curve(ax, rows, key, color, linestyle="-", label=None, band=True):
    qs = np.array([quantiles(select(rows, d=d), key) for d in DIMENSIONS])
    ax.plot(DIMENSIONS, qs[:, 1], color=color, ls=linestyle, marker="o", ms=3, label=label)
    if band:
        ax.fill_between(DIMENSIONS, qs[:, 0], qs[:, 2], color=color, alpha=.13, lw=0)


def legend_budget(fig, y=.985, extra=False):
    handles = [Line2D([], [], color=OBS, marker="o", ms=3, label="Observed"),
               Line2D([], [], color=CERT, ls="--", marker="o", ms=3, label="Certified")]
    if extra:
        handles.append(Line2D([], [], color=GRAY, ls=":", label=r"Exactness: $\lceil d/2\rceil$"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(.52, y), handlelength=2.5)


def format_budget(ax, log=True):
    ax.set_xlim(25, 1040)
    ax.set_xticks([50, 250, 500, 1000])
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_xlabel(r"Number of features $d$")
    ax.grid(axis="y", color=".90", lw=.5)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(2, 650)
        ax.yaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 500]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()


def scaling(rows):
    # Native ICLR text width: no downscaling of the bold labels in the paper.
    fig, axs = plt.subplots(1, 4, figsize=(5.5, 2.12), sharey=True)
    titles = {"rbf": "KRR", "poisson": "Poisson",
              "logistic": "Logistic", "naive_bayes": "Naive Bayes"}
    for ax, family in zip(axs.flat, FAMILIES):
        subset = select(rows, family=family, regime="fixed_feature", value_function="baseline", epsilon=1e-6)
        budget_curve(ax, subset, "m_observed", OBS)
        budget_curve(ax, subset, "m_certified", CERT, "--")
        ax.plot(DIMENSIONS, np.ceil(np.array(DIMENSIONS)/2), color=GRAY, ls=":", lw=1.2)
        ax.set_title(titles[family], loc="center", fontsize=8, pad=6)
        format_budget(ax)
        ax.set_xlabel("")
        ax.set_xticks([50, 500, 1000])
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.tick_params(axis="x", labelsize=6.5, pad=3)
        ax.tick_params(axis="y", labelsize=7)
        if ax is not axs[0]:
            ax.tick_params(axis="y", left=True, labelleft=False)
    axs[0].set_ylabel(r"$m_q$", fontsize=9)
    fig.supxlabel("#d", x=.55, y=.025, fontsize=9, fontweight="bold")
    handles = [Line2D([], [], color=OBS, marker="o", ms=3, label="Observed"),
               Line2D([], [], color=CERT, ls="--", marker="o", ms=3, label="Certified"),
               Line2D([], [], color=GRAY, ls=":", label=r"Exact: $\lceil d/2\rceil$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, 1),
               ncol=3, frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.5)
    fig.subplots_adjust(left=.115, right=.985, bottom=.23, top=.75, wspace=.17)
    return fig


def load_curves(root, rows, family, d):
    subset = select(rows, family=family, d=d, regime="fixed_feature", value_function="baseline", epsilon=1e-6)
    data = []
    for r in subset:
        folder = root / f"{r['regime']}_{family}_d{d}_s{r['seed']}"
        with np.load(folder / r["npz"]) as a:
            # Restrict display to the common consecutive integer prefix. The
            # independent exact-rule diagnostic remains in raw results.
            ms = a["ms"].copy()
            contiguous = np.r_[True, np.diff(ms) == 1]
            observed = a["observed_error"]
            hp_file = folder / (Path(r["npz"]).stem + "_hp.json")
            if hp_file.exists():
                observed = np.asarray(json.loads(hp_file.read_text())["observed_error"])
            data.append((ms[contiguous], observed[contiguous], a["certified_bound"][contiguous]))
    last = min(int(v[0][-1]) for v in data)
    errors = np.array([v[1][:last] for v in data])
    bounds = np.array([v[2][:last] for v in data])
    return subset, np.arange(1, last+1), errors, bounds


def convergence(root, rows, d, curve_rows):
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.8))
    for ax, family, letter in zip(axs.flat, FAMILIES, "abcd"):
        subset, ms, errors, bounds = load_curves(root, rows, family, d)
        for name, matrix, color, ls in (("observed", errors, OBS, "-"), ("certified", bounds, CERT, "--")):
            q = np.quantile(matrix, [.1, .5, .9], axis=0)
            # Zero theoretical truncation error at exactness is undefined on
            # logarithmic axes: omit that zero, do not replace by a fake value.
            valid = q[1] > 0
            ax.plot(ms[valid], q[1, valid], color=color, ls=ls)
            ax.fill_between(ms[valid], np.maximum(q[0, valid], 1e-300), q[2, valid], color=color, alpha=.13, lw=0)
            for j, m in enumerate(ms):
                curve_rows.append(dict(family=family, d=d, m_q=int(m), quantity=name,
                                       p10=float(q[0,j]), median=float(q[1,j]), p90=float(q[2,j]), n=len(matrix)))
        floor = float(np.median([r["roundoff_diagnostic"] for r in subset]))
        if floor > 0:
            ax.axhspan(1e-18, floor, color=".94", zorder=-5)
            ax.axhline(floor, color=".6", ls=":", lw=.7)
        ax.axhline(1e-6, color=".15", ls=(0, (5, 3)), lw=.8)
        ax.text(.985, 1e-6, r" $10^{-6}$", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=7, color=".15")
        ax.set_yscale("log")
        high = 10 ** np.ceil(np.log10(max(np.quantile(bounds[:, 0], .9), 1e-4)))
        ax.set_ylim(1e-18, high * 3)
        ax.set_xlim(.7, ms[-1] + .03 * (ms[-1] - 1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_title(f"({letter}) {LABELS[family]}", loc="left")
        ax.set_ylabel("Maximum absolute error")
        ax.set_xlabel(r"Quadrature nodes $m_q$")
        ax.grid(axis="y", color=".92", lw=.5)
    fig.suptitle(rf"$d={d}$ features | 100 held-out explanations per model", y=.99,
                 fontsize=9, fontweight="bold")
    legend_budget(fig, y=.952)
    fig.subplots_adjust(left=.10, right=.99, bottom=.105, top=.845, wspace=.31, hspace=.57)
    return fig


def controls(rows):
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.9))
    families = FAMILIES[1:]
    for col, family in enumerate(families):
        ax = axs[0,col]
        for regime, ls in (("fixed_feature", "-"), ("fixed_total", ":")):
            subset = select(rows, family=family, regime=regime, value_function="baseline", epsilon=1e-6)
            budget_curve(ax, subset, "m_certified", CERT, ls, band=False)
            budget_curve(ax, subset, "m_observed", OBS, ls, band=False)
        ax.set_title(LABELS[family].replace(" regression", "").replace("Gaussian Naive Bayes", "Naive Bayes"), fontsize=8.5, loc="left")
        format_budget(ax, log=False)
        ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel("Nodes (baseline)")
        ax = axs[1,col]
        for key, color, offset in (("m_observed", OBS, -.08), ("m_certified", CERT, .08)):
            for x, vf in enumerate(("baseline", "interventional")):
                subset = select(rows, family=family, regime="fixed_feature", value_function=vf, d=1000, epsilon=1e-6)
                q = quantiles(subset, key)
                ax.errorbar(x+offset, q[1], yerr=[[q[1]-q[0]], [q[2]-q[1]]],
                            fmt="o", color=color, ms=4, capsize=3)
        ax.set_xticks([0,1], ["Zero baseline", "8-point\nbackground"])
        ax.set_xlim(-.45,1.45)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=".90", lw=.5)
        if col == 0:
            ax.set_ylabel(r"Nodes at $d=1000$")
    # Explicit colored keys avoid asking readers to combine two different legends.
    handles = [Line2D([], [], color=color, ls=ls, label=f"{name}: {signal}")
               for color, name in ((OBS, "Observed"), (CERT, "Certified"))
               for ls, signal in (("-", "fixed per-feature"), (":", "fixed total"))]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(.52,1),
               frameon=False, fontsize=8, handlelength=2.8)
    fig.subplots_adjust(left=.08,right=.99,bottom=.08,top=.81,wspace=.33,hspace=.53)
    return fig


def stress(root):
    rows = json.loads((root/"controlled_stress.json").read_text())
    fig, axs = plt.subplots(1,2,figsize=(7.1,2.85))
    colors = ["#009E73", "#0072B2", "#CC79A7", "#D55E00"]
    for r, color in zip((1.1,1.5,3.,10.), colors):
        subset = [v for v in rows if v["raw_factor"] == r]
        for ax, key in zip(axs, ("m_observed", "m_certified")):
            ax.plot(DIMENSIONS, [v[key] for v in subset], color=color, marker="o", ms=3, label=f"Factor ratio {r:g}")
    for ax, title in zip(axs, ("(a) Observed node budget", "(b) Certified node budget")):
        ax.set_title(title, loc="left", fontsize=9)
        ax.plot(DIMENSIONS, np.ceil(.3*np.array(DIMENSIONS)/2), color=GRAY, ls=":", lw=1.2,
                label="Active-feature exactness")
        format_budget(ax, log=False)
        ax.set_ylim(0,160)
        ax.set_ylabel(r"Nodes for $\varepsilon=10^{-6}$")
        ax.set_yticks([0,25,50,100,150])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.51,.995), fontsize=7.5)
    fig.subplots_adjust(left=.085,right=.99,bottom=.18,top=.75,wspace=.26)
    return fig


def summarize(root, rows):
    groups = sorted(set((r["regime"],r["family"],r["d"],r["value_function"],r["epsilon"]) for r in rows))
    summaries = []
    for regime, family, d, vf, eps in groups:
        values = select(rows, regime=regime, family=family, d=d, value_function=vf, epsilon=eps)
        s = dict(regime=regime,family=family,d=d,value_function=vf,epsilon=eps,n=len(values),n_seeds=len(set(r["seed"] for r in values)))
        for key in ("m_observed", "m_certified", "lambda_max", "A_max", "attribution_max", "reference_disagreement"):
            q = quantiles(values,key)
            s.update({f"{key}_{tag}":float(v) for tag,v in zip(("p10","median","p90"),q)})
        s.update(target_failures=sum(not r["target_met"] for r in values),
                 unresolved_references=sum(not r["reference_resolved"] for r in values),
                 max_error_at_certified=max(r["error_at_certified"] for r in values),
                 max_reference_disagreement=max(r["reference_disagreement"] for r in values),
                 m_certified_max=max(r["m_certified"] for r in values),
                 m_observed_max=max(r["m_observed"] or 0 for r in values))
        summaries.append(s)
    write_csv(root/"summary.csv", summaries)
    write_json(root/"summary.json", summaries)
    # Bootstrap independent training seeds, not the individual instances.
    rng=np.random.default_rng(20260916)
    growth=[]
    for family in FAMILIES:
        subset=select(rows,family=family,regime="fixed_feature",value_function="baseline",epsilon=1e-6)
        seeds=sorted(set(r["seed"] for r in subset))
        for key in ("m_observed","m_certified"):
            low=np.array([np.median([r[key] for r in select(subset,seed=s,d=50)]) for s in seeds])
            high=np.array([np.median([r[key] for r in select(subset,seed=s,d=1000)]) for s in seeds])
            difference=high-low
            boots=difference[rng.integers(0,len(seeds),(10000,len(seeds)))].mean(axis=1)
            growth.append(dict(family=family,quantity=key,mean_seed_median_d50=float(low.mean()),
                               mean_seed_median_d1000=float(high.mean()),mean_increase=float(difference.mean()),
                               bootstrap95_low=float(np.quantile(boots,.025)),bootstrap95_high=float(np.quantile(boots,.975)),
                               n_independent_seeds=len(seeds)))
    write_json(root/"seed_level_growth.json",growth)
    write_csv(root/"seed_level_growth.csv",growth)
    fit_rows=[]
    for f in root.glob("*/fit.json"):
        m=json.loads(f.read_text())
        fit_rows.append(dict(case=f.parent.name,metric=m["metric"],score=m["score"],n_train=m["n_train"],
                             warnings=" | ".join(m["convergence_warnings"]),iterations=str(m["n_iter"])))
    write_csv(root/"model_quality.csv",fit_rows)
    return summaries


def main(root, figures_only=False):
    source=root/"validated_records.json"
    if not source.exists():
        source=root/"records.json"
    rows=json.loads(source.read_text())
    figures=root/"figures"
    figures.mkdir(exist_ok=True)
    previews=figures/"backup"
    previews.mkdir(exist_ok=True)
    write_json(figures/"plot_settings.json", dict(style=STYLE,bands="10th-90th percentile over instances; not confidence intervals",
               main_epsilon=1e-6,observed_color=OBS,certified_color=CERT,
               convergence_display="common contiguous integer prefix; all raw points retained; y minimum 1e-18",
               scaling_layout="1 row x 4 columns; shared #d label; shared left node axis",
               png_directory="backup"))
    summaries=json.loads((root/"summary.json").read_text()) if figures_only else summarize(root,rows)
    curve_rows=[]
    with PdfPages(figures/"paper_figures.pdf",metadata={"Title":"QuadraSHAP synthetic node-budget study"}) as bundle:
        jobs=[("node_budget_scaling",lambda:scaling(rows))]
        jobs += [(f"convergence_d{d}",lambda d=d:convergence(root,rows,d,curve_rows)) for d in DIMENSIONS]
        jobs += [("controls_and_background",lambda:controls(rows)),("controlled_stress",lambda:stress(root))]
        for name,make in jobs:
            fig=make()
            fig.canvas.draw()
            extent=fig.get_tightbbox(fig.canvas.get_renderer())
            width, height=fig.get_size_inches()
            if extent.x0 < -.01 or extent.y0 < -.01 or extent.x1 > width+.01 or extent.y1 > height+.01:
                raise ValueError(f"Figure text extends outside the PDF page: {name}: {extent.bounds}")
            fig.savefig(figures/f"{name}.pdf")
            fig.savefig(previews/f"{name}.png")
            bundle.savefig(fig)
            plt.close(fig)
    write_csv(figures/"convergence_plot_data.csv",curve_rows)
    print("Main d=1000 summaries:")
    for s in summaries:
        if s["regime"]=="fixed_feature" and s["epsilon"]==1e-6 and s["d"]==1000:
            print({k:s[k] for k in ("family","value_function","m_observed_median","m_certified_median","m_certified_max","target_failures","unresolved_references")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--figures-only", action="store_true", help="Preserve all saved statistical summaries")
    args=p.parse_args()
    main(Path(args.output).resolve(), figures_only=args.figures_only)
