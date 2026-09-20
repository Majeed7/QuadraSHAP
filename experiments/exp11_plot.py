"""Regenerate publication figures for the four non-IMDB text benchmarks.

This reads saved measurements only. It never retrains a classifier or reruns an
explainer. The four panels always appear in the same order across figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "exp11_text_classifiers_figures"
DATASETS = (
    ("rotten_tomatoes", "Rotten Tomatoes"),
    ("sst2", "SST-2"),
    ("sms_spam", "SMS spam"),
    ("emotion", "Emotion (OVR)"),
)
EPS = np.array([1e-3, 1e-5, 1e-10, 1e-16])
X = np.arange(len(EPS))
XTICKS = [r"$10^{-3}$", r"$10^{-5}$", r"$10^{-10}$", r"$10^{-16}$"]
BLUE, ORANGE, GREY, DARK = "#0072B2", "#D55E00", "#737373", "#222222"


def load_records():
    result = {}
    for key, title in DATASETS:
        path = ROOT / "results" / f"exp11_pkex_metal_{key}" / "records.csv"
        frame = pd.read_csv(path)
        for method in ("cpu", "metal"):
            rows = frame[(frame.method_key == method) & (frame.status == "complete")]
            if len(rows) != 20 * len(EPS):
                raise ValueError(f"Incomplete {key}/{method}: {len(rows)} of 80 rows")
        baseline_count = len(frame[frame.method_key == "pkex"])
        if not 1 <= baseline_count <= 20:
            raise ValueError(f"Invalid {key}/pkex status count: {baseline_count}")
        result[key] = (title, frame)
    return result


def series(frame, method, column, reduction):
    rows = frame[(frame.method_key == method) & (frame.status == "complete")]
    return np.asarray([
        reduction(rows[np.isclose(rows.eps, eps, rtol=1e-12, atol=0)][column].to_numpy(float))
        for eps in EPS
    ])


def style_axis(ax, row, col, ylabel=None, log=True):
    ax.set_xlim(-.18, 3.18)
    ax.set_xticks(X, XTICKS)
    if row == 0:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel(r"Requested $\varepsilon$", fontweight="bold")
    if ylabel and col == 0:
        ax.set_ylabel(ylabel, fontweight="bold")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", which="major", color="#c9c9c9", alpha=.55, lw=.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def grid(title):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35), sharey=True)
    fig.suptitle(title, fontsize=10.3, fontweight="bold", y=.992)
    return fig, axes


def error_figure(data):
    fig, axes = grid("Certified quadrature bound and observed attribution error")
    for index, (key, _) in enumerate(DATASETS):
        row, col = divmod(index, 2)
        ax = axes[row, col]
        title, frame = data[key]
        ax.set_title(title, fontsize=9.1, fontweight="bold", pad=5)
        ax.plot(X, EPS, ls=":", lw=1.15, color=DARK)
        ax.plot(X, series(frame, "cpu", "certified_bound", np.max), "s--",
                color=GREY, lw=1.3, ms=3.7)
        cpu = series(frame, "cpu", "observed_max_abs_error", np.max)
        ax.plot(X[:3], cpu[:3], "o-", color=BLUE, lw=1.65, ms=4)
        ax.plot(X, series(frame, "metal", "observed_max_abs_error", np.max),
                "D-", color=ORANGE, lw=1.65, ms=3.7)
        ax.set_ylim(1e-16, 2e-2)
        style_axis(ax, row, col, r"Worst $|\hat\phi_i-\phi_i^{\rm ref}|$")
        cpu_rows = frame[(frame.method_key == "cpu") & (frame.status == "complete")]
        node_labels = []
        for eps in EPS:
            mq = cpu_rows[np.isclose(cpu_rows.eps, eps, rtol=1e-12, atol=0)].m_q
            low, high = int(mq.min()), int(mq.max())
            node_labels.append(str(low) if low == high else f"{low}-{high}")
        ax.text(.975, .94, "$m_q$: " + "/".join(node_labels),
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top", color=DARK)
        ax.text(.025, .045, "CPU at $10^{-16}$: reference",
                transform=ax.transAxes, fontsize=6.3, color=BLUE)
    handles = [
        Line2D([0], [0], color=DARK, ls=":", lw=1.2, label="Requested tolerance"),
        Line2D([0], [0], color=GREY, ls="--", marker="s", ms=3.7, label="Certified bound"),
        Line2D([0], [0], color=BLUE, marker="o", ms=4, label="CPU observed"),
        Line2D([0], [0], color=ORANGE, marker="D", ms=3.7, label="Metal observed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .018), fontsize=7.2)
    fig.subplots_adjust(left=.105, right=.985, top=.91, bottom=.15, wspace=.19, hspace=.26)
    fig.savefig(FIG / "text_certified_observed.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def runtime_figure(data):
    fig, axes = grid("Time per text: 5,000 features and a 300 s cap")
    for index, (key, _) in enumerate(DATASETS):
        row, col = divmod(index, 2)
        ax = axes[row, col]
        title, frame = data[key]
        ax.set_title(title, fontsize=9.1, fontweight="bold", pad=5)
        ax.plot(X, series(frame, "cpu", "seconds_end_to_end", np.median),
                "o-", color=BLUE, lw=1.65, ms=4)
        ax.plot(X, series(frame, "metal", "seconds_end_to_end", np.median),
                "D-", color=ORANGE, lw=1.65, ms=3.7)
        baseline = frame[frame.method_key == "pkex"]
        successes = baseline[baseline.status == "complete"]
        if len(successes):
            seconds = float(successes.seconds_end_to_end.median())
            ax.axhline(seconds, color=GREY, lw=1.2, ls="-.")
        ax.axhline(300, color=GREY, lw=1.05, ls="--")
        ax.text(.03, .98, f"PKeX {len(successes)}/{len(baseline)} completed",
                transform=ax.transAxes, ha="left", va="top", color=GREY,
                fontsize=7, fontweight="bold")
        ax.set_ylim(.2, 800)
        style_axis(ax, row, col, "Median time (s)")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", ms=4, label="QuadraSHAP CPU"),
        Line2D([0], [0], color=ORANGE, marker="D", ms=3.7, label="QuadraSHAP Metal"),
        Line2D([0], [0], color=GREY, ls="--", label="300 s limit"),
    ]
    if any((frame[(frame.method_key == "pkex") & (frame.status == "complete")]).shape[0]
           for _, frame in data.values()):
        handles.append(Line2D([0], [0], color=GREY, ls="-.", label="PKeX if completed"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(.5, .018), fontsize=7.2)
    fig.subplots_adjust(left=.105, right=.985, top=.91, bottom=.15, wspace=.19, hspace=.26)
    fig.savefig(FIG / "text_runtime.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def stability_figure(data):
    fig, axes = grid("Floating-point efficiency residual across 20 texts")
    for index, (key, _) in enumerate(DATASETS):
        row, col = divmod(index, 2)
        ax = axes[row, col]
        title, frame = data[key]
        ax.set_title(title, fontsize=9.1, fontweight="bold", pad=5)
        for method, color, marker in (("cpu", BLUE, "o"), ("metal", ORANGE, "D")):
            residual = series(frame, method, "efficiency_residual", np.max)
            ax.plot(X, np.maximum(residual, 1e-16), marker=marker,
                    color=color, lw=1.65, ms=4 if method == "cpu" else 3.7)
        ax.set_ylim(1e-15, 1e-4)
        style_axis(ax, row, col, r"Worst $|\sum_i\hat\phi_i-\Delta f|$")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", ms=4, label="CPU float64"),
        Line2D([0], [0], color=ORANGE, marker="D", ms=3.7, label="Metal float32"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .018), fontsize=7.2)
    fig.subplots_adjust(left=.105, right=.985, top=.91, bottom=.15, wspace=.19, hspace=.26)
    fig.savefig(FIG / "text_numerical_stability.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def write_all_five_summary(data):
    """Include prior IMDB data in a machine-readable cross-dataset table."""
    five = {"imdb": pd.read_csv(ROOT / "results" / "exp10_pkex_metal_imdb" / "records.csv")}
    five.update({key: frame for key, (_, frame) in data.items()})
    rows = []
    for dataset, frame in five.items():
        for method in ("cpu", "metal", "pkex"):
            eps_values = EPS if method != "pkex" else [np.nan]
            for eps in eps_values:
                selected = frame[frame.method_key == method]
                if method != "pkex":
                    selected = selected[np.isclose(selected.eps, eps, rtol=1e-12, atol=0)]
                complete = selected[selected.status == "complete"]
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "requested_eps": eps,
                    "n_instances": len(selected),
                    "n_completed": len(complete),
                    "n_timeout": int((selected.status == "timeout").sum()),
                    "median_nodes": complete.m_q.median() if method != "pkex" else np.nan,
                    "max_certified_bound": complete.certified_bound.max() if len(complete) else np.nan,
                    "worst_observed_error": complete.observed_max_abs_error.max() if len(complete) else np.nan,
                    "median_end_to_end_seconds": complete.seconds_end_to_end.median() if len(complete) else np.nan,
                    "worst_efficiency_residual": complete.efficiency_residual.max() if len(complete) else np.nan,
                })
    pd.DataFrame(rows).to_csv(FIG / "all_five_text_summary.csv", index=False)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.3,
        "axes.titlesize": 9.1, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    FIG.mkdir(parents=True, exist_ok=True)
    data = load_records()
    write_all_five_summary(data)
    error_figure(data)
    runtime_figure(data)
    stability_figure(data)
    print(f"Saved three figures in {FIG}")


if __name__ == "__main__":
    main()
