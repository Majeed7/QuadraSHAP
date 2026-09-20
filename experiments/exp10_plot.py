"""Render paper-ready figures from the saved full-IMDB experiment; no reruns."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results" / "exp10_pkex_metal_imdb"
FIG = OUT / "figures"
EPS = np.array([1e-3, 1e-5, 1e-10, 1e-16])
X = np.arange(4)
TICKS = [r"$10^{-3}$", r"$10^{-5}$", r"$10^{-10}$", r"$10^{-16}$"]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#696969"


def format_axes(ax, title, xlabel=True):
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(-0.17, 3.22)
    ax.set_xticks(X, TICKS)
    if xlabel:
        ax.set_xlabel(r"Requested tolerance $\varepsilon$", fontweight="bold")
    ax.grid(axis="y", which="major", lw=.45, alpha=.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def values(group, col):
    return np.array([group.loc[np.isclose(group.eps, e, rtol=1e-12, atol=0), col].iloc[0] for e in EPS])


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.labelsize": 8.7, "axes.titlesize": 9, "legend.fontsize": 7.1,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    FIG.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(OUT / "records.csv")
    done = records[records.status == "complete"]
    assert len(done) == 160 and (records[records.method_key == "pkex"].status == "timeout").sum() == 20
    aggregate = done.groupby(["method_key", "eps"], as_index=False).agg(
        observed=("observed_max_abs_error", "max"),
        bound=("certified_bound", "max"),
        seconds=("seconds_end_to_end", "median"),
        efficiency=("efficiency_residual", "max"))
    cpu = aggregate[aggregate.method_key == "cpu"]
    metal = aggregate[aggregate.method_key == "metal"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.24), gridspec_kw={"wspace": .34})
    ax.plot(X, EPS, ls=":", lw=1.05, color="#222222", label="Requested tolerance")
    ax.plot(X, values(cpu, "bound"), "s--", lw=1.35, ms=4, color=GREY, label="Certified bound")
    ax.plot(X[:3], values(cpu, "observed")[:3], "o-", lw=1.65, ms=4.5,
            color=BLUE, label="CPU observed")
    ax.plot(X, values(metal, "observed"), "D-", lw=1.65, ms=4.1,
            color=ORANGE, label="Metal observed")
    ax.scatter(3, 1.8e-18, marker="v", color=BLUE, s=32)
    ax.annotate("CPU reference (self-error = 0)", (3, 1.8e-18), (1.25, 1.3e-17),
                color=BLUE, fontsize=6.5, arrowprops={"arrowstyle": "-", "color": BLUE, "lw": .65})
    ax.set_yscale("log"); ax.set_ylim(5e-19, 4e-3)
    ax.set_ylabel(r"Worst $|\hat\phi_i-\phi_i^{\rm ref}|$", fontweight="bold")
    format_axes(ax, "(a) Certified versus observed")
    ax.legend(loc="lower left", frameon=False, labelspacing=.3)

    bx.plot(X, values(cpu, "seconds"), "o-", lw=1.65, ms=4.5, color=BLUE,
            label="QuadraSHAP CPU")
    bx.plot(X, values(metal, "seconds"), "D-", lw=1.65, ms=4.1, color=ORANGE,
            label="QuadraSHAP Metal")
    bx.axhline(300, color=GREY, ls="--", lw=1.1)
    bx.text(0.05, 310, "PKeX: 20/20 timeouts", color=GREY, fontsize=7.1, fontweight="bold")
    bx.set_yscale("log"); bx.set_ylim(2, 800)
    bx.set_yticks([3, 5, 10, 30, 100, 300], ["3", "5", "10", "30", "100", "300"])
    bx.set_ylabel("Median time per review (s)", fontweight="bold")
    format_axes(bx, "(b) End-to-end time")
    bx.legend(loc="center right", frameon=False)
    fig.suptitle("Full IMDB reviews | 5,000 TF-IDF features | 20 test instances",
                 y=.985, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=.085, right=.99, top=.86, bottom=.19)
    fig.savefig(FIG / "imdb_accuracy_runtime.pdf", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.24), gridspec_kw={"wspace": .35})
    for method, color, marker, label in (("cpu", BLUE, "o", "CPU float64"),
                                         ("metal", ORANGE, "D", "Metal float32")):
        sel = done[done.method_key == method]
        series = [sel[np.isclose(sel.eps, e, rtol=1e-12, atol=0)].observed_max_abs_error.to_numpy(float)
                  for e in EPS]
        low = np.array([v.min() for v in series])
        med = np.array([np.median(v) for v in series])
        high = np.array([v.max() for v in series])
        count = 3 if method == "cpu" else 4
        ax.fill_between(X[:count], low[:count], high[:count], color=color, alpha=.16, linewidth=0)
        ax.plot(X[:count], med[:count], color=color, marker=marker, lw=1.75, ms=4.5, label=label)
    ax.plot(X, EPS, ls=":", color="#222222", lw=1, label="Requested tolerance")
    ax.scatter(3, 1.2e-15, marker="v", color=BLUE, s=32)
    ax.text(2.1, 1.8e-15, "CPU reference = 0", color=BLUE, fontsize=6.6)
    ax.set_yscale("log"); ax.set_ylim(5e-16, 4e-3)
    ax.set_ylabel("Observed max absolute error", fontweight="bold")
    format_axes(ax, "(a) Spread over reviews")
    ax.legend(loc="center left", frameon=False)

    bx.plot(X, values(cpu, "efficiency"), "o-", color=BLUE, lw=1.75,
            ms=4.5, label="CPU float64")
    bx.plot(X, values(metal, "efficiency"), "D-", color=ORANGE, lw=1.75,
            ms=4.1, label="Metal float32")
    bx.set_yscale("log"); bx.set_ylim(5e-13, 8e-5)
    bx.set_ylabel(r"Worst efficiency residual $|\sum_i\hat\phi_i-\Delta f|$", fontweight="bold")
    format_axes(bx, "(b) Floating-point stability")
    bx.legend(loc="center right", frameon=False)
    fig.suptitle("Numerical stability across 20 full-length IMDB reviews",
                 y=.985, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=.085, right=.99, top=.86, bottom=.19)
    fig.savefig(FIG / "imdb_numerical_stability.pdf", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)
    print(f"Saved two PDF figures in {FIG}")


if __name__ == "__main__":
    main()
