"""Summarize the manuscript-dimension synthetic benchmark and draw PDF figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "results" / "exp15_synthetic_pkex_metal_precision"
DIMS = (50, 500, 1000, 2000, 5000)
EPS = (1e-3, 1e-5, 1e-10, 1e-16)
COLORS = ("#137F86", "#25709F", "#635BA7", "#944D82")


def load():
    frames = [pd.read_csv(SOURCE / f"d{d}" / "records.csv") for d in DIMS]
    all_rows = pd.concat(frames, ignore_index=True)
    counts = all_rows[all_rows.method == "QuadraSHAP Metal"].groupby(["d", "requested_eps"]).size()
    if any(counts.get((d, eps), 0) != 50 for d in DIMS for eps in EPS):
        raise ValueError("incomplete Metal sweep")
    pkex = all_rows[all_rows.method == "PKeX-Shapley"]
    if any(len(pkex[(pkex.d == d) & (pkex.status == "complete")]) != 50 for d in DIMS[:3]):
        raise ValueError("incomplete PKeX baseline; rerun aggregation when it finishes")
    if any((pkex.d > 1000)):
        raise ValueError("PKeX must be omitted at 2000 and 5000 features")
    summary = all_rows[all_rows.method == "QuadraSHAP Metal"].groupby(
        ["d", "requested_eps"], as_index=False).agg(
            n=("instance", "size"), median_nodes=("m_q", "median"),
            min_nodes=("m_q", "min"), max_nodes=("m_q", "max"),
            median_certified_bound=("certified_quadrature_bound", "median"),
            worst_certified_bound=("certified_quadrature_bound", "max"),
            median_observed_error=("observed_max_abs_error_to_float64_reference", "median"),
            worst_observed_error=("observed_max_abs_error_to_float64_reference", "max"),
            median_seconds=("seconds_end_to_end", "median"),
            q25_seconds=("seconds_end_to_end", lambda x: np.quantile(x, .25)),
            q75_seconds=("seconds_end_to_end", lambda x: np.quantile(x, .75)),
            worst_efficiency_residual=("efficiency_residual", "max"),
            worst_reference_extra_node_gap=("reference_extra_node_gap", "max"))
    baseline = pkex.groupby("d", as_index=False).agg(
        n=("instance", "size"), median_seconds=("seconds_end_to_end", "median"),
        q25_seconds=("seconds_end_to_end", lambda x: np.quantile(x, .25)),
        q75_seconds=("seconds_end_to_end", lambda x: np.quantile(x, .75)),
        median_observed_error=("observed_max_abs_error_to_float64_reference", "median"),
        worst_observed_error=("observed_max_abs_error_to_float64_reference", "max"),
        worst_efficiency_residual=("efficiency_residual", "max"))
    out = SOURCE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out / "all_instances.csv", index=False)
    summary.to_csv(out / "metal_summary.csv", index=False)
    baseline.to_csv(out / "pkex_summary.csv", index=False)
    return summary, baseline


def draw(summary, baseline):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
        "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.18, 2.72))
    x = np.arange(len(DIMS))
    for k, eps in enumerate(EPS):
        part = summary[np.isclose(summary.requested_eps, eps, rtol=1e-12, atol=0)].sort_values("d")
        if len(part) != 5:
            raise ValueError(eps)
        y = part.median_seconds.to_numpy()
        low = y - part.q25_seconds.to_numpy()
        high = part.q75_seconds.to_numpy() - y
        axes[0].errorbar(x, y, yerr=[low, high], color=COLORS[k], marker="o",
                         markersize=4, lw=1.5, capsize=2.2,
                         label=rf"QuadraSHAP $\varepsilon=10^{{{int(np.log10(eps))}}}$")
        axes[1].plot(x, part.median_nodes, color=COLORS[k], marker="o",
                     markersize=4, lw=1.5)
    b = baseline.sort_values("d")
    y = b.median_seconds.to_numpy()
    axes[0].errorbar(x[:3], y, yerr=[y-b.q25_seconds.to_numpy(),
                                      b.q75_seconds.to_numpy()-y],
                     color="#B35021", marker="s", markersize=4.2, lw=1.5,
                     ls="--", capsize=2.2, label="PKeX-Shapley")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-3, 1e3)
    axes[0].set_ylabel("Time per explanation (s)")
    axes[1].set_ylabel(r"Certified node budget $m_q$")
    axes[1].set_ylim(0, 13)
    for ax in axes:
        ax.set_xlim(-.15, 4.15)
        ax.set_xticks(x, ["50", "500", "1k", "2k", "5k"])
        ax.set_xlabel("Number of features $d$")
        ax.grid(axis="y", color=".86", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, width=.7, length=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.025),
               ncol=5, fontsize=6.0, frameon=False, handlelength=1.7,
               columnspacing=.8)
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.81, wspace=.28)
    fig.savefig(SOURCE / "summary" / "synthetic_runtime_nodes.pdf",
                bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for k, eps in enumerate(EPS):
        part = summary[np.isclose(summary.requested_eps, eps, rtol=1e-12, atol=0)].sort_values("d")
        ax.plot(x, part.median_observed_error, color=COLORS[k], marker="o",
                markersize=4, lw=1.5,
                label=rf"QuadraSHAP $\varepsilon=10^{{{int(np.log10(eps))}}}$")
    ax.plot(x[:3], b.median_observed_error, color="#B35021", marker="s",
            ls="--", lw=1.5, markersize=4.2, label="PKeX-Shapley")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-3)
    ax.set_xlim(-.15, 4.15)
    ax.set_xticks(x, ["50", "500", "1k", "2k", "5k"])
    ax.set_xlabel("Number of features $d$")
    ax.set_ylabel("Median maximum attribution error")
    ax.grid(axis="y", color=".86", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.025),
               ncol=3, fontsize=6.0, frameon=False, handlelength=1.7,
               columnspacing=.7)
    fig.subplots_adjust(left=.15, right=.99, bottom=.17, top=.77)
    fig.savefig(SOURCE / "summary" / "synthetic_accuracy.pdf",
                bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


if __name__ == "__main__":
    draw(*load())
