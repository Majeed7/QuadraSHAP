"""Merge matched interventional records and draw a paper-width PDF figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
SOURCE = RESULTS / "exp14_interventional_metal_precision"
OUT = SOURCE / "summary"
DATASETS = ("rotten_tomatoes", "sst2", "sms_spam", "emotion")
LABELS = {"rotten_tomatoes": "RT", "sst2": "SST-2", "sms_spam": "SMS", "emotion": "Emotion"}
METHODS = ("Q 1e-03", "Q 1e-05", "Q 1e-10", "Q 1e-16", "KernelSHAP", "SamplingSHAP")
COLORS = {"Q": "#007B82", "KernelSHAP": "#B14C1D", "SamplingSHAP": "#6552A4"}


def combine():
    rows = []
    for dataset in DATASETS:
        quad = pd.read_csv(SOURCE / dataset / "records.csv")
        if len(quad) != 80 or quad.groupby("requested_eps").size().to_list() != [20] * 4:
            raise ValueError(f"incomplete QuadraSHAP records: {dataset}")
        for _, record in quad.iterrows():
            ref = json.loads((SOURCE / dataset / "raw" /
                             f"reference_instance{int(record.instance):03d}.json").read_text())
            eps = float(record.requested_eps)
            rows.append(dict(dataset=dataset, instance=int(record.instance),
                method=f"Q {eps:.0e}", requested_eps=eps, m_q=int(record.m_q),
                max_abs_error=float(record.observed_max_abs_error_to_float64_reference),
                l2_error=float(record.observed_l2_error_to_float64_reference),
                seconds_core=float(record.seconds_core),
                seconds_end_to_end=float(record.seconds_core + ref["seconds_summary_and_budget"]),
                certified_quadrature_bound=float(record.certified_quadrature_bound),
                efficiency_residual=float(record.efficiency_residual),
                reference_extra_node_gap=float(record.reference_extra_node_gap),
                value_function="30-background interventional",
                phi_path=record.phi_path, reference_phi_path=record.reference_phi_path))
        for instance in range(20):
            ref_path = SOURCE / dataset / "raw" / f"reference_instance{instance:03d}.npz"
            phi_ref = np.load(ref_path)["phi"]
            for method in ("kernel", "sampling"):
                base = RESULTS / f"exp12_shap_background_{dataset}" / "raw"
                meta = json.loads((base / f"{method}_instance{instance:03d}.json").read_text())
                if meta["status"] != "complete":
                    raise ValueError(f"incomplete SHAP baseline {dataset}/{instance}/{method}")
                phi_path = base / f"{method}_instance{instance:03d}.npz"
                phi = np.load(phi_path)["phi"]
                delta = phi - phi_ref
                rows.append(dict(dataset=dataset, instance=instance,
                    method="KernelSHAP" if method == "kernel" else "SamplingSHAP",
                    requested_eps=np.nan, m_q=np.nan,
                    max_abs_error=float(np.max(np.abs(delta))),
                    l2_error=float(np.linalg.norm(delta)),
                    seconds_core=float(meta["seconds_explain"]),
                    seconds_end_to_end=float(meta["seconds_end_to_end"]),
                    certified_quadrature_bound=np.nan,
                    efficiency_residual=float(meta["efficiency_residual"]),
                    reference_extra_node_gap=np.nan,
                    value_function="30-background interventional",
                    phi_path=str(phi_path), reference_phi_path=str(ref_path)))
    frame = pd.DataFrame(rows)
    if len(frame) != 480:
        raise ValueError("expected 480 method-instance rows")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "all_instances.csv", index=False)
    summary = frame.groupby(["dataset", "method"], sort=False).agg(
        n=("instance", "size"), median_nodes=("m_q", "median"),
        median_max_abs_error=("max_abs_error", "median"),
        worst_max_abs_error=("max_abs_error", "max"),
        median_seconds_end_to_end=("seconds_end_to_end", "median"),
        median_seconds_core=("seconds_core", "median"),
        worst_efficiency_residual=("efficiency_residual", "max"),
        max_certified_bound=("certified_quadrature_bound", "max"),
        max_reference_extra_node_gap=("reference_extra_node_gap", "max")
    ).reset_index()
    summary.to_csv(OUT / "method_summary.csv", index=False)
    return frame


def draw(frame):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
        "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(7.18, 3.65), sharey="row",
                             gridspec_kw={"height_ratios": [1.05, 1.0]})
    rng = np.random.default_rng(20260917)
    for col, dataset in enumerate(DATASETS):
        part = frame[frame.dataset == dataset]
        for row, metric in enumerate(("max_abs_error", "seconds_end_to_end")):
            ax = axes[row, col]
            for j, method in enumerate(METHODS):
                values = part[part.method == method][metric].to_numpy(dtype=float)
                values = np.maximum(values, 1e-15)
                color = COLORS["Q"] if method.startswith("Q ") else COLORS[method]
                jitter = rng.uniform(-0.16, 0.16, len(values))
                ax.scatter(j + jitter, values, s=6.5, alpha=.20, color=color,
                           edgecolors="none", rasterized=True, zorder=2)
                lo, hi = np.quantile(values, [.25, .75])
                ax.plot([j, j], [lo, hi], color=color, lw=2.0, zorder=4)
                ax.plot(j, np.median(values), marker="D", markersize=3.6,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=.5, zorder=5)
            ax.axvline(3.5, color=".78", lw=.6)
            ax.set_yscale("log")
            ax.set_xlim(-.45, 5.45)
            ax.grid(axis="y", which="major", color=".88", lw=.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=6, width=.6, length=2.5)
            if row == 0:
                ax.set_ylim(1e-9, 1)
                ax.set_title(LABELS[dataset], fontsize=8, pad=5)
                ax.set_xticks([])
            else:
                ax.set_ylim(5e-2, 3e2)
                ax.set_xticks(range(6), [r"$10^{-3}$", r"$10^{-5}$", r"$10^{-10}$",
                                             r"$10^{-16}$", "Kernel", "Sampling"],
                              rotation=55, ha="right")
                ax.tick_params(axis="x", pad=1)
            if col:
                ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylabel("Max attribution error", fontsize=7)
    axes[1, 0].set_ylabel("Time per text (s)", fontsize=7)
    fig.text(.50, .005, "QuadraSHAP requested tolerance                 SHAP baseline",
             ha="center", fontsize=7, fontweight="bold")
    fig.subplots_adjust(left=.085, right=.995, top=.92, bottom=.23, wspace=.10, hspace=.14)
    fig.savefig(OUT / "interventional_comparison.pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(7.18, 2.15), sharey=True)
    for col, dataset in enumerate(DATASETS):
        ax = axes[col]
        part = frame[(frame.dataset == dataset) & frame.method.str.startswith("Q ")]
        certs, observed = [], []
        for method in METHODS[:4]:
            values = part[part.method == method]
            certs.append(float(values.certified_quadrature_bound.median()))
            observed.append(float(values.max_abs_error.median()))
        ax.plot(range(4), certs, color="#B14C1D", marker="s", markersize=3.8,
                ls="--", lw=1.3, label="Certified quadrature bound")
        ax.plot(range(4), observed, color="#007B82", marker="D", markersize=3.8,
                lw=1.3, label="Observed Metal error")
        ax.set_title(LABELS[dataset], fontsize=8, pad=5)
        ax.set_yscale("log")
        ax.set_ylim(1e-20, 1e-2)
        ax.set_xlim(-.18, 3.18)
        ax.set_xticks(range(4), [r"$10^{-3}$", r"$10^{-5}$",
                                   r"$10^{-10}$", r"$10^{-16}$"], rotation=40, ha="right")
        ax.grid(axis="y", which="major", color=".88", lw=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=6.5, width=.6, length=2.5)
        if col:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("Maximum-coordinate error", fontsize=7)
    fig.text(.5, .015, "Requested quadrature tolerance", ha="center",
             fontsize=7, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.08),
               ncol=2, fontsize=7, frameon=False, handlelength=2)
    fig.subplots_adjust(left=.085, right=.995, top=.81, bottom=.27, wspace=.10)
    fig.savefig(OUT / "interventional_precision_limit.pdf",
                bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    draw(combine())
