"""Summarize and plot saved 30-background SHAP runs without rerunning explainers."""
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
FIG = ROOT / "results" / "exp12_shap_background_figures"
DATASETS = (
    ("rotten_tomatoes", "Rotten Tomatoes"),
    ("sst2", "SST-2"),
    ("sms_spam", "SMS spam"),
    ("emotion", "Emotion (OVR)"),
)
BLUE, ORANGE = "#0072B2", "#D55E00"


def read_results():
    outputs = {}
    paired = []
    summary = []
    for key, title in DATASETS:
        base = ROOT / "results" / f"exp12_shap_background_{key}"
        frame = pd.read_csv(base / "records.csv")
        for method in ("kernel", "sampling"):
            sub = frame[frame.method_key == method]
            if len(sub) != 20 or (sub.status != "complete").any():
                raise ValueError(f"Incomplete {key}/{method}: {len(sub)} rows")
        differences = []
        for instance in range(20):
            kernel = np.load(base / "raw" / f"kernel_instance{instance:03d}.npz")["phi"]
            sampling = np.load(base / "raw" / f"sampling_instance{instance:03d}.npz")["phi"]
            delta = kernel - sampling
            worst = float(np.max(np.abs(delta)))
            differences.append(worst)
            paired.append({
                "dataset": key,
                "instance": instance,
                "max_abs_method_disagreement": worst,
                "l2_method_disagreement": float(np.linalg.norm(delta)),
                "interpretation": "disagreement, not error to an exact reference",
            })
        for method in ("kernel", "sampling"):
            sub = frame[frame.method_key == method]
            summary.append({
                "dataset": key,
                "method": method,
                "n_background": 30,
                "requested_nsamples": 1000,
                "n_instances": len(sub),
                "n_complete": int((sub.status == "complete").sum()),
                "n_finite": int(sub.finite.sum()),
                "median_seconds_end_to_end": float(sub.seconds_end_to_end.median()),
                "max_seconds_end_to_end": float(sub.seconds_end_to_end.max()),
                "median_model_rows": float(sub.actual_model_rows.median()),
                "median_nonzero_attributions": float(sub.n_nonzero_phi.median()),
                "worst_efficiency_residual": float(sub.efficiency_residual.max()),
                "median_max_abs_method_disagreement": float(np.median(differences)),
                "worst_max_abs_method_disagreement": float(np.max(differences)),
                "observed_error_to_exact_interventional_reference": np.nan,
                "certified_bound": np.nan,
            })
        outputs[key] = (title, frame)
    return outputs, pd.DataFrame(summary), pd.DataFrame(paired)


def plot_runtime(outputs):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.1, "axes.labelsize": 8.5,
        "axes.titlesize": 9.2, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), sharey=True)
    fig.suptitle("Interventional SHAP: 1,000 samples, 30 backgrounds, 20 texts",
                 fontsize=10.0, fontweight="bold", y=.991)
    for panel, (key, _) in enumerate(DATASETS):
        row, col = divmod(panel, 2)
        ax = axes[row, col]
        title, frame = outputs[key]
        kernel = frame[frame.method_key == "kernel"].sort_values("instance")
        sampling = frame[frame.method_key == "sampling"].sort_values("instance")
        tk = kernel.seconds_end_to_end.to_numpy(float)
        ts = sampling.seconds_end_to_end.to_numpy(float)
        for a, b in zip(tk, ts):
            ax.plot([0, 1], [a, b], color="#9c9c9c", alpha=.24, lw=.7, zorder=1)
        jitter = np.linspace(-.055, .055, len(tk))
        ax.scatter(jitter, tk, s=10, color=BLUE, alpha=.55, edgecolors="none", zorder=2)
        ax.scatter(1 + jitter, ts, s=10, color=ORANGE, alpha=.55, edgecolors="none", zorder=2)
        for x, values, color in ((0, tk, BLUE), (1, ts, ORANGE)):
            ax.plot([x-.14, x+.14], [np.median(values)] * 2, lw=2.4,
                    color=color, solid_capstyle="round", zorder=3)
        ax.set_title(title, fontweight="bold", pad=5)
        ax.set_xticks([0, 1], ["KernelSHAP", "SamplingSHAP"])
        ax.set_xlim(-.24, 1.24)
        ax.set_yscale("log")
        ax.set_ylim(.15, 100)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color="#d0d0d0", alpha=.6, lw=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel("End-to-end time per text (s)", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=0, label="KernelSHAP: median 30,032 model rows"),
        Line2D([0], [0], color=ORANGE, marker="o", lw=0, label="SamplingSHAP: median 2,032 model rows"),
        Line2D([0], [0], color="#555555", lw=2.4, label="Median of 20 texts"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .004), fontsize=7.0)
    fig.subplots_adjust(left=.115, right=.985, top=.91, bottom=.14,
                        wspace=.19, hspace=.28)
    fig.savefig(FIG / "text_shap_background_runtime.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    outputs, summary, paired = read_results()
    summary.to_csv(FIG / "method_summary.csv", index=False)
    paired.to_csv(FIG / "pairwise_disagreement.csv", index=False)
    plot_runtime(outputs)
    print(f"Saved summary, disagreement diagnostics and runtime PDF in {FIG}")


if __name__ == "__main__":
    main()
