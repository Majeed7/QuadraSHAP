"""
plot_mq_results.py
==================
Standalone plotting script for the m_q convergence benchmark.
Reads mq_agg_d*_krr_rbf.csv from benchmarks/results/mq/ and produces:
  - one individual PDF+PNG per feature count
  - one combined multi-panel PDF+PNG

Run from the repo root:
    python benchmarks/plot_mq_results.py
"""

from __future__ import annotations
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RESULTS    = ROOT / "benchmarks" / "results" / "mq"
PLOT_DIR   = RESULTS          # figures go to the same folder
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── load all KRR aggregated CSVs ──────────────────────────────────────────────
files = sorted(glob.glob(str(RESULTS / "mq_agg_d*_krr_rbf.csv")))
frames = []
for f in files:
    frames.append(pd.read_csv(f))
df_all = pd.concat(frames, ignore_index=True)

feature_counts = sorted(df_all["n_features"].unique())
print(f"Feature counts found: {feature_counts}")

# Feature counts used in the combined figure (skip d=100)
COMBINED_FEATURES = [d for d in feature_counts if d != 100]

# ── publication rcParams ───────────────────────────────────────────────────────
FONT_SIZE  = 15
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          FONT_SIZE,
    "axes.titlesize":     FONT_SIZE + 1,
    "axes.labelsize":     FONT_SIZE,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "axes.linewidth":     1.4,
    "xtick.labelsize":    FONT_SIZE - 2,
    "ytick.labelsize":    FONT_SIZE - 2,
    "xtick.major.width":  1.3,
    "ytick.major.width":  1.3,
    "xtick.minor.width":  0.8,
    "ytick.minor.width":  0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "legend.fontsize":    FONT_SIZE - 2,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.65",
    "lines.linewidth":    2.2,
    "lines.markersize":   8,
    "figure.dpi":         150,
    "pdf.fonttype":       42,   # embed fonts in PDF
    "ps.fonttype":        42,
})

COLOR     = "#2166AC"    # deep blue (ColorBrewer RdBu)
COLOR_STD = "#92C5DE"    # light blue for std band
ZERO_CLR  = "#D6604D"    # red for the exact-zero annotation


def _bold_ticks(ax: mpl.axes.Axes) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _enforce_min_decades(ax: mpl.axes.Axes, min_decades: float = 2.0) -> None:
    """Ensure the y-axis spans at least `min_decades` on a log scale.

    The window is centered at the geometric mean of the current limits,
    so the data stays centred and the axis isn't pushed to a strange range.
    """
    lo, hi = ax.get_ylim()
    span = np.log10(hi) - np.log10(lo)
    if span < min_decades:
        center = 0.5 * (np.log10(lo) + np.log10(hi))
        ax.set_ylim(10 ** (center - min_decades / 2),
                    10 ** (center + min_decades / 2))


def _draw_panel(
    ax: mpl.axes.Axes,
    sub: pd.DataFrame,
    n_features: int,
    *,
    show_ylabel: bool = True,
) -> None:
    """Draw one m_q convergence panel onto ax."""
    mq_exact = int(sub["m_q_exact"].iloc[0])

    # Split zero and non-zero rows
    nz  = sub[sub["mean_shapley_distance"] > 0].sort_values("m_q")
    zer = sub[sub["mean_shapley_distance"] == 0]

    x   = nz["m_q"].to_numpy()
    y   = nz["mean_shapley_distance"].to_numpy()
    std = nz["std_shapley_distance"].to_numpy()

    # ── shaded std band ───────────────────────────────────────────────────────
    ax.fill_between(
        x,
        np.maximum(y - std, y * 0.05),   # floor at 5 % of mean for log scale
        y + std,
        color=COLOR_STD, alpha=0.45, zorder=2, label=r"$\pm 1\,\sigma$",
    )

    # ── mean line ─────────────────────────────────────────────────────────────
    ax.semilogy(
        x, y,
        color=COLOR, marker="o", zorder=3,
        markeredgecolor="white", markeredgewidth=0.7,
        label=r"Mean $\|\hat{\varphi}_{m_q} - \varphi^\star\|_2$",
    )

    # ── lock x-axis to the data range before drawing anything else ───────────
    x_min, x_max = x.min(), x.max()
    x_pad = (x_max - x_min) * 0.06
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    # ── enforce minimum y-axis span (2 decades) ───────────────────────────────
    _enforce_min_decades(ax, min_decades=2.0)

    # ── exact zero annotation (where distance = 0 exactly) ───────────────────
    # if not zer.empty:
    #     ylims = ax.get_ylim()
    #     log_lo, log_hi = np.log10(ylims[0]), np.log10(ylims[1])
    #     y_marker = 10 ** (log_lo + 0.12 * (log_hi - log_lo))
    #     for mq_val in zer["m_q"]:
    #         ax.plot(
    #             mq_val, y_marker, marker="v", color=ZERO_CLR,
    #             markersize=10, zorder=5, clip_on=False,
    #             label=rf"$= 0$ at $m_q = {int(mq_val)}$",
    #         )

    # ── reference line: draw only if m_q_exact is inside the data range ──────
    ylims   = ax.get_ylim()
    log_lo  = np.log10(ylims[0])
    log_hi  = np.log10(ylims[1])
    # if x_min <= mq_exact <= x_max:
    #     # m_q_exact falls within the sweep → draw vertical dashed line
    #     ax.axvline(mq_exact, color="#888888", linestyle="--",
    #                linewidth=1.3, zorder=1)
    #     ax.text(
    #         mq_exact * 1.02,
    #         10 ** (log_lo + 0.85 * (log_hi - log_lo)),
    #         rf"$m_q^\star\!=\!{mq_exact}$",
    #         color="#555555", fontsize=FONT_SIZE - 3, fontweight="bold",
    #         va="top", ha="left",
    #     )
    # else:
    #     # m_q_exact is far outside the sweep → just annotate in the corner
    #     ax.text(
    #         0.97, 0.95,
    #         rf"$m_q^\star = {mq_exact:,}$",
    #         transform=ax.transAxes,
    #         color="#555555", fontsize=FONT_SIZE - 3, fontweight="bold",
    #         va="top", ha="right",
    #         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.8),
    # )

    # ── formatting ────────────────────────────────────────────────────────────
    ax.set_title(rf"$d = {n_features:,}$", pad=5)
    ax.set_xlabel(r"$m_q$", labelpad=3)
    if show_ylabel:
        ax.set_ylabel(r"$\ell_2$ distance", labelpad=4)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=True)   # keep tick numbers, just no axis label
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which="major", linestyle="-",  linewidth=0.5, color="0.88", zorder=0)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.4, color="0.92", zorder=0)
    ax.tick_params(which="both", top=True, right=True)

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate entries (fill_between + line + optional zero marker)
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    leg = ax.legend(h2, l2, loc="upper right", handlelength=1.6,
                    frameon=True, borderpad=0.5, labelspacing=0.3)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    _bold_ticks(ax)


# ═══════════════════════════════════════════════════════════════════════════════
#  A) Individual single-panel figures
# ═══════════════════════════════════════════════════════════════════════════════
for n_features in feature_counts:
    sub = df_all[df_all["n_features"] == n_features].copy()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    _draw_panel(ax, sub, n_features)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = PLOT_DIR / f"mq_convergence_d{n_features}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  B) Combined multi-panel figure  (single row, d=100 excluded)
# ═══════════════════════════════════════════════════════════════════════════════
n_cols = len(COMBINED_FEATURES)   # all panels in one row

fig, axes = plt.subplots(
    1, n_cols,
    figsize=(4 * n_cols, 4.0),
    squeeze=False,
)
axes_flat = axes.flatten()

for idx, (ax, n_features) in enumerate(zip(axes_flat, COMBINED_FEATURES)):
    sub = df_all[df_all["n_features"] == n_features].copy()
    _draw_panel(ax, sub, n_features, show_ylabel=(idx == 0))

fig.tight_layout(w_pad=1.5)

for ext in ("pdf", "png"):
    out = PLOT_DIR / f"mq_convergence_combined.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved → {out}")
plt.close(fig)

# reset
mpl.rcParams.update(mpl.rcParamsDefault)
print("\nDone.")
