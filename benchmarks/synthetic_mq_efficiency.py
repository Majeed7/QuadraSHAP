"""Synthetic m_q sweep: distance to exact Shapley for RBF kernel explainers.

For each feature count d, this script:
1. Generates a normalised synthetic regression dataset with N_SAMPLES samples.
2. Fits an SVR and a KernelRidge model with RBF kernels.
3. Computes the **exact** reference Shapley values at m_q_exact = ceil(d/2)
   using the JAX logspace backend.  At this node count Gauss-Legendre is
   exact for polynomials of degree <= d-1, which covers the product-game
   integrand exactly.
4. Sweeps m_q in [1, ceil(d/2)] and measures the Shapley distance:
       mean_{samples} ||phi(m_q, x) - phi_exact(x)||_2
5. Saves per-run CSV data and a combined multi-panel figure.

The JAX logspace backend is used throughout:
  - it is memory-lean (O(m_q * n) per feature step, not O(m_q * n * d)),
  - it is numerically stable in log-space,
  - it benefits from Metal / GPU JIT on Apple Silicon.
JAX must be installed (pip install jax[metal] for M-chip).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pgshapley.kernels import RBFLocalExplainer


RESULTS_DIR = ROOT / "benchmarks" / "results" / "mq"

# Reference (m_q = ceil(d/2)): use NumPy — avoids JAX JIT compilation at very
# large m_q (e.g. m_q=5000 for d=10000) which exhausts memory on Metal.
BACKEND_REF   = "logspace_numpy"

# Sweep (m_q ≤ 500): use JAX — JIT compiles once and is fast on Metal for all
# repeated calls.  At m_q ≤ 500 the carry tensors are small enough to compile.
BACKEND_SWEEP = "logspace_jax"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_normalized_regression(
    *,
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(5, min(n_features, 20)),
        noise=0.5,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    return X.astype(np.float64), y.astype(np.float64)


def build_mq_values(n_features: int, n_sweep: int, seed: int) -> list[int]:
    """Return n_sweep random m_q values from [1, min(500, ceil(d/2))].

    The upper bound is always included in the sweep.
    """
    upper = min(500, math.ceil(n_features / 2))
    rng = np.random.default_rng(seed)
    # Draw n_sweep-1 values from [1, upper-1], then append upper.
    n_random = min(n_sweep - 1, upper - 1)
    random_vals = rng.choice(np.arange(1, upper), size=n_random, replace=False).tolist()
    values = sorted({int(v) for v in random_vals} | {upper})
    return values


def evaluate_mq_distance(
    *,
    model,
    X_eval: np.ndarray,
    m_q_values: list[int],
    eval_samples: int,
) -> list[dict[str, float]]:
    """Compute mean ||phi(m_q, x) - phi_exact(x)||_2 for each m_q.

    phi_exact is computed at m_q_exact = ceil(d/2) using JAX logspace.
    At that node count Gauss-Legendre integrates the product-game integrand
    (a degree d-1 polynomial in the quadrature variable) exactly.
    """
    explainer = RBFLocalExplainer(model)
    d = explainer.d
    m_q_exact = math.ceil(d / 2)
    n_use = min(eval_samples, len(X_eval))

    # ── Step 1: reference Shapley at m_q = ceil(d/2) — NumPy, no JIT ────────
    # logspace_numpy streams feature-by-feature so peak RAM is O(m_q * n),
    # which is safe even for m_q=5000, d=10000.
    print(f"    [ref  m_q={m_q_exact}, d={d}, n_train={explainer.n}, "
          f"backend={BACKEND_REF}] …", end=" ", flush=True)
    phi_refs: list[np.ndarray] = []
    for row in X_eval[:n_use]:
        phi_ref = np.asarray(
            explainer.explain(row, m_q=m_q_exact, method=BACKEND_REF),
            dtype=np.float64,
        )
        phi_refs.append(phi_ref)
    print("done")

    # ── Step 2: warm up JAX JIT once at the smallest sweep m_q ───────────────
    # m_q ≤ 500 is always small enough to compile on Metal without OOM.
    print(f"    [warm-up JAX JIT  m_q={m_q_values[0]}, backend={BACKEND_SWEEP}] …",
          end=" ", flush=True)
    _ = explainer.explain(X_eval[0], m_q=m_q_values[0], method=BACKEND_SWEEP)
    print("done")

    # ── Step 3: sweep m_q and measure L2 distance to reference ───────────────
    agg_rows: list[dict]  = []   # one row per m_q  (mean / std / max)
    raw_rows: list[dict]  = []   # one row per (m_q, sample_idx)

    for m_q in m_q_values:
        distances: list[float] = []
        for sample_idx, (row, phi_ref) in enumerate(zip(X_eval[:n_use], phi_refs)):
            phi = np.asarray(
                explainer.explain(row, m_q=m_q, method=BACKEND_SWEEP),
                dtype=np.float64,
            )
            dist = float(np.linalg.norm(phi - phi_ref))
            distances.append(dist)
            raw_rows.append({"m_q": m_q, "sample_idx": sample_idx,
                             "shapley_distance": dist})

        agg_rows.append(
            {
                "m_q":                    m_q,
                "m_q_exact":              m_q_exact,
                "mean_shapley_distance":  float(np.mean(distances)),
                "std_shapley_distance":   float(np.std(distances)),
                "max_shapley_distance":   float(np.max(distances)),
                "eval_samples":           n_use,
            }
        )
        print(f"      m_q={m_q:5d}  mean_dist={agg_rows[-1]['mean_shapley_distance']:.4e}")

    return agg_rows, raw_rows


def plot_results(df: pd.DataFrame, feature_counts: list[int]) -> None:
    if df.empty:
        return

    # ── publication-quality global style ─────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          13,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "axes.titleweight":   "bold",
        "axes.labelweight":   "bold",
        "axes.linewidth":     1.4,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "legend.fontsize":    10,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.7",
        "lines.linewidth":    2.0,
        "lines.markersize":   7,
        "figure.dpi":         150,
    })

    # Colour-blind-friendly palette (Paul Tol "bright")
    COLORS = {
        "svr_rbf": "#4477AA",   # blue
        "krr_rbf": "#EE6677",   # red
    }
    MARKERS = {
        "svr_rbf": "o",
        "krr_rbf": "s",
    }
    LABELS = {
        "svr_rbf": "SVR (RBF)",
        "krr_rbf": "KRR (RBF)",
    }

    n_panels = len(feature_counts)
    n_cols   = min(3, n_panels)
    n_rows   = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, n_features in zip(axes_flat, feature_counts, strict=True):
        subset = df[df["n_features"] == n_features]

        for model_kind, model_df in subset.groupby("model_kind"):
            model_df  = model_df.sort_values("m_q")
            x         = model_df["m_q"].to_numpy()
            y         = model_df["mean_shapley_distance"].to_numpy()
            std       = model_df["std_shapley_distance"].to_numpy()
            m_q_exact = int(model_df["m_q_exact"].iloc[0])

            color  = COLORS.get(model_kind, "#444444")
            marker = MARKERS.get(model_kind, "^")
            label  = LABELS.get(model_kind, model_kind)

            ax.semilogy(
                x, np.maximum(y, 1e-16),
                marker=marker, color=color, label=label,
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=3,
            )
            ax.fill_between(
                x,
                np.maximum(y - std, 1e-16),
                y + std,
                color=color, alpha=0.15, zorder=2,
            )

        # Vertical line at the exact reference
        ax.axvline(
            m_q_exact, color="#555555", linestyle="--",
            linewidth=1.2, zorder=1,
            label=rf"$m_q^{{\star}} = \lceil d/2 \rceil = {m_q_exact}$",
        )

        ax.set_title(rf"$d = {n_features}$", pad=6)
        ax.set_xlabel(r"$m_q$")
        ax.set_ylabel(r"$\|\hat{\varphi}(m_q) - \varphi^{\star}\|_2$")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7,
                color="0.75", zorder=0)
        ax.grid(True, which="major", linestyle="-", linewidth=0.5,
                color="0.85", zorder=0)
        ax.tick_params(which="both", top=True, right=True)

        leg = ax.legend(loc="upper right", handlelength=1.8,
                        frameon=True, borderpad=0.6)
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

        # Bold tick labels
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        r"Convergence of Gauss–Legendre Shapley to exact ($m_q = \lceil d/2 \rceil$)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(w_pad=2.5, h_pad=3.0)

    for ext in ("pdf", "png"):
        out = RESULTS_DIR / f"mq_shapley_distance.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved plot → {out}")

    plt.close(fig)

    # Reset rcParams so the rest of the session is unaffected
    plt.rcParams.update(plt.rcParamsDefault)


def run_sweep(args: argparse.Namespace) -> None:
    ensure_dirs()
    all_agg: list[dict] = []   # aggregated rows for the combined CSV + plot
    all_raw: list[dict] = []   # raw per-sample rows

    for n_features in args.feature_counts:
        print(f"\n{'─'*60}")
        print(f"  n_features = {n_features}")
        print(f"{'─'*60}")
        X, y = make_normalized_regression(
            n_samples=args.n_samples,
            n_features=n_features,
            seed=args.seed + n_features,
        )
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, shuffle=True
        )
        m_q_values = build_mq_values(n_features=n_features, n_sweep=args.n_sweep, seed=args.seed)
        print(f"  m_q sweep ({len(m_q_values)} pts): {m_q_values}")

        models: dict[str, object] = {
            "krr_rbf": KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0 / n_features),
        }

        for model_kind, model in models.items():
            print(f"\n  Fitting {model_kind} … ", end="", flush=True)
            model.fit(X_train, y_train)
            print("done")

            agg_rows, raw_rows = evaluate_mq_distance(
                model=model,
                X_eval=X_test,
                m_q_values=m_q_values,
                eval_samples=args.eval_samples,
            )

            # Tag every row with (n_features, model_kind)
            for row in agg_rows:
                row.update({"n_features": n_features, "model_kind": model_kind})
            for row in raw_rows:
                row.update({"n_features": n_features, "model_kind": model_kind})

            all_agg.extend(agg_rows)
            all_raw.extend(raw_rows)

            # ── per-(d, model) individual CSV saved immediately ───────────────
            tag = f"d{n_features}_{model_kind}"
            pd.DataFrame(agg_rows).to_csv(
                RESULTS_DIR / f"mq_agg_{tag}.csv", index=False
            )
            pd.DataFrame(raw_rows).to_csv(
                RESULTS_DIR / f"mq_raw_{tag}.csv", index=False
            )
            print(f"    Saved per-run CSVs: mq_agg_{tag}.csv  mq_raw_{tag}.csv")

            # ── incremental combined CSV (safe checkpoint) ────────────────────
            pd.DataFrame(all_agg).to_csv(
                RESULTS_DIR / "mq_shapley_distance.csv", index=False
            )
            pd.DataFrame(all_raw).to_csv(
                RESULTS_DIR / "mq_shapley_distance_raw.csv", index=False
            )

    # ── final combined save + plot ────────────────────────────────────────────
    df_agg = pd.DataFrame(all_agg)
    df_agg.to_csv(RESULTS_DIR / "mq_shapley_distance.csv", index=False)
    pd.DataFrame(all_raw).to_csv(RESULTS_DIR / "mq_shapley_distance_raw.csv", index=False)
    print(f"\nSaved combined CSV  → {RESULTS_DIR / 'mq_shapley_distance.csv'}")
    print(f"Saved raw distances → {RESULTS_DIR / 'mq_shapley_distance_raw.csv'}")

    plot_results(df_agg, feature_counts=args.feature_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Total dataset size (default: 500).",
    )
    parser.add_argument(
        "--feature-counts", nargs="+", type=int,
        default=[10000],#[50, 100, 1000, 5000, 10000],
        help="Feature counts to sweep (default: 50 100 1000 5000 10000).",
    )
    parser.add_argument(
        "--n-sweep", type=int, default=10,
        help="Number of random m_q values sampled from [1, 500] (default: 10).",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=10,
        help="Test samples used to estimate the mean distance (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(args)
    print(f"\nAll artifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
