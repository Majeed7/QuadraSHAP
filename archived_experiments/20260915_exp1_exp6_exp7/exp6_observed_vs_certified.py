"""Observed versus certified quadrature error, one figure per feature count.

For each ``d`` this experiment independently:

1. generates a synthetic regression problem with 30% informative features and trains one RBF
   kernel-ridge model with the same fixed recipe (``gamma = 10 / d``, ``alpha = 0.1``);
2. explains a population of held-out instances with a fixed grid of Gauss--Legendre node counts;
3. compares the certified max-norm error with the observed max-norm error against the exact rule;
4. writes a separate publication figure for that feature count.

No difficulty calibration or secondary grouping is performed.  The only plotted experimental
variables are feature count and ``m_q``.

Outputs (``experiments/results/exp6_observed_vs_certified/``):
    errors.csv, summary.csv, summary.tex, error_d100.pdf/.png,
    error_d500.pdf/.png, error_d1000.pdf/.png, meta.json
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from common import (MATPLOTLIB, fit_krr, log_line, out_dir, plt, synthetic_regression,
                    write_csv, write_latex_table, write_meta)
from quadrashap import RKHSExplainer
from quadrashap.product_games.budget import certify

NAME = "exp6_observed_vs_certified"
VALUE_FUNCTION = "neutral"
N_TRAIN = 300
GAMMA_SCALE = 10.0                 # fixed recipe: gamma = GAMMA_SCALE / d
INFORMATIVE_FRACTION = 0.30
FLOOR_MARGIN = 2.0
SETTINGS = [(100, 60), (500, 40), (1000, 25)]
QUICK_SETTINGS = [(30, 4), (40, 4), (50, 4)]
M_Q_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
C_OBS, C_CERT = "C0", "C3"


def node_grid(d: int) -> list[int]:
    """Use the same predeclared node counts, truncated below exactness for dimension ``d``."""
    threshold = (d + 1) // 2
    return [m for m in M_Q_VALUES if m < threshold]


def informative_count(d: int) -> int:
    """Thirty percent of the available features, rounded to the nearest integer."""
    return max(1, int(round(INFORMATIVE_FRACTION * d)))


def collect(settings, backend: str, seed: int) -> list[dict]:
    rows = []
    for d, n_instances in settings:
        n_informative = informative_count(d)
        X, y, _ = synthetic_regression(
            N_TRAIN + n_instances, d, n_informative, seed=seed, scale=1.0)
        model = fit_krr(X[:N_TRAIN], y[:N_TRAIN], gamma_scale=GAMMA_SCALE)
        explainer = RKHSExplainer(model, backend=backend)
        ms = node_grid(d)
        threshold = (d + 1) // 2
        log_line(f"[{NAME}] d={d}: {n_informative} informative features (30%); "
                 f"trained KRR on {N_TRAIN} samples; "
                 f"{n_instances} held-out instances; m_q={ms}; exact at {threshold}")
        t0 = time.perf_counter()

        for instance in range(n_instances):
            x = X[N_TRAIN + instance]
            summary = explainer.summarize(x, VALUE_FUNCTION)
            phi_exact = explainer.explain(x, VALUE_FUNCTION, m_q="exact")
            rule_noise = float(np.max(np.abs(
                phi_exact - explainer.explain(x, VALUE_FUNCTION, m_q=threshold + 1))))
            floor = max(FLOOR_MARGIN * rule_noise, np.finfo(float).eps * summary.A_max)

            for m_q in ms:
                phi = explainer.explain(x, VALUE_FUNCTION, m_q=m_q)
                observed = float(np.max(np.abs(phi - phi_exact)))
                certified = float(certify(summary, m_q))
                rows.append(dict(
                    d=d, instance=instance, m_q=m_q, exact_threshold=threshold,
                    n_train=N_TRAIN, n_informative=n_informative, gamma_scale=GAMMA_SCALE,
                    certified_error=certified, observed_error=observed,
                    rounding_floor=float(floor),
                    above_floor=bool(observed > floor),
                    informative=bool(certified > floor),
                    holds=bool(observed <= certified * (1.0 + 1e-12) + floor),
                    holds_strict=bool(observed <= certified * (1.0 + 1e-12))))

        log_line(f"[{NAME}] d={d}: {n_instances * len(ms)} comparisons in "
                 f"{time.perf_counter() - t0:.1f}s")
    return rows


def read_errors(path: Path) -> list[dict]:
    ints = {"d", "instance", "m_q", "exact_threshold", "n_train", "n_informative"}
    bools = {"above_floor", "informative", "holds", "holds_strict"}
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append({key: (int(value) if key in ints else value == "True" if key in bools
                               else float(value)) for key, value in row.items()})
    return rows


def band(rows: list[dict], m_q: int, key: str) -> tuple[float, float, float]:
    values = np.asarray([row[key] for row in rows if row["m_q"] == m_q], dtype=float)
    return (float(np.median(values)), float(np.percentile(values, 5)),
            float(np.percentile(values, 95)))


def make_figure(rows: list[dict], d: int, n_instances: int, path_pdf: Path, path_png: Path):
    if not MATPLOTLIB:
        log_line(f"[{NAME}] matplotlib missing: skipped figure for d={d}")
        return None
    selected = [row for row in rows if row["d"] == d]
    ms = sorted({row["m_q"] for row in selected})
    obs = np.asarray([band(selected, m, "observed_error") for m in ms])
    cert = np.asarray([band(selected, m, "certified_error") for m in ms])

    fig, ax = plt.subplots(figsize=(4.2, 3.35))
    ax.fill_between(ms, cert[:, 1], cert[:, 2], color=C_CERT, alpha=0.14, lw=0)
    ax.fill_between(ms, obs[:, 1], obs[:, 2], color=C_OBS, alpha=0.20, lw=0)
    ax.plot(ms, obs[:, 0], "o-", color=C_OBS, ms=3.4, lw=1.6, label="observed error")
    ax.plot(ms, cert[:, 0], "s--", color=C_CERT, ms=3.2, lw=1.4, label="certified bound")

    floor = float(np.max([row["rounding_floor"] for row in selected]))
    ax.axhspan(1e-300, floor, color="0.9", lw=0, zorder=0)
    ax.text(0.98, 0.96, f"{n_instances} instances\nbands: 5-95th percentile",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.32")
    positive = [row["observed_error"] for row in selected if row["observed_error"] > 0]
    ymin = max(min(positive + [floor]) / 20.0, 1e-22)
    ymax = max(row["certified_error"] for row in selected) * 3.0
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ticks = [m for m in (1, 2, 4, 8, 16, 32, 50) if m <= max(ms)]
    ax.set_xticks(ticks, labels=[str(m) for m in ticks])
    ax.set_xlabel("number of nodes $m_q$")
    ax.set_ylabel(r"absolute error  $\max_i|\hat\phi_i-\phi_i|$")
    ax.set_title(rf"$d={d}$ features ({informative_count(d)} informative)")
    ax.grid(alpha=0.25, which="major")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return path_pdf


def summarize(rows: list[dict], settings) -> list[dict]:
    output = []
    for d, n_instances in settings:
        selected = [row for row in rows if row["d"] == d]
        informative = [row for row in selected if row["informative"]]
        above_floor = [row for row in selected if row["above_floor"]]
        ratios = np.asarray([
            row["certified_error"] / row["observed_error"]
            for row in above_floor if row["observed_error"] > 0
        ])
        output.append(dict(
            d=d, n_informative_features=informative_count(d),
            informative_fraction=INFORMATIVE_FRACTION,
            n_instances=n_instances, n_points=len(selected),
            exact_threshold=(d + 1) // 2,
            n_informative_points=len(informative),
            coverage=sum(row["holds"] for row in selected) / len(selected),
            informative_coverage=(sum(row["holds"] for row in informative)
                                  / max(len(informative), 1)),
            median_log10_slack=float(np.median(np.log10(ratios))),
            p90_log10_slack=float(np.percentile(np.log10(ratios), 90))))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="prefix_scan_numpy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replot", action="store_true",
                        help="redraw figures and summaries from errors.csv")
    args = parser.parse_args()

    settings = QUICK_SETTINGS if args.quick else SETTINGS
    output_dir = out_dir(NAME)
    started = time.perf_counter()
    if args.replot:
        rows = read_errors(output_dir / "errors.csv")
        settings = [(d, len({row["instance"] for row in rows if row["d"] == d}))
                    for d in dict.fromkeys(row["d"] for row in rows)]
    else:
        rows = collect(settings, args.backend, args.seed)
        write_csv(rows, output_dir / "errors.csv")

    summary = summarize(rows, settings)
    write_csv(summary, output_dir / "summary.csv")
    write_latex_table(
        summary, output_dir / "summary.tex",
        columns=["d", "n_informative_features", "n_instances", "n_points", "exact_threshold",
                 "coverage", "informative_coverage", "median_log10_slack", "p90_log10_slack"],
        headers=[r"$d$", "informative features", "instances", "points", r"$\lceil d/2\rceil",
                 "coverage", "informative coverage", r"median $\log_{10}(B/E)$", "90th pct."],
        formats={"coverage": ".4f", "informative_coverage": ".4f",
                 "median_log10_slack": ".1f", "p90_log10_slack": ".1f"},
        caption=(r"Certified and observed QuadraSHAP quadrature error over held-out instances. "
                 r"Observed error is measured against the exact rule; bands in the corresponding "
                 r"figures show the 5th--95th percentiles."),
        label="tab:observed-certified-error",
        note=(r"One RBF kernel-ridge model is trained independently for each feature count using "
              r"the same training recipe; 30\% of the features generate the response."))

    figures = []
    for d, n_instances in settings:
        figures.append(make_figure(rows, d, n_instances,
                                   output_dir / f"error_d{d}.pdf",
                                   output_dir / f"error_d{d}.png"))

    elapsed = time.perf_counter() - started
    if not args.replot:
        write_meta(NAME,
                   settings=[{"d": d, "n_informative": informative_count(d), "n_instances": n}
                             for d, n in settings],
                   informative_fraction=INFORMATIVE_FRACTION,
                   n_train=N_TRAIN, gamma_scale=GAMMA_SCALE, alpha=0.1,
                   value_function=VALUE_FUNCTION, backend=args.backend,
                   node_grids={str(d): node_grid(d) for d, _ in settings},
                   n_points=len(rows), elapsed_seconds=elapsed)
    informative = [row for row in rows if row["informative"]]
    log_line(f"[{NAME}] {len(rows)} comparisons; certificate respected above the numerical floor "
             f"in {sum(row['holds'] for row in informative)}/{len(informative)} points; "
             f"done in {elapsed:.1f}s -> {output_dir}")


if __name__ == "__main__":
    main()
