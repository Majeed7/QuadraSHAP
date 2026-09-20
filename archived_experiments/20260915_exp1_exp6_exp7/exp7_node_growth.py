"""How the required quadrature node count grows with feature count.

For each ``d in {50, 100, 250, 500, 1000}`` this experiment trains an independent RBF
kernel-ridge model using the same fixed raw kernel bandwidth.  Thirty percent of the features
generate the response.  For every held-out instance it records two node counts for the same
absolute attribution-error target:

* ``m_certified``: the a priori node budget returned before quadrature;
* ``m_observed``: the smallest node count whose observed max-norm error against the exact rule
  reaches the target in hindsight.

The single paper figure plots both counts against ``d``.  It is designed to test whether the
required node count increases with dimensionality, and whether that increase remains modest.

Outputs (``experiments/results/exp7_node_growth/``):
    records.csv, summary.csv, summary.tex, node_growth.pdf/.png, meta.json
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from common import (MATPLOTLIB, log_line, out_dir, plt, synthetic_regression, write_csv,
                    write_latex_table, write_meta)
from quadrashap import RKHSExplainer

NAME = "exp7_node_growth"
DIMENSIONS = (50, 100, 250, 500, 1000)
N_INSTANCES = 30
N_TRAIN = 300
INFORMATIVE_FRACTION = 0.30
RBF_GAMMA = 0.005                 # fixed raw bandwidth for every feature count
KRR_ALPHA = 0.1
EPSILON = 1e-6                    # fixed absolute max-attribution-error target
VALUE_FUNCTION = "neutral"
C_OBS, C_CERT = "C0", "C3"


def informative_count(d: int) -> int:
    return max(1, int(round(INFORMATIVE_FRACTION * d)))


def collect(dimensions, n_instances: int, backend: str, seed: int) -> list[dict]:
    rows = []
    for d in dimensions:
        n_informative = informative_count(d)
        X, y, mask = synthetic_regression(
            N_TRAIN + n_instances, d, n_informative, seed=seed, scale=1.0)
        if int(mask.sum()) != n_informative:
            raise RuntimeError("synthetic generator returned the wrong informative-feature count")
        model = KernelRidge(kernel="rbf", gamma=RBF_GAMMA, alpha=KRR_ALPHA).fit(
            X[:N_TRAIN], y[:N_TRAIN])
        explainer = RKHSExplainer(model, backend=backend)
        exact_threshold = (d + 1) // 2
        log_line(f"[{NAME}] d={d}: {n_informative} informative features, {n_instances} instances, "
                 f"absolute tolerance={EPSILON:g}, exact at {exact_threshold}")
        started = time.perf_counter()

        for instance in range(n_instances):
            x = X[N_TRAIN + instance]
            report = explainer.node_budget(x, EPSILON, VALUE_FUNCTION)
            phi_exact = explainer.explain(x, VALUE_FUNCTION, m_q="exact")

            m_observed = None
            observed_at_certified = None
            for m_q in range(1, report.m_q + 1):
                phi = explainer.explain(x, VALUE_FUNCTION, m_q=m_q)
                error = float(np.max(np.abs(phi - phi_exact)))
                if m_q == report.m_q:
                    observed_at_certified = error
                if m_observed is None and error <= EPSILON:
                    m_observed = m_q

            if m_observed is None:
                raise RuntimeError(
                    f"certified budget failed numerically at d={d}, instance={instance}")

            rows.append(dict(
                d=d, n_informative=n_informative, informative_fraction=INFORMATIVE_FRACTION,
                instance=instance, n_train=N_TRAIN, gamma=RBF_GAMMA, alpha=KRR_ALPHA,
                epsilon=EPSILON, m_certified=report.m_q, m_observed=m_observed,
                exact_threshold=exact_threshold, certified_bound=report.bound,
                observed_at_certified=observed_at_certified,
                bound_holds=bool(observed_at_certified <= report.bound + 1e-14),
                tolerance_holds=bool(observed_at_certified <= EPSILON)))

        log_line(f"[{NAME}] d={d}: completed in {time.perf_counter() - started:.1f}s")
    return rows


def read_records(path: Path) -> list[dict]:
    ints = {"d", "n_informative", "instance", "n_train", "m_certified", "m_observed",
            "exact_threshold"}
    bools = {"bound_holds", "tolerance_holds"}
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append({key: (int(value) if key in ints else value == "True" if key in bools
                               else float(value)) for key, value in row.items()})
    return rows


def summarize(rows: list[dict], dimensions) -> list[dict]:
    output = []
    for d in dimensions:
        selected = [row for row in rows if row["d"] == d]
        certified = np.asarray([row["m_certified"] for row in selected], dtype=float)
        observed = np.asarray([row["m_observed"] for row in selected], dtype=float)
        output.append(dict(
            d=d, n_informative=informative_count(d), n_instances=len(selected),
            epsilon=EPSILON, observed_median=float(np.median(observed)),
            observed_p05=float(np.percentile(observed, 5)),
            observed_p95=float(np.percentile(observed, 95)),
            certified_median=float(np.median(certified)),
            certified_p05=float(np.percentile(certified, 5)),
            certified_p95=float(np.percentile(certified, 95)),
            exact_threshold=(d + 1) // 2,
            exact_over_certified=float(((d + 1) // 2) / np.median(certified)),
            bound_coverage=sum(row["bound_holds"] for row in selected) / len(selected),
            tolerance_coverage=sum(row["tolerance_holds"] for row in selected) / len(selected)))
    return output


def make_figure(summary: list[dict], path_pdf: Path, path_png: Path):
    if not MATPLOTLIB:
        log_line(f"[{NAME}] matplotlib missing: figure skipped")
        return None

    d = np.asarray([row["d"] for row in summary], dtype=float)
    obs = np.asarray([row["observed_median"] for row in summary])
    obs_lo = np.asarray([row["observed_p05"] for row in summary])
    obs_hi = np.asarray([row["observed_p95"] for row in summary])
    cert = np.asarray([row["certified_median"] for row in summary])
    cert_lo = np.asarray([row["certified_p05"] for row in summary])
    cert_hi = np.asarray([row["certified_p95"] for row in summary])

    fig, ax = plt.subplots(figsize=(5.0, 3.45))
    ax.fill_between(d, obs_lo, obs_hi, color=C_OBS, alpha=0.18, lw=0)
    ax.fill_between(d, cert_lo, cert_hi, color=C_CERT, alpha=0.14, lw=0)
    ax.plot(d, obs, "o-", color=C_OBS, lw=1.7, ms=4.2,
            label="empirical minimum")
    ax.plot(d, cert, "s--", color=C_CERT, lw=1.5, ms=4.0,
            label="certified budget")

    for x, y in zip(d, obs):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -11), textcoords="offset points",
                    ha="center", fontsize=7, color=C_OBS)
    for x, y in zip(d, cert):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=C_CERT)

    ax.set_xscale("log")
    ax.set_xticks(d, labels=[str(int(v)) for v in d])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.set_ylim(0.5, max(cert_hi) + 1.5)
    ax.set_yticks(range(1, int(np.ceil(max(cert_hi))) + 2))
    ax.set_xlabel("number of features $d$")
    ax.set_ylabel("required nodes $m_q$")
    ax.set_title(r"Node growth for absolute error target $\varepsilon=10^{-6}$", fontsize=10)
    ax.text(0.03, 0.95,
            f"30% informative features\n{int(summary[0]['n_instances'])} instances per feature count",
            transform=ax.transAxes, va="top", fontsize=7, color="0.3")
    ax.grid(alpha=0.25, which="major")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return path_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="prefix_scan_numpy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figure and tables from records.csv")
    args = parser.parse_args()

    dimensions = DIMENSIONS if not args.quick else (50, 100, 250)
    n_instances = N_INSTANCES if not args.quick else 3
    output_dir = out_dir(NAME)
    started = time.perf_counter()
    if args.replot:
        rows = read_records(output_dir / "records.csv")
        dimensions = tuple(dict.fromkeys(row["d"] for row in rows))
    else:
        rows = collect(dimensions, n_instances, args.backend, args.seed)
        write_csv(rows, output_dir / "records.csv")

    table = summarize(rows, dimensions)
    write_csv(table, output_dir / "summary.csv")
    write_latex_table(
        table, output_dir / "summary.tex",
        columns=["d", "n_informative", "n_instances", "observed_median", "certified_median",
                 "exact_threshold", "exact_over_certified", "bound_coverage"],
        headers=[r"$d$", "informative", "instances", r"observed $m_q$",
                 r"certified $m_q$", r"$\lceil d/2\rceil", "exact/certified", "coverage"],
        formats={"observed_median": ".0f", "certified_median": ".0f",
                 "exact_over_certified": ".1f", "bound_coverage": ".3f"},
        caption=(r"Growth of the empirically sufficient and certified Gauss--Legendre node counts "
                 r"with feature dimension for absolute attribution-error tolerance $10^{-6}$."),
        label="tab:node-growth",
        note=(r"Medians over 30 held-out instances. Independent RBF kernel-ridge models use the "
              r"same raw bandwidth and regularisation; 30\% of features are informative."))

    figure_pdf = make_figure(table, output_dir / "node_growth.pdf", output_dir / "node_growth.png")
    elapsed = time.perf_counter() - started
    if not args.replot:
        write_meta(NAME, dimensions=list(dimensions), n_instances=n_instances, n_train=N_TRAIN,
                   informative_fraction=INFORMATIVE_FRACTION, rbf_gamma=RBF_GAMMA,
                   alpha=KRR_ALPHA, epsilon=EPSILON, value_function=VALUE_FUNCTION,
                   backend=args.backend, n_rows=len(rows), elapsed_seconds=elapsed)
    log_line(f"[{NAME}] {len(rows)} instance-level comparisons in {elapsed:.1f}s -> "
             f"{figure_pdf or output_dir}")


if __name__ == "__main__":
    main()
