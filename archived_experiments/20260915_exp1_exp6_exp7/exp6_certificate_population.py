"""
Experiment 6 -- the certificate over a population of instances, not one or two.

For every explained instance ``r`` and every node count ``m_q`` below the exactness
threshold we record two numbers that are directly comparable, because both are absolute
errors in the max norm on the same scale ``A_max``:

    B_r   the certified bound, computed a priori from the factor tables alone
          (``certify(summarize(x), m_q)``), before any quadrature has run;
    E_r   the observed error ``max_i |phi_i(m_q) - phi_i(exact)|`` against QuadraSHAP at
          ``ceil(d/2)`` nodes -- the same code path and the same arithmetic, so what is
          measured is quadrature error and nothing else.

The population is ``d in {100, 500, 1000}`` times ``n_instances`` instances times one shared
node grid.  To isolate the effect of feature count, the kernel scale is calibrated separately
for every ``d`` so that the population mean total relative variation is exactly the requested
``Lambda`` (40 by default).  Thus ``Lambda`` does not change between panels.

The figure has one panel per feature count.  Solid curves are population-median observed errors,
dashed curves are population-median certified bounds, and bands are the 5th--95th percentiles.
Every panel uses the same node counts, so the only intended experimental change is ``d``.

The rounding level of the attributions is measured per instance rather than assumed: at and
above the exactness threshold the rule is exact, so ``m_q = ceil(d/2)`` and ``ceil(d/2) + 1``
are the same number in exact arithmetic and their difference is pure floating-point error.
Below that level the comparison no longer measures quadrature, so those points are shown as a
shaded band and excluded from the slack statistics.

Outputs (experiments/results/exp6_certificate_population/):
    population.csv, coverage.csv, coverage.tex, certificate_population.pdf/.png, meta.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (MATPLOTLIB, fit_krr, log_line, out_dir, plt, synthetic_regression, write_csv,
                    write_latex_table, write_meta)
from quadrashap import RKHSExplainer
from quadrashap.product_games.budget import budget_from_summary, certify

NAME = "exp6_certificate_population"
UNIT_ROUNDOFF = np.finfo(float).eps / 2     # float64: u = 2^-53 = 1.11e-16
VALUE_FUNCTION = "neutral"          # the product-kernel value function; no background needed
N_TRAIN = 300                       # games per instance (the kernel expansion)
FLOOR_MARGIN = 2.0                  # safety margin on the *measured* rounding level of each instance
TARGET_LAMBDA = 40.0                # held fixed (as a population mean) for every feature count
N_NODE_COUNTS = 9                   # identical geometrically spaced m_q values in every panel

# (d, n_instances).  The larger dimensions use fewer instances because their exact reference
# costs more; every curve still reports a population distribution rather than a single example.
SETTINGS = [(100, 60), (500, 40), (1000, 25)]
QUICK_SETTINGS = [(60, 5), (80, 5)]
EPS_MARK = 1e-3                     # the package default tolerance, marked in every panel
C_OBS, C_CERT = "C0", "C3"


def shared_node_grid(settings, target_lambda: float, n: int = N_NODE_COUNTS) -> list[int]:
    """One geometric node grid, strictly below every setting's exactness threshold.

    The informative range is set by ``Lambda``, not by ``d``: the quadrature error decays like
    ``exp(-11 m^2 / (3 Lambda))`` once ``m`` grows past ``Lambda/4``, so a little beyond ``m ~ Lambda``
    it has collapsed into the rounding noise of the attributions and the comparison stops measuring
    quadrature.  The grid runs to ``1.35 Lambda`` or just below the smallest exactness threshold.
    """
    min_d = min(d for d, _ in settings)
    hi = int(min((min_d + 1) // 2 - 1, max(8, round(1.35 * target_lambda))))
    return sorted({int(round(v)) for v in np.geomspace(2, hi, n)})


def population_lambdas(X_train: np.ndarray, X_eval: np.ndarray, gamma_scale: float) -> np.ndarray:
    """Exact neutral-game Lambda for each instance at ``gamma = gamma_scale / d``.

    For an RBF factor ``u_j = exp(-gamma (x_j-x'_j)^2)`` with neutral absent factor one,
    ``Lambda`` is ``max_r sum_j (1-u_j)``.  ``expm1`` keeps the calculation accurate for wide
    kernels.  This is the same quantity later returned by ``ex.summarize``.
    """
    gamma = float(gamma_scale) / X_train.shape[1]
    return np.asarray([
        np.max(np.sum(-np.expm1(-gamma * (X_train - x[None, :]) ** 2), axis=1))
        for x in X_eval
    ])


def calibrate_gamma_scale(X_train: np.ndarray, X_eval: np.ndarray,
                          target_lambda: float) -> tuple[float, np.ndarray]:
    """Choose ``gamma*d`` so the population mean Lambda equals ``target_lambda``.

    Lambda is monotone in the RBF gamma.  Bisection therefore gives a deterministic calibration
    without using any attribution errors.  The evaluation population is used only to set this
    controlled numerical difficulty; no predictive performance is estimated in this experiment.
    """
    d = X_train.shape[1]
    if not 0.0 < target_lambda < d:
        raise ValueError(f"target Lambda must lie strictly between 0 and d={d}")
    lo, hi = 0.0, 1.0
    vals = population_lambdas(X_train, X_eval, hi)
    while vals.mean() < target_lambda:
        hi *= 2.0
        vals = population_lambdas(X_train, X_eval, hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        vals = population_lambdas(X_train, X_eval, mid)
        if vals.mean() < target_lambda:
            lo = mid
        else:
            hi = mid
    gamma_scale = 0.5 * (lo + hi)
    return gamma_scale, population_lambdas(X_train, X_eval, gamma_scale)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided lower confidence bound for a binomial proportion (exact, Beta form)."""
    if n == 0:
        return float("nan")
    if k >= n:
        return float(alpha ** (1.0 / n))
    try:
        from scipy.stats import beta
        return float(beta.ppf(alpha, k, n - k + 1))
    except Exception:  # pragma: no cover - scipy is a dependency of the package
        return float("nan")


def spearman(a, b) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:  # pragma: no cover
        return float("nan")


def read_population(path) -> list[dict]:
    """Reload ``population.csv`` so the figure can be re-tuned without re-running the quadrature."""
    import csv
    ints = {"d", "instance", "m_q", "m_budget_eps"}
    bools = {"above_floor", "informative", "holds", "holds_strict"}
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (int(v) if k in ints else v == "True" if k in bools else float(v))
                         for k, v in r.items()})
    return rows


def collect(settings, backend: str, seed: int, target_lambda: float) -> list[dict]:
    rows = []
    ms = shared_node_grid(settings, target_lambda)
    for d, n_inst in settings:
        X, y, _ = synthetic_regression(N_TRAIN + n_inst, d, min(10, d // 2), seed=seed, scale=1.0)
        gs, calibrated_lambdas = calibrate_gamma_scale(X[:N_TRAIN], X[N_TRAIN:], target_lambda)
        model = fit_krr(X[:N_TRAIN], y[:N_TRAIN], gamma_scale=gs)
        ex = RKHSExplainer(model, backend=backend)
        log_line(f"[{NAME}] d={d:5d}: target mean Lambda={target_lambda:g}, "
                 f"calibrated gamma*d={gs:.6g}, realised mean={calibrated_lambdas.mean():.6g}, "
                 f"range=[{calibrated_lambdas.min():.3g}, {calibrated_lambdas.max():.3g}], "
                 f"{n_inst} instances, m_q in {ms} (exactness threshold {(d + 1) // 2})")
        t0 = time.perf_counter()
        for inst in range(n_inst):
            x = X[N_TRAIN + inst]
            summary = ex.summarize(x, VALUE_FUNCTION)
            if not np.isclose(summary.lambda_max, calibrated_lambdas[inst], rtol=1e-12, atol=1e-12):
                raise RuntimeError("Lambda calibration and explainer summary disagree")
            phi_exact = ex.explain(x, VALUE_FUNCTION, m_q="exact")
            # The rounding level is measured, not assumed: at and above the exactness threshold the
            # rule is exact, so two different node counts there are mathematically the same number
            # and differ only by floating-point error.  It varies by an order of magnitude between
            # instances, which a fixed multiple of eps * A_max does not capture.
            noise = float(np.abs(phi_exact - ex.explain(x, VALUE_FUNCTION, m_q=(d + 1) // 2 + 1)).max())
            floor = max(FLOOR_MARGIN * noise, np.finfo(float).eps * summary.A_max)
            m_eps = budget_from_summary(summary, EPS_MARK).m_q
            for m in ms:
                phi = ex.explain(x, VALUE_FUNCTION, m_q=m)
                err = float(np.abs(phi - phi_exact).max())
                bound = float(certify(summary, m))
                rows.append(dict(d=d, target_lambda=target_lambda, gamma_times_d=gs,
                                 instance=inst, m_q=m, m_budget_eps=m_eps,
                                 lambda_max=float(summary.lambda_max), A_max=float(summary.A_max),
                                 max_abs_phi=float(np.abs(phi_exact).max()),
                                 certified=bound, observed=err, floor=float(floor),
                                 rounding_noise=noise,
                                 above_floor=bool(err > floor), informative=bool(bound > floor),
                                 # the measurement is quadrature error plus rounding, so the
                                 # certificate is respected iff err <= bound + floor
                                 holds=bool(err <= bound * (1 + 1e-12) + floor),
                                 holds_strict=bool(err <= bound * (1 + 1e-12)),
                                 ratio=float(bound / err) if err > 0 else float("inf"),
                                 m_over_lambda=float(m / summary.lambda_max) if summary.lambda_max else float("inf")))
        log_line(f"[{NAME}]   -> {n_inst * len(ms)} points in {time.perf_counter() - t0:.1f}s "
                 f"(mean Lambda={np.mean([r['lambda_max'] for r in rows if r['d'] == d]):.6g})")
    return rows


def make_figure(rows, path_pdf, path_png, settings):
    """One panel per dimension, one colour per kernel bandwidth, two curves each.

    Solid is the observed error against the exact rule, dashed the certified error computed a priori
    from the factor tables.  The panel shows four things: the certified curve is above the observed
    one everywhere (validity), the vertical gap between them (tightness), the equal slopes (the bound
    is right in rate), and -- reading across panels -- that the budget is set by the total relative
    variation ``Lambda`` and not by ``d``: at the same ``Lambda`` the same ``m*`` serves ``d = 100``
    and ``d = 1000``, while the exactness threshold ``ceil(d/2)`` grows from 50 to 500.
    """
    if not MATPLOTLIB:
        log_line(f"[{NAME}] matplotlib missing: figure skipped (all the data is in population.csv)")
        return None
    ds = list(dict.fromkeys(d for d, _, _ in settings))
    gss = sorted({gs for _, gs, _ in settings})
    colors = {gs: c for gs, c in zip(gss, ("C0", "C3", "C2", "C4"))}
    fig, axes = plt.subplots(1, len(ds), figsize=(3.7 * len(ds), 3.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, ds):
        panel = [r for r in rows if r["d"] == d]
        thr = (d + 1) // 2
        for gs in gss:
            sel = [r for r in panel if r["gamma_times_d"] == gs]
            if not sel:
                continue
            ms = sorted({r["m_q"] for r in sel})
            lam = float(np.mean([r["lambda_max"] for r in sel]))
            col = colors[gs]

            def band(key):
                v = [np.array([r[key] for r in sel if r["m_q"] == m], dtype=float) for m in ms]
                return (np.array([np.median(a) for a in v]),
                        np.array([np.percentile(a, 5) for a in v]),
                        np.array([np.percentile(a, 95) for a in v]))

            obs, obs_lo, obs_hi = band("observed")
            cert, cert_lo, cert_hi = band("certified")
            ax.fill_between(ms, cert_lo, cert_hi, color=col, alpha=0.15, lw=0, zorder=1)
            ax.fill_between(ms, obs_lo, obs_hi, color=col, alpha=0.22, lw=0, zorder=2)
            ax.plot(ms, cert, "s--", color=col, ms=3.0, lw=1.2, zorder=4)
            ax.plot(ms, obs, "o-", color=col, ms=3.0, lw=1.4, zorder=5,
                    label=rf"$\Lambda={lam:.0f}$  ($\gamma d={gs:g}$)")

            m_eps = float(np.median([r["m_budget_eps"] for r in sel]))
            ax.axvline(m_eps, color=col, ls=":", lw=0.9, zorder=3)
            ax.annotate(rf"$m^*={m_eps:.0f}$", xy=(m_eps, EPS_MARK), xytext=(2.5, 3),
                        textcoords="offset points", fontsize=6, color=col)

        uA = float(np.median([UNIT_ROUNDOFF * r["A_max"] for r in panel]))
        ax.axhspan(1e-300, max(r["floor"] for r in panel), color="0.9", lw=0, zorder=0)
        ax.axhline(uA, color="0.45", ls=(0, (1, 2)), lw=0.9, zorder=3)
        ax.annotate(r"$u\,A_{\max}$", xy=(0.99, uA), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                    fontsize=6, color="0.35")
        ax.axhline(EPS_MARK, color="0.4", ls=":", lw=0.9, zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of nodes $m_q$")
        ax.set_title(rf"$d={d}$   (exactness threshold $\lceil d/2\rceil={thr}$)", fontsize=8.5)
        ax.grid(alpha=0.25, which="major")
        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.92, title="solid: observed\ndashed: certified",
                  title_fontsize=6)

    ymin = max(min(UNIT_ROUNDOFF * r["A_max"] for r in rows) / 20, 1e-22)
    axes[0].set_ylim(ymin, max(r["certified"] for r in rows) * 5)
    axes[0].set_ylabel(r"absolute error  $\max_i|\hat\phi_i-\phi_i|$")
    axes[-1].text(0.98, 0.96,
                  "bands: 5\u201395 pct. over instances\n"
                  "grey: floor of the float64 rule\n"
                  r"($u=2^{-53}\approx1.1\times10^{-16}$)",
                  transform=axes[-1].transAxes, ha="right", va="top", fontsize=6, color="0.35")

    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path_pdf


def make_fixed_lambda_figure(rows, path_pdf, path_png, settings):
    """Plot observed and certified errors while holding mean Lambda fixed across dimensions."""
    if not MATPLOTLIB:
        log_line(f"[{NAME}] matplotlib missing: figure skipped (all the data is in population.csv)")
        return None
    ds = [d for d, _ in settings]
    fig, axes = plt.subplots(1, len(ds), figsize=(3.7 * len(ds), 3.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, ds):
        panel = [r for r in rows if r["d"] == d]
        thr = (d + 1) // 2
        ms = sorted({r["m_q"] for r in panel})

        def band(key):
            vals = [np.asarray([r[key] for r in panel if r["m_q"] == m], dtype=float) for m in ms]
            return (np.asarray([np.median(v) for v in vals]),
                    np.asarray([np.percentile(v, 5) for v in vals]),
                    np.asarray([np.percentile(v, 95) for v in vals]))

        obs, obs_lo, obs_hi = band("observed")
        cert, cert_lo, cert_hi = band("certified")
        ax.fill_between(ms, cert_lo, cert_hi, color=C_CERT, alpha=0.14, lw=0, zorder=1)
        ax.fill_between(ms, obs_lo, obs_hi, color=C_OBS, alpha=0.20, lw=0, zorder=2)
        ax.plot(ms, cert, "s--", color=C_CERT, ms=3.0, lw=1.3, zorder=4,
                label="certified bound")
        ax.plot(ms, obs, "o-", color=C_OBS, ms=3.0, lw=1.5, zorder=5,
                label="observed error")

        m_eps = float(np.median([r["m_budget_eps"] for r in panel]))
        ax.axvline(m_eps, color="C2", ls=":", lw=1.0, zorder=3)
        ax.annotate(rf"$m^*={m_eps:.0f}$", xy=(m_eps, EPS_MARK), xytext=(2.5, 3),
                    textcoords="offset points", fontsize=6.5, color="C2")

        uA = float(np.median([UNIT_ROUNDOFF * r["A_max"] for r in panel]))
        ax.axhspan(1e-300, max(r["floor"] for r in panel), color="0.9", lw=0, zorder=0)
        ax.axhline(uA, color="0.45", ls=(0, (1, 2)), lw=0.9, zorder=3)
        ax.annotate(r"$u\,A_{\max}$", xy=(0.99, uA), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                    fontsize=6, color="0.35")
        ax.axhline(EPS_MARK, color="0.4", ls=":", lw=0.9, zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of nodes $m_q$")
        ax.set_title(rf"$d={d}$   (exactness threshold $\lceil d/2\rceil={thr}$)", fontsize=8.5)
        ax.grid(alpha=0.25, which="major")
        n_inst = len({r["instance"] for r in panel})
        lam = float(np.mean([r["lambda_max"] for r in panel]))
        ax.text(0.97, 0.96, rf"$\bar{{\Lambda}}={lam:.1f}$" + f"\n{n_inst} instances",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.32")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([handles[1], handles[0]], [labels[1], labels[0]],
                   fontsize=7, loc="lower left", framealpha=0.92)
    ymin = max(min(UNIT_ROUNDOFF * r["A_max"] for r in rows) / 20, 1e-22)
    axes[0].set_ylim(ymin, max(r["certified"] for r in rows) * 5)
    axes[0].set_ylabel(r"absolute error  $\max_i|\hat\phi_i-\phi_i|$")

    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default="prefix_scan_numpy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-lambda", type=float, default=TARGET_LAMBDA,
                    help="population mean Lambda to hold fixed across feature counts")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure and tables from an existing population.csv (no quadrature)")
    args = ap.parse_args()

    settings = QUICK_SETTINGS if args.quick else SETTINGS
    d_out = out_dir(NAME)
    t_start = time.perf_counter()
    if args.replot:
        rows = read_population(d_out / "population.csv")
        if not rows or "target_lambda" not in rows[0]:
            raise ValueError("population.csv predates the fixed-Lambda design; rerun without --replot")
        settings = [(d, len({r["instance"] for r in rows if r["d"] == d}))
                    for d in dict.fromkeys(r["d"] for r in rows)]
        args.target_lambda = float(np.mean([r["target_lambda"] for r in rows]))
        log_line(f"[{NAME}] replotting {len(rows)} points from {d_out / 'population.csv'}")
    else:
        rows = collect(settings, args.backend, args.seed, args.target_lambda)
        write_csv(rows, d_out / "population.csv")
    elapsed = time.perf_counter() - t_start

    # ---------------------------------------------------------------- coverage / tightness table
    cov = []
    for d, n_inst in settings:
        sel = [r for r in rows if r["d"] == d]
        per_instance = [next(r for r in sel if r["instance"] == inst)
                        for inst in sorted({r["instance"] for r in sel})]
        inf = [r for r in sel if r["above_floor"]]
        cert = [r for r in sel if r["informative"]]
        lr = np.log10([r["ratio"] for r in inf]) if inf else np.array([np.nan])
        lambdas = np.asarray([r["lambda_max"] for r in per_instance])
        cov.append(dict(d=d, n_instances=n_inst, n_points=len(sel), target_lambda=args.target_lambda,
                        gamma_times_d=float(np.mean([r["gamma_times_d"] for r in sel])),
                        lambda_mean=float(lambdas.mean()), lambda_sd=float(lambdas.std(ddof=1)),
                        lambda_min=float(lambdas.min()), lambda_max=float(lambdas.max()),
                        m_budget_median=float(np.median([r["m_budget_eps"] for r in per_instance])),
                        n_informative=len(cert),
                        coverage=sum(r["holds"] for r in sel) / max(len(sel), 1),
                        coverage_lower=clopper_pearson_lower(sum(r["holds"] for r in sel), len(sel)),
                        coverage_informative=sum(r["holds_strict"] for r in cert) / max(len(cert), 1),
                        median_slack=float(np.median(lr)), p90_slack=float(np.percentile(lr, 90)),
                        max_slack=float(np.max(lr)),
                        spearman=spearman([r["certified"] for r in inf], [r["observed"] for r in inf])))
        log_line(f"[{NAME}] d={d:5d} gamma*d={cov[-1]['gamma_times_d']:8.4f}  "
                 f"mean Lambda={cov[-1]['lambda_mean']:.6f}  "
                 f"coverage {cov[-1]['coverage']:.4f} (informative {cov[-1]['coverage_informative']:.4f}"
                 f" of {cov[-1]['n_informative']})  median slack 1e{cov[-1]['median_slack']:.1f}  "
                 f"max 1e{cov[-1]['max_slack']:.1f}  Spearman {cov[-1]['spearman']:.3f}")
    write_csv(cov, d_out / "coverage.csv")
    write_latex_table(
        cov, d_out / "coverage.tex",
        columns=["d", "n_instances", "gamma_times_d", "lambda_mean", "lambda_sd", "m_budget_median",
                 "n_points", "coverage", "median_slack", "p90_slack", "spearman"],
        headers=[r"$d$", r"instances", r"$\gamma d$", r"$\bar\Lambda$", r"sd$(\Lambda)$",
                 r"median $m_q^\star$", r"points", r"coverage",
                 r"median $\log_{10}\frac{B}{E}$", r"90th pct.", r"Spearman $(B,E)$"],
        formats={"gamma_times_d": ".2f", "lambda_mean": ".1f", "lambda_sd": ".1f",
                 "m_budget_median": ".0f", "coverage": ".4f", "median_slack": ".1f",
                 "p90_slack": ".1f", "spearman": ".3f"},
        caption=(r"The a priori certificate over a population of instances at fixed total relative variation. "
                 r"For each feature count, the RBF scale is calibrated so the population mean is "
                 rf"$\bar\Lambda={args.target_lambda:g}$. For every instance and every "
                 r"node count below the exactness threshold, the certified bound $B$ is computed from the "
                 r"factor tables before any quadrature runs and compared with the observed error $E$ against "
                 r"the exact rule. A point is \emph{informative} when the certificate is still above the "
                 r"rounding level of the attributions; coverage is the fraction of $(\text{instance}, m_q)$ "
                 r"pairs with $E \le B$ up to that rounding level, and the slack columns are taken over the "
                 r"points whose observed error is above it."),
        label="tab:certificate-population",
        note=(r"Kernel ridge regression with an isotropic RBF kernel, $n=300$ training points, "
              r"neutral-factor value function; the same node grid is used for every $d$."))

    fig_pdf = make_fixed_lambda_figure(
        rows, d_out / "certificate_population.pdf", d_out / "certificate_population.png", settings)
    if args.replot:
        log_line(f"[{NAME}] figure and tables rewritten in {d_out}")
        return
    meta_settings = [{"d": d, "n_instances": n,
                      "gamma_times_d": float(np.mean([r["gamma_times_d"] for r in rows if r["d"] == d])),
                      "lambda_mean": float(np.mean([r["lambda_max"] for r in rows if r["d"] == d]))}
                     for d, n in settings]
    write_meta(NAME, settings=meta_settings, target_lambda=args.target_lambda,
               lambda_control="RBF gamma calibrated per d to fix the population mean Lambda",
               value_function=VALUE_FUNCTION, n_train=N_TRAIN, backend=args.backend,
               node_grid=sorted({r["m_q"] for r in rows}),
               n_points=len(rows), elapsed_seconds=elapsed)
    log_line(f"[{NAME}] {len(rows)} points, coverage {sum(r['holds'] for r in rows)}/{len(rows)}, "
             f"done in {elapsed:.1f}s -> {fig_pdf or d_out}")


if __name__ == "__main__":
    main()
