"""
Experiment 1 -- does the a priori certificate hold, and how tight is it?

For every (model, instance, tolerance eps) we

  1. compute the node budget m*(eps) from the factor tables alone, together with the
     certified error A_max * B(m*, Lambda_max)  (no quadrature has run at this point);
  2. run QuadraSHAP with m*(eps) and with the exactness threshold ceil(d/2);
  3. record the observed error max_i |phi_hat_i - phi_i|, whether it respects both the
     certificate and eps, the tightness ratio certified/observed, and the smallest number
     of nodes that would have met eps in hindsight.

A second pass sweeps m_q from 1 upwards and records the observed error next to the
certified bound and the (free) efficiency residual, which is the figure that connects the
appendix to the experiments.

Outputs (experiments/results/exp1_certified_bound/):
    certificate.csv, certificate_summary.csv, certificate_summary.tex,
    decay.csv, decay.pdf, tightness.tex, meta.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (RESULTS, figure, fit_krr, log_line, out_dir, save_figure, synthetic_regression,
                    timed, write_csv, write_latex_table, write_meta)
from quadrashap import RKHSExplainer
from quadrashap.product_games.budget import certify

NAME = "exp1_certified_bound"

# (label, d, gamma*d, value function, n_background, n_instances)
SETTINGS = [
    ("d=100, wide",    100,   1.0, "neutral",        0,  10),
    ("d=100, narrow",  100,  30.0, "neutral",        0,  10),
    ("d=1000, wide",  1000,   1.0, "neutral",        0,   5),
    ("d=1000, medium",1000,  10.0, "neutral",        0,   5),
    ("d=1000, narrow",1000, 100.0, "neutral",        0,   5),
    ("d=100, interv.", 100,   1.0, "interventional", 5,  10),
    ("d=1000, interv.",1000,  1.0, "interventional", 5,   3),
]
EPSILONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
N_TRAIN = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny version for smoke tests")
    ap.add_argument("--backend", default="logspace_numpy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = SETTINGS if not args.quick else [("quick d=60", 60, 1.0, "neutral", 0, 3),
                                                ("quick interv.", 60, 10.0, "interventional", 3, 2)]
    epsilons = EPSILONS if not args.quick else [1e-2, 1e-4]
    n_train = N_TRAIN if not args.quick else 60

    rows, decay_rows = [], []
    t_start = time.perf_counter()
    for label, d, gs, vf, n_b, n_inst in settings:
        X, y, _ = synthetic_regression(n_train + 20, d, min(10, d // 2), seed=args.seed, scale=1.0)
        model = fit_krr(X[:n_train], y[:n_train], gamma_scale=gs)
        ex = RKHSExplainer(model, backend=args.backend)
        kw = {"background": X[:n_b]} if vf == "interventional" else {}
        exact_threshold = (d + 1) // 2
        log_line(f"[{NAME}] {label}: d={d}, gamma*d={gs}, {vf}, n_b={n_b}, {n_inst} instances")

        for inst in range(n_inst):
            x = X[n_train + inst]
            t_exact = timed(ex.explain, x, vf, m_q="exact", **kw)
            phi_exact = t_exact.value
            summary = ex.summarize(x, vf, **kw)

            # --- budget for each tolerance
            for eps in epsilons:
                t_budget = timed(ex.node_budget, x, eps, vf, **kw)
                rep = t_budget.value
                t_quad = timed(ex.explain, x, vf, m_q=rep.m_q, **kw)
                phi = t_quad.value
                err = float(np.abs(phi - phi_exact).max())
                # smallest number of nodes that meets eps in hindsight
                m_emp = next((m for m in range(1, rep.m_q + 1)
                              if np.abs(ex.explain(x, vf, m_q=m, **kw) - phi_exact).max() <= eps), rep.m_q)
                rows.append(dict(setting=label, d=d, gamma_times_d=gs, value_function=vf, n_background=n_b,
                                 instance=inst, eps=eps, lambda_max=rep.lambda_max, A_max=rep.scale, eta=rep.eta,
                                 m_certified=rep.m_q, m_empirical=m_emp, exact_threshold=exact_threshold,
                                 certified_bound=rep.bound, observed_err=err,
                                 bound_holds=bool(err <= rep.bound + 1e-15),
                                 eps_holds=bool(err <= eps + 1e-15),
                                 tightness=float(rep.bound / err) if err > 0 else np.inf,
                                 efficiency_residual=None, t_budget_s=t_budget.seconds,
                                 t_quad_s=t_quad.seconds, t_exact_s=t_exact.seconds))

            # --- decay curve (first instance only)
            if inst == 0:
                m_top = min(exact_threshold, 64)      # the informative range; beyond it the error is at the floor
                ms = sorted(set(list(range(1, min(exact_threshold, 13))) +
                                [int(round(v)) for v in np.geomspace(12, m_top, 8)] + [exact_threshold]))
                for m in ms:
                    phi_m = ex.explain(x, vf, m_q=m, **kw)
                    decay_rows.append(dict(setting=label, d=d, gamma_times_d=gs, value_function=vf,
                                           lambda_max=summary.lambda_max, m_q=m,
                                           observed_err=float(np.abs(phi_m - phi_exact).max()),
                                           certified_bound=certify(summary, m),
                                           efficiency_residual=float(abs(
                                               phi_m.sum() - (ex.model.predict_scale(x[None, :])[0]
                                                              - ex.value_function_at_empty(vf, **kw))))))
    elapsed = time.perf_counter() - t_start

    d_out = out_dir(NAME)
    write_csv(rows, d_out / "certificate.csv")
    write_csv(decay_rows, d_out / "decay.csv")

    # ---------------------------------------------------------------- summary per (setting, eps)
    summary_rows = []
    for label, d, gs, vf, n_b, _ in settings:
        for eps in epsilons:
            sel = [r for r in rows if r["setting"] == label and r["eps"] == eps]
            if not sel:
                continue
            obs = np.array([r["observed_err"] for r in sel])
            summary_rows.append(dict(
                setting=label, d=d, value_function=vf, eps=eps,
                lambda_max=float(np.median([r["lambda_max"] for r in sel])),
                A_max=float(np.median([r["A_max"] for r in sel])),
                m_certified=int(np.median([r["m_certified"] for r in sel])),
                m_empirical=int(np.median([r["m_empirical"] for r in sel])),
                exact_threshold=sel[0]["exact_threshold"],
                certified_bound=float(np.median([r["certified_bound"] for r in sel])),
                observed_median=float(np.median(obs)), observed_max=float(obs.max()),
                bound_holds=f"{sum(r['bound_holds'] for r in sel)}/{len(sel)}",
                eps_holds=f"{sum(r['eps_holds'] for r in sel)}/{len(sel)}",
                tightness=float(np.median([r["tightness"] for r in sel])),
                t_budget_ms=float(np.median([r["t_budget_s"] for r in sel])) * 1e3,
                t_quad_ms=float(np.median([r["t_quad_s"] for r in sel])) * 1e3,
                t_exact_s=float(np.median([r["t_exact_s"] for r in sel])),
                speedup=float(np.median([r["t_exact_s"] / max(r["t_quad_s"] + r["t_budget_s"], 1e-12) for r in sel]))))
    write_csv(summary_rows, d_out / "certificate_summary.csv")

    n_ok = sum(r["bound_holds"] for r in rows)
    log_line(f"[{NAME}] certificate held in {n_ok}/{len(rows)} runs; "
             f"eps met in {sum(r['eps_holds'] for r in rows)}/{len(rows)}")

    main_rows = [r for r in summary_rows if r["eps"] in (1e-2, 1e-3, 1e-6)]
    write_latex_table(
        main_rows, d_out / "certificate_summary.tex",
        columns=["setting", "eps", "lambda_max", "m_certified", "exact_threshold", "certified_bound",
                 "observed_max", "bound_holds", "t_budget_ms", "speedup"],
        headers=[r"Setting", r"$\varepsilon$", r"$\Lambda$", r"$m_q^\star$", r"$\lceil d/2\rceil$",
                 r"certified", r"observed", r"holds", r"budget (ms)", r"speed-up"],
        formats={"eps": "sci", "lambda_max": ".1f", "certified_bound": "sci", "observed_max": "sci",
                 "t_budget_ms": ".1f", "speedup": ".0f"},
        caption=(r"A priori node budgets and their certificates. For each setting and tolerance $\varepsilon$, "
                 r"$m_q^\star$ is the number of Gauss--Legendre nodes certified from the factor tables alone, "
                 r"``certified'' is $A_{\max}B(m_q^\star,\Lambda)$ and ``observed'' the largest error against the "
                 r"exact rule over all instances and features; ``holds'' counts the runs in which the observed error "
                 r"respected the certificate. The last column is the speed-up of budget selection plus quadrature "
                 r"over the exact rule."),
        label="tab:certified-budget",
        note=r"Medians over instances; $\Lambda$ is the total relative variation of the factor table.")

    write_latex_table(
        [dict(setting=r["setting"], eps=r["eps"], m_certified=r["m_certified"], m_empirical=r["m_empirical"],
              extra=r["m_certified"] - r["m_empirical"], tightness=r["tightness"])
         for r in summary_rows if r["eps"] in (1e-2, 1e-3, 1e-6)],
        d_out / "tightness.tex",
        columns=["setting", "eps", "m_certified", "m_empirical", "extra", "tightness"],
        headers=[r"Setting", r"$\varepsilon$", r"$m_q^\star$", r"$m_q^{\min}$", r"excess", r"certified/observed"],
        formats={"eps": "sci", "tightness": ".1e"},
        caption=(r"Conservativeness of the certificate: $m_q^\star$ is the certified budget, $m_q^{\min}$ the "
                 r"smallest budget that meets $\varepsilon$ in hindsight, and the last column the ratio of the "
                 r"certified bound to the observed error."),
        label="tab:certificate-tightness")

    # ---------------------------------------------------------------- decay figure
    fig, ax = figure(d_out / "decay.pdf", figsize=(6.0, 3.8))
    if fig is not None:
        labels = [s[0] for s in settings]
        colors = {lab: f"C{i}" for i, lab in enumerate(labels)}
        for lab in labels:
            sel = [r for r in decay_rows if r["setting"] == lab]
            if not sel:
                continue
            m = [r["m_q"] for r in sel]
            ax.semilogy(m, np.maximum([r["observed_err"] for r in sel], 1e-17), "-o", ms=3,
                        color=colors[lab], label=f"{lab} ($\\Lambda$={sel[0]['lambda_max']:.0f})")
            ax.semilogy(m, np.maximum([r["certified_bound"] for r in sel], 1e-17), "--", lw=1,
                        color=colors[lab], alpha=0.6)
        ax.set_xlabel("number of Gauss--Legendre nodes $m_q$")
        ax.set_ylabel("max$_i|\\widehat\\phi_i-\\phi_i|$")
        ax.set_title("observed error (solid) and certified bound (dashed)")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        save_figure(fig, d_out / "decay.pdf")

    write_meta(NAME, settings=[dict(zip(["label", "d", "gamma_times_d", "value_function", "n_background",
                                         "n_instances"], s)) for s in settings],
               epsilons=epsilons, n_train=n_train, backend=args.backend, seed=args.seed,
               runs=len(rows), bound_held=int(n_ok), elapsed_seconds=elapsed)
    log_line(f"[{NAME}] done in {elapsed:.1f}s -> {d_out}")


if __name__ == "__main__":
    main()
