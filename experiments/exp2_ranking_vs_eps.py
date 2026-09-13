"""
Experiment 2 -- does the certified tolerance preserve the ranking of the Shapley values?

The certificate bounds an *absolute* error, so what matters for a ranking is the tolerance
relative to the size of the attributions.  We therefore fit the same model to targets of
increasing spread (``y`` standard normal, then scaled by 10 and 100, so that the attributions
grow by the same factor) and, for each tolerance eps, compare the ranking produced with the
certified budget against the ranking of the exact Shapley values:

  * Kendall tau / Spearman over all d features,
  * top-k overlap for k = 1, 5, 10 and whether the full ranking is identical,
  * sign agreement,
  * the realised relative error max_i |phi_hat_i - phi_i| / max_i |phi_i|.

The x-axis that makes every setting collapse onto one curve is eps / max_i |phi_i|, i.e. the
tolerance expressed as a fraction of the largest attribution.

Outputs (experiments/results/exp2_ranking_vs_eps/):
    ranking.csv, ranking_summary.csv, ranking_summary.tex, ranking.pdf, meta.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (figure, fit_krr, log_line, out_dir, rank_metrics, save_figure, synthetic_regression,
                    timed, write_csv, write_latex_table, write_meta)
from quadrashap import RKHSExplainer

NAME = "exp2_ranking_vs_eps"

# (label, d, gamma*d, value function, n_background, n_instances)
SETTINGS = [
    ("d=200, neutral",        200,  1.0, "neutral",        0, 20),
    ("d=200, interventional", 200,  1.0, "interventional", 5, 20),
    ("d=1000, neutral",      1000,  1.0, "neutral",        0,  8),
]
SCALES = [1.0, 10.0, 100.0]          # std of the target, hence of the attributions
EPSILONS = [1e0, 1e-1, 1e-2, 1e-3, 1e-5]
N_TRAIN = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default="logspace_numpy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = SETTINGS if not args.quick else [("quick d=80", 80, 1.0, "neutral", 0, 4)]
    scales = SCALES if not args.quick else [1.0, 100.0]
    epsilons = EPSILONS if not args.quick else [1e-1, 1e-3]
    n_train = N_TRAIN if not args.quick else 60

    rows = []
    t_start = time.perf_counter()
    for label, d, gs, vf, n_b, n_inst in settings:
        for scale in scales:
            X, y, _ = synthetic_regression(n_train + 30, d, min(10, d // 2), seed=args.seed, scale=scale)
            model = fit_krr(X[:n_train], y[:n_train], gamma_scale=gs)
            ex = RKHSExplainer(model, backend=args.backend)
            kw = {"background": X[:n_b]} if vf == "interventional" else {}
            log_line(f"[{NAME}] {label}, target std={scale:g}: {n_inst} instances")
            for inst in range(n_inst):
                x = X[n_train + inst]
                phi_exact = ex.explain(x, vf, m_q="exact", **kw)
                phi_scale = float(np.abs(phi_exact).max())
                for eps in epsilons:
                    rep = ex.node_budget(x, eps, vf, **kw)
                    t = timed(ex.explain, x, vf, m_q=rep.m_q, **kw)
                    phi = t.value
                    m = rank_metrics(phi, phi_exact, ks=(1, 5, 10))
                    rows.append(dict(setting=label, d=d, value_function=vf, target_std=scale, instance=inst,
                                     eps=eps, eps_relative=eps / max(phi_scale, 1e-300), phi_max=phi_scale,
                                     m_certified=rep.m_q, exact_threshold=rep.exact_threshold,
                                     lambda_max=rep.lambda_max, certified_bound=rep.bound,
                                     rel_err=m["max_abs_err"] / max(phi_scale, 1e-300), t_quad_s=t.seconds, **m))
    elapsed = time.perf_counter() - t_start

    d_out = out_dir(NAME)
    write_csv(rows, d_out / "ranking.csv")

    summary = []
    for label, d, gs, vf, n_b, _ in settings:
        for scale in scales:
            for eps in epsilons:
                sel = [r for r in rows if r["setting"] == label and r["target_std"] == scale and r["eps"] == eps]
                if not sel:
                    continue
                agg = lambda k: float(np.mean([r[k] for r in sel]))
                summary.append(dict(
                    setting=label, d=d, target_std=scale, eps=eps,
                    eps_relative=float(np.median([r["eps_relative"] for r in sel])),
                    m_certified=float(np.median([r["m_certified"] for r in sel])),
                    exact_threshold=sel[0]["exact_threshold"],
                    max_abs_err=float(np.max([r["max_abs_err"] for r in sel])),
                    rel_err=float(np.max([r["rel_err"] for r in sel])),
                    kendall_tau=agg("kendall_tau"), spearman=agg("spearman"),
                    top1=agg("top1_overlap"), top5=agg("top5_overlap"), top10=agg("top10_overlap"),
                    sign_agreement=agg("sign_agreement"), ranking_identical=agg("ranking_identical"),
                    t_quad_ms=float(np.median([r["t_quad_s"] for r in sel])) * 1e3, n=len(sel)))
    write_csv(summary, d_out / "ranking_summary.csv")

    main_setting = settings[0][0]
    write_latex_table(
        [r for r in summary if r["setting"] == main_setting],
        d_out / "ranking_summary.tex",
        columns=["target_std", "eps", "eps_relative", "m_certified", "rel_err", "top5", "top10",
                 "kendall_tau", "ranking_identical"],
        headers=[r"std$(y)$", r"$\varepsilon$", r"$\varepsilon/\max_i|\phi_i|$", r"$m_q^\star$",
                 r"rel.\ error", r"top-5", r"top-10", r"Kendall $\tau$", r"identical"],
        formats={"target_std": ".0f", "eps": "sci", "eps_relative": "sci", "m_certified": ".0f",
                 "rel_err": "sci", "top5": ".3f", "top10": ".3f", "kendall_tau": ".4f",
                 "ranking_identical": ".2f"},
        caption=(r"Ranking fidelity of the certified budget against the exact Shapley values "
                 rf"({main_setting.replace('=', '$=$')}). The tolerance $\varepsilon$ is absolute, so the "
                 r"quantity that governs the ranking is $\varepsilon$ relative to the largest attribution; "
                 r"``identical'' is the fraction of instances whose full ranking of $|\phi_i|$ is unchanged."),
        label="tab:ranking-vs-eps",
        note=r"Averages over instances; ``rel.\ error'' is the largest realised error over instances and features.")

    # ---------------------------------------------------------------- figure: agreement vs relative tolerance
    fig, ax = figure(d_out / "ranking.pdf", figsize=(6.0, 3.6))
    if fig is not None:
        for i, (label, *_rest) in enumerate(settings):
            for j, scale in enumerate(scales):
                sel = sorted([r for r in summary if r["setting"] == label and r["target_std"] == scale],
                             key=lambda r: r["eps_relative"])
                if not sel:
                    continue
                ax.semilogx([r["eps_relative"] for r in sel], [r["top10"] for r in sel],
                            marker="os^"[j % 3], color=f"C{i}", ls=["-", "--", ":"][j % 3], ms=4,
                            label=f"{label}, std$(y)$={scale:g}")
        ax.set_xlabel(r"tolerance relative to the largest attribution, $\varepsilon/\max_i|\phi_i|$")
        ax.set_ylabel("top-10 overlap with the exact ranking")
        ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower left")
        save_figure(fig, d_out / "ranking.pdf")

    write_meta(NAME, settings=[dict(zip(["label", "d", "gamma_times_d", "value_function", "n_background",
                                         "n_instances"], s)) for s in settings],
               scales=scales, epsilons=epsilons, n_train=n_train, backend=args.backend, seed=args.seed,
               runs=len(rows), elapsed_seconds=elapsed)
    log_line(f"[{NAME}] done in {elapsed:.1f}s -> {d_out}")


if __name__ == "__main__":
    main()
