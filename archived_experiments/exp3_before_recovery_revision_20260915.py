"""
Experiment 3 -- 300 samples, 1000 features: accuracy, feature recovery and cost against the SHAP family.

One RBF kernel-ridge model on synthetic data with ten informative features gives two references:

  * the **exact Shapley values** of the fitted model, which QuadraSHAP produces at the exactness
    threshold ceil(d/2) -- so every approximation can be scored against the quantity it is
    estimating, rather than against another approximation;
  * the **generative** informative set, against which every method's ranking is scored
    (precision@k, recall@k, AUROC).

All methods explain exactly the same value function -- the empirical interventional one over a
shared background sample -- so the comparison is about the estimator, not about the semantics.
QuadraSHAP is run with its a priori budget for several tolerances; ``shap``'s PermutationExplainer
and KernelExplainer are run at two budgets each (the dependency-free implementations in
``baselines.py`` are used when ``shap`` is not installed, and the table records which ran).

A second block compares the two *exact* methods for the neutral-factor value function,
PKeX-Shapley and QuadraSHAP, on time and on agreement.

Outputs (experiments/results/exp3_synthetic_recovery/):
    runs.csv, summary.csv, accuracy.tex, recovery.tex, exact_methods.tex, tradeoff.pdf, meta.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import r2_score

from baselines import SHAP_AVAILABLE, kernel_shap, permutation_shap, pkex_shapley, random_attribution
from common import (figure, fit_krr, log_line, out_dir, rank_metrics, recovery_metrics, save_figure,
                    synthetic_regression, timed, write_csv, write_latex_table, write_meta)
from quadrashap import RKHSExplainer

NAME = "exp3_synthetic_recovery"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--d", type=int, default=1000)
    ap.add_argument("--n-informative", type=int, default=10)
    ap.add_argument("--amplitude", type=float, default=3.0, help="amplitude of the informative columns")
    ap.add_argument("--n-instances", type=int, default=10)
    ap.add_argument("--n-background", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-5])
    ap.add_argument("--permutations", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--kernel-samples", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--gamma-scale", type=float, default=1.0)
    ap.add_argument("--backend", default="logspace_numpy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.d, args.n_instances, args.n_background = 80, 60, 2, 3
        args.eps, args.permutations, args.kernel_samples = [1e-2, 1e-4], [1], [256]

    t_start = time.perf_counter()
    n_test = args.n_instances
    X, y, informative = synthetic_regression(args.n_train + n_test, args.d, args.n_informative,
                                             seed=args.seed, scale=1.0, amplitude=args.amplitude)
    Xtr, ytr, Xte = X[:args.n_train], y[:args.n_train], X[args.n_train:]
    model = fit_krr(Xtr, ytr, gamma_scale=args.gamma_scale)
    r2 = (r2_score(ytr, model.predict(Xtr)), r2_score(y[args.n_train:], model.predict(Xte)))
    log_line(f"[{NAME}] KRR on {args.n_train}x{args.d}: train R2={r2[0]:.3f}, test R2={r2[1]:.3f}; "
             f"shap package {'available' if SHAP_AVAILABLE else 'NOT available (using fallbacks)'}")

    ex = RKHSExplainer(model, backend=args.backend)
    background = Xtr[:args.n_background]
    f = lambda P: model.predict(np.atleast_2d(P))
    rows, exact_rows = [], []

    for inst in range(n_test):
        x = Xte[inst]
        t_ref = timed(ex.explain, x, "interventional", background=background, m_q="exact")
        phi_exact = t_ref.value
        log_line(f"[{NAME}]  instance {inst}: exact rule ({(args.d + 1) // 2} nodes) in {t_ref.seconds:.1f}s")

        def record(method, phi, seconds, implementation, budget, nodes=None, evals=None):
            rows.append(dict(instance=inst, method=method, implementation=implementation, budget=budget,
                             m_q=nodes, evaluations=evals, seconds=seconds,
                             **rank_metrics(phi, phi_exact, ks=(1, 5, 10)),
                             **recovery_metrics(phi, informative, ks=(5, 10, 20)),
                             phi_max=float(np.abs(phi_exact).max())))

        record("QuadraSHAP (exact)", phi_exact, t_ref.seconds, "quadrashap", r"$\lceil d/2\rceil$",
               nodes=(args.d + 1) // 2)
        for eps in args.eps:
            t_b = timed(ex.node_budget, x, eps, "interventional", background=background)
            t_q = timed(ex.explain, x, "interventional", background=background, m_q=t_b.value.m_q)
            record(f"QuadraSHAP ($\\varepsilon$={eps:g})", t_q.value, t_b.seconds + t_q.seconds,
                   "quadrashap", f"eps={eps:g}", nodes=t_b.value.m_q)
        for n_perm in args.permutations:
            a = permutation_shap(f, x, background, n_permutations=n_perm, seed=args.seed + inst)
            record(f"PermutationSHAP ({n_perm}$\\times$)", a.phi, a.seconds, a.implementation,
                   f"{n_perm} permutations", evals=a.evaluations)
        for ns in args.kernel_samples:
            a = kernel_shap(f, x, background, n_samples=ns, seed=args.seed + inst)
            record(f"KernelSHAP ({ns})", a.phi, a.seconds, a.implementation, f"{ns} samples", evals=a.evaluations)
        a = random_attribution(args.d, seed=args.seed + inst)
        record("random", a.phi, 0.0, "random", "--")

        # ---- exact methods for the neutral-factor value function: PKeX-Shapley vs QuadraSHAP
        t_neutral = timed(ex.explain, x, "neutral", m_q="exact")
        U = ex.factors(x)
        t_pkex = timed(pkex_shapley, U, ex.coef, return_diagnostics=True)
        phi_pkex, diag = t_pkex.value
        rep = ex.node_budget(x, 1e-3, "neutral")
        t_budget = timed(ex.explain, x, "neutral", m_q=rep.m_q)
        exact_rows.append(dict(instance=inst,
                               t_pkex_s=t_pkex.seconds, t_quadrashap_exact_s=t_neutral.seconds,
                               t_quadrashap_eps_s=t_budget.seconds + rep.seconds, m_q_eps=rep.m_q,
                               rel_diff=float(np.abs(phi_pkex - t_neutral.value).max()
                                              / max(np.abs(t_neutral.value).max(), 1e-300)),
                               pkex_max_abs_e=diag["max_abs_e"], pkex_finite=diag["finite"]))

    elapsed = time.perf_counter() - t_start
    d_out = out_dir(NAME)
    write_csv(rows, d_out / "runs.csv")
    write_csv(exact_rows, d_out / "exact_methods.csv")

    impl_note = r"\texttt{shap} package" if SHAP_AVAILABLE else "reference implementations (see baselines.py)"
    methods = list(dict.fromkeys(r["method"] for r in rows))
    summary = []
    for meth in methods:
        sel = [r for r in rows if r["method"] == meth]
        agg = lambda k, fn=np.mean: float(fn([r[k] for r in sel]))
        summary.append(dict(method=meth, implementation=sel[0]["implementation"], budget=sel[0]["budget"],
                            m_q=sel[0]["m_q"], evaluations=sel[0]["evaluations"],
                            seconds=agg("seconds", np.median),
                            max_abs_err=agg("max_abs_err", np.max), rel_l2_err=agg("rel_l2_err"),
                            rel_err=agg("max_abs_err", np.max) / agg("phi_max", np.median),
                            kendall_tau=agg("kendall_tau"), top10_overlap=agg("top10_overlap"),
                            precision10=agg("precision@10"), recall10=agg("recall@10"), auroc=agg("auroc")))
    write_csv(summary, d_out / "summary.csv")

    write_latex_table(
        summary, d_out / "accuracy.tex",
        columns=["method", "budget", "seconds", "max_abs_err", "rel_l2_err", "kendall_tau", "top10_overlap"],
        headers=[r"Method", r"budget", r"time (s)", r"max error", r"rel.\ $\ell_2$", r"Kendall $\tau$", r"top-10"],
        formats={"seconds": ".2f", "max_abs_err": "sci", "rel_l2_err": "sci", "kendall_tau": ".3f",
                 "top10_overlap": ".2f"},
        caption=(rf"Accuracy against the exact Shapley values of the same model "
                 rf"($d={args.d}$, $n={args.n_train}$, empirical interventional value function with "
                 rf"$n_b={args.n_background}$ background rows, {n_test} instances). QuadraSHAP at the "
                 r"exactness threshold is the reference, so its error is zero by construction; the rows below it "
                 r"differ only in how the same value function is estimated."),
        label="tab:synthetic-accuracy",
        note=("Median time, worst-case error over instances and features. Sampling baselines: " + impl_note + "."))

    write_latex_table(
        summary, d_out / "recovery.tex",
        columns=["method", "budget", "seconds", "precision10", "recall10", "auroc"],
        headers=[r"Method", r"budget", r"time (s)", r"precision@10", r"recall@10", r"AUROC"],
        formats={"seconds": ".2f", "precision10": ".2f", "recall10": ".2f", "auroc": ".3f"},
        caption=(rf"Recovery of the {args.n_informative} generative informative features out of $d={args.d}$. "
                 r"The exact Shapley values of the fitted model are the ceiling any exact method can reach; "
                 r"the sampling estimators fall below it at a much larger cost."),
        label="tab:synthetic-recovery")

    if exact_rows:
        e_agg = {k: float(np.median([r[k] for r in exact_rows])) for k in
                 ("t_pkex_s", "t_quadrashap_exact_s", "t_quadrashap_eps_s", "rel_diff", "m_q_eps")}
        write_latex_table(
            [dict(method="PKeX-Shapley", budget=r"$O(nd^2)$ exact", seconds=e_agg["t_pkex_s"], rel_diff=None,
                  speedup=None),
             dict(method="QuadraSHAP (exact rule)", budget=rf"$m_q={(args.d + 1) // 2}$",
                  seconds=e_agg["t_quadrashap_exact_s"], rel_diff=e_agg["rel_diff"],
                  speedup=e_agg["t_pkex_s"] / max(e_agg["t_quadrashap_exact_s"], 1e-12)),
             dict(method=r"QuadraSHAP ($\varepsilon=10^{-3}$)", budget=rf"$m_q={int(e_agg['m_q_eps'])}$",
                  seconds=e_agg["t_quadrashap_eps_s"], rel_diff=None,
                  speedup=e_agg["t_pkex_s"] / max(e_agg["t_quadrashap_eps_s"], 1e-12))],
            d_out / "exact_methods.tex",
            columns=["method", "budget", "seconds", "rel_diff", "speedup"],
            headers=[r"Method", r"budget", r"time (s)", r"rel.\ difference", r"speed-up over PKeX"],
            formats={"seconds": ".3f", "rel_diff": "sci", "speedup": ".1f"},
            caption=(r"The two exact methods for the neutral-factor value function on the same model "
                     rf"($d={args.d}$, $n={args.n_train}$): PKeX-Shapley's $O(nd^2)$ symmetric-polynomial "
                     r"recursion and QuadraSHAP's quadrature, which agree to machine precision. The certified "
                     r"budget needs a few nodes instead of $\lceil d/2\rceil$."),
            label="tab:exact-methods",
            note=r"Medians over instances; PKeX-Shapley is a reimplementation of the published recursion.")

    # ---------------------------------------------------------------- accuracy vs cost figure
    fig, ax = figure(d_out / "tradeoff.pdf", figsize=(5.6, 3.8))
    if fig is not None:
        for i, r in enumerate(summary):
            if r["method"] == "random" or r["max_abs_err"] <= 0:
                continue
            fam = ("QuadraSHAP" if "Quadra" in r["method"] else
                   "PermutationSHAP" if "Permutation" in r["method"] else "KernelSHAP")
            ax.loglog(max(r["seconds"], 1e-3), r["max_abs_err"], marker="o^s"["QPK".index(fam[0])],
                      color={"QuadraSHAP": "C0", "PermutationSHAP": "C1", "KernelSHAP": "C2"}[fam], ms=7)
            ax.annotate(r["budget"].replace("$", "").replace("\\varepsilon", "eps"),
                        (max(r["seconds"], 1e-3), r["max_abs_err"]), fontsize=5,
                        textcoords="offset points", xytext=(4, 3))
        for fam, c, mk in (("QuadraSHAP", "C0", "o"), ("PermutationSHAP", "C1", "^"), ("KernelSHAP", "C2", "s")):
            ax.plot([], [], mk, color=c, label=fam)
        ax.set_xlabel("time per instance (s)"); ax.set_ylabel(r"max$_i|\widehat\phi_i-\phi_i|$")
        ax.set_title(f"accuracy vs cost, $d={args.d}$"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
        save_figure(fig, d_out / "tradeoff.pdf")

    write_meta(NAME, config=vars(args), train_r2=r2[0], test_r2=r2[1], shap_available=SHAP_AVAILABLE,
               n_instances=n_test, elapsed_seconds=elapsed)
    log_line(f"[{NAME}] done in {elapsed:.1f}s -> {d_out}")


if __name__ == "__main__":
    main()
