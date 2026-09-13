"""
Experiment 5 -- the two evaluators: which is faster, and which is stable where?

QuadraSHAP evaluates the same quadrature rule in two ways:

  * **log-space shared product** -- for each node one shared product
    ``P_q = prod_j T_j(t_q)`` is accumulated as a log-magnitude with a separate sign,
    and the leave-one-out products are recovered by subtracting ``log|T_i(t_q)|``.
    Working set ``O(m_q * n)``, two transcendentals per factor, and factor magnitudes
    are clamped at ``eps``, so it is *approximate* whenever a factor is smaller than
    ``eps`` and wrong wherever one vanishes.
  * **prefix--suffix scan** (the division-free algorithm of the appendix) -- the
    leave-one-out products are assembled as ``pref_i * suf_i`` from an exclusive prefix
    and an exclusive suffix product.  No division, no clamping, exact when a factor
    vanishes, no transcendentals -- but the working set is ``O(m_q * n * d)`` and the
    partial products are formed in plain floating point, so they overflow or underflow
    whenever a *prefix* leaves the float64 range, even if the answer does not.

Part A times both over a grid of ``(d, m_q)``, with and without the blockwise plan, so
the cache effect on the scan is visible.

Part B measures both against a ``float128`` prefix--suffix reference (exponent range
1e+-4932, no clamping) on six regimes built as explicit present/absent factor tables:

  near one              wide product kernel, factors just below 1 -- the benign case;
  long products         narrow kernel, d=1000: the shared product reaches 1e-217;
  underflowing          narrow kernel, d=5000: the shared product underflows to zero at the
                        upper nodes -- harmlessly, since those nodes contribute nothing;
  mixed magnitudes      GLM-type factors exp(+-c): the full product is O(1) but the
                        prefix products overflow -- the scan's failure mode;
  vanishing factors      a few factors at 1e-18 (far coordinates of a narrow kernel):
                        below the log-space clamp -- log-space's failure mode;
  sign-changing factor  a factor that is exactly zero at a quadrature node: the scan is
                        exact, log-space clamps the zero.

Outputs (experiments/results/exp5_backend_and_stability/):
    timing.csv, timing.tex, stability.csv, stability.tex, backends.pdf, meta.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import figure, log_line, out_dir, save_figure, write_csv, write_latex_table, write_meta
from quadrashap.product_games.blocks import plan_blocks
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy

NAME = "exp5_backend_and_stability"
CORE = ProductGamesShapleyNumpy()
M_Q_STAB = 16


def nodes_01(m_q: int):
    x, w = np.polynomial.legendre.leggauss(m_q)
    return 0.5 * (x + 1.0), 0.5 * w


def reference_phi(K: np.ndarray, Ut, m_q: int) -> np.ndarray:
    """Prefix--suffix evaluation in float128: no clamping and an exponent range of 1e+-4932."""
    K = np.asarray(K, dtype=np.longdouble)
    m, d = K.shape
    Ut = (np.ones((1, d), dtype=np.longdouble) if Ut is None
          else np.asarray(Ut, dtype=np.longdouble).reshape(-1, d))
    x, w = nodes_01(m_q)
    x = x.astype(np.longdouble)
    w = w.astype(np.longdouble)
    acc = np.zeros((m, d), dtype=np.longdouble)
    one = np.ones((m, 1), dtype=np.longdouble)
    for q in range(m_q):
        B = Ut + x[q] * K
        pref = np.concatenate([one, np.cumprod(B[:, :-1], axis=1)], axis=1)
        suf = np.concatenate([np.cumprod(B[:, :0:-1], axis=1)[:, ::-1], one], axis=1)
        acc += w[q] * pref * suf
    return K * acc


# --------------------------------------------------------------------------- factor tables
def kernel_table(n: int, d: int, decay: float, seed: int = 0):
    """Present factors ``u = exp(-|z|)``, ``E[-log u] = decay``; absent factor neutral.

    ``decay * d`` is how far the shared product travels from one in log-magnitude:
    ``decay * d = 709`` is where a float64 product underflows.
    """
    rng = np.random.default_rng(seed)
    return np.exp(-rng.exponential(decay, size=(n, d))), None


def mixed_table(n: int, d: int, c: float, seed: int = 0):
    """GLM-type factors ``u_j = exp(+-c)``: the full product is 1, the prefix products are not.

    Half the features carry ``exp(+c)`` and half ``exp(-c)``, ordered so that the exclusive
    prefix product climbs to ``exp(c*d/2)`` before coming back down.
    """
    rng = np.random.default_rng(seed)
    s = np.concatenate([np.ones(d // 2), -np.ones(d - d // 2)])
    U = np.exp(c * s + 0.01 * rng.standard_normal((n, d)))
    return U, None


def vanishing_table(n: int, d: int, n_small: int, small: float, seed: int = 0):
    """A narrow product kernel: most factors are O(1), a few are ``small`` in *both* tables.

    A coordinate far from the background has ``u_j = ut_j ~ exp(-gamma r^2)`` tiny, so the
    factor ``T_j(t) = (1-t) ut_j + t u_j`` is tiny at every node -- below the log-space clamp.
    """
    rng = np.random.default_rng(seed)
    U = np.exp(-rng.exponential(1e-2, size=(n, d)))
    Ut = np.exp(-rng.exponential(1e-2, size=(1, d)))
    idx = rng.choice(d, size=n_small, replace=False)
    U[:, idx] = small * np.exp(-rng.exponential(0.1, size=(n, n_small)))
    Ut[:, idx] = small
    return U, Ut


def sign_change_table(n: int, d: int, m_q: int, n_zero: int, seed: int = 0):
    """Factors that are exactly zero at a quadrature node.

    With ``ut_j = t*`` and ``u_j = t* - 1`` the factor is ``T_j(t) = t* - t``, which vanishes
    at ``t = t*``.  Choosing ``t*`` to be a Gauss--Legendre node puts an exact zero into the
    evaluation -- the case the division-free scan exists for.
    """
    rng = np.random.default_rng(seed)
    x, _ = nodes_01(m_q)
    U = np.exp(-rng.exponential(1e-2, size=(n, d)))
    Ut = np.exp(-rng.exponential(1e-2, size=(1, d)))
    idx = rng.choice(d, size=n_zero, replace=False)
    tstar = x[rng.choice(m_q, size=n_zero)]
    Ut[0, idx] = tstar
    U[:, idx] = tstar - 1.0
    return U, Ut


def stability_cases(quick: bool):
    """``(label, builder)``; each builder returns ``(U, Ut)`` -- present and absent factors."""
    if quick:
        return [("near one (wide kernel)", lambda n: kernel_table(n, 200, 1e-3, 1)),
                ("sign-changing factor", lambda n: sign_change_table(n, 200, M_Q_STAB, 3, 1))]
    return [
        ("near one (wide kernel)", lambda n: kernel_table(n, 1000, 1e-3, 1)),
        ("long products (narrow kernel)", lambda n: kernel_table(n, 1000, 0.5, 1)),
        ("underflowing shared product", lambda n: kernel_table(n, 5000, 0.5, 1)),
        ("mixed magnitudes (GLM)", lambda n: mixed_table(n, 1000, 1.5, 1)),
        ("vanishing factors", lambda n: vanishing_table(n, 1000, 5, 1e-18, 1)),
        ("sign-changing factor", lambda n: sign_change_table(n, 1000, M_Q_STAB, 5, 1)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-games", type=int, default=300)
    ap.add_argument("--n-stability", type=int, default=32, help="games per stability regime (float128 is slow)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-unblocked-gb", type=float, default=3.0,
                    help="skip the unblocked scan when its working set exceeds this (it is the point of blocking)")
    args = ap.parse_args()

    grid = ([(200, (4, 16, 64, 100)), (1000, (4, 16, 64, 500)), (5000, (4, 16, 64))] if not args.quick
            else [(200, (4, 16))])
    n = args.n_games
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ Part A: timing
    rows = []
    for d, m_qs in grid:
        U, _ = kernel_table(n, d, 1e-3, seed=args.seed)
        K, Ut = U - 1.0, None
        for m_q in m_qs:
            for backend, fn in (("logspace_numpy", CORE.phi_matrix_logspace),
                                ("prefix_scan_numpy", CORE.phi_matrix_prefix_scan)):
                for blocked in (True, False):
                    if backend.startswith("logspace") and not blocked:
                        continue                      # log-space has no (m_q, n, d) tensor to block
                    plan = plan_blocks(n, m_q, d, backend, block_size="auto" if blocked else -1)
                    if not blocked and plan.estimated_peak_bytes > args.max_unblocked_gb * 2 ** 30:
                        rows.append(dict(d=d, m_q=m_q, n_games=n, backend=backend, blocking="off",
                                         block_size=plan.block_size, node_block=plan.node_block,
                                         peak_mb=plan.estimated_peak_bytes / 2 ** 20, seconds=None))
                        log_line(f"[{NAME}] d={d:5d} m_q={m_q:4d} {backend:18s} blocking=off  -> skipped "
                                 f"(working set ~{rows[-1]['peak_mb'] / 1024:.1f} GB)")
                        continue
                    ts = []
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        if backend.startswith("prefix"):
                            out = np.zeros((n, d))
                            for r0 in range(0, n, plan.block_size):
                                out[r0:r0 + plan.block_size] = fn(K[r0:r0 + plan.block_size], m_q, Ut=Ut,
                                                                  node_block=plan.node_block)
                        else:
                            out = fn(K, m_q, Ut=Ut)
                        ts.append(time.perf_counter() - t0)
                    rows.append(dict(d=d, m_q=m_q, n_games=n, backend=backend,
                                     blocking="auto" if blocked else "off",
                                     block_size=plan.block_size, node_block=plan.node_block,
                                     peak_mb=plan.estimated_peak_bytes / 2 ** 20, seconds=float(np.min(ts))))
                    log_line(f"[{NAME}] d={d:5d} m_q={m_q:4d} {backend:18s} blocking={'auto' if blocked else 'off ':4s}"
                             f" -> {rows[-1]['seconds']:7.3f}s  (peak ~{rows[-1]['peak_mb']:.0f} MB)")

    # ------------------------------------------------------------------ Part B: stability
    ns = args.n_stability
    srows = []
    for label, build in stability_cases(args.quick):
        U, Ut = build(ns)
        d = U.shape[1]
        K = U - (1.0 if Ut is None else Ut)
        ref = reference_phi(K, Ut, M_Q_STAB)
        scale = float(np.abs(ref).max())              # the size of the attribution we are asked to resolve

        # diagnostics in float128: the partial products are what decide whether the scan survives
        Utb = (np.ones((1, d)) if Ut is None else np.asarray(Ut)).astype(np.longdouble)
        x, _ = nodes_01(M_Q_STAB)
        T = Utb[None, :, :] + x[:, None, None].astype(np.longdouble) * K[None, :, :].astype(np.longdouble)
        A = np.abs(T)
        nz = A[A > 0]
        min_factor = nz.min() if nz.size else np.longdouble(0.0)
        pref = np.abs(np.cumprod(T, axis=2))                          # (m_q, ns, d) exclusive-prefix magnitudes
        pnz = pref[pref > 0]
        def log10(v):
            """Rounded decimal exponent; ``v`` may be a float128 outside the float64 range."""
            v = np.longdouble(v)
            if not np.isfinite(v) or v <= 0:
                return float("-inf") if v <= 0 else float("inf")
            return int(round(float(np.log10(v))))
        max_prefix = log10(pref.max())
        min_prefix = log10(pnz.min() if pnz.size else np.longdouble(0.0))
        n_zero_factors = int((A == 0).sum())

        for backend, fn in (("logspace_numpy", CORE.phi_matrix_logspace),
                            ("prefix_scan_numpy", CORE.phi_matrix_prefix_scan)):
            with np.errstate(all="ignore"):
                phi = fn(K, M_Q_STAB, Ut=Ut)
            finite = bool(np.isfinite(phi).all())
            err = np.abs(np.nan_to_num(phi, nan=np.inf, posinf=np.inf, neginf=np.inf).astype(np.longdouble) - ref)
            rel = float(err.max() / max(scale, 1e-300)) if scale > 0 else float(err.max())
            srows.append(dict(case=label, d=d, backend=backend,
                              log10_min_factor=log10(min_factor),
                              log10_max_prefix=max_prefix, log10_min_prefix=min_prefix,
                              log10_max_phi=log10(scale), zero_factors=n_zero_factors,
                              rel_err=rel, finite=finite))
            log_line(f"[{NAME}] {label:30s} d={d:5d} {backend:18s} rel.err {rel:9.2e} "
                     f"finite={finite} (prefix range 1e{min_prefix:.0f}..1e{max_prefix:.0f}, "
                     f"min factor 1e{log10(min_factor):.0f}, max|phi| 1e{log10(scale):.0f})")

    elapsed = time.perf_counter() - t_start
    d_out = out_dir(NAME)
    write_csv(rows, d_out / "timing.csv")
    write_csv(srows, d_out / "stability.csv")

    # ------------------------------------------------------------------ tables
    tbl = []
    for d, m_qs in grid:
        for m_q in m_qs:
            def get(b, bl):
                return next((r["seconds"] for r in rows if r["d"] == d and r["m_q"] == m_q
                             and r["backend"] == b and r["blocking"] == bl), None)
            ls, ps, pu = get("logspace_numpy", "auto"), get("prefix_scan_numpy", "auto"), get("prefix_scan_numpy", "off")
            tbl.append(dict(d=d, m_q=m_q, logspace=ls, prefix=ps, prefix_unblocked=pu,
                            ratio=(ls / ps) if ls and ps else None,
                            peak=next((r["peak_mb"] for r in rows if r["d"] == d and r["m_q"] == m_q
                                       and r["backend"] == "prefix_scan_numpy" and r["blocking"] == "off"), None)))
    write_latex_table(
        tbl, d_out / "timing.tex",
        columns=["d", "m_q", "logspace", "prefix", "prefix_unblocked", "ratio", "peak"],
        headers=[r"$d$", r"$m_q$", r"log-space (s)", r"scan, blocked (s)", r"scan, unblocked (s)",
                 r"log-space/scan", r"scan working set (MB)"],
        formats={"logspace": ".3f", "prefix": ".3f", "prefix_unblocked": ".3f", "ratio": ".2f", "peak": ".0f"},
        caption=(rf"Cost of the two evaluators of the same quadrature rule on {n} product games. The "
                 r"division-free prefix--suffix scan does no transcendental work and is the faster of the two "
                 r"once its $(m_q\times n\times d)$ working set is blocked to fit cache; unblocked, the same "
                 r"arithmetic is memory-bound and loses to the log-space evaluator, whose working set is "
                 r"$O(m_q n)$."),
        label="tab:backends-timing",
        note=(r"Best of three runs, single-threaded NumPy. The blocked column is the automatic plan, which "
              r"leaves the computation whole whenever one $(m_q\times n\times d)$ sweep already fits the "
              r"last-level cache -- hence the identical entries in the first rows."))

    # exact zeros and overflows read better as symbols than as a missing entry
    def tex_row(r):
        r = dict(r)
        for k in ("log10_min_factor", "log10_min_prefix", "log10_max_prefix", "log10_max_phi"):
            if r[k] == float("-inf"):
                r[k] = "$0$"
            elif r[k] == float("inf"):
                r[k] = r"$\infty$"
        if not np.isfinite(r["rel_err"]):
            r["rel_err"] = r"overflow"
        return r

    write_latex_table(
        [tex_row(r) for r in srows], d_out / "stability.tex",
        columns=["case", "d", "backend", "log10_min_factor", "log10_min_prefix", "log10_max_prefix",
                 "log10_max_phi", "zero_factors", "rel_err", "finite"],
        headers=[r"Regime", r"$d$", r"evaluator", r"$\log_{10}\min|T_j|$", r"$\log_{10}\min|\text{prefix}|$",
                 r"$\log_{10}\max|\text{prefix}|$", r"$\log_{10}\max|\phi|$", r"\#\,zeros", r"rel.\ error", r"finite"],
        formats={"rel_err": "sci"},
        column_format="llrrrrrrrc",
        caption=(r"Accuracy of the two evaluators against a \texttt{float128} prefix--suffix reference. The scan "
                 r"forms the partial products in plain arithmetic, so it fails when a \emph{prefix} leaves the "
                 r"float64 range even though the answer does not; the log-space evaluator is immune to that but "
                 r"clamps factor magnitudes at $10^{-12}$, so it is the approximate one exactly where a factor "
                 r"vanishes or is very small. Underflow of the shared product is by itself harmless (row three): "
                 r"it happens at the upper nodes, whose contribution to the integral is negligible anyway. What "
                 r"breaks the scan is a partial product leaving the float64 range while the answer does not."),
        label="tab:backends-stability",
        note=(rf"Largest relative error over all features and games, $m_q={M_Q_STAB}$, on the scale "
              r"$\max_{r,i}|\phi_{r,i}|$."))

    # ------------------------------------------------------------------ figure
    fig, ax = figure(d_out / "backends.pdf", figsize=(5.8, 3.6))
    if fig is not None:
        for i, (d, m_qs) in enumerate(grid):
            for backend, ls in (("logspace_numpy", "-"), ("prefix_scan_numpy", "--")):
                sel = [r for r in rows if r["d"] == d and r["backend"] == backend and r["blocking"] == "auto"]
                sel.sort(key=lambda r: r["m_q"])
                ax.loglog([r["m_q"] for r in sel], [r["seconds"] for r in sel], ls, marker="o", ms=3,
                          color=f"C{i}", label=f"$d={d}$, {'log-space' if 'log' in backend else 'scan'}")
        ax.set_xlabel("number of nodes $m_q$")
        ax.set_ylabel(f"time for {n} product games (s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
        save_figure(fig, d_out / "backends.pdf")

    write_meta(NAME, grid=[{"d": d, "m_q": list(ms)} for d, ms in grid], n_games=n, repeats=args.repeats,
               n_stability=ns, m_q_stability=M_Q_STAB,
               stability_cases=[c[0] for c in stability_cases(args.quick)], elapsed_seconds=elapsed)
    log_line(f"[{NAME}] done in {elapsed:.1f}s -> {d_out}")


if __name__ == "__main__":
    main()
