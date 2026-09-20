"""
Experiment 9 -- the interventional value function over the vocabulary: where sampling hits the wall.

Experiment 7 masked words to zero, so a word absent from the document was a dummy player and the
game had only the document's ``d ~ 15-100`` distinct words.  Here the value function is the
**empirical interventional** one of Section 4 (Proposition 5): an absent word takes the TF-IDF
value it has in each of ``n_b`` background documents, averaged.  Every word that occurs in the
document *or in any background document* is then a genuine player with a non-zero attribution
("the model would have leaned further had this word been present"), nobody can prune, and the game
has ``d`` in the thousands with a 5000-word vocabulary.

For the samplers this is the standard interventional-SHAP setting -- one masked prediction per
background row per coalition, which is exactly what ``shap`` does with a background set -- and it is
where they stop being usable: KernelSHAP needs more than ``d`` coalitions before its regression is
even identifiable, Permutation SHAP needs ``d`` evaluations per sweep, and both converge like
``1/sqrt(n)`` from there.  QuadraSHAP integrates the same ``n_sv * n_b`` product games with a node
budget set by ``Lambda``, which stays small, so its cost is a few nodes' worth of passes over the
factor tables regardless of ``d``.

Cost is wall-clock seconds -- nothing else is comparable across the methods without an assumption.
The samplers are given the fast batched evaluator (one matrix product per batch of coalitions) and
a wall-clock cap per method; their error at the cap is recorded, and the time they would need for
the accuracy QuadraSHAP certifies is *extrapolated* along their fitted ``1/sqrt(n)`` slope and
labelled as such.

The reference is QuadraSHAP at the node count whose a priori certificate is ``1e-14 * A_max``,
checked two ways: agreement with ``m_ref + 8`` nodes at the float64 floor on every document, and,
once per corpus, agreement with the full exactness-threshold rule ``ceil(d/2)``.  In float64 the
latter is not more accurate than the former -- both sit on the same rounding floor -- so the
certified reference is used throughout and the ``ceil(d/2)`` run is a spot check.

Outputs (experiments/results/exp9_text_interventional/):
    records.csv, instances.csv, reference_phi.npz, error_vs_time.pdf, time_to_target.pdf,
    summary.tex, meta.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import MODELS, log_line, out_dir, plt, MATPLOTLIB, write_csv, write_latex_table, write_meta
from text_attribution import ESTIMATORS, MaskGame
from text_datasets import load_dataset
from quadrashap.product_games.blocks import plan_blocks
from quadrashap.product_games.budget import GameSummary, budget_from_summary, certify
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy

NAME = "exp9_text_interventional"
CORE = ProductGamesShapleyNumpy()

DATASETS = ["sst2", "mr", "agnews", "imdb"]
MAX_FEATURES = 5000
N_FIT = 1500
# background rows per corpus: enough that the union of their supports puts d in the thousands
# while n_sv * n_b * d stays within a few GB of factor tables.  Both QuadraSHAP and the samplers
# scale linearly in n_b, so this choice moves the absolute times, not the comparison.
N_BACKGROUND = {"sst2": 60, "mr": 60, "agnews": 40, "imdb": 20}
SVM = dict(C=10.0, gamma=1.0)
EPS_LEVELS = [1e-2, 1e-3, 1e-6]          # certified tolerances, on the scale A_max
REF_REL = 1e-14                           # certificate of the reference rule, on the scale A_max
REF_CHECK_EXTRA = 8                       # the reference is compared with m_ref + this many nodes
SAMPLER_BUDGETS = sorted({int(round(2 ** (k / 2))) for k in range(10, 42)})   # sqrt(2) steps, 32 .. 2^21
TARGET = 1e-3                             # the accuracy the time-to-target figure is measured against
METHODS = ["kernelshap", "permutationshap", "samplingshap", "lime"]
COLORS = {"quadrashap": "C3", "kernelshap": "C0", "permutationshap": "C2", "samplingshap": "C1", "lime": "C4"}
LABEL = {"quadrashap": "QuadraSHAP", "kernelshap": "KernelSHAP", "permutationshap": "Permutation SHAP",
         "samplingshap": "SamplingSHAP", "lime": "LIME"}


# --------------------------------------------------------------------------- model and game
def fit_model(dataset: str, refit: bool = False):
    import joblib
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"exp9_{dataset}_F{MAX_FEATURES}_n{N_FIT}.joblib"
    if path.exists() and not refit:
        return joblib.load(path)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    texts, y, info = load_dataset(dataset)
    vec = TfidfVectorizer(max_features=MAX_FEATURES, sublinear_tf=True, min_df=2, strip_accents="unicode")
    Xtr = vec.fit_transform(texts[:N_FIT])
    clf = SVC(kernel="rbf", **SVM).fit(Xtr, y[:N_FIT])
    Xte = vec.transform(texts[N_FIT:N_FIT + 1000])
    bundle = dict(vec=vec, clf=clf, info=info, accuracy=float(clf.score(Xte, y[N_FIT:N_FIT + 1000])),
                  n_sv=int(clf.n_support_.sum()), vocab=len(vec.vocabulary_), svm=SVM,
                  max_features=MAX_FEATURES, n_fit=N_FIT, Xtr=Xtr)
    joblib.dump(bundle, path)
    return bundle


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A, dtype=np.float64)


def build_interventional_game(bundle, text: str, background):
    """Present/absent factor tables of the ``n_sv * n_b`` product games of one document.

    Players are the words that occur in the document or in any background row; for every other
    word both the document and every background row have TF-IDF 0, so its factor is the same in
    every coalition and folds into the per-support-vector constant.  Games are ordered
    ``(background row b, support vector r) -> b * n_sv + r`` and carry weight ``alpha_r c_r / n_b``.
    """
    vec, clf = bundle["vec"], bundle["clf"]
    x = vec.transform([text])
    players = np.unique(np.concatenate([x.indices, background.indices]))
    d = players.size
    Z = clf.support_vectors_
    n_sv = Z.shape[0]
    alpha = _dense(clf.dual_coef_).ravel()
    gamma = float(clf._gamma if hasattr(clf, "_gamma") else SVM["gamma"])

    Zp = _dense(Z[:, players])                                   # (n_sv, d)
    xp = _dense(x[:, players]).ravel()                           # (d,)
    Bp = _dense(background[:, players])                          # (n_b, d)
    n_b = Bp.shape[0]
    sq_out = _dense(Z.multiply(Z).sum(axis=1)).ravel() - (Zp ** 2).sum(axis=1)
    c = np.exp(-gamma * sq_out)                                  # factors outside the player set

    U = np.exp(-gamma * (xp[None, :] - Zp) ** 2)                 # (n_sv, d), same for every b
    Ut = np.empty((n_b * n_sv, d))
    for b in range(n_b):
        Ut[b * n_sv:(b + 1) * n_sv] = np.exp(-gamma * (Bp[b][None, :] - Zp) ** 2)
    w = np.tile(alpha * c / n_b, n_b)
    words = np.array(vec.get_feature_names_out())[players]
    # U is kept as (n_sv, d): game row r uses U[r % n_sv] -- no (games x d) copy of it
    return dict(U=U, Ut=Ut, w=w, intercept=float(clf.intercept_[0]), d=d, n_sv=n_sv, n_b=n_b,
                words=words, in_document=np.isin(players, x.indices))


def quadrashap_phi(U, Ut, w, m_q: int) -> np.ndarray:
    """The rule through the library's block planner; ``K = U - Ut`` is formed per block.

    ``U`` has ``n_sv`` rows and game row ``r`` uses ``U[r % n_sv]``, so no (games x d) copy of the
    present factors or of ``K`` is ever held.  The core alone is memory-bound at this size, so the
    planner's blocking is part of what is being timed -- it is what the library does by default.
    """
    n, d = Ut.shape
    n_u = U.shape[0]
    plan = plan_blocks(n, m_q, d, "prefix_scan_numpy")
    acc = np.zeros(d)
    for r0 in range(0, n, plan.block_size):
        r1 = min(n, r0 + plan.block_size)
        K = U[np.arange(r0, r1) % n_u] - Ut[r0:r1]
        Phi = CORE.phi_matrix_prefix_scan(K, m_q, Ut=Ut[r0:r1], node_block=plan.node_block)
        acc += (Phi * w[r0:r1, None]).sum(axis=0)
    return acc


def summarize_chunked(U, Ut, w, chunk_rows: int = 4096) -> GameSummary:
    """``GameSummary`` over the games in row chunks (its temporaries are several (rows x d) tables)."""
    n, d = Ut.shape
    n_u = U.shape[0]
    summ = GameSummary(d=d)
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        summ.update(U[np.arange(r0, r1) % n_u] - Ut[r0:r1], Ut[r0:r1], w[r0:r1])
    return summ


def rel_l2(phi, ref):
    return float(np.linalg.norm(phi - ref) / max(np.linalg.norm(ref), 1e-300))


def top_k_overlap(phi, ref, k=10):
    a, b = set(np.argsort(-np.abs(phi))[:k]), set(np.argsort(-np.abs(ref))[:k])
    return len(a & b) / k


# --------------------------------------------------------------------------- the run
def collect(datasets, n_instances, time_cap, exact_check_max_m, seed):
    records, instances, refs = [], [], {}
    for ds in datasets:
        bundle = fit_model(ds)
        texts, y, _ = load_dataset(ds)
        rng = np.random.default_rng(seed)
        n_b = N_BACKGROUND[ds] if isinstance(N_BACKGROUND, dict) else N_BACKGROUND
        bg_rows = rng.choice(N_FIT, n_b, replace=False)
        background = bundle["Xtr"][bg_rows]
        f_bg = float(bundle["clf"].decision_function(background).mean())
        log_line(f"[{NAME}] {ds}: vocab {bundle['vocab']}, {bundle['n_sv']} support vectors, "
                 f"{n_b} background rows, accuracy {bundle['accuracy']:.3f}")
        done = 0
        for k in range(N_FIT, len(texts)):
            if done >= n_instances:
                break
            g = build_interventional_game(bundle, texts[k], background)
            if g["d"] < 50:
                continue
            done += 1
            U, Ut, w, d = g["U"], g["Ut"], g["w"], g["d"]
            n_games = Ut.shape[0]
            game = MaskGame(U, Ut, w, g["intercept"])
            # the value function is the model's own: v(N) = f(x), v(empty) = mean_b f(z_b)
            v_empty, v_full = game.empty_full()
            f_x = float(bundle["clf"].decision_function(bundle["vec"].transform([texts[k]]))[0])
            assert abs(v_full - f_x) < 1e-7 * max(1, abs(f_x)), (v_full, f_x)
            assert abs(v_empty - f_bg) < 1e-7 * max(1, abs(f_bg)), (v_empty, f_bg)

            # ---- a priori: summary, budgets, reference
            t0 = time.perf_counter()
            summ = summarize_chunked(U, Ut, w)
            t_summary = time.perf_counter() - t0
            A = summ.A_max
            m_ref = budget_from_summary(summ, REF_REL * A).m_q
            t0 = time.perf_counter()
            phi_ref = quadrashap_phi(U, Ut, w, m_ref)
            t_ref = time.perf_counter() - t0
            phi_ref2 = quadrashap_phi(U, Ut, w, m_ref + REF_CHECK_EXTRA)
            floor = rel_l2(phi_ref2, phi_ref)
            refs[f"{ds}_{done - 1}"] = phi_ref
            common = dict(dataset=ds, instance=done - 1, d=d, n_sv=g["n_sv"], n_b=g["n_b"], n_games=n_games)

            inst = dict(**common, label=int(y[k]), decision=f_x, mean_background=f_bg,
                        lambda_max=float(summ.lambda_max), A_max=float(A), phi_norm=float(np.linalg.norm(phi_ref)),
                        words_in_document=int(g["in_document"].sum()), t_summary=t_summary,
                        m_ref=m_ref, t_ref=t_ref, ref_floor=floor,
                        efficiency_residual=float(abs(phi_ref.sum() - (v_full - v_empty))))
            # ---- QuadraSHAP at the certified budgets
            for eps in EPS_LEVELS:
                rep = budget_from_summary(summ, eps * A)
                t0 = time.perf_counter()
                phi = quadrashap_phi(U, Ut, w, rep.m_q)
                dt = time.perf_counter() - t0
                records.append(dict(**common, method="quadrashap", eps=eps, m_q=rep.m_q, n_evals=0,
                                    seconds=dt + t_summary, seconds_rule=dt, certified_rel=float(rep.bound / A),
                                    rel_l2=rel_l2(phi, phi_ref), max_rel=float(np.abs(phi - phi_ref).max() / A),
                                    top10=top_k_overlap(phi, phi_ref), identifiable=True))
                inst[f"m_eps{eps:g}"] = rep.m_q
                inst[f"t_eps{eps:g}"] = dt + t_summary
                inst[f"err_eps{eps:g}"] = records[-1]["rel_l2"]
            # ---- spot check: the full exactness-threshold rule, once per corpus
            thr = (d + 1) // 2
            if done == 1 and thr <= exact_check_max_m:
                t0 = time.perf_counter()
                phi_exact = quadrashap_phi(U, Ut, w, thr)
                inst["exact_m"] = thr
                inst["exact_seconds"] = time.perf_counter() - t0
                inst["exact_vs_ref"] = rel_l2(phi_exact, phi_ref)
                log_line(f"[{NAME}]   spot check: ceil(d/2)={thr} nodes in {inst['exact_seconds']:.1f}s, "
                         f"agrees with the certified reference to {inst['exact_vs_ref']:.1e}")
            # ---- the samplers, under a wall-clock cap
            for m in METHODS:
                path = ESTIMATORS[m](MaskGame(U, Ut, w, g["intercept"]), SAMPLER_BUDGETS, seed=seed, time_cap=time_cap)
                for B, ne, sec, phi in zip(path.budgets, path.n_evals, path.seconds, path.phi):
                    records.append(dict(**common, method=m, eps=None, m_q=None, n_evals=ne, seconds=sec,
                                        seconds_rule=sec, certified_rel=None, rel_l2=rel_l2(phi, phi_ref),
                                        max_rel=float(np.abs(phi - phi_ref).max() / A),
                                        top10=top_k_overlap(phi, phi_ref), identifiable=bool(ne > d)))
                inst[f"{m}_evals_at_cap"] = path.n_evals[-1]
                inst[f"{m}_err_at_cap"] = records[-1]["rel_l2"]
                inst[f"{m}_seconds"] = path.seconds[-1]
            instances.append(inst)
            log_line(f"[{NAME}]   doc {done - 1}: d={d}, games={n_games}, Lambda={summ.lambda_max:.2f}, "
                     f"m*(1e-3)={inst['m_eps0.001']} in {inst['t_eps0.001']:.1f}s (err {inst['err_eps0.001']:.1e}), "
                     f"m_ref={m_ref} in {t_ref:.1f}s (floor {floor:.1e}); "
                     + " ".join(f"{m}:{inst[f'{m}_err_at_cap']:.2f}@{inst[f'{m}_evals_at_cap']}" for m in METHODS))
            del U, Ut, w, game, g
    return records, instances, refs


# --------------------------------------------------------------------------- extrapolation
def fit_slope(times, errs):
    """Log-log slope of error against time on the tail of a sampler's path (the 1/sqrt(n) regime)."""
    t, e = np.asarray(times, float), np.asarray(errs, float)
    ok = (t > 0) & (e > 0) & np.isfinite(e)
    t, e = t[ok], e[ok]
    if t.size < 3:
        return float("nan"), float("nan")
    tail = slice(max(0, t.size - 5), t.size)
    b, a = np.polyfit(np.log10(t[tail]), np.log10(e[tail]), 1)
    return float(b), float(a)


def time_to_target(records, target=TARGET):
    """Per (dataset, instance, method): measured or extrapolated seconds to reach ``target``."""
    out = []
    keys = sorted({(r["dataset"], r["instance"]) for r in records})
    for ds, inst in keys:
        rows = [r for r in records if r["dataset"] == ds and r["instance"] == inst]
        d = rows[0]["d"]
        q = [r for r in rows if r["method"] == "quadrashap" and abs(r["eps"] - target) < 1e-15]
        out.append(dict(dataset=ds, instance=inst, d=d, method="quadrashap", seconds=q[0]["seconds"],
                        achieved=q[0]["rel_l2"], extrapolated=False, slope=None))
        for m in METHODS:
            sel = sorted([r for r in rows if r["method"] == m], key=lambda r: r["seconds"])
            if not sel:
                continue
            hit = next((r for r in sel if r["rel_l2"] <= target), None)
            if hit is not None:
                out.append(dict(dataset=ds, instance=inst, d=d, method=m, seconds=hit["seconds"],
                                achieved=hit["rel_l2"], extrapolated=False, slope=None))
                continue
            # The fitted slope decides only whether the estimator is converging at all (LIME is
            # biased and is not).  The extrapolation itself assumes the Monte-Carlo rate
            # error ~ t^{-1/2} from the last checkpoint -- the most favourable assumption for a
            # sampler, since the fitted tails are never steeper than that.
            usable = [r for r in sel if r["identifiable"]] or sel[-3:]
            b, a = fit_slope([r["seconds"] for r in usable], [r["rel_l2"] for r in usable])
            if not np.isfinite(b) or b >= -0.15:
                out.append(dict(dataset=ds, instance=inst, d=d, method=m, seconds=float("inf"),
                                achieved=sel[-1]["rel_l2"], extrapolated=True, slope=b))
                continue
            last = sel[-1]
            t_hat = last["seconds"] * (last["rel_l2"] / target) ** 2
            out.append(dict(dataset=ds, instance=inst, d=d, method=m, seconds=float(t_hat),
                            achieved=last["rel_l2"], extrapolated=True, slope=b))
    return out


# --------------------------------------------------------------------------- figures
def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def figure_error_vs_time(records, datasets, path, target=TARGET):
    if not MATPLOTLIB:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        rows = [r for r in records if r["dataset"] == ds]
        insts = sorted({r["instance"] for r in rows})
        d_med = np.median([r["d"] for r in rows])
        for m in METHODS:
            for inst in insts:
                sel = sorted([r for r in rows if r["method"] == m and r["instance"] == inst],
                             key=lambda r: r["seconds"])
                if not sel:
                    continue
                t = np.array([r["seconds"] for r in sel]); e = np.array([r["rel_l2"] for r in sel])
                ax.plot(t, e, "-", color=COLORS[m], lw=1.0, alpha=0.55,
                        label=LABEL[m] if inst == insts[0] else None)
                if m in ("permutationshap", "kernelshap") and inst == insts[0]:
                    tt = np.geomspace(t[-1], t[-1] * (e[-1] / target) ** 2, 20)
                    ax.plot(tt, e[-1] * np.sqrt(t[-1] / tt), ":", color=COLORS[m], lw=1.0,
                            label=r"extrapolated at $t^{-1/2}$" if m == "permutationshap" else None)
        q = [r for r in rows if r["method"] == "quadrashap"]
        for eps, mk in ((1e-3, "s"), (1e-6, "D")):        # 1e-2 needs the same 3 nodes as 1e-3
            sel = [r for r in q if abs(r["eps"] - eps) < 1e-15]
            ax.scatter([r["seconds"] for r in sel], [max(r["rel_l2"], 1e-16) for r in sel], marker=mk,
                       s=28, color=COLORS["quadrashap"], zorder=6, edgecolor="w", lw=0.5,
                       label=rf"QuadraSHAP, certified $\varepsilon={eps:g}$")
        ax.axhline(target, color="0.6", ls=(0, (2, 2)), lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(rf"{ds}   ($d\approx{d_med:.0f}$ players)", fontsize=9)
        ax.set_xlabel("wall-clock seconds")
        ax.grid(alpha=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"relative error  $\|\hat\phi-\phi\|_2/\|\phi\|_2$")
    axes[0].set_ylim(1e-9, 30)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout()
    _save(fig, path)


def figure_time_to_target(ttt, datasets, path, target=TARGET):
    if not MATPLOTLIB:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.9 * len(datasets), 2.8), sharex=True)
    axes = np.atleast_1d(axes)
    order = ["quadrashap"] + METHODS
    for ax, ds in zip(axes, datasets):
        vals, extrap = [], []
        for m in order:
            g = [r for r in ttt if r["dataset"] == ds and r["method"] == m]
            s = np.array([r["seconds"] for r in g], float)
            vals.append(float(np.median(s)) if len(s) else np.nan)
            extrap.append(bool(g) and all(r["extrapolated"] for r in g))
        ypos = np.arange(len(order))[::-1]
        finite = np.array([v if np.isfinite(v) else np.nan for v in vals])
        cap = np.nanmax(finite) * 30 if np.isfinite(np.nanmax(finite)) else 1e6
        for yp, v, ex, m in zip(ypos, vals, extrap, order):
            if not np.isfinite(v):
                ax.barh(yp, cap, height=0.62, color=COLORS[m], alpha=0.25, hatch="//", zorder=3)
                ax.text(cap * 1.2, yp, "does not converge", va="center", fontsize=6.3, color="0.3")
                continue
            ax.barh(yp, v, height=0.62, color=COLORS[m], alpha=0.45 if ex else 1.0,
                    hatch="//" if ex else None, zorder=3)
            lab = (f"{v:.1f} s" if v < 60 else f"{v / 60:.0f} min" if v < 3600 else
                   f"{v / 3600:.0f} h" if v < 86400 else f"{v / 86400:.0f} d")
            ax.text(v * 1.2, yp, lab + (" (extrap.)" if ex else ""), va="center", fontsize=6.3, color="0.3")
        ax.set_yticks(ypos); ax.set_yticklabels([LABEL[m] for m in order], fontsize=7)
        ax.set_xscale("log"); ax.grid(alpha=0.22, axis="x")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        d_med = np.median([r["d"] for r in ttt if r["dataset"] == ds])
        ax.set_title(rf"{ds}   ($d\approx{d_med:.0f}$)", fontsize=9)
        ax.set_xlabel("seconds")
    xmax = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(0.1, xmax * 40)
    fig.suptitle(rf"wall-clock to reach {target:g} relative error (median over documents; "
                 r"hatched = extrapolated from the capped run at the Monte-Carlo rate $t^{-1/2}$)",
                 fontsize=8.5, y=1.04)
    fig.tight_layout()
    _save(fig, path)


def make_figures_and_tables(records, instances, datasets, d_out):
    ttt = time_to_target(records)
    write_csv(ttt, d_out / "time_to_target.csv")
    figure_error_vs_time(records, datasets, d_out / "error_vs_time.pdf")
    figure_time_to_target(ttt, datasets, d_out / "time_to_target.pdf")
    rows = []
    for ds in datasets:
        ins = [r for r in instances if r["dataset"] == ds]
        if not ins:
            continue
        med = lambda k: float(np.median([float(r[k]) for r in ins if r.get(k) not in (None, "", "None")]))
        tt = lambda m: float(np.median([r["seconds"] for r in ttt if r["dataset"] == ds and r["method"] == m]))
        rows.append(dict(dataset=ds, d=med("d"), n_games=med("n_games"), lambda_max=med("lambda_max"),
                         m_eps=med("m_eps0.001"), t_q=med("t_eps0.001"), err_q=med("err_eps0.001"),
                         m_ref=med("m_ref"), t_ref=med("t_ref"), floor=med("ref_floor"),
                         t_kernel=tt("kernelshap"), t_perm=tt("permutationshap"),
                         kernel_identifiable=float(np.mean([float(r["kernelshap_evals_at_cap"]) > float(r["d"]) for r in ins]))))
    write_csv(rows, d_out / "summary.csv")
    write_latex_table(
        rows, d_out / "summary.tex",
        columns=["dataset", "d", "n_games", "lambda_max", "m_eps", "t_q", "err_q", "m_ref", "t_ref", "t_kernel", "t_perm"],
        headers=[r"corpus", r"$d$", r"games", r"$\Lambda$", r"$m^*(10^{-3})$", r"$t$ (s)", r"error",
                 r"$m_{\mathrm{ref}}$", r"$t_{\mathrm{ref}}$ (s)", r"KernelSHAP (s)", r"Permutation (s)"],
        formats={"d": ".0f", "n_games": ".0f", "lambda_max": ".2f", "m_eps": ".0f", "t_q": ".1f", "err_q": "sci",
                 "m_ref": ".0f", "t_ref": ".1f", "t_kernel": ".3g", "t_perm": ".3g"},
        caption=(r"The empirical interventional value function over a 5000-word vocabulary with a background "
                 r"set of real documents. $d$ is the number of players (words in the document or in any background row) "
                 r"and games $= n_{\mathrm{sv}} n_b$. QuadraSHAP's columns are the certified budget for "
                 r"$\varepsilon = 10^{-3}A_{\max}$, its wall-clock and its achieved relative error, and the "
                 r"reference rule ($10^{-14}A_{\max}$ certified). The last two columns are the seconds "
                 r"KernelSHAP and Permutation SHAP need for $10^{-3}$ relative error, extrapolated from a 60\,s "
                 r"run at the Monte-Carlo rate $t^{-1/2}$ (the most favourable assumption for a sampler)."),
        label="tab:text-interventional",
        note=r"TF-IDF + RBF-SVM, $n_{\mathrm{fit}}=1500$; medians over documents; single-threaded NumPy.")
    return ttt, rows


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-instances", type=int, default=5)
    ap.add_argument("--time-cap", type=float, default=45.0, help="seconds per sampler per document")
    ap.add_argument("--exact-check-max-m", type=int, default=1500, help="skip the ceil(d/2) spot check above this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    d_out = out_dir(NAME)

    if args.replot:
        import csv
        def load(path, floats):
            out = []
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    rec = dict(r)
                    for k in floats:
                        if k in rec:
                            rec[k] = float(rec[k]) if rec[k] not in ("", "None") else None
                    if "identifiable" in rec:
                        rec["identifiable"] = rec["identifiable"] == "True"
                    for k in ("d", "instance", "n_games"):
                        if k in rec and rec[k] not in ("", "None"):
                            rec[k] = int(float(rec[k]))
                    out.append(rec)
            return out
        records = load(d_out / "records.csv", ["eps", "seconds", "rel_l2", "max_rel", "top10", "certified_rel"])
        for r in records:
            r["eps"] = r["eps"] if r["eps"] is not None else float("nan")
        instances = load(d_out / "instances.csv", [])
        datasets = list(dict.fromkeys(r["dataset"] for r in records))
        make_figures_and_tables(records, instances, datasets, d_out)
        log_line(f"[{NAME}] redrawn from {len(records)} records -> {d_out}")
        return

    n_inst = 1 if args.quick else args.n_instances
    cap = 5.0 if args.quick else args.time_cap
    t0 = time.perf_counter()
    records, instances, refs = collect(args.datasets, n_inst, cap, args.exact_check_max_m, args.seed)
    elapsed = time.perf_counter() - t0
    write_csv(records, d_out / "records.csv")
    write_csv(instances, d_out / "instances.csv")
    np.savez_compressed(d_out / "reference_phi.npz", **refs)
    ttt, rows = make_figures_and_tables(records, instances, args.datasets, d_out)
    write_meta(NAME, datasets=args.datasets, n_instances=n_inst, time_cap=cap, max_features=MAX_FEATURES,
               n_fit=N_FIT, n_background=N_BACKGROUND, svm=SVM, eps_levels=EPS_LEVELS, ref_rel=REF_REL,
               sampler_budgets=SAMPLER_BUDGETS, target=TARGET,
               n_records=len(records), elapsed_seconds=elapsed)
    for r in rows:
        log_line(f"[{NAME}] {r['dataset']:7s} d={r['d']:5.0f} games={r['n_games']:6.0f} Lambda={r['lambda_max']:.2f} | "
                 f"QuadraSHAP 1e-3: {r['m_eps']:.0f} nodes, {r['t_q']:.1f}s | KernelSHAP to 1e-3: {r['t_kernel']:.3g}s "
                 f"| Permutation to 1e-3: {r['t_perm']:.3g}s")
    log_line(f"[{NAME}] done in {elapsed / 60:.1f} min -> {d_out}")


if __name__ == "__main__":
    main()
