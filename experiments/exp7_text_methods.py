"""
Experiment 7 -- QuadraSHAP against the SHAP family on text classification.

On a TF-IDF + RBF-SVM the exact Shapley values are available (QuadraSHAP at ``ceil(d/2)`` nodes is
exact in exact arithmetic, and agrees with brute-force enumeration to machine precision), so this is
not a beauty contest of attributions: it is an **error-versus-cost benchmark against ground truth**.

Every method explains the same game.  Masking a word sets its TF-IDF entry to the baseline value 0
("the word is not there"), which is the baseline value function of Section 4 and exactly what the
deletion protocol does.  Words outside a document's support are never masked and fold into a
constant, so the game has ``d = |supp(x)|`` players -- the distinct words of the document.

Four corpora spanning the document lengths that matter (``d ~ 10`` to ``d ~ 150``): SST-2, Rotten
Tomatoes (MR), AG News (World vs Sci/Tech) and IMDB.

Cost is reported in two currencies.  Wall-clock seconds, and the number of value-function
evaluations: one evaluation costs ``n_sv * d`` factor operations and so does one quadrature node, so
QuadraSHAP at ``m_q`` nodes is charged ``m_q`` evaluations.  That makes the x-axis implementation-
independent and puts every method on the same plot.

Compared, all at matched budgets and over several seeds:
    QuadraSHAP        m_q from 1 node to the exactness threshold (deterministic)
    KernelSHAP        coalitions from the Shapley kernel + constrained least squares
    SamplingSHAP      per-feature Monte-Carlo differences
    Permutation SHAP  antithetic permutation sweeps
    LIME              same masking, exponential kernel, weighted ridge
    random            the floor

Outputs (experiments/results/exp7_text_methods/):
    records.csv        one row per (dataset, instance, method, seed, budget)
    instances.csv      one row per explained document: d, text, top words by exact attribution
    exact_phi.npz      the exact attributions, so any figure can be redrawn without recomputing
    models/*.joblib    the fitted vectoriser + SVM per dataset
    *.pdf / *.png      figures;  meta.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import MODELS, log_line, out_dir, plt, MATPLOTLIB, write_csv, write_latex_table, write_meta
from text_attribution import ESTIMATORS, MaskGame, random_path
from text_datasets import load_dataset
from quadrashap.product_games.budget import GameSummary, budget_from_summary, certify
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy

NAME = "exp7_text_methods"
CORE = ProductGamesShapleyNumpy()

DATASETS = ["sst2", "mr", "agnews", "imdb"]
MAX_FEATURES = 20_000
N_FIT = 3000
SVM = dict(C=10.0, gamma=1.0)
BUDGETS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
M_Q_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
EPS_MARKS = [1e-2, 1e-3]
METHOD_ORDER = ["quadrashap", "kernelshap", "permutationshap", "samplingshap", "lime", "random"]
COLORS = {"quadrashap": "C3", "kernelshap": "C0", "permutationshap": "C2",
          "samplingshap": "C1", "lime": "C4", "random": "0.6"}
LABEL = {"quadrashap": "QuadraSHAP", "kernelshap": "KernelSHAP", "permutationshap": "Permutation SHAP",
         "samplingshap": "SamplingSHAP", "lime": "LIME", "random": "random"}


# --------------------------------------------------------------------------- model
def fit_model(dataset: str, seed: int = 0, refit: bool = False):
    """TF-IDF + RBF-SVM, cached under ``models/``."""
    import joblib
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"exp7_{dataset}_F{MAX_FEATURES}_n{N_FIT}.joblib"
    if path.exists() and not refit:
        return joblib.load(path)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    texts, y, info = load_dataset(dataset, seed=seed)
    vec = TfidfVectorizer(max_features=MAX_FEATURES, sublinear_tf=True, min_df=2, strip_accents="unicode")
    Xtr = vec.fit_transform(texts[:N_FIT])
    clf = SVC(kernel="rbf", **SVM).fit(Xtr, y[:N_FIT])
    Xte = vec.transform(texts[N_FIT:N_FIT + 1000])
    bundle = dict(vec=vec, clf=clf, info=info, accuracy=float(clf.score(Xte, y[N_FIT:N_FIT + 1000])),
                  n_sv=int(clf.n_support_.sum()), vocab=len(vec.vocabulary_), svm=SVM,
                  max_features=MAX_FEATURES, n_fit=N_FIT)
    joblib.dump(bundle, path)
    return bundle


def build_game(bundle, text: str):
    """The masked game of one document: present/absent factor tables over its support."""
    vec, clf = bundle["vec"], bundle["clf"]
    x = vec.transform([text])
    supp = x.indices.copy()
    if supp.size == 0:
        return None, None, None
    xs = x.data.copy()                                   # values on the support
    Z = clf.support_vectors_                             # (n_sv, F), sparse for a sparse fit
    dense = lambda A: np.asarray(A.todense() if hasattr(A, "todense") else A, dtype=np.float64)
    alpha = dense(clf.dual_coef_).ravel()                # sklearn keeps this sparse for a sparse fit
    gamma = float(clf._gamma if hasattr(clf, "_gamma") else SVM["gamma"])

    Zs = dense(Z[:, supp])                               # (n_sv, d) support columns
    sq_all = dense(Z.multiply(Z).sum(axis=1)).ravel()    # ||Z_r||^2
    sq_supp = (Zs ** 2).sum(axis=1)
    U = np.exp(-gamma * (xs[None, :] - Zs) ** 2)         # present factors
    Ut = np.exp(-gamma * Zs ** 2)                        # absent factors (baseline entry 0)
    w = alpha * np.exp(-gamma * (sq_all - sq_supp))      # constant outside the support
    game = MaskGame(U, Ut, w, float(clf.intercept_[0]))
    words = np.array(vec.get_feature_names_out())[supp]
    return game, (U, Ut, w), words


def quadrashap_phi(U, Ut, w, m_q: int) -> np.ndarray:
    return (CORE.phi_matrix_prefix_scan(U - Ut, m_q, Ut=Ut) * w[:, None]).sum(axis=0)


# --------------------------------------------------------------------------- metrics
def metrics(phi: np.ndarray, ref: np.ndarray) -> dict:
    from scipy.stats import kendalltau
    nz = np.abs(ref) > 0
    k = min(5, ref.size)
    top_a = set(np.argsort(-np.abs(phi))[:k])
    top_b = set(np.argsort(-np.abs(ref))[:k])
    return dict(max_abs_err=float(np.abs(phi - ref).max()),
                rel_l2=float(np.linalg.norm(phi - ref) / max(np.linalg.norm(ref), 1e-300)),
                top5_overlap=len(top_a & top_b) / k,
                kendall_tau=float(kendalltau(phi, ref).statistic) if ref.size > 2 else float("nan"),
                sign_agreement=float(np.mean(np.sign(phi[nz]) == np.sign(ref[nz]))) if nz.any() else 1.0)


# --------------------------------------------------------------------------- the run
def collect(datasets, n_instances, seeds, budgets, refit, quick):
    records, instances, exact_store = [], [], {}
    for ds in datasets:
        bundle = fit_model(ds, refit=refit)
        texts, y, info = load_dataset(ds)
        log_line(f"[{NAME}] {ds}: vocab {bundle['vocab']}, {bundle['n_sv']} support vectors, "
                 f"test accuracy {bundle['accuracy']:.3f}")
        picked = 0
        for k in range(N_FIT, len(texts)):
            if picked >= n_instances:
                break
            game, tables, words = build_game(bundle, texts[k])
            if game is None or game.d < 4:
                continue
            U, Ut, w = tables
            d = game.d
            if quick and d > 30:
                continue
            picked += 1

            # the value function must be the model's own
            v_full = game.value(np.ones(d)[None])[0]
            f_x = float(bundle["clf"].decision_function(bundle["vec"].transform([texts[k]]))[0])
            assert abs(v_full - f_x) < 1e-8 * max(1.0, abs(f_x)), (v_full, f_x)

            thr = (d + 1) // 2
            t0 = time.perf_counter()
            phi_exact = quadrashap_phi(U, Ut, w, thr)
            t_exact = time.perf_counter() - t0
            summ = GameSummary(d=d).update(U - Ut, Ut, w)
            v_empty = game.value(np.zeros(d)[None])[0]
            eff = abs(phi_exact.sum() - (v_full - v_empty))

            exact_store[f"{ds}_{picked - 1}"] = phi_exact
            order = np.argsort(-np.abs(phi_exact))[:10]
            instances.append(dict(dataset=ds, instance=picked - 1, d=d, label=int(y[k]),
                                  decision=f_x, exactness_threshold=thr, n_sv=bundle["n_sv"],
                                  lambda_max=float(summ.lambda_max), A_max=float(summ.A_max),
                                  efficiency_residual=float(eff),
                                  top_words="|".join(f"{words[i]}:{phi_exact[i]:+.3g}" for i in order),
                                  text=texts[k][:400]))

            common = dict(dataset=ds, instance=picked - 1, d=d, n_sv=bundle["n_sv"])
            # ---- QuadraSHAP: its own cost knob is the node count
            for m_q in [m for m in M_Q_GRID if m < thr] + [thr]:
                t0 = time.perf_counter()
                phi = quadrashap_phi(U, Ut, w, m_q)
                dt = time.perf_counter() - t0
                records.append(dict(**common, method="quadrashap", seed=0, budget=m_q, m_q=m_q,
                                    n_evals=m_q, seconds=dt, certified=float(certify(summ, m_q)),
                                    exact_rule=bool(m_q >= thr), **metrics(phi, phi_exact)))
            for eps in EPS_MARKS:
                rep = budget_from_summary(summ, eps)
                records.append(dict(**common, method="quadrashap_budget", seed=0, budget=rep.m_q,
                                    m_q=rep.m_q, n_evals=rep.m_q, seconds=float("nan"), eps=eps,
                                    certified=float(rep.bound), exact_rule=bool(rep.m_q >= thr),
                                    **metrics(quadrashap_phi(U, Ut, w, rep.m_q), phi_exact)))

            # ---- the sampling estimators, one nested path per seed
            for seed in range(seeds):
                for mname, fn in ESTIMATORS.items():
                    path = fn(MaskGame(U, Ut, w, game.intercept), budgets, seed=seed)
                    for B, ne, sec, phi in zip(path.budgets, path.n_evals, path.seconds, path.phi):
                        records.append(dict(**common, method=mname, seed=seed, budget=B, m_q=None,
                                            n_evals=ne, seconds=sec, **metrics(phi, phi_exact)))
                rp = random_path(d, budgets, seed=seed)
                for B, ne, sec, phi in zip(rp.budgets, rp.n_evals, rp.seconds, rp.phi):
                    records.append(dict(**common, method="random", seed=seed, budget=B, m_q=None,
                                        n_evals=ne, seconds=sec, **metrics(phi, phi_exact)))
        log_line(f"[{NAME}]   {picked} documents explained "
                 f"(d median {np.median([r['d'] for r in instances if r['dataset'] == ds]):.0f}, "
                 f"exact rule {t_exact * 1e3:.1f} ms on the last one)")
    return records, instances, exact_store


# --------------------------------------------------------------------------- figures
def _curve(rows, method, xkey, ykey):
    """Median and inter-quartile band of ``ykey`` against the method's own cost grid."""
    sel = [r for r in rows if r["method"] == method]
    if not sel:
        return None
    xs = sorted({r["budget"] for r in sel})
    x, med, lo, hi = [], [], [], []
    for b in xs:
        g = [r for r in sel if r["budget"] == b]
        x.append(np.median([r[xkey] for r in g]))
        v = np.array([r[ykey] for r in g], dtype=float)
        med.append(np.median(v))
        lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75))
    return np.array(x), np.array(med), np.array(lo), np.array(hi)


FLOOR = 1e-13          # the float64 quadrature rule's own precision (see exp6): nothing below is real
TARGET = 1e-2          # the accuracy a "cost to reach" is measured against
CEILING = 8192 * 2     # where censored ("never reached the target") entries are drawn


def _style_axes(ax, xlabel=None, ylabel=None, logx=True, logy=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.22, which="major", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _shared_legend(fig, axes, ncol=6, y=0.0):
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=7.5, frameon=False, handlelength=1.9, columnspacing=1.5)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- per-document cost
def cost_per_document(records, target=TARGET):
    """For every (dataset, document, method): the first budget whose median error meets ``target``.

    ``None`` means the largest budget was never enough (censored).  QuadraSHAP appears twice: the
    nodes it needs for ``target``, and the exactness threshold, where it stops approximating.
    """
    out = {}
    for r in records:
        if r["method"] in ("random", "quadrashap_budget"):
            continue
        key = (r["dataset"], r["instance"], r["method"], r["budget"])
        out.setdefault(key, []).append(r["rel_l2"])
    per_doc = {}
    for (ds, inst, m, b), errs in out.items():
        per_doc.setdefault((ds, inst, m), []).append((b, float(np.median(errs))))
    rows = []
    d_of = {(r["dataset"], r["instance"]): r["d"] for r in records}
    for (ds, inst, m), pairs in per_doc.items():
        pairs.sort()
        hit = next((b for b, e in pairs if e <= target), None)
        rows.append(dict(dataset=ds, instance=inst, method=m, d=d_of[(ds, inst)],
                         cost=hit, censored=hit is None))
    for (ds, inst), d in d_of.items():
        rows.append(dict(dataset=ds, instance=inst, method="quadrashap_exact", d=d,
                         cost=(d + 1) // 2, censored=False))
    return rows


def _binned(rows, method, bins):
    """Median cost per document-size bin; a bin is censored when most of its documents are."""
    y, cens = [], []
    for a, b in bins:
        g = [r for r in rows if r["method"] == method and a <= r["d"] < b]
        if not g:
            y.append(np.nan); cens.append(False); continue
        done = [r["cost"] for r in g if not r["censored"]]
        if len(done) <= len(g) / 2:
            y.append(CEILING); cens.append(True)
        else:
            y.append(float(np.median(done))); cens.append(False)
    return np.array(y), np.array(cens)


# --------------------------------------------------------------------------- figures
def figure_error_vs_cost(records, datasets, path, xkey="n_evals", ykey="rel_l2"):
    """Error against cost, one panel per corpus.  Every method spans the same x range."""
    if not MATPLOTLIB:
        return None
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        rows = [r for r in records if r["dataset"] == ds]
        d_med = np.median([r["d"] for r in rows])
        ax.axhspan(FLOOR / 10, FLOOR, color="0.92", lw=0, zorder=0)
        ax.axvline(d_med, color="0.72", lw=0.8, ls=(0, (2, 2)), zorder=1)
        for m in METHOD_ORDER:
            c = _curve(rows, m, xkey, ykey)
            if c is None:
                continue
            x, med, lo, hi = c
            med, lo, hi = (np.maximum(v, FLOOR) for v in (med, lo, hi))
            if m == "random":
                ax.axhline(med[0], color=COLORS[m], ls=(0, (1, 2)), lw=1.0, label=LABEL[m], zorder=1)
                continue
            first = m == "quadrashap"
            ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.14, lw=0, zorder=2)
            ax.plot(x, med, "-", color=COLORS[m], lw=2.2 if first else 1.3,
                    zorder=5 if first else 3, label=LABEL[m], solid_capstyle="round")
        ex = [r for r in rows if r["method"] == "quadrashap" and r["exact_rule"]]
        if ex:
            ax.scatter([np.median([r[xkey] for r in ex])], [FLOOR], marker="o", s=34, facecolor="w",
                       edgecolor=COLORS["quadrashap"], lw=1.5, zorder=7,
                       label=r"exact rule, $\lceil d/2\rceil$ nodes")
        _style_axes(ax, "value-function evaluations" if xkey == "n_evals" else "seconds")
        ax.set_title(rf"{ds}   ($d\approx{d_med:.0f}$)", fontsize=9)
        ax.set_ylim(FLOOR / 10, 40)
        if xkey == "n_evals":
            ax.set_xlim(1, 1.2e4)
            ax.annotate(r"$d$", xy=(d_med, 25), fontsize=7, color="0.5", ha="center")
    axes[0].set_ylabel(r"relative error  $\|\hat\phi-\phi\|_2/\|\phi\|_2$")
    axes[0].text(0.02, 0.02, "float64 limit", transform=axes[0].transAxes, fontsize=5.8, color="0.45")
    _shared_legend(fig, axes)
    fig.tight_layout()
    return _save(fig, path)


def figure_cost_vs_d(records, path, target=TARGET):
    """The scaling picture: what each method costs to reach a fixed accuracy, against document size."""
    if not MATPLOTLIB:
        return None
    rows = cost_per_document(records, target)
    bins = [(4, 12), (12, 20), (20, 35), (35, 70), (70, 130), (130, 400)]
    centres = np.array([np.sqrt(a * b) for a, b in bins])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhspan(CEILING / 1.5, CEILING * 2.6, color="0.93", lw=0, zorder=0)
    for k, m in enumerate(["kernelshap", "permutationshap", "samplingshap", "lime"]):
        y, cens = _binned(rows, m, bins)
        y = y.astype(float)
        y[cens] *= 1.0 + 0.16 * k            # stagger, so two censored methods do not coincide
        line = y.copy()
        if cens.all():                        # never reached anywhere: markers only, no line
            ax.scatter(centres, y, marker="^", s=32, facecolor="w", edgecolor=COLORS[m],
                       lw=1.2, zorder=4, label=LABEL[m])
            continue
        ax.plot(centres, line, "-o", color=COLORS[m], ms=3.4, lw=1.3, label=LABEL[m], zorder=3)
        if cens.any():
            ax.scatter(centres[cens], y[cens], marker="^", s=32, facecolor="w",
                       edgecolor=COLORS[m], lw=1.2, zorder=4)
    y, _ = _binned(rows, "quadrashap", bins)
    ax.plot(centres, y, "-o", color=COLORS["quadrashap"], ms=3.8, lw=2.2, zorder=6,
            label=rf"QuadraSHAP (to {target:g})")
    y, _ = _binned(rows, "quadrashap_exact", bins)
    ax.plot(centres, y, "--o", color=COLORS["quadrashap"], ms=3.4, lw=1.6, zorder=6,
            label=r"QuadraSHAP, exact  $\lceil d/2\rceil$")
    _style_axes(ax, "document size $d$ (distinct words)",
                rf"evaluations to reach {target:g} relative error")
    ax.set_ylim(1, CEILING * 2.6)
    ax.text(ax.get_xlim()[0] * 1.05, CEILING * 1.9, "never reached within 8192 evaluations",
            fontsize=6.5, color="0.45")
    _shared_legend(fig, ax, ncol=3, y=0.02)
    fig.tight_layout()
    return _save(fig, path)


def figure_cost_bars(records, datasets, path, target=TARGET):
    """The two-second read: evaluations each method needs for the same accuracy, per corpus."""
    if not MATPLOTLIB:
        return None
    rows = cost_per_document(records, target)
    methods = ["quadrashap", "kernelshap", "permutationshap", "samplingshap", "lime"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.9 * len(datasets), 2.7), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        vals, cens = [], []
        for m in methods:
            g = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            done = [r["cost"] for r in g if not r["censored"]]
            if len(done) <= len(g) / 2:
                vals.append(CEILING); cens.append(True)
            else:
                vals.append(float(np.median(done))); cens.append(False)
        ypos = np.arange(len(methods))[::-1]
        from matplotlib.colors import to_rgba
        face = [to_rgba(COLORS[m], 0.35 if c else 1.0) for m, c in zip(methods, cens)]
        ax.barh(ypos, vals, height=0.62, color=face, zorder=3)
        for yp, v, c in zip(ypos, vals, cens):
            ax.text(v * 1.25, yp, (">8192" if c else f"{v:,.0f}"), va="center", fontsize=6.5,
                    color="0.25")
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABEL[m] for m in methods], fontsize=7)
        ax.set_xscale("log")
        ax.set_xlim(1, CEILING * 9)
        ax.grid(alpha=0.22, axis="x", lw=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        d_med = np.median([r["d"] for r in records if r["dataset"] == ds])
        ax.set_title(rf"{ds}   ($d\approx{d_med:.0f}$)", fontsize=9)
        ax.set_xlabel("evaluations")
    fig.suptitle(rf"value-function evaluations to reach {target:g} relative error "
                 r"(median over documents)", fontsize=9, y=1.04)
    fig.tight_layout()
    return _save(fig, path)


def figure_quality(records, datasets, path):
    """Top-5 word agreement: the mean over documents and seeds, with a standard-error band."""
    if not MATPLOTLIB:
        return None
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        rows = [r for r in records if r["dataset"] == ds]
        for m in METHOD_ORDER:
            sel = [r for r in rows if r["method"] == m]
            if not sel:
                continue
            xs = sorted({r["budget"] for r in sel})
            x = [np.median([r["n_evals"] for r in sel if r["budget"] == b]) for b in xs]
            v = [np.array([r["top5_overlap"] for r in sel if r["budget"] == b]) for b in xs]
            mean = np.array([a.mean() for a in v])
            se = np.array([a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0 for a in v])
            if m == "random":
                ax.axhline(mean[0], color=COLORS[m], ls=(0, (1, 2)), lw=1.0, label=LABEL[m])
                continue
            first = m == "quadrashap"
            ax.fill_between(x, mean - se, mean + se, color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(x, mean, "-", color=COLORS[m], lw=2.2 if first else 1.3,
                    zorder=5 if first else 3, label=LABEL[m], solid_capstyle="round")
        _style_axes(ax, "value-function evaluations", logy=False)
        ax.set_ylim(0, 1.03)
        ax.set_xlim(1, 1.2e4)
        ax.set_title(ds, fontsize=9)
    axes[0].set_ylabel("top-5 word agreement")
    _shared_legend(fig, axes)
    fig.tight_layout()
    return _save(fig, path)


def cost_to_reach(records, datasets, target=1e-2):
    """Evaluations and seconds each sampler needs to reach ``target`` relative error."""
    out = []
    for ds in datasets:
        rows = [r for r in records if r["dataset"] == ds]
        row = {"dataset": ds, "d_median": float(np.median([r["d"] for r in rows]))}
        for m in METHOD_ORDER:
            c = _curve(rows, m, "n_evals", "rel_l2")
            if c is None:
                row[m] = None
                continue
            x, med, _, _ = c
            ok = np.where(med <= target)[0]
            row[m] = float(x[ok[0]]) if ok.size else None
        ex = [r for r in rows if r["method"] == "quadrashap" and r["exact_rule"]]
        row["quadrashap_exact_evals"] = float(np.median([r["n_evals"] for r in ex])) if ex else None
        row["quadrashap_exact_seconds"] = float(np.median([r["seconds"] for r in ex])) if ex else None
        out.append(row)
    return out


def make_all_figures(records, instances, datasets, d_out):
    figure_error_vs_cost(records, datasets, d_out / "error_vs_evals.pdf", "n_evals")
    figure_error_vs_cost(records, datasets, d_out / "error_vs_time.pdf", "seconds")
    figure_cost_bars(records, datasets, d_out / "cost_bars.pdf")
    figure_cost_vs_d(records, d_out / "cost_vs_d.pdf")
    figure_quality(records, datasets, d_out / "top5_vs_evals.pdf")
    write_csv(cost_per_document(records), d_out / "cost_per_document.csv")
    tbl = cost_to_reach(records, datasets)
    write_csv(tbl, d_out / "cost_to_reach.csv")
    write_latex_table(
        tbl, d_out / "cost_to_reach.tex",
        columns=["dataset", "d_median", "kernelshap", "permutationshap", "samplingshap", "lime",
                 "quadrashap", "quadrashap_exact_evals", "quadrashap_exact_seconds"],
        headers=[r"corpus", r"$d$", r"KernelSHAP", r"Permutation", r"Sampling", r"LIME",
                 r"\textbf{QuadraSHAP}", r"\textbf{\quad exact}", r"\textbf{\quad s}"],
        formats={"d_median": ".0f", "kernelshap": ".0f", "permutationshap": ".0f",
                 "samplingshap": ".0f", "lime": ".0f", "quadrashap": ".0f",
                 "quadrashap_exact_evals": ".0f", "quadrashap_exact_seconds": ".3f"},
        caption=(r"Value-function evaluations needed to reach a relative error of $10^{-2}$ against the "
                 r"exact Shapley values of the same masked game (median over documents and seeds; "
                 r"``--'' means the budget of 8192 evaluations was never enough). One evaluation and one "
                 r"quadrature node cost the same $n_{\mathrm{sv}}d$ factor operations, so the columns are "
                 r"directly comparable. The last two give what QuadraSHAP needs for the \emph{exact} "
                 r"values -- the exactness threshold $\lceil d/2\rceil$ -- and the wall-clock seconds there."),
        label="tab:text-cost-to-reach",
        note=r"TF-IDF (20k features) + RBF-SVM, masking a word to its baseline TF-IDF value of zero.")
    return tbl


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-instances", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-budget", type=int, default=8192)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw every figure from records.csv")
    args = ap.parse_args()

    d_out = out_dir(NAME)
    if args.replot:
        import csv
        with open(d_out / "records.csv") as fh:
            records = []
            for r in csv.DictReader(fh):
                rec = dict(r)
                for k in ("d", "n_sv", "seed", "budget", "n_evals"):
                    rec[k] = int(float(rec[k])) if rec[k] not in ("", "None") else None
                for k in ("seconds", "max_abs_err", "rel_l2", "top5_overlap", "kendall_tau",
                          "sign_agreement", "certified", "eps"):
                    rec[k] = float(rec[k]) if rec.get(k) not in (None, "", "None") else float("nan")
                rec["exact_rule"] = rec.get("exact_rule") == "True"
                records.append(rec)
        datasets = list(dict.fromkeys(r["dataset"] for r in records))
        make_all_figures(records, None, datasets, d_out)
        log_line(f"[{NAME}] figures redrawn from {len(records)} records -> {d_out}")
        return

    budgets = [b for b in BUDGETS if b <= args.max_budget]
    n_inst = 3 if args.quick else args.n_instances
    seeds = 1 if args.quick else args.seeds
    t0 = time.perf_counter()
    records, instances, exact = collect(args.datasets, n_inst, seeds, budgets, args.refit, args.quick)
    elapsed = time.perf_counter() - t0

    write_csv(records, d_out / "records.csv")
    write_csv(instances, d_out / "instances.csv")
    np.savez_compressed(d_out / "exact_phi.npz", **exact)
    tbl = make_all_figures(records, instances, args.datasets, d_out)
    write_meta(NAME, datasets=args.datasets, n_instances=n_inst, seeds=seeds, budgets=budgets,
               m_q_grid=M_Q_GRID, max_features=MAX_FEATURES, n_fit=N_FIT, svm=SVM,
               n_records=len(records), elapsed_seconds=elapsed)
    for row in tbl:
        log_line(f"[{NAME}] {row['dataset']:8s} d={row['d_median']:.0f}  evals to 1e-2: " +
                 "  ".join(f"{m}={row[m] if row[m] else '>8192':>6}" for m in
                           ("kernelshap", "permutationshap", "samplingshap", "lime")) +
                 f"   QuadraSHAP exact = {row['quadrashap_exact_evals']:.0f}")
    log_line(f"[{NAME}] {len(records)} records in {elapsed:.1f}s -> {d_out}")


if __name__ == "__main__":
    main()
