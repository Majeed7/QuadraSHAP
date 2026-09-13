"""
Experiment 4 -- sentiment analysis: cost and the quality of the words each method selects.

A TF-IDF + RBF-SVC sentiment classifier (the PKeX-Shapley text setting) is trained once and
cached under ``models/``.  For a sample of positively-classified reviews we

  * time every explainer on the same instance (QuadraSHAP at its certified budgets, PKeX-Shapley
    on its own neutral-factor value function, and the SHAP family on the interventional value
    function shared with QuadraSHAP);
  * rank the words by attribution, delete the top-k *positive* ones from the document (their
    TF-IDF entries are zeroed, which is what "removing a word" means here) and record how far
    the sentiment score falls -- the standard deletion / AOPC protocol.  A method that finds the
    words the model really uses makes the score fall fastest; a random ranking is the floor.

Everything is reported per instance and aggregated, including how often the predicted label flips
after removing five words.

Usage
-----
    python exp4_sentiment_deletion.py                     # rotten_tomatoes via `datasets`
    python exp4_sentiment_deletion.py --dataset synthetic # offline, self-contained corpus
    python exp4_sentiment_deletion.py --dataset sst2 --max-features 2000 --n-instances 20

Outputs (experiments/results/exp4_sentiment_deletion/):
    runs.csv, deletion_curve.csv, timing.tex, deletion.tex, top_words.tex, deletion.pdf, meta.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from baselines import SHAP_AVAILABLE, kernel_shap, permutation_shap, pkex_shapley, random_attribution
from common import MODELS, figure, log_line, out_dir, save_figure, timed, write_csv, write_latex_table, write_meta
from quadrashap import RKHSExplainer

NAME = "exp4_sentiment_deletion"

POSITIVE = ("brilliant delightful gripping luminous tender inventive charming heartfelt exhilarating masterful "
            "witty riveting gorgeous stirring rousing sublime").split()
NEGATIVE = ("dull tedious clumsy lifeless muddled pretentious shallow tiresome incoherent leaden predictable "
            "wooden bland dreary mawkish clunky").split()
FILLER = ("film movie story director actor scene script hour minute screen character plot camera audience "
          "cinema performance sequence ending opening pace theme role cast studio").split()
FILLER += [f"topic{i:03d}" for i in range(400)]     # vocabulary filler, so the offline corpus is not trivially small


def load_corpus(name: str, seed: int = 0):
    """Return ``(texts, labels)`` with 1 = positive. ``synthetic`` needs no network."""
    if name == "synthetic":
        rng = np.random.default_rng(seed)
        texts, labels = [], []
        for _ in range(4000):
            pos = rng.random() < 0.5
            strong, weak = (POSITIVE, NEGATIVE) if pos else (NEGATIVE, POSITIVE)
            words = list(rng.choice(FILLER, size=rng.integers(12, 26)))
            words += list(rng.choice(strong, size=rng.integers(2, 5)))
            if rng.random() < 0.25:                       # a little label noise / mixed reviews
                words += list(rng.choice(weak, size=1))
            rng.shuffle(words)
            texts.append(" ".join(words)); labels.append(int(pos))
        return texts, np.array(labels)
    from datasets import load_dataset
    cfg = {"rotten_tomatoes": ("rotten_tomatoes", None, "text", "label"),
           "sst2": ("glue", "sst2", "sentence", "label"),
           "imdb": ("imdb", None, "text", "label")}[name]
    ds = load_dataset(cfg[0], cfg[1])
    tr = ds["train"]
    return list(tr[cfg[2]]), np.asarray(tr[cfg[3]])


def build_model(dataset: str, max_features: int, seed: int, retrain: bool = False):
    """Fit (or load) TF-IDF + RBF-SVC; cached under ``models/``."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{dataset}_tfidf{max_features}_svc_rbf.joblib"
    if path.exists() and not retrain:
        blob = joblib.load(path)
        log_line(f"[{NAME}] loaded cached model {path.name} (test accuracy {blob['accuracy']:.3f})")
        return blob
    texts, labels = load_corpus(dataset, seed)
    tr_t, te_t, tr_y, te_y = train_test_split(texts, labels, test_size=0.2, random_state=seed, stratify=labels)
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True, stop_words=None, min_df=2)
    Xtr = np.asarray(vec.fit_transform(tr_t).todense())
    Xte = np.asarray(vec.transform(te_t).todense())
    clf = SVC(kernel="rbf", C=2.0, gamma="scale").fit(Xtr, tr_y)
    acc = float((clf.predict(Xte) == te_y).mean())
    blob = dict(vectorizer=vec, model=clf, X_train=Xtr, y_train=tr_y, X_test=Xte, y_test=te_y,
                texts_test=te_t, accuracy=acc, dataset=dataset, max_features=max_features, seed=seed)
    joblib.dump(blob, path, compress=3)
    log_line(f"[{NAME}] trained and cached {path.name}: test accuracy {acc:.3f}, "
             f"{Xtr.shape[0]} train docs, d={Xtr.shape[1]}, {clf.support_vectors_.shape[0]} support vectors")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rotten_tomatoes",
                    choices=["rotten_tomatoes", "sst2", "imdb", "synthetic"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-features", type=int, default=1000)
    ap.add_argument("--n-instances", type=int, default=10)
    ap.add_argument("--n-background", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=10, help="deletion curve length")
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--permutations", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--kernel-samples", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--with-exact", action="store_true", help="also run the exactness threshold (slow)")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--backend", default="logspace_numpy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.dataset, args.max_features, args.n_instances = "synthetic", 200, 3
        args.permutations, args.kernel_samples, args.eps = [1], [256], [1e-2]

    t_start = time.perf_counter()
    blob = build_model(args.dataset, args.max_features, args.seed, retrain=args.retrain)
    vec, clf, Xtr, Xte = blob["vectorizer"], blob["model"], blob["X_train"], blob["X_test"]
    words = np.asarray(vec.get_feature_names_out())
    d = Xtr.shape[1]

    ex = RKHSExplainer(clf, backend=args.backend)
    background = Xtr[:args.n_background]
    f = lambda P: clf.decision_function(np.atleast_2d(P))          # the attributed quantity
    score = lambda P: clf.decision_function(np.atleast_2d(P))

    # instances: confidently positive test documents with enough words to delete
    conf = clf.decision_function(Xte)
    order = np.argsort(-conf)
    chosen = [i for i in order if (Xte[i] != 0).sum() >= args.top_k + 3][:args.n_instances]
    log_line(f"[{NAME}] {args.dataset}: d={d}, {len(ex.coef)} support vectors, "
             f"{len(chosen)} instances, shap package {'available' if SHAP_AVAILABLE else 'NOT available'}")

    rows, curve_rows, word_rows = [], [], []
    for n_inst, idx in enumerate(chosen):
        x = Xte[idx]
        present = np.flatnonzero(x)
        f_x = float(score(x[None, :])[0])

        attributions = {}
        for eps in args.eps:                                        # QuadraSHAP, certified budget
            t_b = timed(ex.node_budget, x, eps, "interventional", background=background)
            t_q = timed(ex.explain, x, "interventional", background=background, m_q=t_b.value.m_q)
            attributions[f"QuadraSHAP ($\\varepsilon$={eps:g})"] = (t_q.value, t_b.seconds + t_q.seconds,
                                                                    f"$m_q={t_b.value.m_q}$", "interventional")
        if args.with_exact:
            t = timed(ex.explain, x, "interventional", background=background, m_q="exact")
            attributions["QuadraSHAP (exact)"] = (t.value, t.seconds, rf"$m_q={(d + 1) // 2}$", "interventional")
        for n_perm in args.permutations:
            a = permutation_shap(f, x, background, n_permutations=n_perm, seed=args.seed + n_inst)
            attributions[f"PermutationSHAP ({n_perm}$\\times$)"] = (a.phi, a.seconds, f"{a.evaluations} evals",
                                                                    "interventional")
        for ns in args.kernel_samples:
            a = kernel_shap(f, x, background, n_samples=ns, seed=args.seed + n_inst)
            attributions[f"KernelSHAP ({ns})"] = (a.phi, a.seconds, f"{ns} samples", "interventional")
        a = random_attribution(d, seed=args.seed + n_inst)
        attributions["random"] = (a.phi, 0.0, "--", "--")

        # PKeX-Shapley: exact, on its own (neutral-factor) value function
        U = ex.factors(x)
        t_p = timed(pkex_shapley, U, ex.coef)
        attributions["PKeX-Shapley"] = (t_p.value, t_p.seconds, r"$O(nd^2)$", "neutral")
        t_qn = timed(ex.explain, x, "neutral", m_q=ex.node_budget(x, 1e-3, "neutral").m_q)
        attributions[r"QuadraSHAP neutral ($\varepsilon=10^{-3}$)"] = (t_qn.value, t_qn.seconds, "certified",
                                                                       "neutral")

        for method, (phi, seconds, budget, vf) in attributions.items():
            # delete the top-k words with the most positive attribution among the words in the document
            phi_present = np.where(np.isin(np.arange(d), present), phi, -np.inf)
            ranked = np.argsort(-phi_present)[:args.top_k]
            x_del = x.copy(); drops = []
            for k, j in enumerate(ranked, start=1):
                x_del = x_del.copy(); x_del[j] = 0.0
                drops.append(f_x - float(score(x_del[None, :])[0]))
                curve_rows.append(dict(instance=n_inst, method=method, k=k, drop=drops[-1],
                                       score=f_x - drops[-1], value_function=vf))
            flipped = bool((f_x - drops[min(4, len(drops) - 1)]) < 0)   # label flip after five deletions
            rows.append(dict(instance=n_inst, doc_index=int(idx), n_words=int(len(present)), method=method,
                             value_function=vf, budget=budget, seconds=seconds, f_x=f_x,
                             drop_at_1=drops[0], drop_at_3=drops[min(2, len(drops) - 1)],
                             drop_at_5=drops[min(4, len(drops) - 1)], drop_at_k=drops[-1],
                             aopc=float(np.mean(drops)), flipped_at_5=flipped,
                             top_words=" ".join(words[ranked[:5]])))
            if n_inst == 0:
                word_rows.append(dict(method=method, value_function=vf,
                                      words=", ".join(words[ranked[:5]]), drop5=drops[min(4, len(drops) - 1)]))

    elapsed = time.perf_counter() - t_start
    d_out = out_dir(NAME)
    write_csv(rows, d_out / "runs.csv")
    write_csv(curve_rows, d_out / "deletion_curve.csv")

    methods = list(dict.fromkeys(r["method"] for r in rows))
    summary = []
    for meth in methods:
        sel = [r for r in rows if r["method"] == meth]
        agg = lambda k, fn=np.mean: float(fn([r[k] for r in sel]))
        summary.append(dict(method=meth, value_function=sel[0]["value_function"], budget=sel[0]["budget"],
                            seconds=agg("seconds", np.median),
                            drop_at_1=agg("drop_at_1"), drop_at_3=agg("drop_at_3"), drop_at_5=agg("drop_at_5"),
                            aopc=agg("aopc"), flipped=float(np.mean([r["flipped_at_5"] for r in sel]))))
    write_csv(summary, d_out / "summary.csv")

    write_latex_table(
        summary, d_out / "timing.tex",
        columns=["method", "value_function", "budget", "seconds"],
        headers=[r"Method", r"value function", r"budget", r"time per instance (s)"],
        formats={"seconds": ".3f"},
        caption=(rf"Cost of one explanation of the TF-IDF + RBF-SVC sentiment classifier "
                 rf"(\texttt{{{args.dataset}}}, $d={d}$, {len(ex.coef)} support vectors, "
                 rf"$n_b={args.n_background}$ background documents). PKeX-Shapley and the neutral-factor "
                 r"QuadraSHAP row solve the same (restricted-kernel) value function; the remaining rows solve "
                 r"the empirical interventional one."),
        label="tab:text-timing", note="Median over instances.")

    write_latex_table(
        [r for r in summary if r["value_function"] != "neutral" or "PKeX" in r["method"]],
        d_out / "deletion.tex",
        columns=["method", "seconds", "drop_at_1", "drop_at_3", "drop_at_5", "aopc", "flipped"],
        headers=[r"Method", r"time (s)", r"$k=1$", r"$k=3$", r"$k=5$", r"AOPC", r"label flipped"],
        formats={"seconds": ".3f", "drop_at_1": ".3f", "drop_at_3": ".3f", "drop_at_5": ".3f",
                 "aopc": ".3f", "flipped": ".0%"},
        caption=(rf"Deletion test on {len(chosen)} positively classified reviews: the drop in the sentiment "
                 rf"score after removing the $k$ words with the most positive attribution, and the fraction of "
                 rf"reviews whose predicted label flips after five removals. AOPC averages the drop over "
                 rf"$k=1,\dots,{args.top_k}$. Larger is better."),
        label="tab:text-deletion")

    write_latex_table(
        word_rows, d_out / "top_words.tex",
        columns=["method", "words", "drop5"],
        headers=[r"Method", r"five most positive words", r"score drop"],
        formats={"drop5": ".3f"}, column_format="llr",
        caption="The five words each method selects in the first review, and the resulting drop in the sentiment score.",
        label="tab:text-words")

    fig, ax = figure(d_out / "deletion.pdf", figsize=(5.6, 3.6))
    if fig is not None:
        for i, meth in enumerate(methods):
            ks = sorted({r["k"] for r in curve_rows})
            ys = [float(np.mean([r["drop"] for r in curve_rows if r["method"] == meth and r["k"] == k])) for k in ks]
            ax.plot(ks, ys, marker="o", ms=3, color=f"C{i % 10}",
                    ls="--" if meth == "random" else "-", label=meth.replace("$", "").replace("\\varepsilon", "eps"))
        ax.set_xlabel("number of words removed $k$"); ax.set_ylabel("drop in sentiment score")
        ax.grid(alpha=0.3); ax.legend(fontsize=5.5)
        save_figure(fig, d_out / "deletion.pdf")

    write_meta(NAME, config=vars(args), dataset=args.dataset, d=d, n_support_vectors=int(len(ex.coef)),
               model_accuracy=blob["accuracy"], instances=[int(i) for i in chosen],
               shap_available=SHAP_AVAILABLE, elapsed_seconds=elapsed,
               model_cache=str((MODELS / f"{args.dataset}_tfidf{args.max_features}_svc_rbf.joblib").relative_to(MODELS.parent)))
    log_line(f"[{NAME}] done in {elapsed:.1f}s -> {d_out}")


if __name__ == "__main__":
    main()
