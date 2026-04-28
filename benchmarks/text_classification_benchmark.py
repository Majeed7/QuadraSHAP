"""
text_classification_benchmark.py
=================================
Simple, no-argparse benchmark for TF-IDF text classification.
Edit the CONFIG block at the top to change any parameter.

Models
------
  - RandomForestClassifier  (tree)
  - ExtraTreesClassifier    (tree)
  - KernelRidge (RBF)       (kernel)

Tree explainers tested
----------------------
  pg_numpy_prefix_scan   our method, NumPy prefix/suffix cumprod
  pg_numpy_logspace      our method, NumPy log-space
  pg_jax_prefix_scan     our method, JAX prefix/suffix (Metal on M-chip)
  pg_jax_logspace        our method, JAX log-space
  shap_tree              shap.TreeExplainer (if shap is installed)

Kernel explainers tested (KRR-RBF)
------------------------------------
  prefix_scan_numpy      our method, NumPy
  prefix_scan_jax        our method, JAX (Metal on M-chip)
  logspace_numpy         our method, NumPy
  logspace_jax           our method, JAX

Output
------
  benchmarks/results/text_clf/timing_results.csv
  benchmarks/results/text_clf/model_metrics.csv
  benchmarks/results/text_clf/timing_plot.png
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ── optional deps ────────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[warn] optuna not found – install with: pip install optuna")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    import jax
    # Metal (M-chip) does not support float64; keep x64 disabled.
    jax.config.update("jax_enable_x64", False)
    JAX_AVAILABLE = True
    print(f"[jax] backend: {jax.default_backend()}  devices: {jax.devices()}")
except Exception:
    JAX_AVAILABLE = False
    print("[jax] JAX not available – JAX methods will be skipped")

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[warn] datasets not found – install with: pip install datasets")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pgshapley import TreeExplainer as PGTreeExplainer
from pgshapley.kernels import RBFLocalExplainer


# ============================================================
#  CONFIG  – edit anything here
# ============================================================
MAX_FEATURES      = 5000    # TF-IDF vocabulary size
EXPLAIN_SAMPLES   = 50      # samples to explain per (dataset, model, method)
N_OPTUNA_TRIALS   = 20      # Optuna search trials per model
TREE_TRAIN_LIMIT  = 5000    # max training rows for tree models
KRR_TRAIN_LIMIT   = 2000    # max training rows for KernelRidge
VAL_FRACTION      = 0.15    # fraction of training set used for validation (Optuna)
MQ_EXPLAIN        = 50      # GL nodes for per-sample explanation (None → ceil(d/2))
SEED              = 42

RESULTS_DIR = ROOT / "benchmarks" / "results" / "text_clf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets to loop over.  All three are binary – KRR works on all of them.
DATASETS = [
    {
        "name":      "rotten_tomatoes",
        "path":      "rotten_tomatoes",
        "config":    None,
        "text_col":  "text",
        "label_col": "label",
    },
    {
        "name":      "imdb",
        "path":      "imdb",
        "config":    None,
        "text_col":  "text",
        "label_col": "label",
    },
    {
        "name":      "sms_spam",
        "path":      "sms_spam",
        "config":    None,
        "text_col":  "sms",
        "label_col": "label",
    },
]
# ============================================================


# ── helpers ──────────────────────────────────────────────────────────────────

def load_and_split(ds_info: dict) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Load a HuggingFace dataset and return (train_texts, test_texts, y_train, y_test)."""
    ds = load_dataset(ds_info["path"], ds_info["config"])
    text_col  = ds_info["text_col"]
    label_col = ds_info["label_col"]

    if "train" in ds and "test" in ds:
        train_ds = ds["train"]
        test_ds  = ds["test"]
    elif "train" in ds:
        split = ds["train"].train_test_split(test_size=0.2, seed=SEED)
        train_ds = split["train"]
        test_ds  = split["test"]
    else:
        raise ValueError(f"Cannot find a usable train split in {ds_info['name']}")

    train_texts: list[str] = [str(t) for t in train_ds[text_col]]
    test_texts:  list[str] = [str(t) for t in test_ds[text_col]]
    y_train_raw = np.array(train_ds[label_col])
    y_test_raw  = np.array(test_ds[label_col])

    # Encode labels to consecutive integers 0, 1, …
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw)

    return train_texts, test_texts, y_train, y_test


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def limit(X, y: np.ndarray, n: int):
    """Return the first n rows (stratified in spirit – just head-slice for speed)."""
    if n is None or len(y) <= n:
        return X, y
    if sparse.issparse(X):
        return X[:n], y[:n]
    return X[:n], y[:n]


def record_timing(
    all_timing: list[dict],
    dataset_name: str,
    model_name: str,
    method_name: str,
    times: list[float],
) -> None:
    all_timing.append(
        {
            "dataset":         dataset_name,
            "model":           model_name,
            "method":          method_name,
            "n_explained":     len(times),
            "total_s":         float(np.sum(times)),
            "mean_s":          float(np.mean(times)),
            "std_s":           float(np.std(times)),
            "median_s":        float(np.median(times)),
        }
    )


# ── Optuna objectives ─────────────────────────────────────────────────────────

def optuna_rf(trial, X_tr, y_tr, X_val, y_val):
    n_estimators    = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth       = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
    max_features    = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return accuracy_score(y_val, model.predict(X_val))


def optuna_et(trial, X_tr, y_tr, X_val, y_val):
    n_estimators    = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth       = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
    max_features    = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return accuracy_score(y_val, model.predict(X_val))


def optuna_krr(trial, X_tr, y_tr_pm1, X_val, y_val_pm1):
    alpha = trial.suggest_float("alpha", 1e-3, 50.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
    model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
    model.fit(X_tr, y_tr_pm1)
    preds = (model.predict(X_val) >= 0.0).astype(int)
    y_val_01 = ((y_val_pm1 + 1) // 2).astype(int)
    return accuracy_score(y_val_01, preds)


# ── per-sample benchmark timing loop ─────────────────────────────────────────

def bench_tree_methods(
    model,
    X_explain: np.ndarray,
    all_timing: list[dict],
    dataset_name: str,
    model_name: str,
) -> None:
    """Benchmark all tree explainer variants on EXPLAIN_SAMPLES rows."""

    # Build one PGTreeExplainer per backend_method.
    pg_methods = [
        ("pg_numpy_prefix_scan", "numpy_prefix_scan"),
        ("pg_numpy_logspace",    "numpy_logspace"),
    ]
    if JAX_AVAILABLE:
        pg_methods += [
            ("pg_jax_prefix_scan", "jax_prefix_scan"),
            ("pg_jax_logspace",    "jax_logspace"),
        ]

    for method_label, backend in pg_methods:
        try:
            explainer = PGTreeExplainer(model, backend_method=backend)
        except Exception as exc:
            print(f"      [skip] {method_label}: {exc}")
            continue

        # Warm-up call (triggers JAX JIT on first pass)
        try:
            _ = explainer.shap_values(X_explain[:1], check_additivity=False)
        except Exception:
            pass

        times: list[float] = []
        for row in X_explain:
            t0 = time.perf_counter()
            _ = explainer.shap_values(row.reshape(1, -1), check_additivity=False)
            times.append(time.perf_counter() - t0)

        record_timing(all_timing, dataset_name, model_name, method_label, times)
        print(f"      {method_label:<26}  mean={np.mean(times)*1e3:7.2f}ms  "
              f"total={np.sum(times):.2f}s")

    # SHAP library TreeExplainer
    if SHAP_AVAILABLE:
        try:
            shap_expl = shap.TreeExplainer(model)
            # warm-up
            _ = shap_expl.shap_values(X_explain[:1])

            times = []
            for row in X_explain:
                t0 = time.perf_counter()
                _ = shap_expl.shap_values(row.reshape(1, -1))
                times.append(time.perf_counter() - t0)

            record_timing(all_timing, dataset_name, model_name, "shap_tree", times)
            print(f"      {'shap_tree':<26}  mean={np.mean(times)*1e3:7.2f}ms  "
                  f"total={np.sum(times):.2f}s")
        except Exception as exc:
            print(f"      [skip] shap_tree: {exc}")


def bench_kernel_methods(
    model,
    X_explain: np.ndarray,
    all_timing: list[dict],
    dataset_name: str,
    model_name: str,
) -> None:
    """Benchmark all kernel explainer variants on EXPLAIN_SAMPLES rows."""
    try:
        explainer = RBFLocalExplainer(model)
    except Exception as exc:
        print(f"      [skip] RBFLocalExplainer: {exc}")
        return

    kernel_methods = [
        "prefix_scan_numpy",
        "logspace_numpy",
    ]
    if JAX_AVAILABLE:
        kernel_methods += [
            "prefix_scan_jax",
            "logspace_jax",
        ]

    for method in kernel_methods:
        # warm-up
        try:
            _ = explainer.explain(X_explain[0], method=method, m_q=MQ_EXPLAIN)
        except Exception:
            pass

        times: list[float] = []
        ok = True
        for row in X_explain:
            try:
                t0 = time.perf_counter()
                _ = explainer.explain(row, method=method, m_q=MQ_EXPLAIN)
                times.append(time.perf_counter() - t0)
            except Exception as exc:
                print(f"      [error] {method} on sample: {exc}")
                ok = False
                break

        if ok and times:
            record_timing(all_timing, dataset_name, model_name, method, times)
            print(f"      {method:<26}  mean={np.mean(times)*1e3:7.2f}ms  "
                  f"total={np.sum(times):.2f}s")


# ── main loop ─────────────────────────────────────────────────────────────────

all_timing:  list[dict] = []
all_metrics: list[dict] = []

for ds_info in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_info['name']}")
    print(f"{'='*60}")

    # ── load & vectorise ──────────────────────────────────────────────────────
    print("  Loading dataset …")
    train_texts, test_texts, y_train_full, y_test = load_and_split(ds_info)
    print(f"  train={len(train_texts)}  test={len(test_texts)}  "
          f"classes={np.unique(y_train_full)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )
    X_train_full_sp = vectorizer.fit_transform(train_texts)
    X_test_sp       = vectorizer.transform(test_texts)
    feat_names       = vectorizer.get_feature_names_out()
    d                = len(feat_names)
    print(f"  TF-IDF vocabulary: {d} features")

    # Validation split (for Optuna) carved out of the training set.
    val_size = max(200, int(VAL_FRACTION * X_train_full_sp.shape[0]))
    X_tr_sp, X_val_sp, y_tr, y_val = train_test_split(
        X_train_full_sp, y_train_full,
        test_size=val_size, random_state=SEED, stratify=y_train_full,
    )

    # Explanation rows: first EXPLAIN_SAMPLES of test set (dense)
    X_explain = to_dense(X_test_sp[:EXPLAIN_SAMPLES])

    # ── ① RandomForest ────────────────────────────────────────────────────────
    print("\n  [RandomForest]")
    X_tr_rf, y_tr_rf   = limit(X_tr_sp, y_tr, TREE_TRAIN_LIMIT)
    X_val_rf, y_val_rf = limit(X_val_sp, y_val, 1000)

    if OPTUNA_AVAILABLE:
        study_rf = optuna.create_study(direction="maximize",
                                       sampler=optuna.samplers.TPESampler(seed=SEED))
        study_rf.optimize(
            lambda t: optuna_rf(t, X_tr_rf, y_tr_rf, X_val_rf, y_val_rf),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        best_rf_params = study_rf.best_params
        print(f"    Best RF params: {best_rf_params}  val_acc={study_rf.best_value:.4f}")
    else:
        best_rf_params = {"n_estimators": 100, "max_depth": 20,
                          "min_samples_leaf": 1, "max_features": "sqrt"}

    rf_model = RandomForestClassifier(**best_rf_params, random_state=SEED, n_jobs=-1)
    # Retrain on full (limited) train set
    X_full_rf, y_full_rf = limit(X_train_full_sp, y_train_full, TREE_TRAIN_LIMIT)
    rf_model.fit(X_full_rf, y_full_rf)

    rf_preds = rf_model.predict(X_test_sp)
    rf_acc   = accuracy_score(y_test, rf_preds)
    rf_f1    = f1_score(y_test, rf_preds, average="macro")
    print(f"    Test  acc={rf_acc:.4f}  macro-f1={rf_f1:.4f}")
    all_metrics.append({"dataset": ds_info["name"], "model": "random_forest",
                        "accuracy": rf_acc, "macro_f1": rf_f1})

    print(f"    Benchmarking tree explainers on {len(X_explain)} samples "
          f"(m_q not used for trees) …")
    bench_tree_methods(rf_model, X_explain, all_timing, ds_info["name"], "random_forest")

    # ── ② ExtraTrees ─────────────────────────────────────────────────────────
    print("\n  [ExtraTrees]")

    if OPTUNA_AVAILABLE:
        study_et = optuna.create_study(direction="maximize",
                                       sampler=optuna.samplers.TPESampler(seed=SEED))
        study_et.optimize(
            lambda t: optuna_et(t, X_tr_rf, y_tr_rf, X_val_rf, y_val_rf),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        best_et_params = study_et.best_params
        print(f"    Best ET params: {best_et_params}  val_acc={study_et.best_value:.4f}")
    else:
        best_et_params = {"n_estimators": 100, "max_depth": 20,
                          "min_samples_leaf": 1, "max_features": "sqrt"}

    et_model = ExtraTreesClassifier(**best_et_params, random_state=SEED, n_jobs=-1)
    et_model.fit(X_full_rf, y_full_rf)

    et_preds = et_model.predict(X_test_sp)
    et_acc   = accuracy_score(y_test, et_preds)
    et_f1    = f1_score(y_test, et_preds, average="macro")
    print(f"    Test  acc={et_acc:.4f}  macro-f1={et_f1:.4f}")
    all_metrics.append({"dataset": ds_info["name"], "model": "extra_trees",
                        "accuracy": et_acc, "macro_f1": et_f1})

    print(f"    Benchmarking tree explainers on {len(X_explain)} samples …")
    bench_tree_methods(et_model, X_explain, all_timing, ds_info["name"], "extra_trees")

    # ── ③ KernelRidge (RBF) ──────────────────────────────────────────────────
    print("\n  [KernelRidge-RBF]")
    X_tr_krr_sp, y_tr_krr   = limit(X_tr_sp, y_tr, KRR_TRAIN_LIMIT)
    X_val_krr_sp, y_val_krr = limit(X_val_sp, y_val, 500)

    # KRR needs dense features
    X_tr_krr  = to_dense(X_tr_krr_sp)
    X_val_krr = to_dense(X_val_krr_sp)
    X_test_dense = to_dense(X_test_sp)

    # Encode labels as {-1, +1} for regression-based binary classification
    y_tr_pm1  = np.where(y_tr_krr == 1, 1.0, -1.0)
    y_val_pm1 = np.where(y_val_krr == 1, 1.0, -1.0)

    if OPTUNA_AVAILABLE:
        study_krr = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=SEED))
        study_krr.optimize(
            lambda t: optuna_krr(t, X_tr_krr, y_tr_pm1, X_val_krr, y_val_pm1),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        best_krr_params = study_krr.best_params
        print(f"    Best KRR params: {best_krr_params}  val_acc={study_krr.best_value:.4f}")
    else:
        best_krr_params = {"alpha": 1.0, "gamma": 1.0 / d}

    # Retrain on full (limited) dense train set
    X_full_krr_sp, y_full_krr = limit(X_train_full_sp, y_train_full, KRR_TRAIN_LIMIT)
    X_full_krr  = to_dense(X_full_krr_sp)
    y_full_pm1  = np.where(y_full_krr == 1, 1.0, -1.0)

    krr_model = KernelRidge(kernel="rbf", **best_krr_params)
    krr_model.fit(X_full_krr, y_full_pm1)

    krr_preds = (krr_model.predict(X_test_dense) >= 0.0).astype(int)
    krr_acc   = accuracy_score(y_test, krr_preds)
    krr_f1    = f1_score(y_test, krr_preds, average="macro")
    print(f"    Test  acc={krr_acc:.4f}  macro-f1={krr_f1:.4f}")
    all_metrics.append({"dataset": ds_info["name"], "model": "krr_rbf",
                        "accuracy": krr_acc, "macro_f1": krr_f1})

    mq_eff = MQ_EXPLAIN if MQ_EXPLAIN is not None else math.ceil(d / 2)
    print(f"    Benchmarking kernel explainers on {len(X_explain)} samples "
          f"(m_q={mq_eff}) …")
    bench_kernel_methods(krr_model, X_explain, all_timing, ds_info["name"], "krr_rbf")


# ── save results ──────────────────────────────────────────────────────────────

df_timing  = pd.DataFrame(all_timing)
df_metrics = pd.DataFrame(all_metrics)

timing_csv  = RESULTS_DIR / "timing_results.csv"
metrics_csv = RESULTS_DIR / "model_metrics.csv"
df_timing.to_csv(timing_csv,  index=False)
df_metrics.to_csv(metrics_csv, index=False)
print(f"\nSaved timing  → {timing_csv}")
print(f"Saved metrics → {metrics_csv}")

# ── timing plot ───────────────────────────────────────────────────────────────

if not df_timing.empty:
    import matplotlib.pyplot as plt

    datasets_list = df_timing["dataset"].unique().tolist()
    n_ds    = len(datasets_list)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)

    for ax, ds_name in zip(axes[0], datasets_list):
        sub = df_timing[df_timing["dataset"] == ds_name].copy()
        sub["label"] = sub["model"] + "\n" + sub["method"]
        sub = sub.sort_values("mean_s")
        bars = ax.barh(sub["label"], sub["mean_s"] * 1e3)
        ax.set_xlabel("Mean time per sample (ms)")
        ax.set_title(ds_name)
        ax.grid(axis="x", alpha=0.3)
        # annotate with total time
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"tot={row['total_s']:.1f}s",
                    va="center", fontsize=7)

    fig.suptitle(
        f"Explainer timing  |  TF-IDF max_features={MAX_FEATURES}  "
        f"explain_samples={EXPLAIN_SAMPLES}  m_q(kernel)={MQ_EXPLAIN}",
        fontsize=9,
    )
    fig.tight_layout()
    plot_path = RESULTS_DIR / "timing_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot    → {plot_path}")

print("\nDone.")
