"""Benchmark text classification with TF-IDF, tree models, and RBF kernel methods.

This script has two main entry points:

1. `catalog`: build a catalog of standard Hugging Face text-classification datasets
   with split sizes and metadata, so a subset of datasets can be selected later.
2. `run`: train/load cached models for selected datasets, benchmark explainer time,
   and evaluate feature-removal faithfulness for sentiment datasets.

Artifacts are written under:
    - `model/`
    - `benchmarks/results/text/`
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, load_dataset_builder
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.kernel_ridge import KernelRidge

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quadrashap import TreeExplainer as PGTreeExplainer
from quadrashap.kernels.explainer import RBFLocalExplainer


MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "text"


# Same explainer set as benchmarks/treeshap_bench.py so timing numbers are
# comparable across the two scripts.
TREE_EXPLAINER_METHOD_NAMES = [
    "shap",
    "fasttreeshap_v1",
    "fasttreeshap_v2",
    "linear_tree_shap",
    "shapiq",
    "pg_quadrature_tree_cpp",
]
TREE_EXPLAINER_REPEATS = 1


_SCRIPT_T0 = time.perf_counter()


def log(message: str) -> None:
    elapsed = time.perf_counter() - _SCRIPT_T0
    print(f"[{elapsed:7.2f}s] {message}", flush=True)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    config: str | None
    task: str
    text_columns: tuple[str, ...]
    label_column: str = "label"
    positive_label: int | None = None
    notes: str = ""
    catalog_dataset: str | None = None
    catalog_config: str | None = None

    @property
    def hub_id(self) -> str:
        return self.path if self.config is None else f"{self.path}/{self.config}"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "imdb": DatasetSpec(
        key="imdb",
        path="imdb",
        config=None,
        task="sentiment",
        text_columns=("text",),
        positive_label=1,
        notes="Binary sentiment; includes unlabeled split ignored here.",
        catalog_dataset="stanfordnlp/imdb",
        catalog_config="plain_text",
    ),
    "rotten_tomatoes": DatasetSpec(
        key="rotten_tomatoes",
        path="rotten_tomatoes",
        config=None,
        task="sentiment",
        text_columns=("text",),
        positive_label=1,
        notes="Binary sentiment.",
        catalog_dataset="cornell-movie-review-data/rotten_tomatoes",
    ),
    "sst2": DatasetSpec(
        key="sst2",
        path="glue",
        config="sst2",
        task="sentiment",
        text_columns=("sentence",),
        positive_label=1,
        notes="GLUE SST-2; validation is used as test because the public test labels are hidden.",
        catalog_dataset="nyu-mll/glue",
        catalog_config="sst2",
    ),
    "yelp_polarity": DatasetSpec(
        key="yelp_polarity",
        path="yelp_polarity",
        config=None,
        task="sentiment",
        text_columns=("text",),
        positive_label=1,
        notes="Binary sentiment with long reviews.",
        catalog_dataset="fancyzhx/yelp_polarity",
    ),
    "amazon_polarity": DatasetSpec(
        key="amazon_polarity",
        path="amazon_polarity",
        config=None,
        task="sentiment",
        text_columns=("title", "content"),
        positive_label=1,
        notes="Binary sentiment using title + content.",
        catalog_dataset="fancyzhx/amazon_polarity",
    ),
    "sms_spam": DatasetSpec(
        key="sms_spam",
        path="sms_spam",
        config=None,
        task="spam",
        text_columns=("sms",),
        positive_label=None,
        notes="Binary spam detection rather than sentiment.",
        catalog_dataset="ucirvine/sms_spam",
    ),
    "emotion": DatasetSpec(
        key="emotion",
        path="emotion",
        config=None,
        task="emotion",
        text_columns=("text",),
        positive_label=None,
        notes="Six-way emotion classification.",
        catalog_dataset="dair-ai/emotion",
        catalog_config="split",
    ),
    "tweet_eval_sentiment": DatasetSpec(
        key="tweet_eval_sentiment",
        path="tweet_eval",
        config="sentiment",
        task="sentiment",
        text_columns=("text",),
        positive_label=2,
        notes="Three-way sentiment; kernel explainer pipeline is limited to binary datasets.",
        catalog_dataset="cardiffnlp/tweet_eval",
        catalog_config="sentiment",
    ),
    "ag_news": DatasetSpec(
        key="ag_news",
        path="ag_news",
        config=None,
        task="topic",
        text_columns=("text",),
        positive_label=None,
        notes="Four-way topic classification.",
        catalog_dataset="fancyzhx/ag_news",
    ),
    "dbpedia_14": DatasetSpec(
        key="dbpedia_14",
        path="dbpedia_14",
        config=None,
        task="topic",
        text_columns=("title", "content"),
        positive_label=None,
        notes="Fourteen-way topic classification.",
        catalog_dataset="fancyzhx/dbpedia_14",
    ),
    "trec": DatasetSpec(
        key="trec",
        path="trec",
        config=None,
        task="question_type",
        text_columns=("text",),
        positive_label=None,
        notes="Six-way question classification.",
        catalog_dataset="CogComp/trec",
    ),
    "banking77": DatasetSpec(
        key="banking77",
        path="PolyAI/banking77",
        config=None,
        task="intent",
        text_columns=("text",),
        positive_label=None,
        notes="Seventy-seven way banking intent classification.",
        catalog_dataset="PolyAI/banking77",
    ),
}


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    return text.replace("/", "__")


def _resolve_catalog_config(spec: DatasetSpec) -> str | None:
    if spec.catalog_config is not None:
        return spec.catalog_config
    dataset = urllib.parse.quote(spec.catalog_dataset or spec.path, safe="")
    url = "https://datasets-server.huggingface.co/splits?dataset=" + dataset
    payload = _curl_json(url)
    splits = payload.get("splits", [])
    configs = [item.get("config") for item in splits if item.get("config") is not None]
    unique_configs = []
    for config in configs:
        if config not in unique_configs:
            unique_configs.append(config)
    if not unique_configs:
        return spec.config
    if spec.config in unique_configs:
        return spec.config
    if len(unique_configs) == 1:
        return unique_configs[0]
    return unique_configs[0]


def _catalog_endpoint(spec: DatasetSpec) -> str:
    dataset = urllib.parse.quote(spec.catalog_dataset or spec.path, safe="")
    params = [f"dataset={dataset}"]
    config = _resolve_catalog_config(spec)
    if config is not None:
        params.append(f"config={urllib.parse.quote(config, safe='')}")
    return "https://datasets-server.huggingface.co/info?" + "&".join(params)


def _curl_json(url: str) -> dict[str, object]:
    proc = subprocess.run(
        ["curl", "-fsSL", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def build_dataset_catalog() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    log(f"Building catalog for {len(DATASET_SPECS)} datasets")
    for i, spec in enumerate(DATASET_SPECS.values(), 1):
        log(f"  [{i}/{len(DATASET_SPECS)}] querying HF datasets-server for '{spec.key}' ({spec.catalog_dataset or spec.hub_id})")
        row: dict[str, object] = {
            "key": spec.key,
            "hub_id": spec.catalog_dataset or spec.hub_id,
            "task": spec.task,
            "text_columns": ", ".join(spec.text_columns),
            "positive_label": spec.positive_label,
            "notes": spec.notes,
            "binary_kernel_ready": False,
        }
        try:
            payload = _curl_json(_catalog_endpoint(spec))
            dataset_info = payload.get("dataset_info", {})
            splits = dataset_info.get("splits", {}) or {}
            split_sizes = {}
            total_examples = 0
            for split_name, split_info in splits.items():
                num_examples = split_info.get("num_examples")
                split_sizes[split_name] = int(num_examples) if num_examples is not None else None
                if num_examples is not None:
                    total_examples += int(num_examples)
            features = dataset_info.get("features", {})
            label_feature = features.get(spec.label_column, {}) if isinstance(features, dict) else {}
            label_names = label_feature.get("names")
            n_classes = len(label_names) if label_names else None
            if n_classes is None and spec.positive_label is not None:
                n_classes = 2

            row.update(
                {
                    "train_size": split_sizes.get("train"),
                    "validation_size": split_sizes.get("validation"),
                    "test_size": split_sizes.get("test"),
                    "unsupervised_size": split_sizes.get("unsupervised"),
                    "total_examples": total_examples or None,
                    "label_names": ", ".join(label_names) if label_names else None,
                    "n_classes": n_classes,
                    "binary_kernel_ready": n_classes == 2,
                }
            )
        except Exception as exc:
            log(f"    failed: {type(exc).__name__}: {exc}")
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    log("Catalog query loop done; sorting rows")
    df = pd.DataFrame(rows).sort_values(["task", "total_examples", "key"], ascending=[True, False, True])
    return df.reset_index(drop=True)


def write_catalog_outputs(df: pd.DataFrame) -> None:
    ensure_dirs()
    csv_path = RESULTS_DIR / "dataset_catalog.csv"
    json_path = RESULTS_DIR / "dataset_catalog.json"
    log(f"Writing catalog CSV  -> {csv_path}")
    df.to_csv(csv_path, index=False)
    log(f"Writing catalog JSON -> {json_path}")
    df.to_json(json_path, orient="records", indent=2)


def sample_indices_by_label(y: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit >= len(y):
        return np.arange(len(y))
    indices: list[int] = []
    labels, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    alloc = np.floor(proportions * limit).astype(int)
    shortfall = limit - alloc.sum()
    if shortfall > 0:
        order = np.argsort(-(proportions * limit - alloc))
        alloc[order[:shortfall]] += 1
    for label, take in zip(labels, alloc, strict=True):
        label_idx = np.flatnonzero(y == label)
        if take >= len(label_idx):
            indices.extend(label_idx.tolist())
        else:
            chosen = rng.choice(label_idx, size=take, replace=False)
            indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return np.array(indices[:limit], dtype=int)


def join_text_columns(dataset: Dataset, text_columns: tuple[str, ...]) -> list[str]:
    columns = [dataset[column] for column in text_columns]
    rows = zip(*columns, strict=True)
    return [" ".join(str(part) for part in parts if part is not None and str(part).strip()) for parts in rows]


def load_dataset_splits(spec: DatasetSpec, seed: int) -> dict[str, Dataset]:
    log(f"  load_dataset(path={spec.path!r}, config={spec.config!r}) - downloading/caching from HF")
    ds = load_dataset(spec.path, spec.config)
    log(f"  loaded splits: {list(ds.keys())}")
    train_ds = ds["train"]
    validation_ds = ds["validation"] if "validation" in ds else None
    test_ds = ds["test"] if "test" in ds else None

    if test_ds is None and validation_ds is not None:
        # Public leaderboard datasets such as SST-2 usually expose validation labels but hide test labels.
        log("  no test split; reusing validation as test and stratify-splitting train -> (train, validation)")
        test_ds = validation_ds
        split = train_ds.train_test_split(test_size=0.15, stratify_by_column=spec.label_column, seed=seed)
        train_ds = split["train"]
        validation_ds = split["test"]
    elif validation_ds is None:
        log("  no validation split; stratify-splitting train -> (train, validation)")
        split = train_ds.train_test_split(test_size=0.2, stratify_by_column=spec.label_column, seed=seed)
        train_ds = split["train"]
        validation_ds = split["test"]
        if test_ds is None:
            log("  no test split either; halving validation -> (validation, test)")
            split = validation_ds.train_test_split(test_size=0.5, stratify_by_column=spec.label_column, seed=seed)
            validation_ds = split["train"]
            test_ds = split["test"]

    assert validation_ds is not None
    assert test_ds is not None
    log(f"  final split sizes: train={len(train_ds)} validation={len(validation_ds)} test={len(test_ds)}")
    return {"train": train_ds, "validation": validation_ds, "test": test_ds}


def dataset_to_xy(dataset: Dataset, spec: DatasetSpec) -> tuple[list[str], np.ndarray]:
    text = join_text_columns(dataset, spec.text_columns)
    y = np.asarray(dataset[spec.label_column])
    return text, y


def maybe_limit_split(
    X,
    y: np.ndarray,
    limit: int | None,
    rng: np.random.Generator,
):
    if limit is None or limit >= len(y):
        return X, y
    idx = sample_indices_by_label(y, limit, rng)
    if sparse.issparse(X):
        return X[idx], y[idx]
    return X[idx, :], y[idx]


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )


def krr_accuracy(estimator: KernelRidge, X, y: np.ndarray) -> float:
    preds = (estimator.predict(X) >= 0.0).astype(int)
    return accuracy_score(y, preds)


def train_or_load_model(
    *,
    bundle_path: Path,
    signature: dict[str, object],
    estimator,
    param_grid: dict[str, list[object]],
    scoring,
    X_train,
    y_train: np.ndarray,
    X_valid,
    y_valid: np.ndarray,
    y_train_fit: np.ndarray,
    y_valid_fit: np.ndarray,
    feature_names: np.ndarray,
    vectorizer: TfidfVectorizer,
    force_retrain: bool,
    model_kind: str,
) -> tuple[dict[str, object], bool]:
    if bundle_path.exists() and not force_retrain:
        log(f"    cache hit candidate at {bundle_path.name} ({model_kind}); loading to compare signature")
        bundle = joblib.load(bundle_path)
        if bundle.get("signature") == signature:
            log(f"    signature matches; reusing cached {model_kind}")
            return bundle, False
        log("    signature mismatch; will retrain")
    else:
        log(f"    no cached bundle (or --force-retrain) for {model_kind}; will train")

    search_X = sparse.vstack([X_train, X_valid]) if sparse.issparse(X_train) else np.vstack([X_train, X_valid])
    search_y = np.concatenate([y_train_fit, y_valid_fit])
    test_fold = np.concatenate([np.full(len(y_train_fit), -1, dtype=int), np.zeros(len(y_valid_fit), dtype=int)])
    cv = PredefinedSplit(test_fold)
    n_combos = math.prod(len(v) for v in param_grid.values()) if param_grid else 1
    log(f"    GridSearchCV: {model_kind} over {n_combos} param combos on {search_X.shape[0]} rows ({len(y_train_fit)} train + {len(y_valid_fit)} valid)")
    grid = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    t0 = time.perf_counter()
    grid.fit(search_X, search_y)
    fit_time = time.perf_counter() - t0
    log(f"    GridSearchCV done in {fit_time:.2f}s; best_params={grid.best_params_} best_score={grid.best_score_:.4f}")

    bundle = {
        "model_kind": model_kind,
        "signature": signature,
        "estimator": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_validation_score": float(grid.best_score_),
        "fit_time_seconds": fit_time,
        "feature_names": feature_names.tolist(),
        "vectorizer": vectorizer,
    }
    log(f"    persisting bundle -> {bundle_path}")
    joblib.dump(bundle, bundle_path)
    return bundle, True


def _cap_max_depth_grid(grid_values: list[object], cap: int) -> list[int]:
    """Apply a hard depth cap to a ``max_depth`` grid: ``None`` becomes ``cap``,
    explicit integers stay if ``<= cap``, otherwise clamped to ``cap``. The
    deduped result preserves order.
    """
    capped: list[int] = []
    for v in grid_values:
        d = cap if v is None else min(int(v), cap)
        if d not in capped:
            capped.append(d)
    return capped


def get_model_specs(is_binary: bool, max_tree_depth: int) -> list[dict[str, object]]:
    specs = [
        {
            "model_kind": "random_forest",
            "family": "tree",
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": _cap_max_depth_grid([20, None], max_tree_depth),
                "min_samples_leaf": [1, 2],
            },
            "scoring": "accuracy",
            "dense_train": False,
        },
    ]
    if is_binary:
        specs.extend(
            [
                {
                    "model_kind": "svc_rbf",
                    "family": "kernel",
                    "estimator": SVC(kernel="rbf"),
                    "param_grid": {
                        "C": [0.5, 1.0, 4.0],
                        "gamma": ["scale", 0.1, 0.01],
                    },
                    "scoring": "accuracy",
                    "dense_train": True,
                },
                {
                    "model_kind": "krr_rbf",
                    "family": "kernel",
                    "estimator": KernelRidge(kernel="rbf"),
                    "param_grid": {
                        "alpha": [0.1, 1.0, 10.0],
                        "gamma": [0.1, 0.01, 0.001],
                    },
                    "scoring": krr_accuracy,
                    "dense_train": True,
                },
            ]
        )
    return specs


def model_score(bundle: dict[str, object], X) -> np.ndarray:
    estimator = bundle["estimator"]
    kind = bundle["model_kind"]
    if kind in {"decision_tree", "random_forest", "extra_trees"}:
        proba = estimator.predict_proba(X)
        return np.asarray(proba)[:, -1]
    if kind == "svc_rbf":
        return np.asarray(estimator.decision_function(X), dtype=float).ravel()
    if kind == "krr_rbf":
        return np.asarray(estimator.predict(X), dtype=float).ravel()
    raise ValueError(f"Unsupported model kind: {kind}")


def model_predict_label(bundle: dict[str, object], X) -> np.ndarray:
    estimator = bundle["estimator"]
    kind = bundle["model_kind"]
    if kind == "krr_rbf":
        return (estimator.predict(X) >= 0.0).astype(int)
    return estimator.predict(X)


def evaluate_model(bundle: dict[str, object], X_test, y_test: np.ndarray) -> dict[str, float]:
    preds = model_predict_label(bundle, X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }


def evaluate_train_test(
    bundle: dict[str, object],
    X_train, y_train: np.ndarray,
    X_test, y_test: np.ndarray,
) -> dict[str, float]:
    """Quality metrics for both training and test/validation sets.

    Returns ``train_*`` and ``test_*`` keys side by side so the per-model row
    can show both at once and we can spot over/underfit in the LaTeX summary.
    """
    train = evaluate_model(bundle, X_train, y_train)
    test = evaluate_model(bundle, X_test, y_test)
    return {
        "train_accuracy": train["accuracy"],
        "train_macro_f1": train["macro_f1"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
    }


def dense_rows(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def extract_tree_pg_contrib(values: np.ndarray, target_class: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        return arr[0, :, target_class]
    raise ValueError(f"Unexpected PG TreeExplainer output shape: {arr.shape}")


def extract_shap_tree_contrib(values, target_class: int) -> np.ndarray:
    arr = values
    if isinstance(arr, list):
        return np.asarray(arr[target_class])[0]
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        return arr[0, :, target_class]
    raise ValueError(f"Unexpected shap.TreeExplainer output shape: {arr.shape}")


def _max_distinct_features_on_path(tree) -> int:
    """Maximum number of distinct features used along any root-to-leaf path."""
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature

    max_count = 0
    stack: list[tuple[int, frozenset[int]]] = [(0, frozenset())]
    while stack:
        node, feats = stack.pop()
        if children_left[node] == -1:
            if len(feats) > max_count:
                max_count = len(feats)
            continue
        new_feats = feats | {int(feature[node])}
        stack.append((int(children_left[node]), new_feats))
        stack.append((int(children_right[node]), new_feats))
    return max_count


def compute_tree_stats(estimator) -> dict[str, int | None]:
    """Aggregate structural stats for a fitted sklearn tree or tree ensemble."""
    if hasattr(estimator, "estimators_"):
        trees = [est.tree_ for est in estimator.estimators_]
    elif hasattr(estimator, "tree_"):
        trees = [estimator.tree_]
    else:
        return {
            "tree_count": None,
            "total_leaves": None,
            "max_depth": None,
            "max_features_on_path": None,
        }

    total_leaves = 0
    max_depth = 0
    max_features_on_path = 0
    for t in trees:
        leaf_mask = t.children_left == -1
        total_leaves += int(leaf_mask.sum())
        max_depth = max(max_depth, int(t.max_depth))
        max_features_on_path = max(max_features_on_path, _max_distinct_features_on_path(t))
    return {
        "tree_count": len(trees),
        "total_leaves": total_leaves,
        "max_depth": max_depth,
        "max_features_on_path": max_features_on_path,
    }


def _project_phi_to_class(sv, target_class: int) -> np.ndarray:
    """Reduce a SHAP-style explanation array to ``(n_samples, n_features)``."""
    if isinstance(sv, list):
        return np.asarray(sv[target_class])
    arr = np.asarray(sv)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, target_class]
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def _project_ev_to_class(ev, target_class: int) -> float:
    arr = np.asarray(ev).ravel()
    if arr.size == 1:
        return float(arr[0])
    return float(arr[target_class])


class _LtsTreeProxy:
    """Read-only shim around sklearn's ``Tree`` exposing a regressor-shaped
    ``value`` array.

    ``linear_tree_shap`` only handles single-output trees: its ``copy_tree``
    does ``n_node_samples / n_node_samples[0] * value.ravel()``, which fails
    to broadcast when ``value.shape == (n_nodes, 1, n_classes)``. We replace
    ``value`` with per-node ``predict_proba_c`` (shape ``(n_nodes, 1, 1)``)
    so the library treats the classifier tree as a regressor of the chosen
    class probability. Per-tree SHAP values then sum to
    ``predict_proba_c(x) - <library baseline>``, and averaging across an
    RF/ET ensemble yields class-c SHAP for the ensemble.
    """

    __slots__ = ("_t", "value")

    def __init__(self, sklearn_tree, target_class: int):
        self._t = sklearn_tree
        v = np.asarray(sklearn_tree.value, dtype=np.float64)
        row_sum = v.sum(axis=-1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        self.value = v[:, :, target_class:target_class + 1] / row_sum

    def __getattr__(self, name):
        return getattr(self._t, name)


class _LtsClfProxy:
    """Wraps a single sklearn classifier tree so ``linear_tree_shap`` sees it
    as a regressor of class-c probability. See ``_LtsTreeProxy``.
    """

    __slots__ = ("tree_",)

    def __init__(self, clf, target_class: int):
        self.tree_ = _LtsTreeProxy(clf.tree_, target_class)


def _build_tree_explainer(method_name: str, model, target_class: int = 0):
    if method_name == "shap":
        if shap is None:
            raise ImportError("shap is not importable")
        return shap.TreeExplainer(model)
    if method_name == "fasttreeshap_v1":
        import fasttreeshap
        return fasttreeshap.TreeExplainer(model, algorithm="v1", n_jobs=1)
    if method_name == "fasttreeshap_v2":
        import fasttreeshap
        return fasttreeshap.TreeExplainer(model, algorithm="v2", n_jobs=1)
    if method_name == "linear_tree_shap":
        import linear_tree_shap
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        clf_types = (RandomForestClassifier, ExtraTreesClassifier)
        clf_tree_types = (DecisionTreeClassifier,)
        forest_types = (RandomForestRegressor, RandomForestClassifier,
                        ExtraTreesRegressor, ExtraTreesClassifier)
        tree_types = (DecisionTreeRegressor, DecisionTreeClassifier)

        def _wrap(est):
            return _LtsClfProxy(est, target_class) if isinstance(est, clf_tree_types) else est

        if isinstance(model, forest_types):
            wrapped = [_wrap(est) for est in model.estimators_] if isinstance(model, clf_types) \
                else list(model.estimators_)
            return ("rf", [linear_tree_shap.TreeExplainer(e) for e in wrapped])
        if isinstance(model, tree_types):
            target = _wrap(model)
            return ("tree", linear_tree_shap.TreeExplainer(target))
        raise NotImplementedError(
            f"linear_tree_shap doesn't handle {type(model).__name__}"
        )
    if method_name == "shapiq":
        import warnings
        from shapiq.explainer.tree import TreeExplainer as ShapIQTreeExplainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ShapIQTreeExplainer(model=model, max_order=1)
    if method_name == "pg_quadrature_tree_cpp":
        return PGTreeExplainer(model, tree_solver="quadrature_tree", use_cpp=True)
    raise ValueError(f"Unknown tree explainer method: {method_name}")


def _raw_explain_tree(method_name: str, handle, X: np.ndarray):
    if method_name == "linear_tree_shap":
        kind, payload = handle
        if kind == "rf":
            svs = [e.shap_values(X) for e in payload]
            return np.mean(np.stack(svs, axis=0), axis=0)
        return payload.shap_values(X)
    if method_name == "shapiq":
        return handle.explain_X(X)
    return handle.shap_values(X, check_additivity=False)


def _lts_ev_single_tree(sklearn_tree, target_class: int = 0) -> float:
    """linear_tree_shap baseline for one sklearn tree.

    Mirrors `_linear_tree_shap_ev_single` in `treeshap_bench_worker.py`: the
    library uses the unique-sample leaf counts (`tree_.n_node_samples`)
    rather than the bootstrap-weighted counts. For classifier trees, we
    project the per-leaf class counts to ``predict_proba_c`` to match the
    proxy passed into ``linear_tree_shap.TreeExplainer``.
    """
    from sklearn.tree import DecisionTreeClassifier

    t = sklearn_tree.tree_
    leaf_mask = t.children_left == -1
    leaf_vals = np.asarray(t.value[leaf_mask], dtype=np.float64)  # (n_leaves, n_out, n_cls)
    leaf_counts = t.n_node_samples[leaf_mask].astype(np.float64)
    if isinstance(sklearn_tree, DecisionTreeClassifier):
        row_sum = leaf_vals[:, 0, :].sum(axis=1)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        leaf_v = leaf_vals[:, 0, target_class] / row_sum
    else:
        leaf_v = leaf_vals.reshape(leaf_vals.shape[0], -1)[:, 0]
    return float((leaf_v * leaf_counts).sum() / leaf_counts.sum())


def _phi_per_sample(method_name: str, raw_sv, target_class: int) -> np.ndarray:
    if method_name == "shapiq":
        return np.stack([np.asarray(r.get_n_order_values(1)) for r in raw_sv], axis=0)
    return _project_phi_to_class(raw_sv, target_class)


def _ev_per_sample(
    method_name: str,
    handle,
    model,
    target_class: int,
    n_samples: int,
    raw_sv,
) -> np.ndarray:
    if method_name == "shapiq":
        return np.array([float(r.baseline_value) for r in raw_sv], dtype=np.float64)
    if method_name == "linear_tree_shap":
        kind, _ = handle
        if kind == "rf":
            evs = [_lts_ev_single_tree(est, target_class) for est in model.estimators_]
            scalar = float(np.mean(evs))
        else:
            scalar = _lts_ev_single_tree(model, target_class)
        return np.full(n_samples, scalar, dtype=np.float64)
    scalar = _project_ev_to_class(handle.expected_value, target_class)
    return np.full(n_samples, scalar, dtype=np.float64)


def benchmark_explainers(
    *,
    dataset_key: str,
    bundle: dict[str, object],
    X_eval,
    explain_samples: int,
    positive_label: int | None,
) -> list[dict[str, object]]:
    estimator = bundle["estimator"]
    feature_names = bundle["feature_names"]
    X_eval_dense = dense_rows(X_eval)
    n_samples = min(explain_samples, len(X_eval_dense))
    log(f"  benchmark_explainers: dataset={dataset_key} model={bundle['model_kind']} n_samples={n_samples}")
    rows: list[dict[str, object]] = []

    if bundle["model_kind"] in {"decision_tree", "random_forest", "extra_trees"}:
        target_class = positive_label if positive_label is not None else 0
        X_batch = np.ascontiguousarray(X_eval_dense[:n_samples])
        try:
            pred_target = estimator.predict_proba(X_batch)[:, target_class]
        except Exception as exc:
            log(f"    predict_proba failed ({exc}); additivity will be skipped")
            pred_target = None

        for method_name in TREE_EXPLAINER_METHOD_NAMES:
            log(f"    timing {method_name} on {n_samples} instances")
            try:
                handle = _build_tree_explainer(method_name, estimator, target_class)
                # Untimed warmup (JIT, caches, any first-call work).
                raw_sv = _raw_explain_tree(method_name, handle, X_batch)
                times: list[float] = []
                for _ in range(max(1, TREE_EXPLAINER_REPEATS)):
                    t0 = time.perf_counter()
                    raw_sv = _raw_explain_tree(method_name, handle, X_batch)
                    times.append(time.perf_counter() - t0)
                    if times[0] > 1.0:
                        break
                total_time = float(np.median(times))

                additivity_error: float | None = None
                if pred_target is not None:
                    phi = _phi_per_sample(method_name, raw_sv, target_class)
                    ev_vec = _ev_per_sample(
                        method_name, handle, estimator, target_class, n_samples, raw_sv
                    )
                    additivity_error = float(
                        np.max(np.abs(ev_vec + phi.sum(axis=1) - pred_target))
                    )

                err_str = "n/a" if additivity_error is None else f"{additivity_error:.2e}"
                log(
                    f"      {method_name}: total={total_time:.4f}s "
                    f"avg={total_time / max(1, n_samples):.4f}s/inst "
                    f"err={err_str} (n_repeats={len(times)})"
                )
                rows.append(
                    {
                        "dataset": dataset_key,
                        "model_kind": bundle["model_kind"],
                        "explainer": method_name,
                        "n_samples": n_samples,
                        "avg_seconds_per_instance": total_time / max(1, n_samples),
                        "total_seconds": total_time,
                        "n_repeats": len(times),
                        "additivity_error": additivity_error,
                        "status": "ok",
                        "message": None,
                    }
                )
            except BaseException as exc:
                msg = f"{type(exc).__name__}: {exc}"
                log(f"      {method_name}: failed: {msg}")
                rows.append(
                    {
                        "dataset": dataset_key,
                        "model_kind": bundle["model_kind"],
                        "explainer": method_name,
                        "n_samples": n_samples,
                        "avg_seconds_per_instance": None,
                        "total_seconds": None,
                        "n_repeats": 0,
                        "additivity_error": None,
                        "status": "failed",
                        "message": msg[:300],
                    }
                )
        return rows

    if bundle["model_kind"] in {"svc_rbf", "krr_rbf"}:
        log("    constructing RBFLocalExplainer")
        kernel_explainer = RBFLocalExplainer(estimator)
        methods = ["logspace_numpy", "prefix_scan_numpy"]
        for method in methods:
            log(f"    timing rbf_local_{method} on {n_samples} instances")
            times = []
            for i, row in enumerate(X_eval_dense[:n_samples], 1):
                t0 = time.perf_counter()
                _ = kernel_explainer.explain(row, method=method)
                times.append(time.perf_counter() - t0)
                if i % max(1, n_samples // 5) == 0 or i == n_samples:
                    log(f"      rbf_local_{method}: {i}/{n_samples} (avg so far: {np.mean(times):.4f}s)")
            rows.append(
                {
                    "dataset": dataset_key,
                    "model_kind": bundle["model_kind"],
                    "explainer": f"rbf_local_{method}",
                    "n_samples": n_samples,
                    "avg_seconds_per_instance": float(np.mean(times)),
                    "total_seconds": float(np.sum(times)),
                    "n_repeats": 1,
                    "additivity_error": None,
                    "status": "ok",
                    "message": None,
                }
            )
    return rows


def explain_single_instance(
    *,
    bundle: dict[str, object],
    row_dense: np.ndarray,
    target_class: int,
) -> np.ndarray:
    estimator = bundle["estimator"]
    if bundle["model_kind"] in {"decision_tree", "random_forest", "extra_trees"}:
        explainer = PGTreeExplainer(estimator, backend_method="numpy_prefix_scan")
        values = explainer.shap_values(row_dense.reshape(1, -1), check_additivity=False)
        return extract_tree_pg_contrib(values, target_class)
    if bundle["model_kind"] in {"svc_rbf", "krr_rbf"}:
        explainer = RBFLocalExplainer(estimator)
        return np.asarray(explainer.explain(row_dense, method="logspace_numpy"), dtype=float)
    raise ValueError(f"Unsupported model kind: {bundle['model_kind']}")


def score_single_instance(bundle: dict[str, object], row_dense: np.ndarray) -> float:
    return float(model_score(bundle, row_dense.reshape(1, -1))[0])


def perturbation_faithfulness(
    *,
    dataset_key: str,
    spec: DatasetSpec,
    bundle: dict[str, object],
    X_eval,
    y_eval: np.ndarray,
    removal_sizes: list[int],
    max_instances: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if spec.positive_label is None:
        log(f"  perturbation_faithfulness: dataset={dataset_key} has no positive_label; skipping")
        return []
    mask = y_eval == spec.positive_label
    positive_indices = np.flatnonzero(mask)
    if len(positive_indices) == 0:
        log(f"  perturbation_faithfulness: dataset={dataset_key} has zero positive instances; skipping")
        return []
    if len(positive_indices) > max_instances:
        positive_indices = rng.choice(positive_indices, size=max_instances, replace=False)

    log(f"  perturbation_faithfulness: dataset={dataset_key} model={bundle['model_kind']} on {len(positive_indices)} positive instances; removal_sizes={removal_sizes}")

    X_eval_dense = dense_rows(X_eval)
    rows: list[dict[str, object]] = []
    per_k: dict[int, list[float]] = {k: [] for k in removal_sizes}
    per_k_removed: dict[int, list[int]] = {k: [] for k in removal_sizes}

    n_total = len(positive_indices)
    for j, idx in enumerate(positive_indices, 1):
        if j % max(1, n_total // 5) == 0 or j == n_total:
            log(f"    perturbation progress: {j}/{n_total}")
        row_dense = np.asarray(X_eval_dense[idx], dtype=float)
        original_score = score_single_instance(bundle, row_dense)
        contributions = explain_single_instance(
            bundle=bundle,
            row_dense=row_dense,
            target_class=spec.positive_label,
        )
        present = np.flatnonzero(row_dense > 0.0)
        positive_present = present[contributions[present] > 0.0]
        if len(positive_present) == 0:
            continue
        ranked = positive_present[np.argsort(contributions[positive_present])[::-1]]

        for k in removal_sizes:
            chosen = ranked[:k]
            mutated = row_dense.copy()
            mutated[chosen] = 0.0
            mutated_score = score_single_instance(bundle, mutated)
            per_k[k].append(original_score - mutated_score)
            per_k_removed[k].append(int(len(chosen)))

    for k in removal_sizes:
        if not per_k[k]:
            continue
        rows.append(
            {
                "dataset": dataset_key,
                "model_kind": bundle["model_kind"],
                "removed_top_k": k,
                "n_instances": len(per_k[k]),
                "mean_score_drop": float(np.mean(per_k[k])),
                "std_score_drop": float(np.std(per_k[k])),
                "mean_actual_removed": float(np.mean(per_k_removed[k])),
            }
        )
    return rows


def plot_perturbation_results(df: pd.DataFrame) -> None:
    if df.empty:
        log("  perturbation dataframe empty; skipping plots")
        return
    for dataset_key, dataset_df in df.groupby("dataset"):
        log(f"  plotting perturbation results for dataset={dataset_key}")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for model_kind, model_df in dataset_df.groupby("model_kind"):
            model_df = model_df.sort_values("removed_top_k")
            x = model_df["removed_top_k"].to_numpy()
            mean = model_df["mean_score_drop"].to_numpy()
            std = model_df["std_score_drop"].to_numpy()
            ax.plot(x, mean, marker="o", label=model_kind)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"Prediction drop after removing top positive terms: {dataset_key}")
        ax.set_xlabel("Number of removed terms")
        ax.set_ylabel("Score drop")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"perturbation_{dataset_key}.png", dpi=160)
        plt.close(fig)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("^", r"\^{}")
         .replace("~", r"\~{}")
    )


def _fmt_time_ms(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    ms = seconds * 1000.0
    if ms >= 100:   return f"{ms:.0f}"
    if ms >= 10:    return f"{ms:.1f}"
    if ms >= 1:     return f"{ms:.2f}"
    return f"{ms:.3f}"


def _fmt_err(err: float | None) -> str:
    if err is None or not np.isfinite(err):
        return "—"
    if err == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(err))))
    mant = err / (10 ** exp)
    body = rf"{mant:.0f}{{\cdot}}10^{{{exp}}}"
    # Flag numerical breakdown loudly, mirroring table_format.tex.
    if err >= 1e-2:
        return rf"\textcolor{{red}}{{${body}$}}"
    return rf"${body}$"


def write_summary_latex(
    metrics_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render a LaTeX summary table mirroring ``table_format.tex``.

    Columns are datasets. Top rows describe the trained ``random_forest``
    (depth, total leaves, tree count, plus train/test accuracy). Then each
    explainer method gets a two-row block (timing in ms, additivity error).
    """
    if metrics_df.empty:
        log("  metrics dataframe empty; skipping summary_table.tex")
        return
    rf_df = metrics_df[metrics_df["model_kind"] == "random_forest"].copy()
    if rf_df.empty:
        log("  no random_forest rows; skipping summary_table.tex")
        return
    datasets = list(rf_df["dataset"].drop_duplicates())
    rf_by_dataset = {row["dataset"]: row for _, row in rf_df.iterrows()}

    methods: list[str] = []
    if not timing_df.empty:
        rf_timing = timing_df[timing_df["model_kind"] == "random_forest"]
        methods = list(rf_timing["explainer"].drop_duplicates())

    def lookup_timing(ds: str, method: str) -> tuple[float | None, float | None, str]:
        if timing_df.empty:
            return None, None, "missing"
        sub = timing_df[
            (timing_df["dataset"] == ds)
            & (timing_df["model_kind"] == "random_forest")
            & (timing_df["explainer"] == method)
        ]
        if sub.empty:
            return None, None, "missing"
        row = sub.iloc[0]
        status = str(row.get("status") or "ok")
        t = row.get("avg_seconds_per_instance")
        e = row.get("additivity_error")
        t = float(t) if t is not None and pd.notna(t) else None
        e = float(e) if e is not None and pd.notna(e) else None
        return t, e, status

    n = len(datasets)
    col_spec = "@{\\hspace{5pt}}l" + "c" * n + "@{}"
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{Single-threaded tree-explainer benchmark on TF-IDF + "
        r"\texttt{RandomForestClassifier} across text-classification datasets. "
        r"Time is mean ms per explained instance; precision is the worst-case "
        r"additivity residual $\max_i|\mathbb{E}[f] + \sum_j \phi_{ij} - "
        r"\mathrm{predict\_proba}(x_i)|$. {\color{red}Red} flags numerical "
        r"breakdown ($\geq 10^{-2}$).}")
    lines.append(r"\label{tab:text_treeshap_summary}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_cells = [_latex_escape(d) for d in datasets]
    lines.append(" & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    def stat_row(label: str, getter):
        cells = [getter(rf_by_dataset[d]) for d in datasets]
        return label + " & " + " & ".join(cells) + r" \\"

    def fmt_int(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        try: return f"{int(x)}"
        except Exception: return "—"

    def fmt_pct(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{100*float(x):.1f}\\%"

    lines.append(stat_row(r"max depth",  lambda r: fmt_int(r.get("max_depth"))))
    lines.append(stat_row(r"total leaves", lambda r: fmt_int(r.get("total_leaves"))))
    lines.append(stat_row(r"tree count", lambda r: fmt_int(r.get("tree_count"))))
    lines.append(stat_row(r"train acc.", lambda r: fmt_pct(r.get("train_accuracy"))))
    lines.append(stat_row(r"test acc.",  lambda r: fmt_pct(r.get("accuracy"))))

    if methods:
        lines.append(r"\midrule")
        for mi, method in enumerate(methods):
            method_label = r"\textsc{" + _latex_escape(method) + "}"
            time_cells, err_cells = [], []
            for d in datasets:
                t, e, status = lookup_timing(d, method)
                if status != "ok":
                    time_cells.append("—")
                    err_cells.append("—")
                else:
                    time_cells.append(_fmt_time_ms(t))
                    err_cells.append(r"{\scriptsize " + _fmt_err(e) + "}")
            lines.append(method_label + " & " + " & ".join(time_cells) + r" \\")
            tail = r" \\[2pt]" if mi < len(methods) - 1 else r" \\"
            lines.append(" & " + " & ".join(err_cells) + tail)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    log(f"  wrote {out_path}")


def plot_timing_results(df: pd.DataFrame) -> None:
    if df.empty:
        log("  timing dataframe empty; skipping plot")
        return
    plot_df = df.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"] == "ok"]
    plot_df = plot_df.dropna(subset=["avg_seconds_per_instance"])
    if plot_df.empty:
        log("  no successful timing rows to plot; skipping")
        return
    log(f"  rendering explainer_timing.png from {len(plot_df)} rows")
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df["label"] = plot_df["dataset"] + "\n" + plot_df["model_kind"] + "\n" + plot_df["explainer"]
    plot_df = plot_df.sort_values("avg_seconds_per_instance")
    ax.bar(plot_df["label"], plot_df["avg_seconds_per_instance"])
    ax.set_ylabel("Average seconds per explained instance")
    ax.set_title("Explainer runtime benchmark (100-instance average when available)")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "explainer_timing.png", dpi=160)
    plt.close(fig)


def run_text_benchmark(args: argparse.Namespace) -> None:
    ensure_dirs()
    rng = np.random.default_rng(args.seed)
    selected_specs = [DATASET_SPECS[key] for key in args.datasets]
    log(f"run_text_benchmark: {len(selected_specs)} datasets selected -> {[s.key for s in selected_specs]}")
    log(f"  args: max_features={args.max_features} vectorizer_fit_limit={args.vectorizer_fit_limit} "
        f"tree_train={args.tree_train_limit} tree_valid={args.tree_valid_limit} "
        f"kernel_train={args.kernel_train_limit} kernel_valid={args.kernel_valid_limit} "
        f"explain_samples={args.explain_samples} perturbation_samples={args.perturbation_samples} "
        f"removal_sizes={args.removal_sizes} families={args.families} "
        f"max_tree_depth={args.max_tree_depth} force_retrain={args.force_retrain} seed={args.seed}")
    metrics_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    perturb_rows: list[dict[str, object]] = []

    for di, spec in enumerate(selected_specs, 1):
        log(f"=== dataset [{di}/{len(selected_specs)}] {spec.key} (task={spec.task}) ===")
        splits = load_dataset_splits(spec, seed=args.seed)
        log("  joining text columns and extracting labels for each split")
        train_text, y_train = dataset_to_xy(splits["train"], spec)
        valid_text, y_valid = dataset_to_xy(splits["validation"], spec)
        test_text, y_test = dataset_to_xy(splits["test"], spec)
        log(f"  label distribution (train): {dict(zip(*[a.tolist() for a in np.unique(y_train, return_counts=True)]))}")

        vectorizer = build_vectorizer(args.max_features)
        fit_text = train_text
        if args.vectorizer_fit_limit is not None and len(fit_text) > args.vectorizer_fit_limit:
            log(f"  subsampling vectorizer fit set: {len(fit_text)} -> {args.vectorizer_fit_limit}")
            fit_idx = sample_indices_by_label(y_train, args.vectorizer_fit_limit, rng)
            fit_text = [train_text[i] for i in fit_idx]
        log(f"  fitting TF-IDF vectorizer on {len(fit_text)} documents (max_features={args.max_features})")
        t0 = time.perf_counter()
        vectorizer.fit(fit_text)
        log(f"  vectorizer fit done in {time.perf_counter() - t0:.2f}s")

        log(f"  transforming train/valid/test ({len(train_text)}/{len(valid_text)}/{len(test_text)} docs)")
        X_train = vectorizer.transform(train_text)
        X_valid = vectorizer.transform(valid_text)
        X_test = vectorizer.transform(test_text)
        feature_names = vectorizer.get_feature_names_out()
        log(f"  vocabulary size = {len(feature_names)}; X_train shape = {X_train.shape}")

        is_binary = len(np.unique(y_train)) == 2
        model_specs = get_model_specs(is_binary=is_binary, max_tree_depth=args.max_tree_depth)
        if args.families != "both":
            model_specs = [m for m in model_specs if m["family"] == args.families]
        log(f"  is_binary={is_binary}; families={args.families}; max_tree_depth={args.max_tree_depth}; "
            f"will train {len(model_specs)} models: {[m['model_kind'] for m in model_specs]}")
        dataset_dir = MODEL_DIR / slugify(spec.key)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for mi, model_spec in enumerate(model_specs, 1):
            log(f"  --- model [{mi}/{len(model_specs)}] {model_spec['model_kind']} (family={model_spec['family']}) ---")
            family = model_spec["family"]
            train_limit = args.kernel_train_limit if family == "kernel" else args.tree_train_limit
            valid_limit = args.kernel_valid_limit if family == "kernel" else args.tree_valid_limit

            X_train_use, y_train_use = maybe_limit_split(X_train, y_train, train_limit, rng)
            X_valid_use, y_valid_use = maybe_limit_split(X_valid, y_valid, valid_limit, rng)
            log(f"    train/valid usage sizes: {len(y_train_use)}/{len(y_valid_use)} (limits: {train_limit}/{valid_limit})")

            if model_spec["dense_train"]:
                log("    densifying X_train/X_valid (dense_train=True)")
                X_train_fit = dense_rows(X_train_use)
                X_valid_fit = dense_rows(X_valid_use)
            else:
                X_train_fit = X_train_use
                X_valid_fit = X_valid_use

            y_train_fit = y_train_use
            y_valid_fit = y_valid_use
            if model_spec["model_kind"] == "krr_rbf":
                if not is_binary:
                    log("    krr_rbf requires binary labels; skipping")
                    continue
                log("    remapping labels {0,1} -> {-1,+1} for krr_rbf regression target")
                y_train_fit = np.where(y_train_use == 1, 1.0, -1.0)
                y_valid_fit = np.where(y_valid_use == 1, 1.0, -1.0)

            signature = {
                "dataset": spec.key,
                "model_kind": model_spec["model_kind"],
                "param_grid": model_spec["param_grid"],
                "max_features": args.max_features,
                "train_limit": train_limit,
                "valid_limit": valid_limit,
                "vectorizer_fit_limit": args.vectorizer_fit_limit,
                "max_tree_depth": args.max_tree_depth,
                "seed": args.seed,
            }
            bundle_path = dataset_dir / f"{model_spec['model_kind']}.joblib"
            bundle, trained_now = train_or_load_model(
                bundle_path=bundle_path,
                signature=signature,
                estimator=model_spec["estimator"],
                param_grid=model_spec["param_grid"],
                scoring=model_spec["scoring"],
                X_train=X_train_fit,
                y_train=y_train_use,
                X_valid=X_valid_fit,
                y_valid=y_valid_use,
                y_train_fit=np.asarray(y_train_fit),
                y_valid_fit=np.asarray(y_valid_fit),
                feature_names=feature_names,
                vectorizer=vectorizer,
                force_retrain=args.force_retrain,
                model_kind=model_spec["model_kind"],
            )

            X_test_eval = dense_rows(X_test) if model_spec["dense_train"] else X_test
            X_train_eval = dense_rows(X_train_use) if model_spec["dense_train"] else X_train_use
            log(f"    evaluating {model_spec['model_kind']} on train ({X_train_eval.shape[0]} rows) and test ({X_test_eval.shape[0]} rows)")
            metrics = evaluate_train_test(bundle, X_train_eval, y_train_use, X_test_eval, y_test)
            log(f"    train metrics: accuracy={metrics['train_accuracy']:.4f} macro_f1={metrics['train_macro_f1']:.4f}")
            log(f"    test  metrics: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
            tree_stats = compute_tree_stats(bundle["estimator"]) if model_spec["family"] == "tree" else {
                "tree_count": None,
                "total_leaves": None,
                "max_depth": None,
                "max_features_on_path": None,
            }
            if model_spec["family"] == "tree":
                log(
                    f"    tree stats: trees={tree_stats['tree_count']} "
                    f"leaves={tree_stats['total_leaves']} "
                    f"depth={tree_stats['max_depth']} "
                    f"max_feats_on_path={tree_stats['max_features_on_path']}"
                )
            metrics_rows.append(
                {
                    "dataset": spec.key,
                    "model_kind": model_spec["model_kind"],
                    "task": spec.task,
                    "trained_now": trained_now,
                    "fit_time_seconds": bundle["fit_time_seconds"],
                    "validation_score": bundle["best_validation_score"],
                    "train_accuracy": metrics["train_accuracy"],
                    "train_macro_f1": metrics["train_macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "best_params": json.dumps(bundle["best_params"], sort_keys=True),
                    "tree_count": tree_stats["tree_count"],
                    "total_leaves": tree_stats["total_leaves"],
                    "max_depth": tree_stats["max_depth"],
                    "max_features_on_path": tree_stats["max_features_on_path"],
                }
            )

            timing_rows.extend(
                benchmark_explainers(
                    dataset_key=spec.key,
                    bundle=bundle,
                    X_eval=X_test_eval,
                    explain_samples=args.explain_samples,
                    positive_label=spec.positive_label,
                )
            )

            # perturb_rows.extend(
            #     perturbation_faithfulness(
            #         dataset_key=spec.key,
            #         spec=spec,
            #         bundle=bundle,
            #         X_eval=X_test_eval,
            #         y_eval=y_test,
            #         removal_sizes=args.removal_sizes,
            #         max_instances=args.perturbation_samples,
            #         rng=rng,
            #     )
            # )

    log(f"All datasets done; aggregating {len(metrics_rows)} metric rows, "
        f"{len(timing_rows)} timing rows, {len(perturb_rows)} perturbation rows")
    metrics_df = pd.DataFrame(metrics_rows)
    timing_df = pd.DataFrame(timing_rows)
    perturb_df = pd.DataFrame(perturb_rows)

    log(f"Writing model_metrics.csv -> {RESULTS_DIR}")
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    log(f"Writing explainer_timing.csv -> {RESULTS_DIR}")
    timing_df.to_csv(RESULTS_DIR / "explainer_timing.csv", index=False)
    log(f"Writing perturbation_scores.csv -> {RESULTS_DIR}")
    perturb_df.to_csv(RESULTS_DIR / "perturbation_scores.csv", index=False)
    log("Writing summary_table.tex")
    write_summary_latex(metrics_df, timing_df, RESULTS_DIR / "summary_table.tex")
    log("Plotting timing results")
    plot_timing_results(timing_df)
    log("Plotting perturbation results")
    plot_perturbation_results(perturb_df)
    log("run_text_benchmark complete")


DEBUG_DEFAULT_ARGS = ["run", "--datasets", "rotten_tomatoes"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="Build the dataset catalog.")

    run_parser = subparsers.add_parser("run", help="Run the text benchmark on selected datasets.")
    run_parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset keys to benchmark.",
    )
    run_parser.add_argument("--max-features", type=int, default=300)
    run_parser.add_argument("--vectorizer-fit-limit", type=int, default=5000)
    run_parser.add_argument("--tree-train-limit", type=int, default=5000)
    run_parser.add_argument("--tree-valid-limit", type=int, default=2000)
    run_parser.add_argument("--kernel-train-limit", type=int, default=2000)
    run_parser.add_argument("--kernel-valid-limit", type=int, default=1000)
    run_parser.add_argument("--explain-samples", type=int, default=100)
    run_parser.add_argument("--perturbation-samples", type=int, default=50)
    run_parser.add_argument("--removal-sizes", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    run_parser.add_argument(
        "--families",
        choices=["tree", "kernel", "both"],
        default="both",
        help="Which model families to run: 'tree' (random_forest), "
        "'kernel' (svc_rbf, krr_rbf), or 'both' (default).",
    )
    run_parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=100,
        help="Hard upper bound on tree depth applied to the RandomForest "
        "param grid. ``None`` entries in the grid are replaced with this "
        "value; explicit integers are clamped down to it.",
    )
    run_parser.add_argument("--force-retrain", action="store_true")
    run_parser.add_argument("--seed", type=int, default=42)

    _ = catalog_parser

    if argv is None and len(sys.argv) == 1:
        argv = DEBUG_DEFAULT_ARGS

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    log(f"Starting text_classification_experiment with command={args.command!r}")
    if args.command == "catalog":
        df = build_dataset_catalog()
        write_catalog_outputs(df)
        log("Catalog command finished; printing dataframe")
        print(df.to_string(index=False))
        return
    if args.command == "run":
        run_text_benchmark(args)
        log(f"Wrote benchmark artifacts to {RESULTS_DIR}")
        print(f"Wrote benchmark artifacts to {RESULTS_DIR}")
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
