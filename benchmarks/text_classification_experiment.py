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

from pgshapley import TreeExplainer as PGTreeExplainer
from pgshapley.kernels import RBFLocalExplainer


MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "text"


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
    for spec in DATASET_SPECS.values():
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
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["task", "total_examples", "key"], ascending=[True, False, True])
    return df.reset_index(drop=True)


def write_catalog_outputs(df: pd.DataFrame) -> None:
    ensure_dirs()
    csv_path = RESULTS_DIR / "dataset_catalog.csv"
    json_path = RESULTS_DIR / "dataset_catalog.json"
    df.to_csv(csv_path, index=False)
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
    ds = load_dataset(spec.path, spec.config)
    train_ds = ds["train"]
    validation_ds = ds["validation"] if "validation" in ds else None
    test_ds = ds["test"] if "test" in ds else None

    if test_ds is None and validation_ds is not None:
        # Public leaderboard datasets such as SST-2 usually expose validation labels but hide test labels.
        test_ds = validation_ds
        split = train_ds.train_test_split(test_size=0.15, stratify_by_column=spec.label_column, seed=seed)
        train_ds = split["train"]
        validation_ds = split["test"]
    elif validation_ds is None:
        split = train_ds.train_test_split(test_size=0.2, stratify_by_column=spec.label_column, seed=seed)
        train_ds = split["train"]
        validation_ds = split["test"]
        if test_ds is None:
            split = validation_ds.train_test_split(test_size=0.5, stratify_by_column=spec.label_column, seed=seed)
            validation_ds = split["train"]
            test_ds = split["test"]

    assert validation_ds is not None
    assert test_ds is not None
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
        bundle = joblib.load(bundle_path)
        if bundle.get("signature") == signature:
            return bundle, False

    search_X = sparse.vstack([X_train, X_valid]) if sparse.issparse(X_train) else np.vstack([X_train, X_valid])
    search_y = np.concatenate([y_train_fit, y_valid_fit])
    test_fold = np.concatenate([np.full(len(y_train_fit), -1, dtype=int), np.zeros(len(y_valid_fit), dtype=int)])
    cv = PredefinedSplit(test_fold)
    grid = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    t0 = time.perf_counter()
    grid.fit(search_X, search_y)
    fit_time = time.perf_counter() - t0

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
    joblib.dump(bundle, bundle_path)
    return bundle, True


def get_model_specs(is_binary: bool) -> list[dict[str, object]]:
    specs = [
        {
            "model_kind": "decision_tree",
            "family": "tree",
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 2, 5],
            },
            "scoring": "accuracy",
            "dense_train": False,
        },
        {
            "model_kind": "random_forest",
            "family": "tree",
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [20, None],
                "min_samples_leaf": [1, 2],
            },
            "scoring": "accuracy",
            "dense_train": False,
        },
        {
            "model_kind": "extra_trees",
            "family": "tree",
            "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [20, None],
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
    rows: list[dict[str, object]] = []

    if bundle["model_kind"] in {"decision_tree", "random_forest", "extra_trees"}:
        tree_explainers: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
        target_class = positive_label if positive_label is not None else 0

        pg_prefix = PGTreeExplainer(
            estimator,
            feature_names=feature_names,
            backend_method="numpy_prefix_scan",
        )
        tree_explainers.append(
            (
                "pg_tree_numpy_prefix_scan",
                lambda row: extract_tree_pg_contrib(pg_prefix.shap_values(row.reshape(1, -1), check_additivity=False), target_class),
            )
        )

        pg_log = PGTreeExplainer(
            estimator,
            feature_names=feature_names,
            backend_method="numpy_logspace",
        )
        tree_explainers.append(
            (
                "pg_tree_numpy_logspace",
                lambda row: extract_tree_pg_contrib(pg_log.shap_values(row.reshape(1, -1), check_additivity=False), target_class),
            )
        )

        if shap is not None:
            shap_tree = shap.TreeExplainer(estimator)
            tree_explainers.append(
                (
                    "shap_tree",
                    lambda row: extract_shap_tree_contrib(shap_tree.shap_values(row.reshape(1, -1)), target_class),
                )
            )

        for explainer_name, fn in tree_explainers:
            times = []
            for row in X_eval_dense[:n_samples]:
                t0 = time.perf_counter()
                _ = fn(row)
                times.append(time.perf_counter() - t0)
            rows.append(
                {
                    "dataset": dataset_key,
                    "model_kind": bundle["model_kind"],
                    "explainer": explainer_name,
                    "n_samples": n_samples,
                    "avg_seconds_per_instance": float(np.mean(times)),
                    "std_seconds_per_instance": float(np.std(times)),
                }
            )
        return rows

    if bundle["model_kind"] in {"svc_rbf", "krr_rbf"}:
        kernel_explainer = RBFLocalExplainer(estimator)
        methods = ["logspace_numpy", "prefix_scan_numpy"]
        for method in methods:
            times = []
            for row in X_eval_dense[:n_samples]:
                t0 = time.perf_counter()
                _ = kernel_explainer.explain(row, method=method)
                times.append(time.perf_counter() - t0)
            rows.append(
                {
                    "dataset": dataset_key,
                    "model_kind": bundle["model_kind"],
                    "explainer": f"rbf_local_{method}",
                    "n_samples": n_samples,
                    "avg_seconds_per_instance": float(np.mean(times)),
                    "std_seconds_per_instance": float(np.std(times)),
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
        return []
    mask = y_eval == spec.positive_label
    positive_indices = np.flatnonzero(mask)
    if len(positive_indices) == 0:
        return []
    if len(positive_indices) > max_instances:
        positive_indices = rng.choice(positive_indices, size=max_instances, replace=False)

    X_eval_dense = dense_rows(X_eval)
    rows: list[dict[str, object]] = []
    per_k: dict[int, list[float]] = {k: [] for k in removal_sizes}
    per_k_removed: dict[int, list[int]] = {k: [] for k in removal_sizes}

    for idx in positive_indices:
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
        return
    for dataset_key, dataset_df in df.groupby("dataset"):
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


def plot_timing_results(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = df.copy()
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
    metrics_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    perturb_rows: list[dict[str, object]] = []

    for spec in selected_specs:
        splits = load_dataset_splits(spec, seed=args.seed)
        train_text, y_train = dataset_to_xy(splits["train"], spec)
        valid_text, y_valid = dataset_to_xy(splits["validation"], spec)
        test_text, y_test = dataset_to_xy(splits["test"], spec)

        vectorizer = build_vectorizer(args.max_features)
        fit_text = train_text
        if args.vectorizer_fit_limit is not None and len(fit_text) > args.vectorizer_fit_limit:
            fit_idx = sample_indices_by_label(y_train, args.vectorizer_fit_limit, rng)
            fit_text = [train_text[i] for i in fit_idx]
        vectorizer.fit(fit_text)

        X_train = vectorizer.transform(train_text)
        X_valid = vectorizer.transform(valid_text)
        X_test = vectorizer.transform(test_text)
        feature_names = vectorizer.get_feature_names_out()

        is_binary = len(np.unique(y_train)) == 2
        model_specs = get_model_specs(is_binary=is_binary)
        dataset_dir = MODEL_DIR / slugify(spec.key)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for model_spec in model_specs:
            family = model_spec["family"]
            train_limit = args.kernel_train_limit if family == "kernel" else args.tree_train_limit
            valid_limit = args.kernel_valid_limit if family == "kernel" else args.tree_valid_limit

            X_train_use, y_train_use = maybe_limit_split(X_train, y_train, train_limit, rng)
            X_valid_use, y_valid_use = maybe_limit_split(X_valid, y_valid, valid_limit, rng)

            if model_spec["dense_train"]:
                X_train_fit = dense_rows(X_train_use)
                X_valid_fit = dense_rows(X_valid_use)
            else:
                X_train_fit = X_train_use
                X_valid_fit = X_valid_use

            y_train_fit = y_train_use
            y_valid_fit = y_valid_use
            if model_spec["model_kind"] == "krr_rbf":
                if not is_binary:
                    continue
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
            metrics = evaluate_model(bundle, X_test_eval, y_test)
            metrics_rows.append(
                {
                    "dataset": spec.key,
                    "model_kind": model_spec["model_kind"],
                    "task": spec.task,
                    "trained_now": trained_now,
                    "fit_time_seconds": bundle["fit_time_seconds"],
                    "validation_score": bundle["best_validation_score"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "best_params": json.dumps(bundle["best_params"], sort_keys=True),
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

            perturb_rows.extend(
                perturbation_faithfulness(
                    dataset_key=spec.key,
                    spec=spec,
                    bundle=bundle,
                    X_eval=X_test_eval,
                    y_eval=y_test,
                    removal_sizes=args.removal_sizes,
                    max_instances=args.perturbation_samples,
                    rng=rng,
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    timing_df = pd.DataFrame(timing_rows)
    perturb_df = pd.DataFrame(perturb_rows)

    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    timing_df.to_csv(RESULTS_DIR / "explainer_timing.csv", index=False)
    perturb_df.to_csv(RESULTS_DIR / "perturbation_scores.csv", index=False)
    plot_timing_results(timing_df)
    plot_perturbation_results(perturb_df)


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
    run_parser.add_argument("--force-retrain", action="store_true")
    run_parser.add_argument("--seed", type=int, default=42)

    _ = catalog_parser

    if argv is None and len(sys.argv) == 1:
        argv = DEBUG_DEFAULT_ARGS

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "catalog":
        df = build_dataset_catalog()
        write_catalog_outputs(df)
        print(df.to_string(index=False))
        return
    if args.command == "run":
        run_text_benchmark(args)
        print(f"Wrote benchmark artifacts to {RESULTS_DIR}")
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
