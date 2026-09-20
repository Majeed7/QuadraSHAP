"""Text-classifier PKeX-Shapley versus QuadraSHAP CPU/Apple-Metal benchmark.

The IMDB benchmark is deliberately isolated from ``text_datasets.py``: it reads the
canonical Hugging Face Arrow cache (25,000 train and 25,000 test reviews) without
applying a review-length filter. Rotten Tomatoes, SST-2, SMS spam and emotion use
their complete source splits (or a documented SMS split). All are represented by
5,000-dimensional TF-IDF inputs.

For each selected test text we compare the same neutral-factor product-kernel game using

* QuadraSHAP prefix scan on CPU (NumPy, float64),
* QuadraSHAP prefix scan on Apple Metal (JAX, float32), and
* the quadratic ``ESPComputer`` in the supplied RKHS-ExactSHAP repository.

The CPU result requested at eps=1e-16 is the experiment's reference solution.  The a-priori
certificate concerns quadrature in exact arithmetic; observed errors also contain floating-point
error.  PKeX-Shapley is algebraically exact, so its exact-arithmetic approximation certificate is
zero, while any observed discrepancy is numerical.

Examples
--------
Prepare the full model and the fixed stratified sample of 20 reviews::

    .venv/bin/python experiments/exp10_pkex_metal_imdb.py prepare

Run one pilot (the parent process must be allowed to access Metal)::

    ENABLE_PJRT_COMPATIBILITY=1 .venv/bin/python \
      experiments/exp10_pkex_metal_imdb.py run --n-instances 1 --timeout 300

Run all 20 and assemble the CSV files::

    ENABLE_PJRT_COMPATIBILITY=1 .venv/bin/python \
      experiments/exp10_pkex_metal_imdb.py run --n-instances 20 --timeout 300
    .venv/bin/python experiments/exp10_pkex_metal_imdb.py summarize
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quadrashap.product_games.budget import GameSummary, budget_from_summary
from quadrashap.product_games.shapley import ProductGamesShapleyJax, ProductGamesShapleyNumpy


NAME = "exp10_pkex_metal_imdb"
OUT = ROOT / "experiments" / "results" / NAME
MODEL_PATH = OUT / "imdb_full_tfidf5000_svc.joblib"
INSTANCES_PATH = OUT / "instances.npz"
EPS_LEVELS = (1e-3, 1e-5, 1e-10, 1e-16)
REFERENCE_EPS = 1e-16
DEFAULT_PKEX_REPO = ROOT.parent / "RKHS-ExactSHAP"
ARROW_CACHE_GLOB = (
    Path.home() / ".cache" / "huggingface" / "datasets" / "imdb" / "plain_text"
)
DATASET_CHOICES = ("imdb", "rotten_tomatoes", "sst2", "sms_spam", "emotion")
REMOTE_PARQUET = {
    "emotion": {
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "repo": "dair-ai/emotion",
        "config": "split",
        "splits": ("train", "test"),
        "text_column": "text",
    },
    "sms_spam": {
        "revision": "cae486f927c250fe1d4a5b55f11357964ed1646c",
        "repo": "ucirvine/sms_spam",
        "config": "plain_text",
        "splits": ("train",),
        "text_column": "sms",
    },
}


def set_dataset(name: str) -> None:
    """Select an output namespace without changing the original IMDB artifacts."""
    global NAME, OUT, MODEL_PATH, INSTANCES_PATH
    if name not in DATASET_CHOICES:
        raise ValueError(name)
    if name == "imdb":
        NAME = "exp10_pkex_metal_imdb"
        OUT = ROOT / "experiments" / "results" / NAME
        MODEL_PATH = OUT / "imdb_full_tfidf5000_svc.joblib"
    else:
        NAME = f"exp11_pkex_metal_{name}"
        OUT = ROOT / "experiments" / "results" / NAME
        MODEL_PATH = OUT / f"{name}_full_tfidf5000_svc.joblib"
    INSTANCES_PATH = OUT / "instances.npz"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arrow_paths(cache_root: Path | None = None) -> tuple[Path, Path]:
    """Locate the complete canonical Hugging Face IMDB train/test Arrow files."""
    root = ARROW_CACHE_GLOB if cache_root is None else Path(cache_root)
    candidates = sorted(root.glob("*/*/imdb-train.arrow"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No complete Hugging Face IMDB Arrow cache below {root}. "
            "Load the 'imdb' dataset once with Hugging Face datasets first."
        )
    train = candidates[0]
    test = train.with_name("imdb-test.arrow")
    if not test.exists():
        raise FileNotFoundError(f"Missing matching test split: {test}")
    return train, test


def read_arrow(path: Path) -> tuple[list[str], np.ndarray]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        try:
            table = ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = ipc.open_file(source).read_all()
    return table["text"].to_pylist(), np.asarray(table["label"].to_numpy(), dtype=np.int64)


def cached_arrow_files(dataset: str) -> tuple[Path, Path]:
    base = Path.home() / ".cache" / "huggingface" / "datasets"
    if dataset == "rotten_tomatoes":
        matches = sorted(base.glob("rotten_tomatoes/default/*/*/rotten_tomatoes-train.arrow"))
        test_name = "rotten_tomatoes-test.arrow"
    elif dataset == "sst2":
        matches = sorted(base.glob("glue/sst2/*/*/glue-train.arrow"))
        test_name = "glue-validation.arrow"  # public GLUE test labels are hidden
    else:
        raise ValueError(dataset)
    if not matches:
        raise FileNotFoundError(f"Hugging Face Arrow cache for {dataset} not found below {base}")
    train = matches[-1]
    test = train.with_name(test_name)
    if not test.exists():
        raise FileNotFoundError(test)
    return train, test


def _download_parquet(dataset: str, split: str) -> Path:
    spec = REMOTE_PARQUET[dataset]
    target = OUT / "source" / f"{split}.parquet"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/"
           f"{spec['config']}/{split}-00000-of-00001.parquet")
    tmp = target.with_suffix(".parquet.tmp")
    log(f"downloading pinned {dataset}/{split} from {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.URLError as exc:
        # Some macOS framework-Python installations lack a configured CA bundle.
        # curl uses the system trust store and still validates TLS; never use -k.
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        log("Python CA bundle unavailable; retrying with TLS-verified system curl")
        subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error",
                        "--max-time", "120", url, "--output", str(tmp)], check=True)
    tmp.replace(target)
    return target


def _read_parquet(path: Path, text_column: str) -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=[text_column, "label"])
    return table[text_column].to_pylist(), np.asarray(table["label"].to_numpy(), dtype=np.int64)


def load_full_corpus(name: str, seed: int, arrow_cache: Path | None = None):
    """Full source splits, without review-length or document-count filtering."""
    if name == "imdb":
        train_path, test_path = arrow_paths(arrow_cache)
        train_text, y_train = read_arrow(train_path)
        test_text, y_test = read_arrow(test_path)
        assert (len(y_train), len(y_test)) == (25_000, 25_000)
        source = {"train": train_path, "test": test_path, "split_note": "official train/test"}
    elif name in {"rotten_tomatoes", "sst2"}:
        train_path, test_path = cached_arrow_files(name)
        text_col = "sentence" if name == "sst2" else "text"
        train_text, y_train = read_arrow_columns(train_path, text_col)
        test_text, y_test = read_arrow_columns(test_path, text_col)
        source = {"train": train_path, "test": test_path,
                  "split_note": "official train/validation" if name == "sst2" else "official train/test"}
    elif name == "emotion":
        train_path = _download_parquet(name, "train")
        test_path = _download_parquet(name, "test")
        train_text, y_train = _read_parquet(train_path, "text")
        test_text, y_test = _read_parquet(test_path, "text")
        assert (len(y_train), len(y_test)) == (16_000, 2_000)
        source = {"train": train_path, "test": test_path, "split_note": "official train/test"}
    elif name == "sms_spam":
        path = _download_parquet(name, "train")
        texts, labels = _read_parquet(path, "sms")
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(labels)), test_size=.2,
                                  stratify=labels, random_state=seed)
        train_text, y_train = [texts[i] for i in tr], labels[tr]
        test_text, y_test = [texts[i] for i in te], labels[te]
        source = {"train": path, "test": path,
                  "split_note": f"full {len(labels)} messages, stratified 80/20 split, seed {seed}"}
    else:
        raise ValueError(name)
    return train_text, y_train, test_text, y_test, source


def read_arrow_columns(path: Path, text_column: str) -> tuple[list[str], np.ndarray]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all()
    return table[text_column].to_pylist(), np.asarray(table["label"].to_numpy(), dtype=np.int64)


def stratified_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """A fixed balanced sample, shuffled after selection."""
    if n > len(y):
        raise ValueError(f"requested {n} instances from a test set of {len(y)}")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    pieces = []
    base, rem = divmod(n, len(classes))
    for k, label in enumerate(classes):
        count = base + (k < rem)
        pool = np.flatnonzero(y == label)
        pieces.append(rng.choice(pool, count, replace=False))
    selected = np.concatenate(pieces)
    rng.shuffle(selected)
    return selected.astype(np.int64)


def prepare(args: argparse.Namespace) -> None:
    """Train/cache a full-data SVC and save 20 fixed, unfiltered test inputs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    OUT.mkdir(parents=True, exist_ok=True)
    train_text, y_train, test_text, y_test, source = load_full_corpus(
        args.dataset, args.seed, args.arrow_cache)
    log(f"using full {args.dataset} corpus: train={len(y_train)}, test={len(y_test)}")

    if MODEL_PATH.exists() and not args.force:
        log(f"using cached model {MODEL_PATH}")
        bundle = joblib.load(MODEL_PATH)
        vectorizer, model = bundle["vectorizer"], bundle["model"]
        X_test = vectorizer.transform(test_text)
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            # Full SMS training data has only 3,692 terms at min_df=3.  The smallest
            # change that actually yields d=5000 is min_df=2; record the exception.
            min_df=2 if args.dataset == "sms_spam" else 3,
            max_df=0.95,
            max_features=5000,
            sublinear_tf=True,
            dtype=np.float64,
        )
        t0 = time.perf_counter()
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)
        vectorize_seconds = time.perf_counter() - t0
        if X_train.shape != (len(y_train), 5_000):
            raise RuntimeError(f"unexpected train design shape {X_train.shape}; expected d=5000")
        log(f"TF-IDF ready in {vectorize_seconds:.1f}s; fitting full RBF-SVC")
        base_model = SVC(
            kernel="rbf",
            C=float(args.C),
            gamma=args.gamma if args.gamma in {"scale", "auto"} else float(args.gamma),
            cache_size=float(args.svc_cache_mb),
        )
        model = OneVsRestClassifier(base_model, n_jobs=1) if args.dataset == "emotion" else base_model
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        accuracy = float(accuracy_score(y_test, model.predict(X_test)))
        bundle = {
            "model": model,
            "vectorizer": vectorizer,
            "protocol": {
                "dataset": args.dataset,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "review_length_filter": None,
                "tfidf": {
                    "lowercase": True,
                    "stop_words": "english",
                    "ngram_range": [1, 2],
                    "min_df": 2 if args.dataset == "sms_spam" else 3,
                    "max_df": 0.95,
                    "max_features": 5000,
                    "sublinear_tf": True,
                },
                "svc": {"kernel": "rbf", "C": float(args.C), "gamma": args.gamma,
                        "multiclass": "one-vs-rest" if args.dataset == "emotion" else None},
                "source_split_note": source["split_note"],
            },
            "vectorize_seconds": vectorize_seconds,
            "fit_seconds": fit_seconds,
            "test_accuracy": accuracy,
            "source_train": str(source["train"]),
            "source_test": str(source["test"]),
            "source_train_sha256": file_sha256(source["train"]),
            "source_test_sha256": file_sha256(source["test"]),
        }
        joblib.dump(bundle, MODEL_PATH, compress=3)
        components = model.estimators_ if args.dataset == "emotion" else [model]
        log(
            f"model fit in {fit_seconds:.1f}s: accuracy={accuracy:.4f}, "
            f"support vectors={[int(m.support_.size) for m in components]}, "
            f"gamma={components[0]._gamma:.6g}"
        )

    selected = stratified_indices(y_test, args.n_instances, args.seed)
    X_selected = vectorizer.transform([test_text[i] for i in selected]).toarray().astype(np.float64)
    predicted = np.asarray(model.predict(X_selected), dtype=np.int64)
    lengths_words = np.asarray([len(test_text[i].split()) for i in selected], dtype=np.int64)
    lengths_chars = np.asarray([len(test_text[i]) for i in selected], dtype=np.int64)
    np.savez_compressed(
        INSTANCES_PATH,
        X=X_selected,
        y=y_test[selected],
        target_classes=predicted,
        test_indices=selected,
        word_counts=lengths_words,
        char_counts=lengths_chars,
    )
    meta = {
        "name": NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": args.dataset,
        "canonical_split_sizes": [len(train_text), len(test_text)],
        "split_note": source["split_note"],
        "source_train": str(source["train"]),
        "source_test": str(source["test"]),
        "review_length_filter": None,
        "selected_instances": selected.tolist(),
        "selected_labels": y_test[selected].tolist(),
        "selected_target_classes": predicted.tolist(),
        "selected_word_counts": lengths_words.tolist(),
        "selected_word_count_summary": {
            "min": int(lengths_words.min()),
            "median": float(np.median(lengths_words)),
            "mean": float(lengths_words.mean()),
            "max": int(lengths_words.max()),
        },
        "d": int(X_selected.shape[1]),
        "n_support_vectors": ([int(m.support_.size) for m in model.estimators_]
                              if args.dataset == "emotion" else int(model.support_.size)),
        "gamma_effective": (float(model.estimators_[0]._gamma)
                            if args.dataset == "emotion" else float(model._gamma)),
        "model_score": ("predicted-class one-vs-rest decision score" if args.dataset == "emotion"
                        else "binary SVC decision score"),
        "test_accuracy": float(bundle.get("test_accuracy", np.nan)),
        "model_path": str(MODEL_PATH),
        "instances_path": str(INSTANCES_PATH),
    }
    atomic_json(OUT / "prepare_meta.json", meta)
    log(
        f"saved {len(selected)} unfiltered texts: words min/median/mean/max = "
        f"{lengths_words.min()}/{np.median(lengths_words):.0f}/{lengths_words.mean():.1f}/{lengths_words.max()}"
    )


def _dense_rows(matrix, start: int, stop: int, dtype) -> np.ndarray:
    block = matrix[start:stop]
    if hasattr(block, "toarray"):
        block = block.toarray()
    return np.asarray(block, dtype=dtype)


def selected_model(bundle: dict, target_class: int):
    model = bundle["model"]
    return model.estimators_[target_class] if hasattr(model, "estimators_") else model


def _model_arrays(model) -> tuple[object, np.ndarray, float, float]:
    support = model.support_vectors_
    dual = model.dual_coef_
    if hasattr(dual, "toarray"):
        dual = dual.toarray()
    alpha = np.asarray(dual, dtype=np.float64).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    return support, alpha, float(model._gamma), intercept


def factor_block(support, x: np.ndarray, gamma: float, start: int, stop: int, dtype) -> np.ndarray:
    z = _dense_rows(support, start, stop, dtype=np.float64)
    u = np.exp(-gamma * np.square(z - x[None, :]))
    return u.astype(dtype, copy=False)


def summarize_games(support, alpha: np.ndarray, x: np.ndarray, gamma: float, chunk: int) -> tuple[GameSummary, dict]:
    d = x.size
    summary = GameSummary(d=d)
    u_min, u_max = 1.0, 0.0
    t0 = time.perf_counter()
    for start in range(0, len(alpha), chunk):
        stop = min(start + chunk, len(alpha))
        U = factor_block(support, x, gamma, start, stop, np.float64)
        u_min = min(u_min, float(U.min()))
        u_max = max(u_max, float(U.max()))
        summary.update(U - 1.0, None, alpha[start:stop])
    seconds = time.perf_counter() - t0
    return summary, {
        "summary_seconds": seconds,
        "factor_min": u_min,
        "factor_max": u_max,
        "A_max": summary.A_max,
        "lambda_max": summary.lambda_max,
        "n_games": summary.n_games,
    }


def expansion_target(model, x: np.ndarray, alpha: np.ndarray, intercept: float) -> tuple[float, float, float]:
    score = float(np.asarray(model.decision_function(x[None, :])).ravel()[0])
    f_kernel = score - intercept
    v_empty = float(alpha.sum())
    return f_kernel, v_empty, f_kernel - v_empty


def evaluate_quadrature(
    method: str,
    support,
    alpha: np.ndarray,
    x: np.ndarray,
    gamma: float,
    m_q: int,
    chunk: int,
    node_block: int,
) -> tuple[np.ndarray, float]:
    if method == "cpu":
        engine = ProductGamesShapleyNumpy()
        dtype = np.float64
    elif method == "metal":
        engine = ProductGamesShapleyJax()
        dtype = np.float32
    else:
        raise ValueError(method)

    phi = np.zeros(x.size, dtype=np.float64)
    t0 = time.perf_counter()
    for start in range(0, len(alpha), chunk):
        stop = min(start + chunk, len(alpha))
        U = factor_block(support, x, gamma, start, stop, dtype)
        weights = alpha[start:stop]
        if method == "metal" and stop - start < chunk:
            pad = chunk - (stop - start)
            U = np.pad(U, ((0, pad), (0, 0)), constant_values=1.0)
            weights = np.pad(weights, (0, pad))
        Phi = engine.phi_matrix_prefix_scan(U - dtype(1.0), m_q, Ut=None, node_block=node_block)
        phi += (np.asarray(Phi, dtype=np.float64) * weights[:, None]).sum(axis=0)
    return phi, time.perf_counter() - t0


def eps_tag(eps: float) -> str:
    return f"{eps:.0e}".replace("+", "")


def record_paths(method: str, instance: int, eps: float | None = None) -> tuple[Path, Path]:
    stem = f"{method}_instance{instance:03d}"
    if eps is not None:
        stem += f"_eps{eps_tag(eps)}"
    return OUT / "raw" / f"{stem}.json", OUT / "raw" / f"{stem}.npz"


def worker_quadrashap(args: argparse.Namespace, model, x: np.ndarray) -> None:
    support, alpha, gamma, intercept = _model_arrays(model)
    summary, shared = summarize_games(support, alpha, x, gamma, args.chunk)
    budgets = {eps: budget_from_summary(summary, eps, norm="max") for eps in EPS_LEVELS}
    f_kernel, v_empty, target = expansion_target(model, x, alpha, intercept)

    # The tightest result is first so it survives even if a later stress run hits the cap.
    ordered = (REFERENCE_EPS, 1e-10, 1e-5, 1e-3)
    for eps in ordered:
        report = budgets[eps]
        json_path, phi_path = record_paths(args.method, args.instance, eps)
        if json_path.exists() and phi_path.exists() and not args.force:
            log(f"worker cache hit: {json_path.name}")
            continue

        # Warm the single Metal executable shape once; compilation is reported separately.
        compile_seconds = 0.0
        if args.method == "metal" and eps == ordered[0]:
            first = min(args.chunk, len(alpha))
            K0 = factor_block(support, x, gamma, 0, first, np.float32) - np.float32(1.0)
            if first < args.chunk:
                K0 = np.pad(K0, ((0, args.chunk - first), (0, 0)))
            t_compile = time.perf_counter()
            ProductGamesShapleyJax().phi_matrix_prefix_scan(
                K0, report.m_q, Ut=None, node_block=args.node_block
            )
            compile_seconds = time.perf_counter() - t_compile

        values, timings = [], []
        for _ in range(args.repeats):
            phi, seconds = evaluate_quadrature(
                args.method,
                support,
                alpha,
                x,
                gamma,
                report.m_q,
                args.chunk,
                args.node_block,
            )
            values.append(phi)
            timings.append(seconds)
        stack = np.stack(values)
        phi = stack[0]
        repeat_max_abs = float(np.max(np.abs(stack - stack[0])))
        efficiency = abs(float(phi.sum()) - target)
        np.savez_compressed(phi_path, phi=phi, repeats=stack)
        row = {
            "method": "QuadraSHAP CPU" if args.method == "cpu" else "QuadraSHAP Metal",
            "method_key": args.method,
            "instance": args.instance,
            "eps": eps,
            "status": "complete",
            "m_q": int(report.m_q),
            "certified_bound": float(report.bound),
            "certificate_scope": "quadrature error in exact arithmetic",
            "dtype": "float64" if args.method == "cpu" else "float32",
            "seconds_core_median": float(np.median(timings)),
            "seconds_core_all": timings,
            "seconds_summary": float(shared["summary_seconds"]),
            "seconds_end_to_end": float(shared["summary_seconds"] + np.median(timings)),
            "compile_seconds_excluded": compile_seconds,
            "repeat_max_abs": repeat_max_abs,
            "efficiency_residual": float(efficiency),
            "finite": bool(np.isfinite(stack).all()),
            "f_kernel": f_kernel,
            "v_empty": v_empty,
            "efficiency_target": target,
            "phi_l2": float(np.linalg.norm(phi)),
            "phi_max_abs": float(np.max(np.abs(phi))),
            **shared,
            "phi_path": str(phi_path),
        }
        atomic_json(json_path, row)
        log(
            f"{args.method} instance={args.instance} eps={eps:.0e} m_q={report.m_q} "
            f"cert={report.bound:.2e} core={np.median(timings):.3f}s eff={efficiency:.2e}"
        )


def worker_pkex(args: argparse.Namespace, model, x: np.ndarray) -> None:
    repo = Path(args.pkex_repo).resolve()
    esp_dir = repo / "explainer"
    if not (esp_dir / "esp.py").exists():
        raise FileNotFoundError(f"PKeX-Shapley repository not found at {repo}")
    sys.path.insert(0, str(esp_dir))
    from esp import ESPComputer

    support, alpha, gamma, intercept = _model_arrays(model)
    f_kernel, v_empty, target = expansion_target(model, x, alpha, intercept)
    phi = np.zeros(x.size, dtype=np.float64)
    t0 = time.perf_counter()
    factor_seconds = 0.0
    esp = ESPComputer(method="quadratic", use_scaling=True)
    for start in range(0, len(alpha), args.pkex_chunk):
        stop = min(start + args.pkex_chunk, len(alpha))
        tf = time.perf_counter()
        U = factor_block(support, x, gamma, start, stop, np.float64)
        factor_seconds += time.perf_counter() - tf
        Omega = esp.compute_weight_vectors(U)
        phi += (alpha[start:stop, None] * ((U - 1.0) * Omega)).sum(axis=0)
    seconds = time.perf_counter() - t0
    json_path, phi_path = record_paths("pkex", args.instance)
    np.savez_compressed(phi_path, phi=phi)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    row = {
        "method": "PKeX-Shapley",
        "method_key": "pkex",
        "instance": args.instance,
        "eps": None,
        "status": "complete",
        "m_q": None,
        "certified_bound": 0.0,
        "certificate_scope": "algebraically exact in exact arithmetic",
        "dtype": "float64",
        "seconds_core_median": seconds,
        "seconds_core_all": [seconds],
        "seconds_summary": 0.0,
        "seconds_end_to_end": seconds,
        "factor_seconds": factor_seconds,
        "compile_seconds_excluded": 0.0,
        "repeat_max_abs": None,
        "efficiency_residual": abs(float(phi.sum()) - target),
        "finite": bool(np.isfinite(phi).all()),
        "f_kernel": f_kernel,
        "v_empty": v_empty,
        "efficiency_target": target,
        "phi_l2": float(np.linalg.norm(phi)),
        "phi_max_abs": float(np.max(np.abs(phi))),
        "pkex_repo": str(repo),
        "pkex_commit": commit,
        "pkex_esp_method": "quadratic",
        "pkex_use_scaling": True,
        "pkex_chunk": args.pkex_chunk,
        "phi_path": str(phi_path),
    }
    atomic_json(json_path, row)
    log(f"pkex instance={args.instance} complete in {seconds:.3f}s")


def worker(args: argparse.Namespace) -> None:
    bundle = joblib.load(MODEL_PATH)
    arrays = np.load(INSTANCES_PATH)
    x = np.asarray(arrays["X"][args.instance], dtype=np.float64)
    target_class = int(arrays["target_classes"][args.instance]) if "target_classes" in arrays.files else 0
    model = selected_model(bundle, target_class)
    if args.method in {"cpu", "metal"}:
        worker_quadrashap(args, model, x)
    else:
        worker_pkex(args, model, x)


def run(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists() or not INSTANCES_PATH.exists():
        raise FileNotFoundError("run the 'prepare' command first")
    arrays = np.load(INSTANCES_PATH)
    n_available = len(arrays["X"])
    n = min(args.n_instances, n_available)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("PYTHONHASHSEED", "0")

    run_meta = {
        "dataset": args.dataset,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_instances": n,
        "timeout_seconds_per_method_per_instance": args.timeout,
        "eps_levels": EPS_LEVELS,
        "reference": "QuadraSHAP CPU float64 at requested eps=1e-16; not an arbitrary-precision proof",
        "methods": args.methods,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": "Apple M5 Max, 32 GPU cores",
        "pkex_repo": str(Path(args.pkex_repo).resolve()),
        "pkex_esp_method": "quadratic with scaling, repository implementation",
        "pkex_chunk": args.pkex_chunk,
        "cpu_chunk": args.chunk,
        "node_block": args.node_block,
        "repeats": args.repeats,
        "script_sha256": file_sha256(script),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "jax", "jaxlib", "jax-metal", "pyarrow")
        },
    }
    repo = Path(args.pkex_repo).resolve()
    if (repo / "explainer" / "esp.py").exists():
        run_meta["pkex_esp_sha256"] = file_sha256(repo / "explainer" / "esp.py")
        run_meta["pkex_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if (OUT / "run_meta.json").exists():
        old = json.loads((OUT / "run_meta.json").read_text())
        run_meta["first_started_at"] = old.get("first_started_at", old["started_at"])
        run_meta["methods"] = sorted(set(old.get("methods", [])) | set(args.methods))
    atomic_json(OUT / "run_meta.json", run_meta)

    def run_one(method: str, instance: int) -> None:
            status_path = raw / f"status_{method}_instance{instance:03d}.json"
            if status_path.exists() and not args.force:
                previous = json.loads(status_path.read_text())
                if previous.get("status") in {"complete", "timeout"}:
                    log(f"status cache hit: {status_path.name} ({previous['status']})")
                    return
            cmd = [
                sys.executable,
                str(script),
                "worker",
                "--method",
                method,
                "--dataset",
                args.dataset,
                "--instance",
                str(instance),
                "--chunk",
                str(args.chunk),
                "--node-block",
                str(args.node_block),
                "--repeats",
                str(args.repeats),
                "--pkex-chunk",
                str(args.pkex_chunk),
                "--pkex-repo",
                str(args.pkex_repo),
            ]
            if args.force:
                cmd.append("--force")
            log(f"starting {method}, instance {instance}, hard cap {args.timeout}s")
            t0 = time.perf_counter()
            status = "complete"
            returncode = None
            stdout_path = raw / f"log_{method}_instance{instance:03d}.txt"
            try:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                returncode = result.returncode
                stdout_path.write_text(result.stdout)
                if result.returncode != 0:
                    status = "error"
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                captured = exc.stdout or ""
                if isinstance(captured, bytes):
                    captured = captured.decode("utf-8", "replace")
                stdout_path.write_text(captured + f"\nTIMEOUT after {args.timeout} seconds\n")
            seconds = time.perf_counter() - t0
            atomic_json(
                status_path,
                {
                    "method_key": method,
                    "instance": instance,
                    "status": status,
                    "wall_seconds": seconds,
                    "timeout_seconds": args.timeout,
                    "returncode": returncode,
                    "log_path": str(stdout_path),
                },
            )
            log(f"finished {method}, instance {instance}: {status} in {seconds:.1f}s")

    for method in args.methods:
        # Metal owns the single GPU, and concurrent NumPy scans distort CPU timing.  Only the
        # independent CPU-only PKeX timeout jobs are parallelized.
        workers = args.parallel_pkex if method == "pkex" else 1
        if workers == 1:
            for instance in range(n):
                run_one(method, instance)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_one, method, instance) for instance in range(n)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    # A pilot run may follow a completed full run. Reassemble every cached
    # prepared instance so its CSV does not temporarily drop prior results.
    summarize(argparse.Namespace(n_instances=n_available))


def _load_phi(row: dict) -> np.ndarray:
    return np.asarray(np.load(row["phi_path"])["phi"], dtype=np.float64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> None:
    raw = OUT / "raw"
    rows = []
    for path in sorted(raw.glob("*_instance*_eps*.json")) + sorted(raw.glob("pkex_instance*.json")):
        row = json.loads(path.read_text())
        instance = int(row["instance"])
        if hasattr(args, "n_instances") and instance >= args.n_instances:
            continue
        ref_json, _ = record_paths("cpu", instance, REFERENCE_EPS)
        if ref_json.exists():
            ref_row = json.loads(ref_json.read_text())
            reference = _load_phi(ref_row)
            phi = _load_phi(row)
            delta = phi - reference
            row["observed_max_abs_error"] = float(np.max(np.abs(delta)))
            denom = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
            row["observed_rel_l2_error"] = float(np.linalg.norm(delta) / denom)
            row["observed_l2_error"] = float(np.linalg.norm(delta))
            row["target_met_observed"] = (
                None if row.get("eps") is None else bool(row["observed_max_abs_error"] <= float(row["eps"]))
            )
            row["observed_within_certificate"] = bool(
                row["observed_max_abs_error"] <= float(row["certified_bound"]) + 32 * np.finfo(float).eps
            )
        else:
            row["observed_max_abs_error"] = None
            row["observed_rel_l2_error"] = None
            row["observed_l2_error"] = None
            row["target_met_observed"] = None
            row["observed_within_certificate"] = None
        rows.append(row)

    # Add explicit timeout/error records, preserving NA error fields rather than inventing timings.
    completed = {(r["method_key"], int(r["instance"])) for r in rows}
    for path in sorted(raw.glob("status_*_instance*.json")):
        status = json.loads(path.read_text())
        key = (status["method_key"], int(status["instance"]))
        if status["status"] == "complete" or key in completed:
            continue
        rows.append(
            {
                "method": {"cpu": "QuadraSHAP CPU", "metal": "QuadraSHAP Metal", "pkex": "PKeX-Shapley"}[key[0]],
                "method_key": key[0],
                "instance": key[1],
                "eps": None,
                "status": status["status"],
                "m_q": None,
                "certified_bound": 0.0 if key[0] == "pkex" else None,
                "certificate_scope": "algebraically exact in exact arithmetic" if key[0] == "pkex" else None,
                "seconds_core_median": None,
                "seconds_end_to_end": None,
                "wall_seconds_until_failure": status["wall_seconds"],
                "observed_max_abs_error": None,
                "observed_rel_l2_error": None,
                "efficiency_residual": None,
                "finite": None,
            }
        )

    rows.sort(key=lambda r: (int(r["instance"]), str(r["method_key"]), str(r.get("eps"))))
    _write_csv(OUT / "records.csv", rows)

    groups = {}
    for row in rows:
        group_key = (row["method"], row.get("eps"))
        groups.setdefault(group_key, []).append(row)
    summary_rows = []
    for (method, eps), selected in groups.items():
        good = [r for r in selected if r.get("status") == "complete"]

        def finite_values(key: str) -> list[float]:
            return [float(r[key]) for r in good if r.get(key) is not None and np.isfinite(float(r[key]))]

        runtimes = finite_values("seconds_end_to_end")
        errors = finite_values("observed_max_abs_error")
        rels = finite_values("observed_rel_l2_error")
        effs = finite_values("efficiency_residual")
        mqs = finite_values("m_q")
        certs = finite_values("certified_bound")
        summary_rows.append(
            {
                "method": method,
                "eps": eps,
                "n_requested": len(selected),
                "n_complete": len(good),
                "completion_rate": len(good) / len(selected),
                "median_seconds": float(np.median(runtimes)) if runtimes else None,
                "mean_seconds": float(np.mean(runtimes)) if runtimes else None,
                "median_m_q": float(np.median(mqs)) if mqs else None,
                "max_m_q": float(np.max(mqs)) if mqs else None,
                "median_certified_bound": (
                    float(np.median(certs)) if certs else (0.0 if method == "PKeX-Shapley" else None)
                ),
                "max_observed_error": float(np.max(errors)) if errors else None,
                "median_observed_error": float(np.median(errors)) if errors else None,
                "max_observed_rel_l2": float(np.max(rels)) if rels else None,
                "max_efficiency_residual": float(np.max(effs)) if effs else None,
                "all_finite": bool(all(r.get("finite", False) for r in good)) if good else None,
                "observed_target_success_rate": (
                    float(np.mean([r["target_met_observed"] for r in good if r.get("target_met_observed") is not None]))
                    if any(r.get("target_met_observed") is not None for r in good)
                    else None
                ),
            }
        )
    summary_rows.sort(key=lambda r: (r["method"], str(r["eps"])))
    _write_csv(OUT / "summary.csv", summary_rows)
    atomic_json(
        OUT / "result_manifest.json",
        {
            "reference": "QuadraSHAP CPU float64 at requested eps=1e-16",
            "certificate_interpretation": "exact-arithmetic quadrature bound; floating-point error is separate",
            "pkex_certificate_interpretation": "zero approximation error in exact arithmetic; observed error is numerical",
            "records": len(rows),
            "summary_rows": len(summary_rows),
            "machine_eps_float64": float(np.finfo(np.float64).eps),
            "machine_eps_float32": float(np.finfo(np.float32).eps),
            "files": ["prepare_meta.json", "run_meta.json", "instances.npz", "records.csv", "summary.csv"],
        },
    )
    log(f"assembled {len(rows)} records -> {OUT / 'records.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="train/cache the full-data model and fixed instances")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="imdb")
    p.add_argument("--arrow-cache", type=Path, default=None)
    p.add_argument("--n-instances", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--gamma", default="scale")
    p.add_argument("--svc-cache-mb", type=float, default=8000.0)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run", help="run timeout-isolated workers")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="imdb")
    p.add_argument("--n-instances", type=int, default=20)
    p.add_argument("--methods", nargs="+", choices=["cpu", "metal", "pkex"], default=["cpu", "metal", "pkex"])
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--node-block", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--pkex-chunk", type=int, default=1)
    p.add_argument("--parallel-pkex", type=int, default=1,
                   help="number of independent PKeX workers (CPU/Metal always remain serial)")
    p.add_argument("--pkex-repo", type=Path, default=DEFAULT_PKEX_REPO)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("worker", help=argparse.SUPPRESS)
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="imdb")
    p.add_argument("--method", required=True, choices=["cpu", "metal", "pkex"])
    p.add_argument("--instance", required=True, type=int)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--node-block", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--pkex-chunk", type=int, default=1)
    p.add_argument("--pkex-repo", type=Path, default=DEFAULT_PKEX_REPO)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("summarize", help="assemble raw worker files into CSV summaries")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="imdb")
    p.add_argument("--n-instances", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_dataset(args.dataset)
    try:
        {"prepare": prepare, "run": run, "worker": worker, "summarize": summarize}[args.command](args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
