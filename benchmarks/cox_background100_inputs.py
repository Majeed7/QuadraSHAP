"""Prepare and validate immutable B=100 inputs for the existing Cox study.

The fitted model and five held-out patients are reused unchanged. Background
rows extend the saved seed-42 ordering, so the first 30 rows reproduce B=30.
This module performs no model fitting, quadrature, or JAX initialization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/cox_background100"
PREVIOUS = ROOT / "benchmarks/results/cox_background30_gpu"
FULL = ROOT / "benchmarks/results/cox_background_sensitivity"
FITTED = ROOT / "benchmarks/results/cox_gpu_tolerance"
DATASETS = ("gse24080", "tcga_lgg_methylation")
EXPECTED = {"gse24080": (339, 54675), "tcga_lgg_methylation": (383, 396065)}
KEYS = ("beta", "background", "X", "patient_ids", "test_rows",
        "background_indices", "training_rows")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _source_arrays(name):
    _require(name in DATASETS, f"Unknown dataset: {name}")
    previous_design_path = PREVIOUS / "design.json"
    previous_design = read_json(previous_design_path)[name]
    previous_path = PREVIOUS / name / "inputs.npz"
    _require(file_hash(previous_path) == previous_design["prepared_inputs_sha256"],
             f"Saved B=30 inputs changed: {name}")
    for path, expected in previous_design["source_sha256"].items():
        _require(file_hash(path) == expected, f"Saved full-background cache changed: {path}")

    full_folder = FULL / name
    full_design_path = full_folder / "design.json"
    full_design = read_json(full_design_path)
    for filename, expected in full_design["source_sha256"].items():
        _require(file_hash(FITTED / name / filename) == expected,
                 f"Original fitted-model input changed: {name}/{filename}")
    model_path = FITTED / name / "model_and_preprocessing.npz"
    _require(file_hash(model_path) == previous_design["model_sha256"],
             f"Model hash disagrees with B=30: {name}")
    previous = _load_npz(previous_path)
    full = _load_npz(full_folder / "inputs.npz")
    model = _load_npz(model_path)
    training = np.load(full_folder / "training_background.npy", mmap_mode="r")
    n_train, d = EXPECTED[name]
    _require(training.shape == (n_train, d), f"Unexpected training cache shape: {name}")
    _require((full_design["n_train"], full_design["d"]) == (n_train, d),
             f"Unexpected full-background design: {name}")
    _require((previous_design["n_train"], previous_design["d"],
              previous_design["background_size"], previous_design["selection_seed"])
             == (n_train, d, 30, 42), f"Unexpected B=30 design: {name}")
    for left, right, label in ((full["beta"], model["coef"], "coefficients"),
                               (full["training_rows"], model["train_rows"], "training split"),
                               (full["test_rows"], model["test_rows"], "test split"),
                               (previous["beta"], full["beta"], "B=30 coefficients")):
        _require(np.array_equal(left, right), f"Mismatched {label}: {name}")
    train_rows, test_rows = full["training_rows"], full["test_rows"]
    _require(len(np.unique(train_rows)) == n_train and len(np.unique(test_rows)) == len(test_rows),
             f"Duplicate split rows: {name}")
    _require(not np.intersect1d(train_rows, test_rows).size, f"Train/test overlap: {name}")
    order = full["background_order"]
    _require(np.array_equal(order, np.random.default_rng(42).permutation(n_train)),
             f"Background order is not the saved seed-42 permutation: {name}")
    indices = order[:100]
    _require(len(np.unique(indices)) == 100, f"Duplicate background row: {name}")
    _require(np.array_equal(indices[:30], previous["background_indices"]),
             f"B=30 background indices do not form the B=100 prefix: {name}")
    _require(np.array_equal(previous["test_rows"], test_rows[:5]),
             f"Held-out patients changed: {name}")
    _require(previous["patient_ids"].tolist() == previous_design["patient_ids"],
             f"B=30 patient IDs differ from the design: {name}")
    background = np.asarray(training[indices])
    _require(np.array_equal(background[:30], previous["background"]),
             f"B=30 background values do not form the B=100 prefix: {name}")
    _require(np.array_equal(train_rows[indices[:30]], previous["training_rows"]),
             f"B=30 training rows differ from the B=100 prefix: {name}")
    arrays = {key: previous[key] for key in ("beta", "X", "patient_ids", "test_rows")}
    arrays.update(background=background, background_indices=indices, training_rows=train_rows[indices])
    _validate_arrays(name, arrays)

    selection_path = full_folder / "background_selection.csv"
    selection = pd.read_csv(selection_path, dtype={"patient_id": str, "sample_id": str})
    _require(len(selection) == n_train and selection.patient_id.is_unique,
             f"Invalid training-patient selection metadata: {name}")
    chosen = selection.iloc[indices].copy().reset_index(drop=True)
    _require(np.array_equal(chosen.background_selection_rank.to_numpy(), np.arange(100)),
             f"Background ranks do not match the saved ordering: {name}")
    _require(not set(chosen.patient_id) & set(arrays["patient_ids"]),
             f"Held-out patient IDs occur in backgrounds: {name}")
    chosen.insert(0, "background_position", np.arange(100))
    chosen.insert(1, "training_cache_index", indices)
    chosen.insert(2, "dataset_row", arrays["training_rows"])
    chosen["in_nested_background_30"] = np.arange(100) < 30
    patients = pd.DataFrame({"patient_index": np.arange(5), "patient_id": arrays["patient_ids"],
                             "dataset_row": arrays["test_rows"]})
    source_paths = (previous_design_path, previous_path, full_design_path,
                    full_folder / "inputs.npz", full_folder / "training_background.npy",
                    selection_path, model_path, FITTED / name / "explanation_inputs.npz")
    design = {
        "dataset": name, "n_train": n_train, "n_test": len(test_rows), "d": d,
        "background_size": 100, "selection_seed": 42,
        "selection": "First 100 entries of the saved seed-42 permutation of training rows, without replacement",
        "nesting": "First 30 background rows, indices and dataset rows exactly equal the saved B=30 study",
        "background_weights": "Uniform, 1/100 per background patient",
        "patient_ids": arrays["patient_ids"].tolist(), "test_rows": arrays["test_rows"].tolist(),
        "patient_selection": "Same five held-out patients and preprocessed inputs as the saved B=30 study",
        "test_data_used_for_background": False, "model_refitted": False,
        "model_sha256": previous_design["model_sha256"],
        "source_sha256": {str(path): file_hash(path) for path in source_paths},
        "preparation_script_sha256": file_hash(__file__),
    }
    return arrays, chosen, patients, design


def _validate_arrays(name, arrays):
    n_train, d = EXPECTED[name]
    _require(set(arrays) == set(KEYS), f"Unexpected input keys: {name}")
    for key, shape in (("beta", (d,)), ("background", (100, d)), ("X", (5, d)),
                       ("patient_ids", (5,)), ("test_rows", (5,)),
                       ("background_indices", (100,)), ("training_rows", (100,))):
        _require(arrays[key].shape == shape, f"Invalid {key} shape: {name}")
    for key in ("beta", "background", "X"):
        _require(np.isfinite(arrays[key]).all(), f"Nonfinite {key}: {name}")
    for key in ("patient_ids", "test_rows", "background_indices", "training_rows"):
        _require(len(np.unique(arrays[key])) == len(arrays[key]), f"Duplicate {key}: {name}")
    indices = arrays["background_indices"]
    _require(np.issubdtype(indices.dtype, np.integer) and (indices >= 0).all()
             and (indices < n_train).all(), f"Invalid training-cache indices: {name}")
    _require(not np.intersect1d(arrays["test_rows"], arrays["training_rows"]).size,
             f"Held-out rows occur in background: {name}")


def _save_once(path, writer):
    """Install a complete new file atomically, without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer(handle)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_text_once(path, content):
    encoded = content.encode("utf-8")
    if path.exists():
        _require(path.read_bytes() == encoded, f"Refusing conflicting existing artifact: {path}")
    else:
        _save_once(path, lambda stream: stream.write(encoded))


def prepare():
    designs = {}
    for name in DATASETS:
        arrays, selected, patients, design = _source_arrays(name)
        folder = OUT / name
        input_path = folder / "inputs.npz"
        if input_path.exists():
            existing = _load_npz(input_path)
            _require(set(existing) == set(arrays)
                     and all(np.array_equal(existing[k], arrays[k]) for k in arrays),
                     f"Refusing conflicting prepared inputs: {name}")
        else:
            _save_once(input_path, lambda stream: np.savez_compressed(stream, **arrays))
        for filename, table in (("background_selection.csv", selected), ("patient_selection.csv", patients)):
            _save_text_once(folder / filename, table.to_csv(index=False))
        design["prepared_inputs_sha256"] = file_hash(input_path)
        design["selection_sha256"] = {f: file_hash(folder / f)
                                      for f in ("background_selection.csv", "patient_selection.csv")}
        _save_text_once(folder / "design.json", json.dumps(design, indent=2) + "\n")
        designs[name] = design
        print(f"{name}: B=100, five unchanged patients, d={design['d']:,}; B=30 prefix verified", flush=True)
    _save_text_once(OUT / "design.json", json.dumps(designs, indent=2) + "\n")
    return designs


def checked_inputs(name, *, verify_sources=True):
    """Load validated arrays and their design; no reference or JAX computation."""
    _require(name in DATASETS, f"Unknown dataset: {name}")
    folder = OUT / name
    design = read_json(OUT / "design.json")[name]
    _require(read_json(folder / "design.json") == design, f"Root/dataset design mismatch: {name}")
    _require(file_hash(folder / "inputs.npz") == design["prepared_inputs_sha256"],
             f"Prepared B=100 input hash mismatch: {name}")
    for filename, expected in design["selection_sha256"].items():
        _require(file_hash(folder / filename) == expected, f"Selection metadata changed: {name}/{filename}")
    if verify_sources:
        for path, expected in design["source_sha256"].items():
            _require(file_hash(path) == expected, f"Prepared-input source changed: {path}")
    arrays = _load_npz(folder / "inputs.npz")
    _validate_arrays(name, arrays)
    _require(arrays["patient_ids"].tolist() == design["patient_ids"]
             and arrays["test_rows"].tolist() == design["test_rows"], f"Prepared patient selection changed: {name}")
    return {**arrays, "design": design}


if __name__ == "__main__":
    prepare()
    for dataset in DATASETS:
        checked_inputs(dataset)
    print("Both immutable B=100 input sets and provenance checks passed.", flush=True)
