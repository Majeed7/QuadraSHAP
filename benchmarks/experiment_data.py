"""Load prepared research datasets; fit preprocessing on training observations only."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "experiments" / "processed"


@dataclass
class SurvivalData:
    X: np.ndarray
    y: np.ndarray
    samples: pd.DataFrame
    feature_names: np.ndarray
    metadata: dict


def survival_target(time, event) -> np.ndarray:
    """Return scikit-survival's structured target with True meaning an event."""
    time, event = np.asarray(time, dtype=float), np.asarray(event)
    if time.ndim != 1 or event.shape != time.shape:
        raise ValueError("Time and event must be matching vectors")
    if not np.isfinite(time).all() or (time <= 0).any():
        raise ValueError("Survival times must be finite and strictly positive")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Event must be 0=censored or 1=event")
    y = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    y["event"], y["time"] = event.astype(bool), time
    return y


def load_survival(name: str, endpoint: str = "OS") -> SurvivalData:
    folder = DATA / name
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["kind"] != "survival" or endpoint not in metadata["endpoints"]:
        raise ValueError(f"Unsupported survival dataset/endpoint: {name}/{endpoint}")
    X = np.load(folder / "X.npy", mmap_mode="r", allow_pickle=False)
    samples = pd.read_csv(folder / "samples.tsv", sep="\t", dtype={"patient_id": str})
    features = pd.read_csv(folder / "features.tsv", sep="\t").feature_id.to_numpy()
    if X.shape != (len(samples), len(features)) or samples.patient_id.duplicated().any():
        raise ValueError("Misaligned matrix or duplicated patient IDs")
    time, event = samples[f"{endpoint}_time"], samples[f"{endpoint}_event"]
    valid = (np.isfinite(time) & time.gt(0) & event.isin([0, 1])).to_numpy()
    if not valid.any():
        raise ValueError(f"No valid outcomes for {endpoint}")
    if not valid.all():
        X = X[valid]
        samples = samples.loc[valid].reset_index(drop=True)
    y = survival_target(time[valid], event[valid])
    return SurvivalData(X, y, samples, features, {
        **metadata, "selected_endpoint": endpoint,
        "prepared_n_samples": metadata["n_samples"], "n_samples": len(y),
        "events": int(y["event"].sum()),
    })


def load_motorimagery(split: str = "train", flatten: bool = True):
    """Return (X, class labels, trial metadata) using the official session split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    folder = DATA / "motorimagery"
    X = np.load(folder / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
    samples = pd.read_csv(folder / f"samples_{split}.tsv", sep="\t")
    if len(X) != len(samples):
        raise ValueError("Misaligned MotorImagery labels")
    if flatten:
        X = X.reshape(len(X), -1)  # feature = channel * 3000 + time index
    return X, samples.label.to_numpy(), samples


def load_bearing(bearing_id: str, flatten: bool = True):
    """Return vibration windows, minutes to last record, and bearing/window IDs."""
    folder = DATA / "xjtu_sy"
    windows = pd.read_csv(folder / "windows.tsv", sep="\t")
    samples = windows.loc[windows.bearing_id.eq(bearing_id)].reset_index(drop=True)
    if samples.empty:
        raise ValueError(f"Unknown bearing: {bearing_id}")
    X = np.load(folder / f"{bearing_id}.npy", mmap_mode="r", allow_pickle=False)
    if len(X) != len(samples):
        raise ValueError("Misaligned bearing metadata")
    if flatten:
        X = X.reshape(len(X), -1)  # feature = time index * 2 + channel
    return X, samples.remaining_minutes_to_last_record.to_numpy(), samples


def load_bearing_landmark(window_index: int = 1):
    """Use one observed window per bearing for a small survival experiment.

    Every bearing must still have positive time to its last record. All events
    are observed; no artificial censoring is introduced. Split by bearing ID.
    The label uses the final recording as the operational failure endpoint.
    """
    if not isinstance(window_index, int) or window_index < 1:
        raise ValueError("window_index is a positive one-based recording index")
    folder = DATA / "xjtu_sy"
    windows = pd.read_csv(folder / "windows.tsv", sep="\t")
    selected = windows.loc[windows.window_index.eq(window_index)].copy()
    if len(selected) != 15 or selected.remaining_minutes_to_last_record.le(0).any():
        raise ValueError("Choose an earlier window with positive follow-up for all 15 bearings")
    X = np.stack([
        np.load(folder / f"{row.bearing_id}.npy", mmap_mode="r", allow_pickle=False)[row.array_row].reshape(-1)
        for row in selected.itertuples()
    ])
    y = survival_target(selected.remaining_minutes_to_last_record, np.ones(len(selected)))
    return X, y, selected.reset_index(drop=True)


@dataclass
class TrainingPreprocessor:
    """Median imputation and standardization learned from training rows only.

    All-missing or constant training features are removed. Features with tiny but
    nonzero variation are retained. Processing uses blocks to bound temporary RAM.
    """
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, block_size: int = 4096):
        if X.ndim != 2 or not len(X):
            raise ValueError("Expected a nonempty training matrix")
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        d = X.shape[1]
        median, mean, scale = (np.zeros(d) for _ in range(3))
        keep = np.zeros(d, dtype=bool)
        for start in range(0, d, block_size):
            end = min(d, start + block_size)
            block = np.array(X[:, start:end], dtype=np.float64, copy=True)
            if np.isinf(block).any():
                raise ValueError("Infinite training value")
            has_data = ~np.isnan(block).all(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(block, axis=0)
            med[~has_data] = 0
            block = np.where(np.isnan(block), med, block)
            median[start:end] = med
            mean[start:end] = block.mean(axis=0)
            scale[start:end] = block.std(axis=0)
            keep[start:end] = has_data & (scale[start:end] > 0)
        if not keep.any():
            raise ValueError("No nonconstant training features")
        return cls(median, mean, scale, keep)

    def transform(self, X: np.ndarray, block_size: int = 4096) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != len(self.keep):
            raise ValueError("Feature dimension differs from fitted training data")
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        selected = np.flatnonzero(self.keep)
        out = np.empty((len(X), len(selected)), dtype=np.float32)
        for start in range(0, len(selected), block_size):
            indices = selected[start:start + block_size]
            block = np.array(X[:, indices], dtype=np.float64, copy=True)
            if np.isinf(block).any():
                raise ValueError("Infinite measurement")
            block = np.where(np.isnan(block), self.median[indices], block)
            out[:, start:start + len(indices)] = (block - self.mean[indices]) / self.scale[indices]
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite output after scaling")
        return out
