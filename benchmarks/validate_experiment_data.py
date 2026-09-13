"""Verify downloaded bytes, prepared dimensions, endpoint coding, and source values."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import io
import json
import zipfile

import numpy as np
import pandas as pd

from experiment_data import load_bearing, load_bearing_landmark, load_motorimagery, load_survival, TrainingPreprocessor
from fetch_experiment_data import DATA, DATASETS, OPENML, sha256
from prepare_experiment_data import arff_header


def check_source_values(name, dataset):
    """Compare values and IDs directly against raw source records."""
    if name in OPENML:
        offset = 0
        for source_id in OPENML[name]:
            with (DATA / "raw" / "openml" / f"{source_id}.arff").open() as stream:
                names = arff_header(stream)
                row = next(csv.reader(stream))
            if "bcr_patient_barcode" in names:
                assert row[names.index("bcr_patient_barcode")] == dataset.samples.patient_id.iloc[0]
            indices = [i for i, field in enumerate(names) if field not in {"bcr_patient_barcode", "time", "status"}]
            expected = np.array([np.nan if row[i] == "?" else float(row[i]) for i in indices], dtype=np.float32)
            np.testing.assert_array_equal(dataset.feature_names[offset:offset + len(indices)], [names[i] for i in indices])
            np.testing.assert_array_equal(dataset.X[0, offset:offset + len(indices)], expected)
            offset += len(indices)
        return "First raw row of both parts, feature names, and available patient ID"
    if name == "tcga_lgg_methylation":
        raw = pd.read_csv(DATA / "raw" / name / "methylation.tsv.gz", sep="\t", index_col=0, nrows=128)
    else:
        with gzip.open(DATA / "raw" / name / "series_matrix.txt.gz", "rt") as stream:
            for line in stream:
                if line.startswith("!series_matrix_table_begin"):
                    break
            raw = pd.read_csv(stream, sep="\t", index_col=0, nrows=128)
    raw.index = raw.index.astype(str)
    positions = np.flatnonzero(np.isin(dataset.feature_names, raw.index))
    assert len(positions) > 0
    patient_rows = [0, len(dataset.y) // 2, len(dataset.y) - 1]
    sample_ids = dataset.samples.sample_id.iloc[patient_rows].tolist()
    expected = raw.loc[dataset.feature_names[positions], sample_ids].to_numpy(dtype=np.float32).T
    np.testing.assert_array_equal(dataset.X[np.ix_(patient_rows, positions)], expected)
    return f"{len(positions)} source features matched by ID for first/middle/last retained patient"


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert set(manifest["datasets"]) == set(DATASETS), "Prepare all datasets first"
    sources = manifest["source_files"]
    assert len(sources) == 23, "Missing source receipts"
    for entry in sources:
        path = DATA / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path) == entry["sha256"], f"Raw checksum mismatch: {path}"
    report = {"validated_utc": datetime.now(timezone.utc).isoformat(),
              "raw_files_verified": len(sources), "datasets": {}, "prepared_files": []}
    for name, metadata in manifest["datasets"].items():
        folder = DATA / metadata["folder"]
        assert json.loads((folder / "metadata.json").read_text()) == {
            key: value for key, value in metadata.items() if key != "folder"
        }
        if metadata["kind"] == "survival":
            dataset = load_survival(name)
            assert dataset.X.shape == (metadata["n_samples"], metadata["n_features"])
            assert dataset.X.dtype == np.float32
            assert dataset.y["event"].sum() == metadata["events"]
            assert len(set(dataset.feature_names)) == dataset.X.shape[1]
            missing = 0
            for start in range(0, dataset.X.shape[1], 4096):
                block = dataset.X[:, start:start + 4096]
                assert not np.isinf(block).any()
                missing += int(np.isnan(block).sum())
            assert missing == metadata["missing_values"]
            endpoints = {}
            for endpoint in metadata["endpoints"]:
                selected = load_survival(name, endpoint)
                endpoints[endpoint] = {"n": len(selected.y), "events": int(selected.y["event"].sum())}
            report["datasets"][name] = {"shape": list(dataset.X.shape), "missing_values": missing,
                                         "endpoints": endpoints, "source_comparison": check_source_values(name, dataset)}
        elif name == "motorimagery":
            splits = {}
            with zipfile.ZipFile(DATA / "raw" / name / "MotorImagery.zip") as archive:
                for split in ("train", "test"):
                    X, labels, samples = load_motorimagery(split, flatten=False)
                    assert X.shape == (metadata[f"n_{split}"], 64, 3000)
                    assert np.isfinite(X).all()
                    assert set(labels) == {"finger", "tongue"}
                    with io.TextIOWrapper(archive.open(f"MotorImagery_{split.upper()}.ts")) as stream:
                        for line in stream:
                            if line.strip().lower() == "@data":
                                break
                        fields = next(line for line in stream if line.strip()).strip().split(":")
                    assert fields[-1] == labels[0]
                    for channel in (0, 32, 63):
                        np.testing.assert_array_equal(X[0, channel], np.array(fields[channel].split(","), dtype=np.float32))
                    splits[split] = {"shape": list(X.shape), "class_counts": samples.label.value_counts().to_dict()}
            report["datasets"][name] = {"splits": splits, "source_comparison": "First trial, three channels, and label in both official splits"}
        else:
            windows = pd.read_csv(folder / "windows.tsv", sep="\t")
            assert len(windows) == metadata["n_windows"] == 9216
            assert windows.bearing_id.nunique() == 15
            with zipfile.ZipFile(DATA / "raw" / name / "XJTU-SY.zip") as archive:
                for bearing_id in windows.bearing_id.unique():
                    X, remaining, samples = load_bearing(bearing_id, flatten=False)
                    assert X.shape == (len(samples), 32768, 2)
                    np.testing.assert_array_equal(samples.window_index, np.arange(1, len(samples) + 1))
                    np.testing.assert_array_equal(remaining, np.arange(len(samples) - 1, -1, -1))
                    for row in (0, len(samples) - 1):
                        with archive.open(samples.archive_member.iloc[row]) as stream:
                            expected = pd.read_csv(stream).to_numpy(dtype=np.float32)
                        np.testing.assert_array_equal(X[row], expected)
            X, y, bearings = load_bearing_landmark()
            assert X.shape == (15, 65536) and y["event"].all()
            assert bearings.bearing_id.is_unique
            report["datasets"][name] = {"n_bearings": 15, "n_windows": len(windows),
                                         "landmark_shape": list(X.shape),
                                         "source_comparison": "First and final recording of every bearing; consecutive indices and remaining-time order"}
        print(f"Validated {name}", flush=True)
        for path in sorted(folder.iterdir()):
            if path.is_file():
                report["prepared_files"].append({"path": str(path.relative_to(DATA)),
                                                  "bytes": path.stat().st_size, "sha256": sha256(path)})
    small = load_survival("tcga_laml")
    preprocessor = TrainingPreprocessor.fit(small.X[:24])
    assert np.isfinite(preprocessor.transform(small.X[24:])).all()
    report["preprocessing_smoke_check"] = {
        "dataset": "tcga_laml", "training_rows": 24, "held_out_rows": 11,
        "retained_features": int(preprocessor.keep.sum()), "finite_held_out_output": True,
    }
    (DATA / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("All checks passed; wrote data/experiments/validation.json", flush=True)


if __name__ == "__main__":
    main()
