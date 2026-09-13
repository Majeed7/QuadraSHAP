"""Prepare aligned arrays without fitting models or learning preprocessing statistics.

Raw files remain unchanged. Missing measurements are retained for imputation using
training patients only. Outcomes and identifiers are never included in X.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from fetch_experiment_data import DATA, DATASETS, OPENML

EXPECTED = {
    # The published totals include barcode, time, and status: exclude all three.
    "tcga_hnsc": (443, 97536, 152),
    "tcga_lusc": (418, 100892, 132),
    "tcga_lgg_multiomics": (419, 90151, 77),
    "tcga_laml": (35, 90159, 14),
}
BEARING_LENGTHS = {
    "Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158,
    "Bearing1_4": 122, "Bearing1_5": 52,
    "Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533,
    "Bearing2_4": 42, "Bearing2_5": 339,
    "Bearing3_1": 2538, "Bearing3_2": 2496, "Bearing3_3": 371,
    "Bearing3_4": 1515, "Bearing3_5": 114,
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_metadata(name: str, metadata: dict) -> None:
    folder = DATA / "processed" / name
    metadata.update(dataset=name, dtype="float32", preprocessing="No fitted imputation, scaling, or feature selection")
    write_json(folder / "metadata.json", metadata)
    print(f"Prepared {name}: {metadata}", flush=True)


def require_survival(samples: pd.DataFrame, endpoint: str = "OS") -> None:
    if samples.patient_id.isna().any() or samples.patient_id.duplicated().any():
        raise ValueError("Patient identifiers must be present and unique")
    if not samples[f"{endpoint}_event"].isin([0, 1]).all():
        raise ValueError("Event indicator must be 0=censored, 1=event")
    times = samples[f"{endpoint}_time"].to_numpy(dtype=float)
    if not (np.isfinite(times).all() and (times > 0).all()):
        raise ValueError("Survival durations must be finite and strictly positive")


def arff_header(stream) -> list[str]:
    names = []
    for line in stream:
        if line.lower().startswith("@attribute"):
            match = re.fullmatch(r"@attribute\s+'((?:\\.|[^'])*)'\s+(.+)\s*", line.strip(), re.I)
            if not match:
                raise ValueError(f"Unsupported ARFF attribute: {line[:120]}")
            name, kind = match.groups()
            if kind.lower() not in {"numeric", "real", "integer", "string"}:
                raise ValueError(f"Unexpected ARFF type {kind}")
            if kind.lower() == "string" and name != "bcr_patient_barcode":
                raise ValueError(f"Unexpected string predictor: {name}")
            names.append(name)
        elif line.strip().lower() == "@data":
            return names
    raise ValueError("ARFF has no @data section")


def read_arff(path: Path, n: int) -> tuple[list[str], np.ndarray, list[str] | None]:
    with path.open() as stream:
        names = arff_header(stream)
        id_index = names.index("bcr_patient_barcode") if "bcr_patient_barcode" in names else None
        indices = [i for i in range(len(names)) if i != id_index]
        data = np.empty((n, len(indices)), dtype=np.float32)
        ids = [] if id_index is not None else None
        count = 0
        for row in csv.reader(stream):
            if not row or row[0].startswith("%"):
                continue
            if count >= n or len(row) != len(names):
                raise ValueError(f"Unexpected ARFF shape in {path} at row {count}")
            if ids is not None:
                ids.append(row[id_index])
            data[count] = [np.nan if row[i] == "?" else float(row[i]) for i in indices]
            count += 1
        if count != n:
            raise ValueError(f"Expected {n} ARFF rows, found {count}")
    return [names[i] for i in indices], data, ids


def prepare_openml(name: str, folder: Path) -> None:
    n, d, expected_events = EXPECTED[name]
    parts = [read_arff(DATA / "raw" / "openml" / f"{i}.arff", n) for i in OPENML[name]]
    ids = parts[0][2]
    if ids is None:
        raise ValueError("First OpenML part lacks patient identifiers")
    for _, _, other_ids in parts[1:]:
        if other_ids is not None and other_ids != ids:
            raise ValueError("Patient ordering differs across OpenML parts")
    feature_names = [col for cols, _, _ in parts for col in cols if col not in {"time", "status"}]
    if len(feature_names) != d or len(set(feature_names)) != d:
        raise ValueError("Feature count or uniqueness differs from the published benchmark")
    X = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype=np.float32, shape=(n, d))
    outcomes = {}
    offset = 0
    for cols, values, _ in parts:
        for label in ("time", "status"):
            if label in cols:
                value = values[:, cols.index(label)]
                if label in outcomes and not np.array_equal(outcomes[label], value):
                    raise ValueError(f"Inconsistent duplicated endpoint {label}")
                outcomes[label] = value
        take = [i for i, col in enumerate(cols) if col not in {"time", "status"}]
        X[:, offset:offset + len(take)] = values[:, take]
        offset += len(take)
    samples = pd.DataFrame({"patient_id": ids, "sample_id": ids,
                            "OS_time": outcomes["time"], "OS_event": outcomes["status"]})
    require_survival(samples)
    if samples.OS_event.sum() != expected_events:
        raise ValueError("Event count differs from the published benchmark")
    samples.OS_event = samples.OS_event.astype(int)
    samples.to_csv(folder / "samples.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_id": feature_names, "group": [x.rsplit("_", 1)[-1] for x in feature_names]}).to_csv(folder / "features.tsv", sep="\t", index=False)
    X.flush()
    save_metadata(name, {
        "kind": "survival", "n_samples": n, "n_features": d, "events": expected_events,
        "published_total_columns": d + 3,
        "time_unit": "days", "endpoints": ["OS"], "openml_ids": list(OPENML[name]),
        "part_alignment": "Provider-defined row position; compare IDs wherever supplied",
        "upstream_preprocessing": "Published Herrmann et al. data, including upstream clinical encoding/filtering",
        "missing_values": int(np.isnan(X).sum()),
    })


def prepare_methylation(folder: Path) -> None:
    raw = DATA / "raw" / "tcga_lgg_methylation"
    with gzip.open(raw / "methylation.tsv.gz", "rt") as stream:
        source_ids = stream.readline().rstrip().split("\t")[1:]
    specimens = pd.DataFrame({"sample_id": source_ids})
    specimens["patient_id"] = specimens.sample_id.str[:12]
    specimens["sample_type"] = specimens.sample_id.str[13:15]
    clinical = pd.read_excel(raw / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
    clinical = clinical.set_index("bcr_patient_barcode")
    if not clinical.index.is_unique:
        raise ValueError("Duplicated patient identifier in CDR")
    joined = specimens.join(clinical, on="patient_id", validate="many_to_one")
    reasons = np.full(len(joined), "", dtype=object)
    reasons[joined.sample_type.ne("01")] = "non-primary tumor (sample type 02 is recurrent)"
    reasons[(reasons == "") & joined.type.ne("LGG")] = "no matching LGG CDR patient"
    valid_os = joined.OS.isin([0, 1]) & np.isfinite(joined["OS.time"]) & joined["OS.time"].gt(0)
    reasons[(reasons == "") & ~valid_os] = "missing or nonpositive overall survival duration"
    keep = reasons == ""
    excluded = specimens.loc[~keep].copy()
    excluded["reason"] = reasons[~keep]
    excluded.to_csv(folder / "excluded_samples.tsv", sep="\t", index=False)
    samples = joined.loc[keep, ["patient_id", "sample_id", "OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]].copy()
    samples.rename(columns={ep: f"{ep}_event" for ep in ["OS", "DSS", "DFI", "PFI"]}, inplace=True)
    samples.rename(columns={f"{ep}.time": f"{ep}_time" for ep in ["OS", "DSS", "DFI", "PFI"]}, inplace=True)
    require_survival(samples)
    samples.to_csv(folder / "samples.tsv", sep="\t", index=False)
    baseline = ["age_at_initial_pathologic_diagnosis", "gender", "histological_type", "histological_grade"]
    joined.loc[keep, ["patient_id"] + baseline].to_csv(folder / "clinical_covariates.tsv", sep="\t", index=False)
    selected_columns = np.flatnonzero(keep)
    d = 485577
    X = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype=np.float32, shape=(len(samples), d))
    features, missing, all_missing, offset = [], 0, 0, 0
    observed = np.zeros(d, dtype=bool)
    measurement_types = {sample: np.float32 for sample in source_ids}
    for chunk in pd.read_csv(raw / "methylation.tsv.gz", sep="\t", index_col=0, chunksize=4096, dtype=measurement_types):
        values = chunk.to_numpy()[:, selected_columns]
        if offset + len(chunk) > d:
            raise ValueError("Methylation feature count exceeds source documentation")
        finite = values[np.isfinite(values)]
        if finite.size and (finite.min() < 0 or finite.max() > 1):
            raise ValueError("Methylation beta value is outside [0,1]")
        if np.isinf(values).any():
            raise ValueError("Infinite methylation measurement")
        missing += int(np.isnan(values).sum())
        all_missing += int(np.isnan(values).all(axis=1).sum())
        observed[offset:offset + len(chunk)] = ~np.isnan(values).all(axis=1)
        X[:, offset:offset + len(chunk)] = values.T
        features.extend(chunk.index.astype(str))
        offset += len(chunk)
        if offset % 65536 == 0:
            print(f"Converted {offset:,}/{d:,} methylation features", flush=True)
    if offset != d or len(set(features)) != d:
        raise ValueError(f"Unexpected methylation features: {offset}")
    X.flush()
    feature_table = pd.DataFrame({"feature_id": features, "group": "methylation"})
    excluded_features = feature_table.loc[~observed, ["feature_id"]].copy()
    excluded_features["reason"] = "No observed measurement in any eligible patient"
    excluded_features.to_csv(folder / "excluded_features.tsv", sep="\t", index=False)
    if not observed.all():
        compact_path = folder / "X.observed.npy"
        compact = np.lib.format.open_memmap(compact_path, mode="w+", dtype=np.float32, shape=(len(samples), int(observed.sum())))
        indices = np.flatnonzero(observed)
        for start in range(0, len(indices), 4096):
            block = indices[start:start + 4096]
            compact[:, start:start + len(block)] = X[:, block]
        compact.flush()
        del compact, X
        compact_path.replace(folder / "X.npy")
    feature_table.loc[observed].to_csv(folder / "features.tsv", sep="\t", index=False)
    save_metadata("tcga_lgg_methylation", {
        "kind": "survival", "n_source_profiles": len(source_ids), "n_samples": len(samples),
        "n_source_features": d, "n_features": int(observed.sum()), "events": int(samples.OS_event.sum()), "time_unit": "days",
        "endpoints": ["OS", "DSS", "DFI", "PFI"], "default_endpoint": "OS",
        "missing_values": missing - all_missing * len(samples), "excluded_all_missing_features": all_missing,
        "source_missing_values": missing,
        "excluded_profiles": len(excluded), "measurement": "methylation beta value, not Cox coefficient beta",
        "notes": "Primary tumors only; alternate endpoints need their own valid-duration mask",
    })


def prepare_geo(folder: Path) -> None:
    raw = DATA / "raw" / "gse24080"
    metadata = {}
    with gzip.open(raw / "series_matrix.txt.gz", "rt") as stream:
        for line in stream:
            if line.startswith(("!Sample_title\t", "!Sample_geo_accession\t")):
                fields = next(csv.reader([line], delimiter="\t"))
                metadata[fields[0]] = fields[1:]
            if line.startswith("!series_matrix_table_begin"):
                break
        expression = pd.read_csv(stream, sep="\t", index_col=0, comment="!")
    ids = metadata["!Sample_geo_accession"]
    if expression.columns.tolist() != ids or expression.shape != (54675, 559):
        raise ValueError("Unexpected GEO matrix shape or sample order")
    clinical = pd.read_excel(io.BytesIO(gzip.decompress((raw / "clinical.xls.gz").read_bytes())), sheet_name="ClinInfo")
    clinical["sample_title"] = clinical.CELfilename.str.replace(r"\.CEL$", "", regex=True)
    clinical = clinical.set_index("sample_title")
    if not clinical.index.is_unique:
        raise ValueError("Duplicated array filename in GEO clinical table")
    specimens = pd.DataFrame({"sample_id": ids, "sample_title": metadata["!Sample_title"]})
    joined = specimens.join(clinical, on="sample_title", validate="one_to_one")
    if joined.PATID.isna().any():
        raise ValueError("A GEO expression sample has no matching clinical row")
    os_time = "OS TIME JUN2008"
    os_event = "OS CENSOR (1=death) JUN2008"
    valid = joined[os_event].isin([0, 1]) & joined[os_time].gt(0) & np.isfinite(joined[os_time])
    reasons = np.full(len(joined), "", dtype=object)
    reasons[joined.MAQC_Distribution_Status.eq("MAQC_Q")] = "MAQC_Q: outlier flagged by dataset provider"
    reasons[(reasons == "") & ~valid] = "missing or nonpositive overall survival duration"
    keep = reasons == ""
    excluded = specimens.loc[~keep].copy()
    excluded["reason"] = reasons[~keep]
    excluded.to_csv(folder / "excluded_samples.tsv", sep="\t", index=False)
    rows = joined.loc[keep]
    samples = pd.DataFrame({
        "patient_id": rows.PATID.astype(int).astype(str), "sample_id": rows.sample_id,
        "OS_time": rows[os_time], "OS_event": rows[os_event].astype(int),
        "EFS_time": rows["EFS TIME JUN2008"], "EFS_event": rows["EFS CENSOR (1=event) JUN2008"].astype(int),
        "split": rows.MAQC_Distribution_Status.str.lower(), "treatment_protocol": rows.PROT,
    })
    require_survival(samples)
    require_survival(samples, "EFS")
    samples.to_csv(folder / "samples.tsv", sep="\t", index=False)
    covars = ["AGE", "SEX", "RACE", "ISOTYPE", "B2M", "CRP", "CREAT", "LDH", "ALB", "HGB", "ASPC", "BMPC", "MRI", "Cyto Abn"]
    clinical_out = rows[covars].copy()
    clinical_out.insert(0, "patient_id", samples.patient_id)
    clinical_out.to_csv(folder / "clinical_covariates.tsv", sep="\t", index=False)
    X = expression.to_numpy(dtype=np.float32).T[keep]
    np.save(folder / "X.npy", X, allow_pickle=False)
    pd.DataFrame({"feature_id": expression.index.astype(str), "group": "expression_probe"}).to_csv(folder / "features.tsv", sep="\t", index=False)
    save_metadata("gse24080", {
        "kind": "survival", "n_source_profiles": 559, "n_clinical_rows": len(clinical),
        "n_samples": len(samples), "n_features": 54675, "events": int(samples.OS_event.sum()),
        "efs_events": int(samples.EFS_event.sum()), "time_unit": "months", "endpoints": ["OS", "EFS"],
        "excluded_profiles": len(excluded), "missing_values": int(np.isnan(X).sum()),
        "split_counts": samples.split.value_counts().to_dict(),
        "measurement": "Deposited MAS5 log2 intensity; probe sets retained without gene aggregation",
        "notes": "June 2008 time/event columns, not 24-month classification labels",
    })


def prepare_motor(folder: Path) -> None:
    raw = DATA / "raw" / "motorimagery" / "MotorImagery.zip"
    with zipfile.ZipFile(raw) as archive:
        for split, count in [("TRAIN", 278), ("TEST", 100)]:
            name = f"MotorImagery_{split}.ts"
            X = np.lib.format.open_memmap(folder / f"X_{split.lower()}.npy", mode="w+", dtype=np.float32, shape=(count, 64, 3000))
            labels, started = [], False
            with io.TextIOWrapper(archive.open(name)) as stream:
                for line in stream:
                    if not started:
                        started = line.strip().lower() == "@data"
                        continue
                    if not line.strip():
                        continue
                    fields = line.strip().split(":")
                    if len(fields) != 65 or len(labels) >= count:
                        raise ValueError("Unexpected MotorImagery record shape")
                    for channel, field in enumerate(fields[:-1]):
                        signal = np.fromstring(field, sep=",", dtype=np.float32)
                        if signal.shape != (3000,) or not np.isfinite(signal).all():
                            raise ValueError("Invalid MotorImagery signal")
                        X[len(labels), channel] = signal
                    if fields[-1] not in {"finger", "tongue"}:
                        raise ValueError("Unexpected MotorImagery class")
                    labels.append(fields[-1])
            if len(labels) != count:
                raise ValueError("Incorrect MotorImagery split size")
            X.flush()
            pd.DataFrame({"sample_id": [f"{split.lower()}_{i:04d}" for i in range(count)],
                          "label": labels, "subject_id": "single_subject", "session": split.lower()}).to_csv(folder / f"samples_{split.lower()}.tsv", sep="\t", index=False)
        (folder / "source_description.txt").write_bytes(archive.read("MotorImagery.txt"))
    save_metadata("motorimagery", {
        "kind": "classification_time_series", "n_train": 278, "n_test": 100,
        "channels": 64, "time_points": 3000, "n_features_flat": 192000,
        "axis_order": ["trial", "channel", "time"], "sampling_hz": 1000,
        "subjects": 1, "split": "Official split: two recording sessions about one week apart",
    })


def bearing_members(archive: zipfile.ZipFile) -> dict[str, list[tuple[int, str]]]:
    groups = {}
    for member in archive.namelist():
        match = re.search(r"(Bearing[123]_[1-5])/(\d+)\.csv$", member)
        if match:
            bearing, number = match.groups()
            groups.setdefault(bearing, []).append((int(number), member))
    for items in groups.values():
        items.sort()
    return groups


def prepare_xjtu(folder: Path) -> None:
    raw = DATA / "raw" / "xjtu_sy" / "XJTU-SY.zip"
    manifest = []
    with zipfile.ZipFile(raw) as archive:
        groups = bearing_members(archive)
        if set(groups) != set(BEARING_LENGTHS):
            raise ValueError(f"Missing or unexpected bearings: {sorted(groups)}")
        for bearing, items in sorted(groups.items()):
            n = BEARING_LENGTHS[bearing]
            if [i for i, _ in items] != list(range(1, n + 1)):
                raise ValueError(f"Incomplete or duplicated windows for {bearing}")
            condition = int(bearing[7])
            X = np.lib.format.open_memmap(folder / f"{bearing}.npy", mode="w+", dtype=np.float32, shape=(n, 32768, 2))
            for row, (number, member) in enumerate(items):
                with archive.open(member) as stream:
                    values = pd.read_csv(stream).to_numpy(dtype=np.float32)
                if values.shape != (32768, 2) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid vibration recording: {member} {values.shape}")
                X[row] = values
                manifest.append({"bearing_id": bearing, "condition": condition, "window_index": number,
                                 "array_row": row, "remaining_minutes_to_last_record": n - number,
                                 "archive_member": member})
            X.flush()
            print(f"Converted {bearing}: {n} windows", flush=True)
    pd.DataFrame(manifest).to_csv(folder / "windows.tsv", sep="\t", index=False)
    save_metadata("xjtu_sy", {
        "kind": "run_to_failure_time_series", "n_bearings": 15, "n_windows": len(manifest),
        "channels": 2, "time_points": 32768, "n_features_flat": 65536,
        "axis_order": ["window", "time", "channel"], "sampling_hz": 25600,
        "measurement_interval_minutes": 1, "window_seconds": 1.28,
        "target": "Minutes to final recorded window: last window has zero remaining time",
        "notes": "No imposed censoring or official split. Split by bearing; windows are repeated measurements, not independent failures.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--force", action="store_true", help="Rebuild generated arrays from unchanged raw files")
    args = parser.parse_args()
    for name in args.datasets:
        folder = DATA / "processed" / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "metadata.json").exists() and not args.force:
            print(f"Already prepared {name}; retained existing output", flush=True)
            continue
        # A failed rebuild must not leave a marker claiming it completed.
        (folder / "metadata.json").unlink(missing_ok=True)
        if name in OPENML:
            prepare_openml(name, folder)
        else:
            {"tcga_lgg_methylation": prepare_methylation, "gse24080": prepare_geo,
             "motorimagery": prepare_motor, "xjtu_sy": prepare_xjtu}[name](folder)
    entries = {}
    for path in sorted((DATA / "processed").glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        metadata["folder"] = str(path.parent.relative_to(DATA))
        entries[path.parent.name] = metadata
    sources = [json.loads(path.read_text()) for path in sorted((DATA / "raw").glob("*/*.source.json"))]
    write_json(DATA / "manifest.json", {"datasets": entries, "source_files": sources})


if __name__ == "__main__":
    main()
