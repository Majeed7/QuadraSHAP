"""Fetch the public high-dimensional datasets documented in data/experiments/README.md.

Source downloads are resumable and have SHA-256 receipts. No model is trained and
no downloaded code is executed. Run preparation separately after fetching.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "experiments"
OPENML = {
    "tcga_hnsc": (42285, 42286),
    "tcga_lusc": (42299, 42300),
    "tcga_lgg_multiomics": (42293, 42294),
    "tcga_laml": (42291, 42292),
}
FILES = {
    "tcga_lgg_methylation": [
        ("methylation.json", "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LGG.sampleMap%2FHumanMethylation450.json"),
        ("methylation.tsv.gz", "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LGG.sampleMap%2FHumanMethylation450.gz"),
        ("TCGA-CDR-SupplementalTableS1.xlsx", "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"),
    ],
    "gse24080": [
        ("series_matrix.txt.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE24nnn/GSE24080/matrix/GSE24080_series_matrix.txt.gz"),
        ("clinical.xls.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE24nnn/GSE24080/suppl/GSE24080_MM_UAMS565_ClinInfo_27Jun2008_LS_clean.xls.gz"),
    ],
    "motorimagery": [
        ("MotorImagery.zip", "https://www.timeseriesclassification.com/aeon-toolkit/MotorImagery.zip"),
    ],
    "xjtu_sy": [
        ("XJTU-SY.zip", "https://kr0k0tsch.de/rul-datasets/XJTU-SY.zip"),
    ],
}
DATASETS = tuple(OPENML) + tuple(FILES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_name(path.name + ".source.json")
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["bytes"] != path.stat().st_size or receipt["sha256"] != sha256(path):
            raise ValueError(f"Checksum mismatch: {path}; inspect before re-downloading")
        print(f"Verified cached {path.relative_to(DATA)}", flush=True)
        return receipt
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        print(f"Downloading {path.relative_to(DATA)}", flush=True)
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "3", "--connect-timeout", "20", "--max-time", "7200",
            "--continue-at", "-", url, "--output", str(partial),
        ], check=True)
        if partial.stat().st_size == 0:
            raise ValueError(f"Empty response: {url}")
        partial.replace(path)
    receipt = {
        "url": url,
        "path": str(path.relative_to(DATA)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "checksum_kind": "locally computed; not an upstream authenticity signature",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Ready {path.relative_to(DATA)} ({receipt['bytes']:,} bytes)", flush=True)
    return receipt


def fetch(name: str) -> None:
    if name in OPENML:
        for dataset_id in OPENML[name]:
            metadata = DATA / "raw" / "openml" / f"{dataset_id}.json"
            download(f"https://www.openml.org/api/v1/json/data/{dataset_id}", metadata)
            desc = json.loads(metadata.read_text())["data_set_description"]
            # These cohorts advertise Parquet URLs that currently return 404.
            # Keep the original ARFF representation used by the study instead.
            download(desc["url"], metadata.with_suffix(".arff"))
    else:
        for filename, url in FILES[name]:
            download(url, DATA / "raw" / name / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, name): name for name in args.datasets}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(futures[future])
                print(f"FAILED {futures[future]}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Incomplete downloads: {', '.join(failures)}")


if __name__ == "__main__":
    main()
