"""Text-classification corpora for the attribution benchmark, with an on-disk cache.

Four datasets that span the document lengths that matter for the comparison -- the number of
*distinct words in a document* is the dimension ``d`` of the product game, and it is what decides
both the exactness threshold of QuadraSHAP and how hard the sampling estimators have to work:

    sst2     Stanford Sentiment Treebank, binary, sentence level   (d ~ 10)
    mr       Rotten Tomatoes polarity (Pang & Lee), sentence level  (d ~ 15)
    agnews   AG News, World vs Sci/Tech, headline + lead            (d ~ 30)
    imdb     IMDB movie reviews, full documents                     (d ~ 150)

``load_dataset`` returns ``(texts, labels, info)`` from ``experiments/data/<name>.csv.gz``,
downloading and subsampling once if the cache is missing.  The cache is what the experiment reads,
so a machine without network access can run everything as long as the cache has been built (see
``python text_datasets.py --build`` on a machine that does have it).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
TIMEOUT = 60

SOURCES = {
    "sst2": ["https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"],
    "mr": ["https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.pos",
           "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.neg"],
    "agnews": ["https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"],
    "imdb": ["https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"],
}
# (task, positive class name, negative class name, documents kept in the cache, word-count window)
SPEC = {
    "sst2": ("sentiment", "positive", "negative", 8000, (3, 60)),
    "mr": ("sentiment", "fresh", "rotten", 8000, (3, 60)),
    "agnews": ("topic", "Sci/Tech", "World", 8000, (10, 120)),
    "imdb": ("sentiment", "positive", "negative", 6000, (60, 400)),
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.read()


def _raw(name: str) -> Tuple[List[str], List[int]]:
    """Download and parse one corpus into (texts, labels in {0, 1})."""
    if name == "sst2":
        texts, labels = [], []
        for line in _get(SOURCES[name][0]).decode("utf-8", "replace").splitlines():
            if "\t" not in line:
                continue
            t, lab = line.rsplit("\t", 1)
            if lab.strip() in ("0", "1"):
                texts.append(t.strip())
                labels.append(int(lab))
        return texts, labels
    if name == "mr":
        pos = _get(SOURCES[name][0]).decode("latin-1").splitlines()
        neg = _get(SOURCES[name][1]).decode("latin-1").splitlines()
        return [t.strip() for t in pos + neg], [1] * len(pos) + [0] * len(neg)
    if name == "agnews":
        texts, labels = [], []
        blob = _get(SOURCES[name][0]).decode("utf-8", "replace")
        for row in csv.reader(io.StringIO(blob)):
            if len(row) != 3:
                continue
            cls = int(row[0])                      # 1 World, 2 Sports, 3 Business, 4 Sci/Tech
            if cls in (1, 4):
                texts.append((row[1] + " " + row[2]).replace("\\", " ").strip())
                labels.append(1 if cls == 4 else 0)
        return texts, labels
    if name == "imdb":
        texts, labels = [], []
        blob = _get(SOURCES[name][0]).decode("utf-8", "replace")
        for row in csv.reader(io.StringIO(blob)):
            if len(row) != 2 or row[1] not in ("positive", "negative"):
                continue
            texts.append(row[0].replace("<br />", " ").strip())
            labels.append(1 if row[1] == "positive" else 0)
        return texts, labels
    raise ValueError(f"unknown dataset {name!r}; known: {sorted(SOURCES)}")


def build_cache(name: str, seed: int = 0) -> Path:
    """Download, filter by length, balance, subsample and write ``data/<name>.csv.gz``."""
    _, _, _, n_keep, (lo, hi) = SPEC[name]
    texts, labels = _raw(name)
    keep = [i for i, t in enumerate(texts) if lo <= len(t.split()) <= hi]
    rng = np.random.default_rng(seed)
    idx_by_class = {c: [i for i in keep if labels[i] == c] for c in (0, 1)}
    per_class = min(n_keep // 2, *(len(v) for v in idx_by_class.values()))
    chosen = np.concatenate([rng.choice(idx_by_class[c], per_class, replace=False) for c in (0, 1)])
    rng.shuffle(chosen)
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "text"])
        for i in chosen:
            w.writerow([labels[i], " ".join(texts[i].split())])
    return path


def load_dataset(name: str, n_docs: int | None = None, seed: int = 0) -> Tuple[List[str], np.ndarray, Dict]:
    path = DATA / f"{name}.csv.gz"
    if not path.exists():
        try:
            build_cache(name, seed=seed)
        except Exception as exc:
            raise RuntimeError(
                f"no cache at {path} and the download failed ({type(exc).__name__}: {exc}). "
                f"Run `python text_datasets.py --build {name}` on a machine with network access; "
                f"the cache is a few MB and is all the experiment needs.") from exc
    texts, labels = [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    labels = np.asarray(labels)
    if n_docs is not None and n_docs < len(texts):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(texts), n_docs, replace=False)
        texts = [texts[i] for i in sel]
        labels = labels[sel]
    task, pos, neg, _, window = SPEC[name]
    info = {"name": name, "task": task, "classes": (neg, pos), "n_docs": len(texts),
            "word_window": window, "mean_words": float(np.mean([len(t.split()) for t in texts]))}
    return texts, labels, info


def main() -> None:
    ap = argparse.ArgumentParser(description="build the on-disk caches (needs network)")
    ap.add_argument("--build", nargs="*", default=None, help="dataset names (default: all)")
    args = ap.parse_args()
    for name in (args.build or sorted(SOURCES)):
        path = build_cache(name)
        texts, labels, info = load_dataset(name)
        print(f"{name:8s} -> {path}  ({path.stat().st_size / 2 ** 20:.1f} MB, {len(texts)} docs, "
              f"{info['mean_words']:.0f} words on average, {labels.mean():.2f} positive)")


if __name__ == "__main__":
    main()
