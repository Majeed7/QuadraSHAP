"""Archive reproducibility sources and hash the completed experiment artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from exp8_multimodel_budget import REPO, write_json


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def archive(root):
    config=json.loads((root/"config.json").read_text())
    snapshot=root/"source_snapshot"
    for name,expected in config["source_hashes"].items():
        source=REPO/name
        assert digest(source)==expected, f"Numerical core changed: {name}"
        destination=snapshot/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    experiment_dir=Path(__file__).resolve().parent
    for source in experiment_dir.glob("exp8_*"):
        if source.suffix in (".py",".md"):
            destination=snapshot/"experiments"/source.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    assert digest(experiment_dir/"exp8_multimodel_budget.py")==config["source_sha256"]
    manuscript=Path("/Users/Majid/surfdrive/Research/Overleaf/QuadraSHAP")
    for name in ("main.tex","sections/03_product_games_glq.tex","sections/04_multiplicative_models.tex",
                 "sections/05_experiments.tex","sections/appendix/node_budget.tex"):
        source=manuscript/name
        if source.exists():
            destination=snapshot/"manuscript"/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    files=[p for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"]
    entries={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
    write_json(root/"manifest.json",dict(files=entries,n_files=len(entries),total_bytes=sum(v["bytes"] for v in entries.values())))
    print(f"Archived sources and hashed {len(entries)} files ({sum(v['bytes'] for v in entries.values())/1e9:.3f} GB)",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    archive(Path(p.parse_args().output).resolve())
