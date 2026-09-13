#!/usr/bin/env python3
"""
Run every QuadraSHAP paper experiment and report where the results landed.

    python run_all.py                 # everything, full size
    python run_all.py --quick         # smoke test (seconds)
    python run_all.py --only 1 3      # a subset
    python run_all.py --exp4-dataset synthetic   # offline text experiment

Each experiment is a standalone script and can equally be run on its own; see its ``--help``.
"""
from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = {
    1: ("exp1_certified_bound.py", "does the a priori certificate hold, and how tight is it"),
    2: ("exp2_ranking_vs_eps.py", "does the certified tolerance preserve the ranking"),
    3: ("exp3_synthetic_recovery.py", "300x1000 synthetic: accuracy, recovery and cost vs the SHAP family"),
    4: ("exp4_sentiment_deletion.py", "sentiment analysis: cost and deletion quality"),
    5: ("exp5_backend_and_stability.py", "the two evaluators: cost, blocking, and numerical stability"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--exp4-dataset", default=None, help="rotten_tomatoes (default), sst2, imdb or synthetic")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    t0 = time.perf_counter()
    for n in args.only:
        script, blurb = EXPERIMENTS[n]
        print(f"\n{'=' * 78}\n Experiment {n}: {blurb}\n{'=' * 78}", flush=True)
        argv = [script] + (["--quick"] if args.quick else [])
        if n == 4 and args.exp4_dataset:
            argv += ["--dataset", args.exp4_dataset]
        sys.argv = argv
        try:
            runpy.run_path(str(HERE / script), run_name="__main__")
        except Exception as exc:  # keep going: one missing optional dependency should not stop the rest
            print(f"!! experiment {n} failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"\nall done in {time.perf_counter() - t0:.1f}s; results in {HERE / 'results'}")


if __name__ == "__main__":
    main()
