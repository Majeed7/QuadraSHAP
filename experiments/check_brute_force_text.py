"""Ground truth by enumeration, on real documents.

The short-sentence corpora have documents with only a dozen or so distinct words, and that is
exactly what makes them useful: with ``d <= ~18`` all ``2^d`` coalitions can be enumerated, so the
Shapley values can be computed from the definition rather than from any clever identity.  This
script does that for real SST-2 / MR / AG News documents and compares three things:

    the definition          sum over all 2^d coalitions, weights s!(d-1-s)!/d!
    QuadraSHAP              the m_q = ceil(d/2) quadrature rule
    the sampling estimators  at their largest budget

It is the check that the ``exact'' reference of experiment 7 deserves: on the corpora where the
answer can be computed a second way, it is.

    python check_brute_force_text.py --max-d 16 --n-docs 6
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from exp7_text_methods import build_game, fit_model, quadrashap_phi
from text_attribution import MaskGame, kernel_shap_path, permutation_shap_path
from text_datasets import load_dataset


def shapley_by_enumeration(game: MaskGame) -> np.ndarray:
    """Shapley values from the definition: every one of the ``2^d`` coalitions, exactly once."""
    d = game.d
    idx = np.arange(1 << d, dtype=np.int64)
    masks = ((idx[:, None] >> np.arange(d)[None, :]) & 1).astype(np.float64)
    v = game.value(masks)                                   # v(S) for every coalition
    sizes = masks.sum(axis=1).astype(int)
    from math import factorial
    w = np.array([factorial(s) * factorial(d - 1 - s) / factorial(d) for s in range(d)])
    phi = np.empty(d)
    for i in range(d):
        without = idx[((idx >> i) & 1) == 0]
        phi[i] = (w[sizes[without]] * (v[without | (1 << i)] - v[without])).sum()
    return phi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sst2", "mr", "agnews"])
    ap.add_argument("--max-d", type=int, default=16, help="2**max_d coalitions are enumerated")
    ap.add_argument("--n-docs", type=int, default=4)
    ap.add_argument("--budget", type=int, default=8192)
    args = ap.parse_args()

    print(f"{'corpus':8s} {'d':>3s} {'coalitions':>11s} {'||phi||':>10s} "
          f"{'QuadraSHAP':>12s} {'KernelSHAP':>12s} {'Permutation':>12s} {'enum. time':>11s}")
    for ds in args.datasets:
        bundle = fit_model(ds)
        texts, _, _ = load_dataset(ds)
        done = 0
        for k in range(3000, len(texts)):
            if done >= args.n_docs:
                break
            game, tables, _ = build_game(bundle, texts[k])
            if game is None or not (6 <= game.d <= args.max_d):
                continue
            done += 1
            U, Ut, w = tables
            d = game.d

            t0 = time.perf_counter()
            phi_enum = shapley_by_enumeration(MaskGame(U, Ut, w, game.intercept))
            t_enum = time.perf_counter() - t0

            phi_q = quadrashap_phi(U, Ut, w, (d + 1) // 2)
            rel = lambda p: np.linalg.norm(p - phi_enum) / np.linalg.norm(phi_enum)
            ks = kernel_shap_path(MaskGame(U, Ut, w, game.intercept), [args.budget], seed=0).phi[-1]
            ps = permutation_shap_path(MaskGame(U, Ut, w, game.intercept), [args.budget], seed=0).phi[-1]
            print(f"{ds:8s} {d:3d} {1 << d:11,d} {np.linalg.norm(phi_enum):10.4f} "
                  f"{rel(phi_q):12.2e} {rel(ks):12.2e} {rel(ps):12.2e} {t_enum:10.2f}s")
    print("\nRelative errors against the enumerated Shapley values; the samplers used "
          f"{args.budget} evaluations, QuadraSHAP ceil(d/2) nodes.")


if __name__ == "__main__":
    main()
