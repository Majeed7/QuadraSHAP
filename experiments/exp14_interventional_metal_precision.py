"""Metal QuadraSHAP precision sweep on exp12's matched interventional text games.

The timed method is Metal float32. A separate NumPy float64 result at the
1e-16 requested quadrature tolerance is used only as an empirical reference.
Every raw vector and its per-instance diagnostics are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from exp10_pkex_metal_imdb import EPS_LEVELS, atomic_json, file_sha256, selected_model, _model_arrays, set_dataset
import exp10_pkex_metal_imdb as exp11
from exp13_interventional_accuracy import DATASETS, reference, select_nodes, tables_for_background
from quadrashap.product_games.shapley import ProductGamesShapleyJax

OUT = ROOT / "experiments" / "results" / "exp14_interventional_metal_precision"


def metal_attributions(support, alpha, gamma, x, background, dist2_x, m_q, chunk, node_block):
    """Exact same finite-background game as exp13, with dummy-player compression."""
    engine = ProductGamesShapleyJax()
    phi = np.zeros(len(x), dtype=np.float64)
    for b in background:
        varying, K, Ut, weight = tables_for_background(
            support, alpha, gamma, x, b, dist2_x, len(background))
        if not len(varying):
            continue
        # Padding introduces only dummy players, preserving the game and one
        # reusable JIT shape per small width bucket and fixed support chunk.
        width = ((len(varying) + 7) // 8) * 8
        for start in range(0, len(alpha), chunk):
            stop = min(start + chunk, len(alpha))
            n = stop - start
            kp = np.zeros((chunk, width), dtype=np.float32)
            up = np.ones((chunk, width), dtype=np.float32)
            kp[:n, :len(varying)] = K[start:stop].astype(np.float32)
            up[:n, :len(varying)] = Ut[start:stop].astype(np.float32)
            factors = engine.phi_matrix_prefix_scan(kp, m_q, Ut=up, node_block=node_block)
            phi[varying] += weight[start:stop] @ np.asarray(factors[:n, :len(varying)], dtype=np.float64)
    return phi


def run_one(dataset, instance, bundle, instances, background, args):
    raw = OUT / dataset / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    x = np.asarray(instances["X"][instance], dtype=np.float64)
    target_class = int(instances["target_classes"][instance])
    model = selected_model(bundle, target_class)
    support, alpha, gamma, _ = _model_arrays(model)
    phi_ref, phi_extra, info = reference(support, alpha, gamma, x, background,
                                         eps=1e-16, node_block=2, check_extra_nodes=True)
    support_sq = np.asarray(support.multiply(support).sum(axis=1)).ravel() if sparse.issparse(support) else np.square(support).sum(axis=1)
    dist2_x = np.maximum(support_sq + x @ x - 2 * np.asarray(support @ x).ravel(), 0.0)
    ref_path = raw / f"reference_instance{instance:03d}.npz"
    np.savez_compressed(ref_path, phi=phi_ref,
                        phi_extra=phi_extra if phi_extra is not None else np.array([]))
    info.update(dataset=dataset, instance=instance, target_class=target_class,
                reference_phi_path=str(ref_path), reference_dtype="float64")
    atomic_json(raw / f"reference_instance{instance:03d}.json", info)
    score_x = float(np.asarray(model.decision_function(sparse.csr_matrix(x[None, :]))).ravel()[0])
    score_bg = float(np.mean(model.decision_function(sparse.csr_matrix(background))))
    rows = []
    for eps in EPS_LEVELS:
        m_q, bound, exact = select_nodes(info["A_max"], info["lambda_max"],
                                          info["max_varying_features_per_background"], eps)
        t0 = time.perf_counter()
        chunk = len(alpha) if args.chunk == 0 else args.chunk
        phi = metal_attributions(support, alpha, gamma, x, background, dist2_x,
                                 m_q, chunk, args.node_block)
        seconds = time.perf_counter() - t0
        err = float(np.max(np.abs(phi - phi_ref)))
        vec_path = raw / f"metal_instance{instance:03d}_eps{exp11.eps_tag(eps)}.npz"
        np.savez_compressed(vec_path, phi=phi)
        row = dict(dataset=dataset, instance=instance, method="QuadraSHAP Metal",
                   value_function="30-background empirical interventional", d=len(x),
                   n_background=len(background), n_support_vectors=len(alpha),
                   requested_eps=eps, m_q=m_q, exactness_threshold_reduced_games=exact,
                   certified_quadrature_bound=bound, certificate_scope="exact arithmetic only",
                   seconds_core=seconds, seconds_reference=info["seconds_reference"],
                   observed_max_abs_error_to_float64_reference=err,
                   observed_l2_error_to_float64_reference=float(np.linalg.norm(phi - phi_ref)),
                   efficiency_residual=abs(float(phi.sum()) - (score_x - score_bg)),
                   reference_extra_node_gap=info["max_abs_change_at_extra_nodes"],
                   finite=bool(np.isfinite(phi).all()), dtype="float32 Metal; float64 weighted reduction",
                   phi_path=str(vec_path), reference_phi_path=str(ref_path))
        rows.append(row)
        atomic_json(raw / f"metal_instance{instance:03d}_eps{exp11.eps_tag(eps)}.json", row)
        print(f"{dataset} {instance:02d} eps={eps:.0e} m={m_q} bound={bound:.2e} "
              f"observed={err:.2e} time={seconds:.2f}s", flush=True)
    return rows


def assemble(dataset):
    raw = OUT / dataset / "raw"
    rows = [json.loads(p.read_text()) for p in sorted(raw.glob("metal_instance*_eps*.json"))]
    if rows:
        with (OUT / dataset / "records.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=0,
                    help="support rows per Metal call; 0 uses one call per background")
    ap.add_argument("--node-block", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import jax
    devices = [str(d) for d in jax.devices()]
    if not any("metal" in s.lower() for s in devices):
        raise RuntimeError(f"Apple Metal device required; got {devices}")
    set_dataset(args.dataset)
    # set_dataset imported function updates the exp11 module globals.
    base = ROOT / "experiments" / "results" / f"exp12_shap_background_{args.dataset}"
    background_path = base / "background.npz"
    background = np.load(background_path)["X"].astype(np.float64)
    bundle = joblib.load(exp11.MODEL_PATH)
    instances = np.load(exp11.INSTANCES_PATH)
    out = OUT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "run_meta.json", dict(dataset=args.dataset, devices=devices,
        eps_levels=EPS_LEVELS, n_instances=min(args.n_instances, len(instances["X"])),
        n_background=len(background), backend="Apple Metal JAX float32",
        reference_backend="NumPy float64", value_function="same as exp12 and exp13",
        model_sha256=file_sha256(exp11.MODEL_PATH),
        instances_sha256=file_sha256(exp11.INSTANCES_PATH),
        background_sha256=file_sha256(background_path),
        script_sha256=file_sha256(Path(__file__)), machine=platform.machine(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    for i in range(min(args.n_instances, len(instances["X"]))):
        marker = out / "raw" / f"metal_instance{i:03d}_eps{exp11.eps_tag(EPS_LEVELS[-1])}.json"
        if marker.exists() and not args.force:
            print(f"{args.dataset} {i:02d}: cached", flush=True)
            continue
        run_one(args.dataset, i, bundle, instances, background, args)
        assemble(args.dataset)


if __name__ == "__main__":
    main()
