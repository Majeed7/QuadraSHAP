"""Compare saved SHAP baselines to a matched, certified interventional reference.

The 30-background KernelSHAP and SamplingSHAP vectors from exp12 are reused.
Only the QuadraSHAP reference is newly evaluated.  Coordinates for which an
explained text and a background text agree are dummy players and can be
factored out exactly, making the 5,000-feature reference practical.
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
import exp10_pkex_metal_imdb as exp11
from quadrashap.product_games.budget import GameSummary, ellipse_bound
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy

DATASETS = ("rotten_tomatoes", "sst2", "sms_spam", "emotion")
EPS = 1e-6
N_BACKGROUND = 30


def output_dir(dataset: str) -> Path:
    return ROOT / "experiments" / "results" / f"exp13_interventional_accuracy_{dataset}"


def tables_for_background(support, alpha, gamma, x, b, dist2_x, n_background):
    """Compress dummy features without changing the interventional game.

    Outside ``varying``, x_j == b_j and the RBF factor is independent of the
    coalition.  Its product is absorbed into the game's coefficient.  The
    reduced game's polynomial degree therefore depends on the number of
    varying features, not all 5,000 TF-IDF coordinates.
    """
    varying = np.flatnonzero(x != b)
    if not len(varying):
        return varying, None, None, None
    z = support[:, varying].toarray() if sparse.issparse(support) else np.asarray(support[:, varying])
    dx2 = np.square(z - x[varying][None, :])
    db2 = np.square(z - b[varying][None, :])
    fixed_dist2 = np.maximum(dist2_x - dx2.sum(axis=1), 0.0)
    weight = (alpha / n_background) * np.exp(-gamma * fixed_dist2)
    present = np.exp(-gamma * dx2)
    absent = np.exp(-gamma * db2)
    return varying, present - absent, absent, weight


def select_nodes(A_max: float, lambda_max: float, max_varying: int, eps: float):
    """One node set for all background games; certify max coordinate error."""
    exact = max(1, (max_varying + 1) // 2)
    if A_max == 0 or lambda_max == 0:
        return 1, 0.0, exact
    for m_q in range(1, exact):
        bound = A_max * ellipse_bound(m_q, lambda_max)
        if bound <= eps:
            return m_q, bound, exact
    return exact, 0.0, exact


def reference(support, alpha, gamma, x, background, eps=EPS, node_block=2, check_extra_nodes=True):
    """Return the empirical-interventional Shapley vector and diagnostics.

    The budget is obtained from all (background, support-vector) product games
    with weights alpha/B, including the fixed-factor multiplier.  It certifies
    quadrature error for this *finite-background game* in exact arithmetic.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    background = np.asarray(background, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    d = len(x)
    if background.ndim != 2 or background.shape[1] != d:
        raise ValueError("background must have shape (n_background, d)")
    if support.shape != (len(alpha), d):
        raise ValueError("support and alpha dimensions differ")
    if not len(background):
        raise ValueError("the background set cannot be empty")

    t0 = time.perf_counter()
    support_sq = np.asarray(support.multiply(support).sum(axis=1)).ravel() if sparse.issparse(support) else np.square(support).sum(axis=1)
    support_dot_x = np.asarray(support @ x).ravel()
    dist2_x = np.maximum(support_sq + float(x @ x) - 2.0 * support_dot_x, 0.0)
    A = np.zeros(d, dtype=np.float64)
    lambda_max = 0.0
    max_varying = 0
    varying_counts = []
    for b in background:
        varying, K, Ut, weight = tables_for_background(support, alpha, gamma, x, b, dist2_x, len(background))
        varying_counts.append(len(varying))
        max_varying = max(max_varying, len(varying))
        if not len(varying):
            continue
        summary = GameSummary(d=len(varying)).update(K, Ut, weight)
        A[varying] += summary.A
        lambda_max = max(lambda_max, summary.lambda_max)
    A_max = float(A.max()) if d else 0.0
    m_q, certified_bound, exact_threshold = select_nodes(A_max, lambda_max, max_varying, eps)
    summary_seconds = time.perf_counter() - t0

    engine = ProductGamesShapleyNumpy()

    def evaluate(nodes):
        phi = np.zeros(d, dtype=np.float64)
        for b in background:
            varying, K, Ut, weight = tables_for_background(support, alpha, gamma, x, b, dist2_x, len(background))
            if not len(varying):
                continue
            terms = engine.phi_matrix_prefix_scan(K, nodes, Ut=Ut, node_block=node_block)
            phi[varying] += weight @ terms
        return phi

    t1 = time.perf_counter()
    phi = evaluate(m_q)
    quadrature_seconds = time.perf_counter() - t1
    extra_phi = None
    extra_nodes = None
    extra_gap = None
    if check_extra_nodes:
        extra_nodes = min(exact_threshold, m_q + 2)
        if extra_nodes > m_q:
            extra_phi = evaluate(extra_nodes)
            extra_gap = float(np.max(np.abs(extra_phi - phi)))
    diagnostics = {
        "requested_eps": eps,
        "m_q": m_q,
        "certified_bound": certified_bound,
        "A_max": A_max,
        "lambda_max": lambda_max,
        "max_varying_features_per_background": max_varying,
        "min_varying_features_per_background": min(varying_counts),
        "median_varying_features_per_background": float(np.median(varying_counts)),
        "exactness_threshold_reduced_games": exact_threshold,
        "n_background": len(background),
        "n_support_vectors": len(alpha),
        "n_product_games": len(alpha) * len(background),
        "seconds_summary_and_budget": summary_seconds,
        "seconds_quadrature": quadrature_seconds,
        "seconds_reference": time.perf_counter() - t0,
        "extra_nodes": extra_nodes,
        "max_abs_change_at_extra_nodes": extra_gap,
    }
    return phi, extra_phi, diagnostics


def compare(dataset: str, instance: int, bundle, arrays, background, args):
    x = arrays["X"][instance].astype(np.float64)
    target_class = int(arrays["target_classes"][instance]) if "target_classes" in arrays.files else 0
    model = exp11.selected_model(bundle, target_class)
    support, alpha, gamma, _ = exp11._model_arrays(model)
    phi, phi_extra, info = reference(support, alpha, gamma, x, background, args.eps,
                                     args.node_block, args.check_extra_nodes)
    scores_bg = np.asarray(model.decision_function(sparse.csr_matrix(background)), dtype=np.float64).reshape(-1)
    score_x = float(np.asarray(model.decision_function(sparse.csr_matrix(x[None, :])), dtype=np.float64).reshape(-1)[0])
    target = score_x - float(scores_bg.mean())
    info.update({
        "dataset": dataset,
        "instance": instance,
        "target_class": target_class,
        "model_score_x": score_x,
        "background_mean_model_score": float(scores_bg.mean()),
        "efficiency_target": target,
        "reference_efficiency_residual": abs(float(phi.sum()) - target),
        "reference_finite": bool(np.isfinite(phi).all()),
    })
    if not info["reference_finite"]:
        raise FloatingPointError(f"non-finite reference for {dataset} text {instance}")
    out = output_dir(dataset)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    phi_path = raw / f"reference_instance{instance:03d}.npz"
    np.savez_compressed(phi_path, phi=phi, phi_extra=phi_extra if phi_extra is not None else np.array([]))
    info["reference_phi_path"] = str(phi_path)
    baseline = ROOT / "experiments" / "results" / f"exp12_shap_background_{dataset}" / "raw"
    records = []
    for method in ("kernel", "sampling"):
        baseline_path = baseline / f"{method}_instance{instance:03d}.npz"
        estimate = np.load(baseline_path)["phi"]
        if estimate.shape != phi.shape or not np.isfinite(estimate).all():
            raise ValueError(f"invalid saved {method} vector: {baseline_path}")
        delta = estimate - phi
        error = float(np.max(np.abs(delta)))
        records.append({
            "dataset": dataset,
            "instance": instance,
            "method": "KernelSHAP" if method == "kernel" else "SamplingSHAP",
            "d": len(phi),
            "n_background": len(background),
            "requested_nsamples": 1000,
            "reference_eps": args.eps,
            "reference_m_q": info["m_q"],
            "reference_certified_bound": info["certified_bound"],
            "reference_efficiency_residual": info["reference_efficiency_residual"],
            "reference_extra_node_gap": info["max_abs_change_at_extra_nodes"],
            "max_abs_error_to_reference": error,
            "l2_error_to_reference": float(np.linalg.norm(delta)),
            "relative_l2_error_to_reference": float(np.linalg.norm(delta) / np.linalg.norm(phi)) if np.linalg.norm(phi) else np.nan,
            "nominal_true_error_lower": max(0.0, error - info["certified_bound"]),
            "nominal_true_error_upper": error + info["certified_bound"],
            "baseline_phi_path": str(baseline_path),
            "reference_phi_path": str(phi_path),
        })
    exp11.atomic_json(raw / f"reference_instance{instance:03d}.json", info)
    return records, info


def assemble(dataset: str):
    out = output_dir(dataset)
    rows = []
    for path in sorted((out / "raw").glob("comparison_instance*.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "records.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for method in ("KernelSHAP", "SamplingSHAP"):
        errors = np.array([r["max_abs_error_to_reference"] for r in rows if r["method"] == method])
        print(f"{dataset} {method}: n={len(errors)}, median max-error={np.median(errors):.6g}, "
              f"worst max-error={np.max(errors):.6g}", flush=True)


def run(args):
    exp11.set_dataset(args.dataset)
    out = output_dir(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    baseline_out = ROOT / "experiments" / "results" / f"exp12_shap_background_{args.dataset}"
    background_file = baseline_out / "background.npz"
    background = np.load(background_file)["X"].astype(np.float64)
    if background.shape != (N_BACKGROUND, 5000):
        raise ValueError(f"unexpected background shape: {background.shape}")
    bundle = joblib.load(exp11.MODEL_PATH)
    arrays = np.load(exp11.INSTANCES_PATH)
    n = min(args.n_instances, len(arrays["X"]))
    exp11.atomic_json(out / "run_meta.json", {
        "dataset": args.dataset,
        "eps": args.eps,
        "n_instances_requested": n,
        "n_background": N_BACKGROUND,
        "value_function": "finite 30-background interventional replacement game, uniform weights",
        "reference_backend": "NumPy float64 division-free prefix scan",
        "certificate_scope": "exact-arithmetic quadrature error for the finite-background game only",
        "background_path": str(background_file),
        "background_sha256": exp11.file_sha256(background_file),
        "model_path": str(exp11.MODEL_PATH),
        "model_sha256": exp11.file_sha256(exp11.MODEL_PATH),
        "instances_path": str(exp11.INSTANCES_PATH),
        "instances_sha256": exp11.file_sha256(exp11.INSTANCES_PATH),
        "script_sha256": exp11.file_sha256(Path(__file__)),
        "machine": platform.machine(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    for instance in range(n):
        compare_path = raw / f"comparison_instance{instance:03d}.json"
        if compare_path.exists() and not args.force:
            print(f"{args.dataset} text {instance}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        records, info = compare(args.dataset, instance, bundle, arrays, background, args)
        exp11.atomic_json(compare_path, records)
        print(f"{args.dataset} text {instance}: m_q={info['m_q']}, "
              f"bound={info['certified_bound']:.2e}, max extra-node gap="
              f"{info['max_abs_change_at_extra_nodes']}, {time.perf_counter()-t0:.2f}s", flush=True)
        assemble(args.dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--node-block", type=int, default=2)
    ap.add_argument("--check-extra-nodes", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.eps <= 0:
        ap.error("--eps must be positive")
    run(args)


if __name__ == "__main__":
    main()
