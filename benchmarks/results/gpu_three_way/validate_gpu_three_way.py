"""Untimed attribution agreement check for the three-GPU-method benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from xgboost_sklearn_bridge import sklearn_forest_to_xgboost  # noqa: E402


def validate(args) -> None:
    import cupy as cp
    import joblib
    import shap
    import shap._cext_gpu  # noqa: F401
    import xgboost as xgb

    from quadrashap import TreeExplainer

    results = {}
    for path in sorted(args.cache.glob("*_depth24_s*.joblib")):
        bundle = joblib.load(path)
        model = bundle["estimator"]
        X = np.ascontiguousarray(bundle["X"][: args.rows], dtype=np.float64)
        prediction = np.asarray(model.predict(X), dtype=np.float64)

        exact = TreeExplainer(model, tree_solver="quadrature_tree", device="cuda")
        exact_phi = np.asarray(
            exact.shap_values(X, check_additivity=False), dtype=np.float64
        )
        exact_bias = float(np.asarray(exact.expected_value).ravel()[0])

        mq8 = TreeExplainer(
            model, tree_solver="quadrature_tree", device="cuda", m_q=8
        )
        mq8_phi = np.asarray(
            mq8.shap_values(X, check_additivity=False), dtype=np.float64
        )
        mq8_bias = float(np.asarray(mq8.expected_value).ravel()[0])

        booster = sklearn_forest_to_xgboost(model, device="cuda:0")
        dmatrix = xgb.DMatrix(np.array(X, copy=True, order="C"))
        qts_values = np.asarray(
            booster.predict(dmatrix, pred_contribs=True), dtype=np.float64
        )
        qts_phi = qts_values[:, :-1]
        qts_bias = float(qts_values[0, -1])

        gpu = shap.GPUTreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )
        gpu_phi = gpu.shap_values(X, check_additivity=False)
        if isinstance(gpu_phi, list):
            gpu_phi = np.stack([np.asarray(value) for value in gpu_phi], axis=-1)
        gpu_phi = np.asarray(gpu_phi)
        if gpu_phi.ndim == 3:
            gpu_phi = gpu_phi[:, :, 0]
        gpu_phi = np.asarray(gpu_phi, dtype=np.float64)
        gpu_bias = float(np.asarray(gpu.expected_value).ravel()[0])
        cp.cuda.runtime.deviceSynchronize()

        qts_prediction = np.asarray(
            booster.predict(dmatrix, output_margin=True), dtype=np.float64
        )
        results[path.stem] = {
            "rows": int(len(X)),
            "actual_total_leaves": int(bundle["actual_total_leaves"]),
            "max_depth": int(bundle["max_depth"]),
            "quadra_exact_mq": int(exact._backend._cuda_solver.host.m_q),
            "quadra_mq8_phi_max_abs_difference": float(
                np.max(np.abs(mq8_phi - exact_phi))
            ),
            "quadra_mq8_phi_mean_abs_difference": float(
                np.mean(np.abs(mq8_phi - exact_phi))
            ),
            "quadra_mq8_bias_abs_difference": float(abs(mq8_bias - exact_bias)),
            "qts_phi_max_abs_difference": float(np.max(np.abs(qts_phi - exact_phi))),
            "qts_phi_mean_abs_difference": float(np.mean(np.abs(qts_phi - exact_phi))),
            "qts_bias_abs_difference": float(abs(qts_bias - exact_bias)),
            "qts_bridge_prediction_max_abs_difference": float(
                np.max(np.abs(qts_prediction - prediction))
            ),
            "gputreeshap_phi_max_abs_difference": float(
                np.max(np.abs(gpu_phi - exact_phi))
            ),
            "gputreeshap_phi_mean_abs_difference": float(
                np.mean(np.abs(gpu_phi - exact_phi))
            ),
            "gputreeshap_bias_abs_difference": float(abs(gpu_bias - exact_bias)),
            "gputreeshap_additivity_max_error": float(
                np.max(np.abs(gpu_bias + gpu_phi.sum(axis=1) - prediction))
            ),
            "quadra_exact_additivity_max_error": float(
                np.max(np.abs(exact_bias + exact_phi.sum(axis=1) - prediction))
            ),
        }

    maximum_keys = (
        "quadra_mq8_phi_max_abs_difference",
        "quadra_mq8_phi_mean_abs_difference",
        "quadra_mq8_bias_abs_difference",
        "qts_phi_max_abs_difference",
        "qts_phi_mean_abs_difference",
        "qts_bias_abs_difference",
        "qts_bridge_prediction_max_abs_difference",
        "gputreeshap_phi_max_abs_difference",
        "gputreeshap_phi_mean_abs_difference",
        "gputreeshap_bias_abs_difference",
        "gputreeshap_additivity_max_error",
        "quadra_exact_additivity_max_error",
    )
    payload = {
        "quadra_revision": args.quadra_revision,
        "shap_revision": args.shap_revision,
        "rows_per_model": args.rows,
        "results": results,
        "maxima": {
            key: max(result[key] for result in results.values())
            for key in maximum_keys
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["maxima"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--quadra-revision", required=True)
    parser.add_argument("--shap-revision", required=True)
    validate(parser.parse_args())
