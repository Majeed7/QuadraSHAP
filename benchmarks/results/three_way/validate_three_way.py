"""Untimed attribution-level validation for the three-way benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks"))

from xgboost_sklearn_bridge import sklearn_forest_to_xgboost  # noqa: E402


def treegrad_forest(model, X, treegrad_shap):
    estimators = list(np.ravel(model.estimators_))
    output = np.zeros((len(X), X.shape[1]), dtype=np.float64)
    for row_index, row in enumerate(X):
        for estimator in estimators:
            output[row_index] += treegrad_shap(estimator, row, (1, 1))
    return output / len(estimators)


def main(args):
    import cupy as cp
    from quadrashap import TreeExplainer

    sys.path.insert(0, str(args.treegrad_repo))
    from TreeGrad import treegrad_shap

    results = {}
    for model_path in sorted(args.cache.glob("*.joblib")):
        bundle = joblib.load(model_path)
        model = bundle["estimator"]
        X = np.ascontiguousarray(bundle["X"][: args.rows], dtype=np.float64)

        quadra = TreeExplainer(
            model, tree_solver="quadrature_tree", device="cuda"
        )
        quadra_phi = np.asarray(
            quadra.shap_values(X, check_additivity=True), dtype=np.float64
        )
        quadra_bias = float(np.asarray(quadra.expected_value).ravel()[0])

        treegrad_phi = treegrad_forest(model, X, treegrad_shap)

        booster = sklearn_forest_to_xgboost(model, device="cuda:0")
        dmatrix = xgb.DMatrix(np.array(X, copy=True, order="C"))
        qts_values = np.asarray(
            booster.predict(dmatrix, pred_contribs=True), dtype=np.float64
        )
        cp.cuda.runtime.deviceSynchronize()

        treegrad_delta = np.abs(treegrad_phi - quadra_phi)
        qts_delta = np.abs(qts_values[:, :-1] - quadra_phi)
        results[model_path.stem] = {
            "rows": int(len(X)),
            "quadra_exact_mq": int(quadra._backend._cuda_solver.host.m_q),
            "treegrad_phi_max_abs_difference": float(np.max(treegrad_delta)),
            "treegrad_phi_mean_abs_difference": float(np.mean(treegrad_delta)),
            "treegrad_additivity_max_error": float(
                np.max(np.abs(quadra_bias + treegrad_phi.sum(axis=1) - model.predict(X)))
            ),
            "qts_phi_max_abs_difference": float(np.max(qts_delta)),
            "qts_phi_mean_abs_difference": float(np.mean(qts_delta)),
            "qts_bias_abs_difference": float(
                abs(float(qts_values[0, -1]) - quadra_bias)
            ),
        }
        print(model_path.stem, results[model_path.stem], flush=True)

    payload = {
        "quadra_revision": args.quadra_revision,
        "treegrad_revision": args.treegrad_revision,
        "rows_per_model": args.rows,
        "results": results,
        "maxima": {
            key: float(max(row[key] for row in results.values()))
            for key in (
                "treegrad_phi_max_abs_difference",
                "treegrad_phi_mean_abs_difference",
                "treegrad_additivity_max_error",
                "qts_phi_max_abs_difference",
                "qts_phi_mean_abs_difference",
                "qts_bias_abs_difference",
            )
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--treegrad-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--quadra-revision", required=True)
    parser.add_argument("--treegrad-revision", required=True)
    main(parser.parse_args())
