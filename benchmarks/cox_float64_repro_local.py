"""Reproduce, on real genomic survival data, the float32-vs-float64 gap in QuadraSHAP's
Cox certificate (src/quadrashap/product_games/budget.py).

Background
----------
``quadrashap.CoxExplainer`` (src/quadrashap/multiplicative/engine.py, built on
src/quadrashap/product_games/shapley.py) silently computes in float32 whenever
JAX's active platform is not "cpu" -- see ``_jax_work_dtype`` in shapley.py,
which forces float32 for any metal/gpu/tpu platform regardless of
``jax_enable_x64``. The certificate machinery in budget.py is explicit that
its bound "concerns the quadrature error ... not floating-point rounding" --
i.e. it is a real-arithmetic guarantee. This script fits a real ridge Cox
model on real TCGA genomic data and shows, with actual numbers, that:

(a) the default GPU/Metal float32 path can produce an observed Shapley error
    that exceeds both the requested tolerance ``eps`` and the certified
    bound the library itself reports for the ``m_q`` it chose;
(b) the identical ``CoxExplainer``/``logspace_jax`` code path forced onto the
    CPU with ``jax_enable_x64=True`` (genuine float64) does not.

Pipeline
--------
1. Fetch + prepare the smallest real dataset in data/experiments/README.md
   that still has genomic-scale dimensionality: ``tcga_laml`` (35 patients,
   90,159 features, 14 events; OpenML ids 42291/42292).
2. Preprocess with ``benchmarks.experiment_data.TrainingPreprocessor`` (median
   impute + standardize on the training split only), then take a fixed random
   subsample of the kept features (default 6,000 of ~90,115) so that the
   *exact* Gauss-Legendre reference (``ceil(d_active/2)`` nodes, run in plain
   float64 NumPy via ``quadrashap.cox.CoxPHExplainer``) is tractable within a
   few seconds per patient -- reproducing the certificate mechanism at a real,
   still-substantial, d >> n scale rather than the full 90k features. This is
   a deliberate, disclosed scope reduction (see the printed run manifest and
   the final report), not a synthetic-data substitution: every value used is
   a real measurement from the real dataset.
3. Fit ``benchmarks.cox_model.SampleSpaceRidgeCox`` (dense ridge Cox via
   ``sksurv``) on the training rows over the subsampled features. Ridge keeps
   effectively all coefficients nonzero (d_active ~= d_sub).
4. Background = 25 training rows; test patients = the 5 held-out rows.
5. Ground truth per test patient: ``quadrashap.cox.CoxPHExplainer`` (pure
   NumPy, float64, exact by construction at ``m_q = ceil(d_active/2)``).
6. Two dtype paths for ``quadrashap.CoxExplainer(..., backend="logspace_jax")``,
   each run in its own subprocess (JAX's platform is fixed at import time):
     - "metal_float32": default environment on this Apple Silicon machine ->
       JAX picks the Metal GPU backend -> forced float32 (the current
       shipped default whenever JAX is installed with a non-CPU backend).
     - "cpu_float64": subprocess launched with ``JAX_PLATFORMS=cpu``, and
       ``jax.config.update("jax_enable_x64", True)`` before building the
       explainer -> genuine CPU float64 through the identical code path.
7. For every (patient, eps in {1e-3, 1e-6}, dtype path): record requested eps,
   the certified bound, the observed max-abs-error vs. the float64 exact
   reference, and whether each of eps/bound was met. Save to CSV + JSON under
   benchmarks/results/cox_float64_repro_local/.

Run: ``uv run python benchmarks/cox_float64_repro_local.py``
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.experiment_data import load_survival, TrainingPreprocessor  # noqa: E402
from benchmarks.cox_model import SampleSpaceRidgeCox  # noqa: E402
from quadrashap.cox import CoxPHExplainer  # noqa: E402

DATASET = "tcga_laml"
ENDPOINT = "OS"
N_TEST_PATIENTS = 5
N_BACKGROUND = 25
D_SUB = 6000
RIDGE_ALPHA = 300.0
EPS_LIST = [1e-3, 1e-6]
RANDOM_STATE = 42
COLUMN_SEED = 42

RESULTS_DIR = ROOT / "benchmarks" / "results" / "cox_float64_repro_local"


def fetch_and_prepare() -> None:
    print(f"== Fetching {DATASET} (cached if already downloaded) ==", flush=True)
    subprocess.run([sys.executable, str(ROOT / "benchmarks" / "fetch_experiment_data.py"),
                    "--datasets", DATASET], check=True, cwd=ROOT)
    print(f"== Preparing {DATASET} (cached if already prepared) ==", flush=True)
    subprocess.run([sys.executable, str(ROOT / "benchmarks" / "prepare_experiment_data.py"),
                    "--datasets", DATASET], check=True, cwd=ROOT)


def build_model_and_data():
    ds = load_survival(DATASET, endpoint=ENDPOINT)
    n_total, d_raw = ds.X.shape
    events_total = ds.metadata["events"]

    idx = np.arange(len(ds.y))
    train_idx, test_idx = train_test_split(
        idx, test_size=N_TEST_PATIENTS, random_state=RANDOM_STATE, stratify=ds.y["event"],
    )

    pre = TrainingPreprocessor.fit(ds.X[train_idx])
    X_train_full = pre.transform(ds.X[train_idx])
    X_test_full = pre.transform(ds.X[test_idx])
    y_train = ds.y[train_idx]
    d_full = X_train_full.shape[1]

    rng = np.random.default_rng(COLUMN_SEED)
    d_sub = min(D_SUB, d_full)
    cols = np.sort(rng.choice(d_full, size=d_sub, replace=False))
    X_train_sub = np.asarray(X_train_full[:, cols], dtype=np.float64)
    X_test_sub = np.asarray(X_test_full[:, cols], dtype=np.float64)

    model = SampleSpaceRidgeCox(alpha=RIDGE_ALPHA)
    model.fit(X_train_sub, y_train)
    coef = model.coef_
    d_active = int((coef != 0).sum())

    background = X_train_sub[:N_BACKGROUND]

    manifest = {
        "dataset": DATASET, "endpoint": ENDPOINT,
        "n_patients_total": int(n_total), "d_features_raw": int(d_raw), "events_total": int(events_total),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "events_train": int(y_train["event"].sum()),
        "d_kept_after_preprocessing": int(d_full),
        "d_subsample_used": int(d_sub), "column_subsample_seed": COLUMN_SEED,
        "ridge_alpha": RIDGE_ALPHA, "ridge_rank": int(model.rank_),
        "d_active_nonzero_coef": d_active,
        "n_background_rows": int(background.shape[0]),
        "test_patient_indices_in_full_dataset": [int(i) for i in test_idx],
        "test_patient_events": [bool(e) for e in ds.y["event"][test_idx]],
        "eps_list": EPS_LIST,
        "note": ("d_subsample_used is a deliberate, disclosed reduction from d_kept_after_preprocessing "
                 "so the exact float64 reference (ceil(d_active/2) Gauss-Legendre nodes) is tractable "
                 "within the time budget; every feature value used is real TCGA LAML data, not synthetic."),
    }
    return coef, background, X_test_sub, manifest


def exact_reference(coef: np.ndarray, background: np.ndarray, test_patients: np.ndarray):
    ref = CoxPHExplainer(coef, background)
    out = []
    for i in range(test_patients.shape[0]):
        t0 = time.perf_counter()
        expl = ref.explain(test_patients[i])
        out.append({
            "patient_index": i, "phi": expl.values.tolist(),
            "prediction_relative_hazard": expl.prediction, "base_value": expl.base_value,
            "exact_nodes": expl.exact_nodes, "active_features": expl.active_features,
            "efficiency_residual": expl.efficiency_residual, "wall_seconds": time.perf_counter() - t0,
        })
        print(f"  exact reference patient {i}: exact_nodes={expl.exact_nodes} "
              f"active={expl.active_features} residual={expl.efficiency_residual:.3e} "
              f"({time.perf_counter() - t0:.2f}s)", flush=True)
    return out


def run_backend_leg(npz_path: Path, out_path: Path, *, force_cpu_x64: bool) -> dict:
    env = None
    if force_cpu_x64:
        import os
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
    cmd = [sys.executable, str(ROOT / "benchmarks" / "_cox_repro_backend_worker.py"),
           "--npz", str(npz_path), "--eps", *[str(e) for e in EPS_LIST],
           "--backend", "logspace_jax", "--out", str(out_path)]
    if force_cpu_x64:
        cmd.append("--force-cpu-x64")
    print(f"== Running worker leg (force_cpu_x64={force_cpu_x64}) ==", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"worker leg failed (force_cpu_x64={force_cpu_x64})")
    return json.loads(out_path.read_text())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fetch_and_prepare()

    print("== Building ridge Cox model and background/test split ==", flush=True)
    coef, background, test_patients, manifest = build_model_and_data()
    print(json.dumps(manifest, indent=2), flush=True)

    print("== Computing float64 exact reference (quadrashap.cox.CoxPHExplainer, pure NumPy) ==", flush=True)
    exact = exact_reference(coef, background, test_patients)
    exact_phi = {row["patient_index"]: np.asarray(row["phi"]) for row in exact}

    npz_path = RESULTS_DIR / "inputs.npz"
    np.savez(npz_path, coef=coef, background=background, test_patients=test_patients)

    metal_out = RESULTS_DIR / "metal_float32_raw.json"
    cpu_out = RESULTS_DIR / "cpu_float64_raw.json"
    metal_result = run_backend_leg(npz_path, metal_out, force_cpu_x64=False)
    cpu_result = run_backend_leg(npz_path, cpu_out, force_cpu_x64=True)

    print("== Environment used by each leg ==")
    print("metal_float32:", json.dumps(metal_result["environment"]))
    print("cpu_float64  :", json.dumps(cpu_result["environment"]))

    rows = []
    for leg_name, leg in (("metal_float32", metal_result), ("cpu_float64", cpu_result)):
        for case in leg["results"]:
            i, eps = case["patient_index"], case["eps_requested"]
            phi = np.asarray(case["phi"])
            observed = float(np.max(np.abs(phi - exact_phi[i])))
            rows.append({
                "dtype_path": leg_name,
                "patient_index": i,
                "eps_requested": eps,
                "m_q": case["m_q"],
                "exact_threshold": case["exact_threshold"],
                "scale_A_max": case["scale_A_max"],
                "lambda_max": case["lambda_max"],
                "bound": case["bound"],
                "max_absolute_error": observed,
                "observed_tolerance_met": observed <= eps,
                "observed_bound_met": observed <= case["bound"],
                "efficiency_residual": case["efficiency_residual"],
                "wall_seconds": case["wall_seconds"],
            })
    summary = pd.DataFrame(rows).sort_values(["patient_index", "eps_requested", "dtype_path"]).reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (RESULTS_DIR / "exact_reference.json").write_text(json.dumps(exact, indent=2))

    print("\n== Summary ==")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))

    for leg_name in ("metal_float32", "cpu_float64"):
        leg = summary[summary.dtype_path == leg_name]
        n = len(leg)
        eps_viol = int((~leg.observed_tolerance_met).sum())
        bound_viol = int((~leg.observed_bound_met).sum())
        print(f"\n{leg_name}: {eps_viol}/{n} cases violate requested eps, "
              f"{bound_viol}/{n} cases violate the certified bound")

    print(f"\nSaved: {RESULTS_DIR / 'summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'manifest.json'}")
    print(f"Saved: {RESULTS_DIR / 'exact_reference.json'}")
    print(f"Saved: {metal_out}")
    print(f"Saved: {cpu_out}")


if __name__ == "__main__":
    main()
