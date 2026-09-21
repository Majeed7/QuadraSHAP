"""Worker process for cox_float64_repro_local.py.

Runs in its own interpreter so the JAX platform/precision is fixed once at
import time, exactly as it would be for a real user of ``quadrashap.CoxExplainer``:

* Default environment on this machine -> JAX picks the Metal GPU backend, and
  ``_jax_work_dtype`` (src/quadrashap/product_games/shapley.py) forces float32
  for any non-CPU platform regardless of ``jax_enable_x64`` -- this is the
  currently-shipped default behavior.
* ``--force-cpu-x64`` (combined with the caller setting ``JAX_PLATFORMS=cpu``
  in the environment *before* this process starts) -> JAX's only device is the
  CPU, and calling ``jax.config.update("jax_enable_x64", True)`` before any
  array is created gives genuine float64 arithmetic through the same
  ``logspace_jax`` backend code path.

Reads a small ``.npz`` with the fitted ridge-Cox coefficients, the background
matrix and the test patients (all produced by the orchestrator in plain
NumPy/scikit-survival, no JAX involved), computes ``CoxExplainer.explain(...,
return_report=True)`` for every (patient, eps) pair, and writes the resulting
Shapley vectors and certificate reports to a JSON file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Path to the inputs .npz (coef, background, test_patients)")
    parser.add_argument("--eps", nargs="+", type=float, required=True, help="Requested tolerances to test")
    parser.add_argument("--backend", default="logspace_jax", choices=["logspace_jax", "prefix_scan_jax"])
    parser.add_argument("--force-cpu-x64", action="store_true",
                        help="Enable jax_enable_x64 before building the explainer (only meaningful "
                             "when JAX_PLATFORMS=cpu was set in the environment before this process started)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    import jax
    if args.force_cpu_x64:
        jax.config.update("jax_enable_x64", True)

    from quadrashap import CoxExplainer

    data = np.load(args.npz)
    coef = data["coef"]
    background = data["background"]
    test_patients = data["test_patients"]

    platforms = sorted({dev.platform for dev in jax.devices()})
    env_report = {
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_platforms": platforms,
        "jax_enable_x64_config": bool(jax.config.jax_enable_x64),
        "force_cpu_x64_flag": args.force_cpu_x64,
    }

    explainer = CoxExplainer(coef, background=background, backend=args.backend)

    results = []
    for i in range(test_patients.shape[0]):
        x = test_patients[i]
        for eps in args.eps:
            t0 = time.perf_counter()
            phi, report = explainer.explain(x, eps=eps, backend=args.backend, return_report=True)
            elapsed = time.perf_counter() - t0
            results.append({
                "patient_index": i,
                "eps_requested": eps,
                "m_q": report.m_q,
                "scale_A_max": report.scale,
                "lambda_max": report.lambda_max,
                "bound": report.bound,
                "exact_threshold": report.exact_threshold,
                "efficiency_residual": report.efficiency_residual,
                "phi": phi.tolist(),
                "wall_seconds": elapsed,
            })

    Path(args.out).write_text(json.dumps({"environment": env_report, "results": results}, indent=2))
    print(f"Wrote {args.out} ({len(results)} (patient, eps) cases); backend={args.backend} "
          f"default_backend={env_report['jax_default_backend']} x64={env_report['jax_enable_x64_config']}")


if __name__ == "__main__":
    main()
