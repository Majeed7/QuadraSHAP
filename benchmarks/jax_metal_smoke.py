"""Fail explicitly on CPU fallback; test JIT and Cox explanations on Metal.

Run each case in a separate process so a native plugin failure is isolated::

    PYTHONPATH=src JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 \
      .venv-metal/bin/python benchmarks/jax_metal_smoke.py --case primitive
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


def brute_force_cox(beta, x, background):
    """Enumerate every coalition using the prediction function, for tiny d only."""
    d = len(beta)
    if d > 16:
        raise ValueError("Exhaustive enumeration is only intended for a tiny fixture.")
    values = np.empty(1 << d)
    for mask in range(1 << d):
        present = np.array([bool(mask & (1 << j)) for j in range(d)])
        values[mask] = np.exp(np.where(present, x, background) @ beta).mean()
    phi = np.zeros(d)
    for i in range(d):
        for mask in range(1 << d):
            if not mask & (1 << i):
                size = mask.bit_count()
                weight = 1 / (d * math.comb(d - 1, size))
                phi[i] += weight * (values[mask | (1 << i)] - values[mask])
    return phi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("primitive", "prefix_scan_jax", "logspace_jax"), required=True)
    parser.add_argument("--features", type=int, default=8)
    args = parser.parse_args()
    devices = jax.devices()
    print(json.dumps({"python": platform.python_version(), "devices": [str(d) for d in devices],
                      "backend": jax.default_backend(), "jit_disabled": jax.config.jax_disable_jit,
                      "versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "jax-metal", "numpy")}}), flush=True)
    if not devices or any(d.platform.lower() != "metal" for d in devices):
        raise RuntimeError("This smoke test requires Metal; CPU fallback is a failure.")
    if jax.config.jax_disable_jit:
        raise RuntimeError("JIT must be enabled for this GPU test.")
    if args.case == "primitive":
        a_np = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 4096
        a = jax.device_put(a_np)
        fn = jax.jit(lambda z: z @ z.T)
        t = perf_counter()
        result = fn(a).block_until_ready()
        first = perf_counter() - t
        t = perf_counter()
        result = fn(a).block_until_ready()
        warm = perf_counter() - t
        np.testing.assert_allclose(np.asarray(result), a_np @ a_np.T, rtol=2e-5, atol=2e-5)
        print(json.dumps({"case": args.case, "passed": True, "result_device": str(result.device),
                          "first_seconds": first, "warm_seconds": warm}), flush=True)
        return
    from quadrashap import CoxExplainer
    rng = np.random.default_rng(42)
    d = args.features
    beta = rng.normal(size=d) * (0.2 / np.sqrt(d))
    x, bg = rng.normal(size=d), rng.normal(size=(4, d))
    ex = CoxExplainer(beta, background=bg, backend=args.case, memory_budget="512MB")
    m = (d + 1) // 2 if d <= 16 else 8
    t = perf_counter()
    reference = brute_force_cox(beta, x, bg) if d <= 16 else CoxExplainer(beta, background=bg, backend="prefix_scan_numpy").explain(x, m_q=m)
    reference_seconds = perf_counter() - t
    t = perf_counter()
    phi = ex.explain(x, m_q=m)
    first = perf_counter() - t
    t = perf_counter()
    phi = ex.explain(x, m_q=m)
    warm = perf_counter() - t
    max_error = float(np.max(np.abs(phi - reference)))
    np.testing.assert_allclose(phi, reference, atol=2e-5, rtol=2e-4)
    print(json.dumps({"case": args.case, "passed": True, "d": d, "m_q": m,
                      "first_seconds": first, "warm_seconds": warm, "reference_seconds": reference_seconds,
                      "reference": "exhaustive coalitions" if d <= 16 else "float64 NumPy at same nodes",
                      "max_absolute_error": max_error, "block_plan": asdict(ex.last_block_plan)}), flush=True)


if __name__ == "__main__":
    main()
