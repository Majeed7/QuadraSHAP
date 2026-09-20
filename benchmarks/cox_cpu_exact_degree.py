"""Checkpointed CPU FP64 degree-exact baselines for the five-patient B30 study."""
import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games import shapley
from benchmarks.cox_tolerance_rerun import SOURCE, OUT as GPU_SOURCE, DATASETS, checked_inputs, read_json
from benchmarks.cox_cpu_precision_timing import OUT as CPU_SOURCE
from benchmarks.cox_full_exact_gpu import atomic_json, integrate, saved_rule, validate_rule
from benchmarks.cox_gpu_experiment import environment, file_hash

OUT = CPU_SOURCE / "cpu_exact_degree"


def make_ex(data):
    ex = CoxExplainer(data["beta"], background=data["background"],
                      backend="logspace_jax", memory_budget="8GB", block_size=1)
    ex._jax = shapley.ProductGamesShapleyJax()
    ex._jax._phi_logspace_core = ex._jax._phi_logspace_parallel_core
    return ex


def warm_and_pilot(ex, x, m, nodes, weights):
    """Warm precisely the core shape used by every exact block, then time three calls."""
    import jax.numpy as jnp
    plan = ex.plan_blocks(m, backend="logspace_jax")
    K, Ut, _ = next(ex.games(x, block_size=plan.block_size))
    assert K.shape[0] == 1 and shapley._jax_work_dtype(K) == np.dtype(np.float64)
    Ut = np.broadcast_to(shapley._absent_factors_numpy(K, Ut), K.shape)
    nb = plan.node_block
    args = (jnp.asarray(K, dtype=jnp.float64), jnp.asarray(Ut, dtype=jnp.float64),
            jnp.asarray(nodes[:nb], dtype=jnp.float64),
            jnp.asarray(weights[:nb], dtype=jnp.float64), 1e-100)
    core = ex._jax._phi_logspace_parallel_core
    t = perf_counter()
    result = np.asarray(core(*args))
    warmup_seconds = perf_counter() - t
    assert result.dtype == np.float64 and np.isfinite(result).all()
    times = []
    for _ in range(3):
        t = perf_counter()
        result = np.asarray(core(*args))
        times.append(perf_counter() - t)
    return {"warmup_seconds": warmup_seconds, "pilot_core_seconds": times,
            "core_calls_per_patient": plan.n_blocks * ((m + nb - 1) // nb),
            "projected_seconds_per_patient": float(np.median(times)) * plan.n_blocks * ((m + nb - 1) // nb),
            "block_plan": asdict(plan), "core_dtype": str(result.dtype),
            "warmup_protocol": "One synchronized real first-background node block with the full exact core shape; three additional pilot calls excluded from integration timing"}


def validate_streamed_api():
    """Check the inherited checkpoint loop against the public CPU parallel API."""
    rng = np.random.default_rng(53)
    data = {"beta": rng.normal(size=19) * 0.1, "background": rng.normal(size=(3, 19))}
    x = rng.normal(size=19)
    ex = make_ex(data)
    folder = OUT / "stream_validation"
    if (folder / "validation.json").exists():
        assert read_json(folder / "validation.json")["bitwise_equal_to_public_api"]
        return
    folder.mkdir(exist_ok=True)
    m = 10
    public = ex.explain(x, m_q=m)
    got, _, _ = integrate(ex, x, m, folder, {"dataset": "stream_validation"}, interval=30)
    assert np.array_equal(got, public)
    atomic_json(folder / "validation.json", {"bitwise_equal_to_public_api": True,
                                              "d": 19, "B": 3, "m_q": m})


def run(args):
    import jax
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert not jax.config.jax_disable_jit
    OUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    env = environment(require_gpu=False)
    matched = read_json(CPU_SOURCE / "cpu_fp64_environment.json")
    assert env["packages"] == matched["packages"] and env["source_sha256"] == matched["source_sha256"]
    env.update(jax_enable_x64=True, work_dtype="float64", host_accumulation_dtype="float64",
               core_formula="parallel logspace core also used by GPU",
               benchmark_source_sha256=file_hash(__file__),
               integration_driver_sha256=file_hash(Path(__file__).with_name("cox_full_exact_gpu.py")))
    atomic_json(OUT / "environment.json", env)
    provenance = {
        "input_sha256": {n: file_hash(SOURCE / n / "inputs.npz") for n in DATASETS},
        "cpu_approx_sha256": {f: file_hash(CPU_SOURCE / f) for f in
                              ("cpu_fp64_summary.csv", "cpu_fp64_attributions.npz", "cpu_fp64_environment.json")},
        "gpu_approx_sha256": {f: file_hash(GPU_SOURCE / f) for f in
                              ("gpu_summary.csv", "gpu_attributions.npz", "gpu_environment.json")},
        "gpu_exact_sha256": {f: file_hash(SOURCE / f) for f in
                             ("summary.csv", "attributions.npz", "environment.json", "rule_setup.json")},
        "background_size": 30, "distinct_patients_per_dataset": 5,
        "measured_calls_per_patient_method": 1, "exact_cpu_measured": True,
        "timing_protocol": "Checkpointed integration reproduces public API factor/block order; active integration time excludes checkpoint IO, cached rule load/construction and kernel-shape warmup/pilot",
    }
    if (OUT / "provenance.json").exists():
        assert read_json(OUT / "provenance.json") == provenance
    else:
        atomic_json(OUT / "provenance.json", provenance)
    validate_streamed_api()
    rows = pd.read_csv(OUT / "summary.csv").to_dict("records") if (OUT / "summary.csv").exists() else []
    raw = pd.read_csv(OUT / "raw_timings.csv").to_dict("records") if (OUT / "raw_timings.csv").exists() else []
    done = {(r["dataset"], int(r["patient"])) for r in rows}
    rule_specs = read_json(SOURCE / "rule_setup.json")
    for name in args.datasets:
        data = checked_inputs(name)
        ex = make_ex(data)
        m = (len(data["beta"]) + 1) // 2
        spec = rule_specs[name]
        assert spec["exact_nodes"] == m and file_hash(spec["source"]) == spec["source_sha256"]
        with np.load(spec["source"]) as rule:
            nodes, weights = rule["nodes"], rule["weights"]
        validation = validate_rule(nodes, weights, m)
        nodes.setflags(write=False); weights.setflags(write=False)
        atomic_json(OUT / "status.json", {"phase": "warming_exact_shape", "dataset": name, "cases": len(rows)})
        pilot = warm_and_pilot(ex, data["X"][0], m, nodes, weights)
        atomic_json(OUT / f"{name}_pilot.json", {**pilot, "rule": spec, "rule_validation": validation})
        print(f"PILOT {name}: exact m={m}, node_block={pilot['block_plan']['node_block']}; "
              f"core median {np.median(pilot['pilot_core_seconds']):.4f}s; "
              f"projected per patient {pilot['projected_seconds_per_patient']:.1f}s", flush=True)
        if args.pilot:
            continue
        for patient, x in enumerate(data["X"]):
            if (name, patient) in done:
                assert (OUT / f"{name}_p{patient}_exact.npy").exists()
                continue
            folder = OUT / name / f"patient_{patient}"
            folder.mkdir(parents=True, exist_ok=True)
            manifest = {"dataset": name, "patient": patient, "patient_id": str(data["patient_ids"][patient]),
                        "d": len(x), "B": 30, "m_q": m, "core_dtype": "float64", "backend": "cpu",
                        "memory_budget": "8GB", "block_size": 1, "packages": env["packages"],
                        "source_sha256": env["source_sha256"],
                        "integration_driver_sha256": env["integration_driver_sha256"],
                        "input_sha256": provenance["input_sha256"][name], "rule_sha256": spec["source_sha256"]}
            atomic_json(OUT / "status.json", {"phase": "running", "dataset": name, "patient": patient,
                        "cases": len(rows), "progress_file": str(folder / "exact_progress.json")})
            with saved_rule(m, nodes, weights):
                phi, timer, plan = integrate(ex, x, m, folder, manifest,
                                            resume=(folder / "exact_checkpoint.npz").exists(), interval=30)
            assert phi is not None and np.isfinite(phi).all()
            np.save(OUT / f"{name}_p{patient}_exact.npy", phi)
            error = float(np.max(np.abs(phi - data["reference"][patient])))
            row = {"dataset": name, "patient": patient, "patient_id": str(data["patient_ids"][patient]),
                   "B": 30, "d": len(x), "m_q": m, "seconds": timer.compute,
                   "warmup_seconds": pilot["warmup_seconds"] if patient == 0 else 0.0,
                   "memory_budget": "8GB", "block_size": 1, "node_block": plan.node_block,
                   "core_dtype": "float64", "checkpoint_io_seconds": timer.io, "attempts": timer.attempts,
                   "max_absolute_error_vs_auxiliary_reference": error,
                   "attribution_sha256": file_hash(OUT / f"{name}_p{patient}_exact.npy")}
            rows.append(row)
            raw.append({"dataset": name, "patient": patient, "phase": "measured",
                        "seconds": timer.compute, "m_q": m, "checkpoint_io_seconds": timer.io})
            pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
            pd.DataFrame(raw).to_csv(OUT / "raw_timings.csv", index=False)
            print(f"COMPLETE {name} p{patient}: {timer.compute:.3f}s; "
                  f"auxiliary FP64reference gap={error:.6g}", flush=True)
    atomic_json(OUT / "status.json", {"phase": "pilot_complete" if args.pilot else
                "complete" if len(rows) == 10 else "partial", "cases": len(rows),
                "wall_seconds_this_invocation": perf_counter() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)
