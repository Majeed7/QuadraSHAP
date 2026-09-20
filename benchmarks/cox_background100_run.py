"""Matched B100 GPU FP32 / CPU FP64 Cox experiments, with resumable exact runs.

Run ``all`` to execute GPU then CPU in separate processes without concurrent
benchmark work. Completed case records are immutable; interrupted exact cases
resume only with matching inputs, sources, packages, precision and rules.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games import shapley
from benchmarks.cox_background100_inputs import DATASETS, OUT, checked_inputs
from benchmarks.cox_full_exact_gpu import atomic_json, integrate, saved_rule, validate_rule
from benchmarks.cox_gpu_experiment import ROOT, environment, file_hash

MODES = (("1e-1", 0.1), ("1e-2", 0.01), ("1e-3", 0.001))
B30 = ROOT / "benchmarks/results/cox_background30_gpu"


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    """An OS lock prevents a second benchmark from using the same output tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another benchmark still holds {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def unchanged_json(path, record):
    if path.exists():
        if read_json(path) != record:
            raise ValueError(f"Refusing to mix changed study settings: {path}")
    else:
        atomic_json(path, record)


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def make_ex(data, memory_budget="8GB"):
    ex = CoxExplainer(data["beta"], background=data["background"],
                      backend="logspace_jax", memory_budget=memory_budget, block_size=1)
    ex._jax = shapley.ProductGamesShapleyJax()
    ex._jax._phi_logspace_core = ex._jax._phi_logspace_parallel_core
    return ex


def prepare_rules():
    rules, metadata = {}, {}
    for dataset, record in read_json(B30 / "rule_setup.json").items():
        path = Path(record["source"])
        assert file_hash(path) == record["source_sha256"]
        tick = perf_counter()
        with np.load(path, allow_pickle=False) as saved:
            nodes, weights = saved["nodes"], saved["weights"]
        load_seconds = perf_counter() - tick
        m = int(record["exact_nodes"])
        validation = validate_rule(nodes, weights, m)
        nodes.setflags(write=False)
        weights.setflags(write=False)
        rules[dataset] = nodes, weights
        metadata[dataset] = {**record, **validation, "cached_rule_load_seconds": load_seconds,
                             "construction_timing_provenance": "Historical saved rule; not newly constructed for B100"}
    path = OUT / "rule_setup.json"
    if path.exists():
        prior = read_json(path)
        for dataset in DATASETS:
            assert prior[dataset]["source_sha256"] == metadata[dataset]["source_sha256"]
            assert prior[dataset]["exact_nodes"] == metadata[dataset]["exact_nodes"]
    else:
        atomic_json(path, metadata)
    return rules


def validate_streamed_api(folder, dtype):
    """Small forced-padding check of the exact loop and a within-block resume."""
    target = folder / "stream_validation"
    if (target / "validation.json").exists():
        prior = read_json(target / "validation.json")
        assert prior["bitwise_equal_to_public_api"] and prior["bitwise_resume_equal"]
        assert prior["core_dtype"] == str(dtype)
        return
    rng = np.random.default_rng(5300)
    data = {"beta": rng.normal(size=19) * 0.1, "background": rng.normal(size=(3, 19))}
    x = rng.normal(size=19)
    ex = make_ex(data, memory_budget="8KB")
    m = 10
    public = ex.explain(x, m_q=m)
    for name in ("continuous", "resumed"):
        (target / name).mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "small_stream_validation", "core_dtype": str(dtype)}
    continuous, _, plan = integrate(ex, x, m, target / "continuous", manifest,
                                    resume=(target / "continuous/exact_checkpoint.npz").exists())
    assert 1 < plan.node_block < m and m % plan.node_block
    resumed_path = target / "resumed"
    if not (resumed_path / "exact_checkpoint.npz").exists():
        partial, _, _ = integrate(ex, x, m, resumed_path, manifest, max_node_blocks=1)
        assert partial is None
    resumed, _, _ = integrate(ex, x, m, resumed_path, manifest, resume=True)
    np.testing.assert_array_equal(continuous, public)
    np.testing.assert_array_equal(resumed, public)
    atomic_json(target / "validation.json", {"bitwise_equal_to_public_api": True,
                "bitwise_resume_equal": True, "core_dtype": str(dtype), "d": 19,
                "B": 3, "m_q": m, "node_block": plan.node_block})


def warm_exact(ex, x, nodes, weights, dtype):
    import jax.numpy as jnp
    m = len(nodes)
    plan = ex.plan_blocks(m, backend="logspace_jax")
    K, Ut, _ = next(ex.games(x, block_size=plan.block_size))
    Ut = np.broadcast_to(shapley._absent_factors_numpy(K, Ut), K.shape)
    nb = plan.node_block
    args = (jnp.asarray(K, dtype=dtype), jnp.asarray(Ut, dtype=dtype),
            jnp.asarray(nodes[:nb], dtype=dtype), jnp.asarray(weights[:nb], dtype=dtype), 1e-100)
    tick = perf_counter()
    result = np.asarray(ex._jax._phi_logspace_parallel_core(*args))
    warmup = perf_counter() - tick
    assert result.dtype == dtype and np.isfinite(result).all()
    times = []
    for _ in range(3):
        tick = perf_counter()
        result = np.asarray(ex._jax._phi_logspace_parallel_core(*args))
        times.append(perf_counter() - tick)
    calls = plan.n_blocks * ((m + nb - 1) // nb)
    return {"warmup_seconds": warmup, "pilot_core_seconds": times,
            "projected_seconds_per_patient": float(np.median(times)) * calls,
            "core_calls_per_patient": calls, "block_plan": asdict(plan), "core_dtype": str(dtype),
            "protocol": "One real exact-shape core warm-up and three pilot core calls, all excluded from integration timing"}


def run_device(device):
    import jax
    import jax.numpy as jnp
    folder = OUT / device
    with lock(OUT / ".device.lock"):
        folder.mkdir(parents=True, exist_ok=True)
        for sub in ("cases", "attributions", "checkpoints"):
            (folder / sub).mkdir(exist_ok=True)
        dtype = np.dtype(np.float32 if device == "gpu" else np.float64)
        env = environment(require_gpu=device == "gpu")
        assert env["backend"].lower() == ("metal" if device == "gpu" else "cpu")
        assert bool(jax.config.jax_enable_x64) == (device == "cpu")
        assert not jax.config.jax_disable_jit
        previous = read_json(B30 / "environment.json")
        assert env["source_sha256"] == previous["source_sha256"]
        assert env["packages"] == previous["packages"]
        extras = ["src/quadrashap/product_games/budget.py", "src/quadrashap/cox.py",
                  "benchmarks/cox_full_exact_gpu.py", "benchmarks/cox_background100_run.py",
                  "benchmarks/cox_background100_inputs.py"]
        env.update(jax_enable_x64=bool(jax.config.jax_enable_x64), core_dtype=str(dtype),
                   work_dtype=str(dtype), host_accumulation_dtype="float64",
                   benchmark_source_sha256=file_hash(__file__),
                   extra_source_sha256={p: file_hash(ROOT / p) for p in extras},
                   thread_environment={key: os.environ.get(key) for key in
                                       ("XLA_FLAGS", "OMP_NUM_THREADS", "JAX_NUM_THREADS")})
        if (folder / "environment.json").exists():
            old_env = read_json(folder / "environment.json")
            for key in ("backend", "packages", "source_sha256", "extra_source_sha256", "core_dtype", "jax_enable_x64", "thread_environment"):
                assert old_env[key] == env[key], f"Changed environment on resume: {key}"
        else:
            atomic_json(folder / "environment.json", env)
        rules = prepare_rules()
        provenance = {"input_sha256": {n: file_hash(OUT / n / "inputs.npz") for n in DATASETS},
                      "design_sha256": file_hash(OUT / "design.json"),
                      "source_sha256": env["source_sha256"], "extra_source_sha256": env["extra_source_sha256"],
                      "rule_setup_sha256": file_hash(OUT / "rule_setup.json"),
                      "background_size": 100, "distinct_patients_per_dataset": 5,
                      "tolerances": [eps for _, eps in MODES], "measured_calls_per_patient_method": 1,
                      "exact_references": "New B100 full degree-exact results, same device and precision",
                      "approximate_timing": "Synchronized public API including node-budget selection, transfers and host accumulation; excludes rule construction and per-shape warm-up",
                      "exact_timing": "Accumulated active checkpointed integration including factors, planning, core calls and host sums; excludes rule preparation, kernel warm-up/pilots, checkpoint I/O, initial output allocations, resume loading, downtime and lost uncheckpointed work"}
        unchanged_json(folder / "provenance.json", provenance)
        probe = shapley.ProductGamesShapleyJax._phi_logspace_parallel_core(
            jnp.ones((1, 2), dtype=dtype), jnp.ones((1, 2), dtype=dtype),
            jnp.asarray([0.5], dtype=dtype), jnp.asarray([1.0], dtype=dtype), 1e-100)
        assert np.asarray(probe).dtype == dtype
        validate_streamed_api(folder, dtype)
        rows = {}
        design = read_json(OUT / "design.json")
        expected = {(name, patient, mode) for name in DATASETS for patient in range(5)
                    for mode in ("exact", *(label for label, _ in MODES))}
        for path in sorted((folder / "cases").glob("*.json")):
            record = read_json(path)
            key = record["dataset"], int(record["patient"]), record["mode"]
            assert key in expected and key not in rows
            name, patient, mode = key
            assert record["B"] == 100 and record["d"] == design[name]["d"]
            assert record["patient_id"] == str(design[name]["patient_ids"][patient])
            assert record["core_dtype"] == str(dtype) and record["block_size"] == 1 and record["memory_budget"] == "8GB"
            assert record["m_q"] == (record["d"] + 1) // 2 if mode == "exact" else record["eps"] == dict(MODES)[mode]
            assert file_hash(folder / record["attribution_file"]) == record["attribution_sha256"]
            rows[key] = record
        start = perf_counter()
        state = {"phase": "starting", "pid": os.getpid(), "cases": len(rows), "device": device}

        def status(**fields):
            state.update(fields, cases=len(rows), pid=os.getpid())
            atomic_json(folder / "status.json", state)

        def persist(phi, record):
            dataset, patient, mode = record["dataset"], record["patient"], record["mode"]
            key = dataset, patient, mode
            assert key not in rows and phi.shape == (record["d"],)
            assert phi.dtype == np.float64 and np.isfinite(phi).all()
            name = f"{dataset}_p{patient}_{mode}"
            destination = folder / "attributions" / f"{name}.npy"
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, phi)
            temporary.replace(destination)
            record.update(attribution_file=str(destination.relative_to(folder)),
                          attribution_sha256=file_hash(destination))
            atomic_json(folder / "cases" / f"{name}.json", record)
            rows[key] = record
            write_csv(folder / "summary.csv", list(rows.values()))
            print(f"COMPLETE {device} {dataset} p{patient} {mode}: {record['seconds']:.6f}s; m={record['m_q']}", flush=True)

        original_rule = shapley._gauss_legendre_01_numpy
        setup_path = folder / "small_rule_timings.csv"
        setup_rows = pd.read_csv(setup_path).to_dict("records") if setup_path.exists() else []
        done_setup = {(r["dataset"], int(r["m_q"])) for r in setup_rows}
        try:
            for dataset in DATASETS:
                data = checked_inputs(dataset)
                beta, X, ids = (data[k] for k in ("beta", "X", "patient_ids"))
                assert len(data["background"]) == 100 and len(X) == 5
                ex = make_ex(data)
                assert shapley._jax_work_dtype(beta) == dtype
                # The enlarged game is checked before launching expensive full rules.
                for x in X:
                    for K, Ut, weights in ex.games(x, block_size=1):
                        assert np.isfinite(K).all() and np.isfinite(Ut).all()
                        assert not (Ut == 0).any() and not ((K + Ut) == 0).any()
                        assert np.isfinite(weights).all() and np.allclose(weights, 0.01, rtol=0, atol=1e-15)
                budgets = {(p, mode): ex.node_budget(x, eps=eps)
                           for mode, eps in MODES for p, x in enumerate(X)}
                small_rules = {int(r.m_q): original_rule(int(r.m_q)) for r in budgets.values()}
                warmed = set()
                for mode, eps in MODES:
                    for patient, x in enumerate(X):
                        if (dataset, patient, mode) in rows:
                            continue
                        budget = budgets[(patient, mode)]
                        m = int(budget.m_q)
                        assert budget.bound <= eps
                        nodes, weights = small_rules[m]
                        status(phase="running", dataset=dataset, patient=patient, mode=mode, call_phase="warmup")
                        with saved_rule(m, nodes, weights):
                            warmup = 0.0
                            if m not in warmed:
                                tick = perf_counter()
                                warm_phi = ex.explain(x, m_q=m)
                                warmup = perf_counter() - tick
                                assert np.isfinite(warm_phi).all()
                                warmed.add(m)
                            status(call_phase="measured")
                            tick = perf_counter()
                            phi, report = ex.explain(x, eps=eps, return_report=True)
                            seconds = perf_counter() - tick
                        assert report.m_q == m and report.bound <= eps and ex.last_block_plan.m_q == m
                        assert np.isclose(report.bound, budget.bound, rtol=1e-12, atol=0)
                        persist(phi, {"dataset": dataset, "patient": patient, "patient_id": str(ids[patient]),
                                "B": 100, "d": len(beta), "mode": mode, "eps": eps, "m_q": m,
                                "seconds": seconds, "warmup_seconds": warmup, "bound": float(report.bound),
                                "budget_seconds": float(report.seconds), "core_dtype": str(dtype),
                                "memory_budget": "8GB", "block_size": 1, "node_block": ex.last_block_plan.node_block,
                                "checkpoint_io_seconds": 0.0, "attempts": 1})
                # Measure only setup here; these calls never enter explanation timing.
                for m in sorted(small_rules):
                    if (dataset, m) in done_setup:
                        continue
                    construction = []
                    for _ in range(5):
                        original_rule.cache_clear()
                        tick = perf_counter()
                        original_rule(m)
                        construction.append(perf_counter() - tick)
                    setup_rows.append({"dataset": dataset, "m_q": m,
                        "median_seconds": float(np.median(construction)), "min_seconds": min(construction),
                        "max_seconds": max(construction), "repeats": 5})
                write_csv(folder / "small_rule_timings.csv", setup_rows)
                pending = [p for p in range(5) if (dataset, p, "exact") not in rows]
                if not pending:
                    continue
                nodes, weights = rules[dataset]
                m = (len(beta) + 1) // 2
                assert len(nodes) == m
                status(phase="warming_exact_shape", dataset=dataset, mode="exact", patient=pending[0])
                pilot = warm_exact(ex, X[pending[0]], nodes, weights, dtype)
                attempts_dir = folder / "pilot_attempts"
                attempts_dir.mkdir(exist_ok=True)
                pilot_name = f"{dataset}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
                atomic_json(attempts_dir / pilot_name, pilot)
                print(f"PILOT {device} {dataset}: projected exact per patient {pilot['projected_seconds_per_patient']:.1f}s", flush=True)
                for patient in pending:
                    checkpoint_folder = folder / "checkpoints" / dataset / f"patient_{patient}"
                    checkpoint_folder.mkdir(parents=True, exist_ok=True)
                    manifest = {"dataset": dataset, "patient": patient, "patient_id": str(ids[patient]),
                                "d": len(beta), "B": 100, "m_q": m, "core_dtype": str(dtype),
                                "backend": env["backend"], "memory_budget": "8GB", "block_size": 1,
                                "packages": env["packages"], "source_sha256": env["source_sha256"],
                                "extra_source_sha256": env["extra_source_sha256"],
                                "thread_environment": env["thread_environment"],
                                "input_sha256": provenance["input_sha256"][dataset],
                                "rule_sha256": read_json(OUT / "rule_setup.json")[dataset]["source_sha256"]}
                    status(phase="running", patient=patient, mode="exact", call_phase="measured",
                           progress_file=str(checkpoint_folder / "exact_progress.json"))
                    with saved_rule(m, nodes, weights):
                        phi, timer, plan = integrate(ex, X[patient], m, checkpoint_folder, manifest,
                            resume=(checkpoint_folder / "exact_checkpoint.npz").exists(), interval=30)
                    assert phi is not None
                    persist(phi, {"dataset": dataset, "patient": patient, "patient_id": str(ids[patient]),
                            "B": 100, "d": len(beta), "mode": "exact", "eps": None, "m_q": m,
                            "seconds": timer.compute, "warmup_seconds": pilot["warmup_seconds"] if patient == pending[0] else 0.0,
                            "bound": 0.0, "budget_seconds": 0.0, "core_dtype": str(dtype),
                            "memory_budget": "8GB", "block_size": 1, "node_block": plan.node_block,
                            "checkpoint_io_seconds": timer.io, "attempts": timer.attempts})
            assert set(rows) == expected
            assert environment(require_gpu=device == "gpu")["source_sha256"] == env["source_sha256"]
            for path, sha in env["extra_source_sha256"].items():
                assert file_hash(ROOT / path) == sha
            status(phase="complete", measured_calls=40, wall_seconds_this_invocation=perf_counter() - start)
        except BaseException as exc:
            status(phase="interrupted", error=repr(exc))
            raise


def sequence():
    """Run one device at a time and keep idle sleep disabled only while running."""
    OUT.mkdir(parents=True, exist_ok=True)
    with lock(OUT / ".sequence.lock"):
        keep_awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
        state = {"phase": "starting", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(OUT / "run_status.json", state)
        child = None
        try:
            for device in ("gpu", "cpu"):
                path = OUT / device / "status.json"
                if path.exists() and read_json(path).get("phase") == "complete":
                    assert read_json(path)["cases"] == 40
                    continue
                child_env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
                             "JAX_ENABLE_X64": "0" if device == "gpu" else "1"}
                if device == "gpu":
                    child_env.pop("JAX_PLATFORMS", None)
                    child_env.pop("JAX_PLATFORM_NAME", None)
                    child_env["ENABLE_PJRT_COMPATIBILITY"] = "1"
                else:
                    child_env["JAX_PLATFORMS"] = "cpu"
                child = subprocess.Popen([sys.executable, "-u", "-m", "benchmarks.cox_background100_run", device],
                                         cwd=ROOT, env=child_env)
                state.update(phase="running", device=device, child_pid=child.pid)
                atomic_json(OUT / "run_status.json", state)
                code = child.wait()
                if code:
                    raise RuntimeError(f"{device} benchmark exited with code {code}")
            state.update(phase="complete", completed_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(OUT / "run_status.json", state)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                except ProcessLookupError:
                    pass
            state.update(phase="interrupted", error=repr(exc))
            atomic_json(OUT / "run_status.json", state)
            raise
        finally:
            keep_awake.terminate()
            keep_awake.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "gpu", "cpu"))
    args = parser.parse_args()
    if args.action == "all":
        sequence()
    else:
        with threadpool_limits(limits=4):
            run_device(args.action)
