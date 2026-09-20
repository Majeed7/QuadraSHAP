"""Checkpointed full-background Cox integration using the existing JAX core.

This driver reproduces ``CoxExplainer.explain``'s blocking, games and reductions
so long exact-degree runs can resume between node blocks. Numerical library
sources are not modified. Quadrature-rule preparation is outside integration
timing; checkpoint/progress IO is subtracted and reported separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from quadrashap import CoxExplainer
from quadrashap.product_games import shapley
from quadrashap.product_games.blocks import pad_block
from benchmarks.cox_background_experiment import load
from benchmarks.cox_gpu_experiment import environment, error_metrics, file_hash

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/cox_gpu_tolerance/eps_1e6_followup"


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_rule(nodes, weights, m):
    if nodes.shape != (m,) or weights.shape != (m,):
        raise ValueError("Saved quadrature rule has the wrong shape")
    if not (np.isfinite(nodes).all() and np.isfinite(weights).all()
            and (nodes > 0).all() and (nodes < 1).all()
            and (np.diff(nodes) > 0).all() and (weights > 0).all()):
        raise ValueError("Invalid quadrature nodes or weights")
    symmetry = max(float(np.max(np.abs(nodes + nodes[::-1] - 1))),
                   float(np.max(np.abs(weights - weights[::-1]))))
    moment = max(float(abs(weights @ nodes ** k - 1 / (k + 1)))
                 for k in (0, 1, 2, 8, 16) if k <= 2 * m - 1)
    if symmetry > 1e-12 or moment > 1e-8:
        raise ValueError(f"Rule validation failed: symmetry={symmetry}, moments={moment}")
    return {"symmetry_max_error": symmetry, "moment_max_error": moment}


def prepare_rule(dataset, m, folder, exact):
    """Use the saved TCGA rule; save a newly generated GSE rule for resumes."""
    archive = ROOT / "benchmarks/results/cox_survival/tcga_lgg_methylation/exact_quadrature_rule.npz"
    path = archive if dataset == "tcga_lgg_methylation" and exact else folder / "quadrature_rule.npz"
    metadata_path = folder / "rule_preparation.json"
    t = perf_counter()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            nodes, weights = saved["nodes"], saved["weights"]
        preparation = "Loaded saved rule"
        fresh_seconds = None
    else:
        if path == archive:
            raise FileNotFoundError(f"Required archived TCGA rule is missing: {path}")
        shapley._gauss_legendre_01_numpy.cache_clear()
        start = perf_counter()
        nodes, weights = shapley._gauss_legendre_01_numpy(m)
        fresh_seconds = perf_counter() - start
        np.savez(path, nodes=nodes, weights=weights)
        preparation = "Fresh rule generated on CPU and saved"
    validation = validate_rule(nodes, weights, m)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    current = {"source": str(path.relative_to(ROOT)), "sha256": file_hash(path),
               "m_q": m, "preparation": preparation, "fresh_rule_seconds": fresh_seconds,
               "preparation_wall_seconds": perf_counter() - t, **validation}
    if metadata_path.exists():
        original = json.loads(metadata_path.read_text())
        if original["sha256"] != current["sha256"] or original["m_q"] != m:
            raise ValueError("Quadrature rule changed since the original attempt")
        current["original_preparation"] = original.get("original_preparation", original)
    atomic_json(metadata_path, current)
    return nodes, weights, current


@contextmanager
def saved_rule(m, nodes, weights):
    """Process-local replacement, restored even if integration fails."""
    original = shapley._gauss_legendre_01_numpy

    def lookup(requested, dtype=np.float64):
        if int(requested) != m:
            return original(requested, dtype=dtype)
        return nodes.astype(dtype, copy=False), weights.astype(dtype, copy=False)

    shapley._gauss_legendre_01_numpy = lookup
    try:
        yield
    finally:
        shapley._gauss_legendre_01_numpy = original


class Progress:
    """Timer with explicit pauses for atomic checkpoint/progress writes."""
    def __init__(self, folder, manifest, resumed=None):
        self.folder, self.manifest = folder, manifest
        self.compute = float((resumed or {}).get("integration_compute_seconds", 0))
        self.io = float((resumed or {}).get("checkpoint_io_seconds", 0))
        self.generations = int((resumed or {}).get("generation", 0))
        self.attempts = int((resumed or {}).get("attempts", 0)) + 1
        self.tick = perf_counter()
        self.last = self.tick

    def checkpoint(self, phi, partial, block, node, plan, phase="running"):
        before = perf_counter()
        io_before = self.io
        self.compute += before - self.tick
        self.generations += 1
        fraction = min(1.0, (block + node / plan.m_q) / plan.n_blocks)
        state = {"manifest": self.manifest, "generation": self.generations,
                 "attempts": self.attempts, "block": block, "next_node": node,
                 "integration_compute_seconds": self.compute,
                 "checkpoint_io_seconds": self.io, "phase": phase,
                 "completed_fraction": fraction, "block_plan": asdict(plan)}
        target = self.folder / "exact_checkpoint.npz"
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.savez(f, phi=phi, partial=partial, metadata=np.array(json.dumps(state)))
        temporary.replace(target)
        state["checkpoint_io_seconds"] = io_before + perf_counter() - before
        state["active_integration_wall_seconds"] = self.compute + state["checkpoint_io_seconds"]
        atomic_json(self.folder / "exact_progress.json", state)
        print(f"{self.manifest['dataset']}: {fraction:.2%}, block {block}/{plan.n_blocks}, "
              f"node {node}/{plan.m_q}, compute {self.compute:.1f}s", flush=True)
        # Account for JSON/console progress overhead too, but not as GPU time.
        self.io = io_before + perf_counter() - before
        self.tick = self.last = perf_counter()
        return state

    def finish(self):
        self.compute += perf_counter() - self.tick
        self.tick = perf_counter()


def integrate(ex, x, m, folder, manifest, resume=False, interval=30.0, max_node_blocks=None):
    """Same block order, padding, GPU core and float64 host sums as public API."""
    import jax.numpy as jnp

    checkpoint = folder / "exact_checkpoint.npz"
    restored, start_block, start_node = None, 0, 0
    phi, partial = np.zeros(ex.d, dtype=np.float64), np.empty((0, ex.d))
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Checkpoint exists; pass --resume: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as saved:
            restored = json.loads(str(saved["metadata"]))
            phi, partial = saved["phi"], saved["partial"]
        if restored["manifest"] != manifest:
            raise ValueError("Refusing resume: source, inputs, rule or settings changed")
        progress_path = folder / "exact_progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("generation") == restored["generation"]:
                restored["checkpoint_io_seconds"] = progress["checkpoint_io_seconds"]
        start_block, start_node = restored["block"], restored["next_node"]
    elif resume:
        raise FileNotFoundError("--resume requested but checkpoint is absent")

    timer = Progress(folder, manifest, restored)
    plan = ex.plan_blocks(m, backend="logspace_jax")
    ex.last_block_plan = plan
    if restored and restored["block_plan"] != asdict(plan):
        raise ValueError("Block plan changed; refuse to mix numerical reduction orders")
    core = shapley.ProductGamesShapleyJax._phi_logspace_parallel_core
    nodes, weights = shapley._gauss_legendre_01_numpy(m)
    seen = 0
    for block, (K, Ut, w) in enumerate(ex.games(x, block_size=plan.block_size)):
        if block < start_block:
            continue
        if plan.n_blocks > 1:
            K, Ut, w = pad_block(K, Ut, w, plan.block_size)
        if (Ut == 0).any() or ((K + Ut) == 0).any():
            raise ValueError("Unexpected vanishing Cox factors; inspect before exact run")
        Ut = np.broadcast_to(shapley._absent_factors_numpy(K, Ut), K.shape)
        dtype = shapley._jax_work_dtype(K)
        K = K.astype(dtype, copy=False)
        Kj, Utj = jnp.asarray(K, dtype=dtype), jnp.asarray(Ut, dtype=dtype)
        offset = start_node if block == start_block else 0
        if not offset:
            partial = np.zeros(K.shape, dtype=np.float64)
        elif partial.shape != K.shape:
            raise ValueError("Partial block checkpoint shape mismatch")
        nb = max(1, min(int(plan.node_block), m))
        for q0 in range(offset, m, nb):
            xb, wb = nodes[q0:q0 + nb], weights[q0:q0 + nb]
            if len(xb) < nb:
                pad = nb - len(xb)
                xb = np.concatenate([xb, np.full(pad, 0.5)])
                wb = np.concatenate([wb, np.zeros(pad)])
            result = core(Kj, Utj, jnp.asarray(xb, dtype=dtype), jnp.asarray(wb, dtype=dtype), 1e-100)
            partial += np.asarray(result)  # synchronization and host float64 sum
            seen += 1
            next_node = min(q0 + nb, m)
            should_pause = max_node_blocks is not None and seen >= max_node_blocks
            if next_node < m and (should_pause or perf_counter() - timer.last >= interval):
                timer.checkpoint(phi, partial, block, next_node, plan,
                                 phase="paused" if should_pause else "running")
                if should_pause:
                    return None, timer, plan
        phi += (partial * w[:, None]).sum(axis=0)
        partial = np.empty((0, ex.d))
        if not np.isfinite(phi).all():
            raise FloatingPointError("Nonfinite accumulated attribution")
        should_pause = max_node_blocks is not None and seen >= max_node_blocks
        if should_pause or perf_counter() - timer.last >= interval:
            timer.checkpoint(phi, partial, block + 1, 0, plan,
                             phase="paused" if should_pause else "running")
            if should_pause and block + 1 < plan.n_blocks:
                return None, timer, plan
    timer.checkpoint(phi, partial, plan.n_blocks, 0, plan, phase="complete")
    timer.finish()
    return phi, timer, plan


def run(args):
    exp = load(args.dataset)
    exact_m = (len(exp["beta"]) + 1) // 2
    m = args.nodes or exact_m
    smoke = args.nodes is not None
    if args.background_limit and not smoke:
        raise ValueError("--background-limit is allowed only with --nodes smoke runs")
    bg = exp["training"][:args.background_limit] if args.background_limit else exp["training"]
    rule_folder = OUTPUT / args.dataset
    if smoke:
        rule_folder /= f"smoke_m{m}_b{len(bg)}"
    folder = rule_folder / f"patient_{args.patient}"
    folder.mkdir(parents=True, exist_ok=True)
    env = environment(require_gpu=True)
    nodes, weights, rule = prepare_rule(args.dataset, m, rule_folder, exact=not smoke)
    sources = {**env["source_sha256"], str(Path(__file__).relative_to(ROOT)): file_hash(__file__)}
    inputs = {str((exp["folder"] / name).relative_to(ROOT)): file_hash(exp["folder"] / name)
              for name in ("inputs.npz", "training_background.npy", "reference_full_64.npy")}
    manifest = {"dataset": args.dataset, "patient": args.patient, "patient_id": str(exp["patient_ids"][args.patient]),
                "d": len(exp["beta"]), "background_size": len(bg), "m_q": m,
                "exact_nodes": exact_m, "exact_degree": not smoke,
                "backend": "logspace_jax", "memory_budget": args.memory_budget,
                "block_size": args.block_size,
                "source_sha256": sources, "input_sha256": inputs,
                "rule_sha256": rule["sha256"], "packages": env["packages"],
                "device": env["devices"], "blas_threads": 4}
    atomic_json(folder / "exact_environment.json", env)
    ex = CoxExplainer(exp["beta"], background=bg, backend="logspace_jax",
                      memory_budget=args.memory_budget, block_size=args.block_size)
    with saved_rule(m, nodes, weights):
        phi, timer, plan = integrate(ex, exp["X"][args.patient], m, folder, manifest,
                                    resume=args.resume, interval=args.checkpoint_seconds,
                                    max_node_blocks=args.max_node_blocks)
        if phi is None:
            return
        metrics = {}
        if len(bg) == len(exp["training"]):
            reference = np.load(exp["folder"] / "reference_full_64.npy", mmap_mode="r")[args.patient]
            metrics = error_metrics(phi, reference)
            reference_label = "Independent float64 64-node quadrature with certified bound <1e-10; NOT exact-degree reference"
        else:
            reference_label = "Smoke test with subset background; no full-background reference comparison"
        if smoke:
            public = ex.explain(exp["X"][args.patient], m_q=m)
            np.testing.assert_array_equal(phi, public)
            metrics["bitwise_equal_to_public_api"] = True
    record = {**manifest, "repeats": 1, "block_plan": asdict(plan),
              "integration_seconds": timer.compute, "checkpoint_io_seconds": timer.io,
              "active_integration_wall_seconds": timer.compute + timer.io,
              "attempts": timer.attempts, "rule_preparation": rule,
              "timing_scope": "Factor preparation, planning, GPU transfers/JIT/core, synchronized host float64 accumulation; excludes explainer construction, rule preparation, checkpoint IO, validation and resume input loads",
              "resume_timing_scope": "Accumulated recorded active work across attempts; excludes downtime and any uncheckpointed work lost to interruption",
              "reference": reference_label, "qualification": "Full exact-degree quadrature in real arithmetic; GPU FP32 with host FP64 block accumulation" if not smoke else "Non-exact bounded smoke; explicit node override",
              "efficiency_residual": float(abs(phi.sum() - (ex.model.predict_scale(exp["X"][args.patient:args.patient + 1])[0] - ex.expected_value))),
              **metrics}
    np.save(folder / ("smoke_attributions.npy" if smoke else f"exact_gpu_attributions_patient_{args.patient}.npy"), phi)
    atomic_json(folder / ("smoke_timing.json" if smoke else "exact_gpu_timing.json"), record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("gse24080", "tcga_lgg_methylation"))
    parser.add_argument("--patient", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--memory-budget", default="512MB")
    parser.add_argument("--block-size", default="auto", type=lambda v: "auto" if v == "auto" else int(v))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-seconds", type=float, default=30.0)
    parser.add_argument("--nodes", type=int, help="Positive override for isolated smoke results; default is full exact rule")
    parser.add_argument("--background-limit", type=int, help="First B training rows, allowed only for smoke runs")
    parser.add_argument("--max-node-blocks", type=int, help="Pause after this many node blocks and save resumable state")
    args = parser.parse_args()
    if not 0 < args.checkpoint_seconds <= 30:
        parser.error("--checkpoint-seconds must be in (0,30]")
    if any(v is not None and v < 1 for v in (args.nodes, args.background_limit, args.max_node_blocks)):
        parser.error("node/background/block counts must be positive")
    if args.block_size != "auto" and args.block_size < 1:
        parser.error("--block-size must be auto or a positive integer")
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
