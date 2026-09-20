"""KernelSHAP and SamplingSHAP on fixed 30-background text-classifier games.

These are conventional *interventional* SHAP games and must not be compared
as observed errors against the neutral-factor CPU reference from exp11.
Each explainer gets nsamples=1000, one fixed 30-training-text background,
and an independent 300-second per-method/per-text process cap.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import exp10_pkex_metal_imdb as exp11

DATASETS = ("imdb", "rotten_tomatoes", "sst2", "sms_spam", "emotion")
N_BACKGROUND = 30
N_SAMPLES = 1000
SEED = 20260916


def result_dir(dataset: str) -> Path:
    return ROOT / "experiments" / "results" / f"exp12_shap_background_{dataset}"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare(args: argparse.Namespace) -> None:
    exp11.set_dataset(args.dataset)
    out = result_dir(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    train_text, y_train, _, _, source = exp11.load_full_corpus(args.dataset, SEED)
    idx = exp11.stratified_indices(y_train, N_BACKGROUND, SEED)
    bundle = joblib.load(exp11.MODEL_PATH)
    vectorizer = bundle["vectorizer"]
    background = vectorizer.transform([train_text[i] for i in idx]).toarray().astype(np.float64)
    assert background.shape == (N_BACKGROUND, 5000)
    np.savez_compressed(out / "background.npz", X=background, train_indices=idx, labels=y_train[idx])
    exp11.atomic_json(out / "prepare_meta.json", {
        "dataset": args.dataset,
        "n_background": N_BACKGROUND,
        "background_sampling": "fixed stratified training sample without replacement",
        "seed": SEED,
        "background_indices": idx.tolist(),
        "background_labels": y_train[idx].tolist(),
        "background_nonzero_features": int(np.count_nonzero(np.any(background != 0, axis=0))),
        "model_path": str(exp11.MODEL_PATH),
        "model_sha256": exp11.file_sha256(exp11.MODEL_PATH),
        "source_train": str(source["train"]),
        "source_train_sha256": exp11.file_sha256(source["train"]),
        "instances_path": str(exp11.INSTANCES_PATH),
        "instance_inputs_sha256": exp11.file_sha256(exp11.INSTANCES_PATH),
        "value_function": "30-background interventional feature replacement, uniform weights",
        "shap_version": importlib.metadata.version("shap"),
    })
    log(f"saved {N_BACKGROUND} balanced training backgrounds for {args.dataset}")


def worker(args: argparse.Namespace) -> None:
    import shap

    exp11.set_dataset(args.dataset)
    out = result_dir(args.dataset)
    bundle = joblib.load(exp11.MODEL_PATH)
    arrays = np.load(exp11.INSTANCES_PATH)
    bg = np.load(out / "background.npz")["X"].astype(np.float64)
    x = arrays["X"][args.instance].astype(np.float64)
    target_class = (int(arrays["target_classes"][args.instance])
                    if "target_classes" in arrays.files else 0)
    model = exp11.selected_model(bundle, target_class)
    counts = {"calls": 0, "rows": 0}
    varying_features = int(np.count_nonzero(np.any(bg != x[None, :], axis=0)))
    if args.method == "sampling" and 2 * varying_features > N_SAMPLES:
        raise ValueError(
            f"SamplingExplainer cannot allocate one paired sample per varying feature: "
            f"nsamples={N_SAMPLES}, varying_features={varying_features}; "
            f"at least {2 * varying_features} samples are needed by SHAP 0.51.0"
        )

    def score(matrix):
        matrix = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
        counts["calls"] += 1
        counts["rows"] += matrix.shape[0]
        return np.asarray(model.decision_function(matrix), dtype=np.float64).reshape(-1)

    np.random.seed(SEED + args.instance)
    t0 = time.perf_counter()
    if args.method == "kernel":
        explainer = shap.KernelExplainer(score, bg, link="identity")
    else:
        explainer = shap.SamplingExplainer(score, bg)
    init_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    phi = np.asarray(explainer.shap_values(x[None, :], nsamples=N_SAMPLES, silent=True), dtype=np.float64).reshape(-1)
    explain_seconds = time.perf_counter() - t1
    total_seconds = time.perf_counter() - t0
    score_x = float(score(x[None, :])[0])
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    efficiency = abs(float(phi.sum()) - (score_x - base_value))
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    phi_path = raw / f"{args.method}_instance{args.instance:03d}.npz"
    np.savez_compressed(phi_path, phi=phi)
    row = {
        "dataset": args.dataset,
        "method_key": args.method,
        "method": "KernelSHAP" if args.method == "kernel" else "SamplingSHAP",
        "instance": args.instance,
        "status": "complete",
        "d": len(phi),
        "n_background": len(bg),
        "requested_nsamples": N_SAMPLES,
        "actual_nsamples": int(getattr(explainer, "nsamples", N_SAMPLES)),
        "actual_model_calls": counts["calls"],
        "actual_model_rows": counts["rows"],
        "target_class": target_class,
        "n_varying_features": varying_features,
        "seconds_init": init_seconds,
        "seconds_explain": explain_seconds,
        "seconds_end_to_end": total_seconds,
        "model_score": score_x,
        "base_value": base_value,
        "efficiency_target": score_x - base_value,
        "efficiency_residual": efficiency,
        "phi_l2": float(np.linalg.norm(phi)),
        "phi_max_abs": float(np.max(np.abs(phi))),
        "n_nonzero_phi": int(np.count_nonzero(phi)),
        "finite": bool(np.isfinite(phi).all()),
        "certified_bound": None,
        "observed_error_to_interventional_reference": None,
        "reference_scope": "no exact 30-background interventional reference in this experiment",
        "kernel_l1_reg": "num_features(10)" if args.method == "kernel" else None,
        "phi_path": str(phi_path),
    }
    exp11.atomic_json(raw / f"{args.method}_instance{args.instance:03d}.json", row)
    log(f"{args.dataset} {args.method} text {args.instance}: {total_seconds:.2f}s, "
        f"model rows {counts['rows']}, finite={row['finite']}, efficiency={efficiency:.2e}")


def run(args: argparse.Namespace) -> None:
    out = result_dir(args.dataset)
    if not (out / "background.npz").exists():
        raise FileNotFoundError(f"prepare {args.dataset} first")
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    exp11.atomic_json(out / "run_meta.json", {
        "dataset": args.dataset,
        "methods": args.methods,
        "n_instances_requested": args.n_instances,
        "timeout_seconds_per_method_per_instance": args.timeout,
        "n_background": N_BACKGROUND,
        "nsamples": N_SAMPLES,
        "shap_version": importlib.metadata.version("shap"),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "script_sha256": exp11.file_sha256(script),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    for method in args.methods:
        for instance in range(args.n_instances):
            status_path = raw / f"status_{method}_instance{instance:03d}.json"
            if status_path.exists() and not args.force:
                log(f"cache hit: {status_path.name}")
                continue
            cmd = [sys.executable, str(script), "worker", "--dataset", args.dataset,
                   "--method", method, "--instance", str(instance)]
            log(f"starting {args.dataset} {method} text {instance}, cap {args.timeout}s")
            t0 = time.perf_counter()
            status, returncode = "complete", None
            log_path = raw / f"log_{method}_instance{instance:03d}.txt"
            try:
                completed = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(),
                                           capture_output=True, text=True, timeout=args.timeout)
                log_path.write_text(completed.stdout + completed.stderr)
                returncode = completed.returncode
                if returncode != 0:
                    status = "error"
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                captured = exc.stdout or b""
                if isinstance(captured, bytes):
                    captured = captured.decode("utf-8", "replace")
                log_path.write_text(captured + f"\nTIMEOUT after {args.timeout} seconds\n")
            exp11.atomic_json(status_path, {
                "dataset": args.dataset, "method_key": method, "instance": instance,
                "status": status, "wall_seconds": time.perf_counter() - t0,
                "timeout_seconds": args.timeout, "returncode": returncode,
                "log_path": str(log_path),
            })
            log(f"finished {args.dataset} {method} text {instance}: {status}")
    summarize(argparse.Namespace(dataset=args.dataset))


def summarize(args: argparse.Namespace) -> None:
    out = result_dir(args.dataset)
    raw = out / "raw"
    rows = [json.loads(p.read_text()) for p in sorted(raw.glob("kernel_instance*.json"))]
    rows += [json.loads(p.read_text()) for p in sorted(raw.glob("sampling_instance*.json"))]
    keys = {(r["method_key"], r["instance"]) for r in rows}
    for path in sorted(raw.glob("status_*_instance*.json")):
        status = json.loads(path.read_text())
        key = (status["method_key"], status["instance"])
        if key not in keys:
            rows.append({
                "dataset": args.dataset, "method_key": key[0], "instance": key[1],
                "status": status["status"], "wall_seconds": status["wall_seconds"],
                "requested_nsamples": N_SAMPLES, "n_background": N_BACKGROUND,
                "observed_error_to_interventional_reference": None,
                "certified_bound": None, "finite": None,
            })
    rows.sort(key=lambda r: (r["method_key"], r["instance"]))
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (out / "records.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    log(f"assembled {len(rows)} rows for {args.dataset}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for action in ("prepare", "run", "worker", "summarize"):
        p = sub.add_parser(action)
        p.add_argument("--dataset", choices=DATASETS, required=True)
        if action == "worker":
            p.add_argument("--method", choices=("kernel", "sampling"), required=True)
            p.add_argument("--instance", type=int, required=True)
        if action == "run":
            p.add_argument("--methods", nargs="+", choices=("kernel", "sampling"), default=["kernel", "sampling"])
            p.add_argument("--n-instances", type=int, default=20)
            p.add_argument("--timeout", type=float, default=300)
            p.add_argument("--force", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
    try:
        {"prepare": prepare, "run": run, "worker": worker, "summarize": summarize}[args.command](args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
