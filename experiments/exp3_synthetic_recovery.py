"""Experiment 3 v2: synthetic recovery, estimator fidelity, and cost.

Default: a separate pilot, not the publication run. See README_exp3.md.
Each estimator is an isolated, time-limited job. Completed jobs are resumable.
No manuscript edits, downloads, AI calls, GPU, or silent baseline fallbacks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import sklearn
from scipy.special import roots_legendre
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quadrashap.multiplicative.rkhs import RKHSExplainer
from quadrashap.product_games.budget import GameSummary, budget_from_summary
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy

VERSION = 2
PROFILES = {
    "quick": dict(d=20, n_train=160, n_val=80, n_test=100, n_instances=3,
                  n_background=2, seeds=[97], sampler_repeats=1,
                  kernel_multipliers=[2, 4], permutations=[1, 4]),
    "pilot": dict(d=1000, n_train=2000, n_val=500, n_test=1000, n_instances=3,
                  n_background=5, seeds=[19], sampler_repeats=1,
                  kernel_multipliers=[2, 4], permutations=[1, 4]),
    "full": dict(d=1000, n_train=2000, n_val=500, n_test=1000, n_instances=6,
                 n_background=5, seeds=[0, 1, 2, 3, 4], sampler_repeats=3,
                 kernel_multipliers=[2, 4, 8], permutations=[1, 4, 16]),
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    files = [Path(__file__)] + sorted((REPO / "src/quadrashap").rglob("*.py"))
    return {str(p.relative_to(REPO)): digest(p) for p in files}


def make_data(cfg, seed):
    """Equal population variances; independent train/validation/test streams."""
    d = cfg["d"]
    s = round(cfg["informative_fraction"] * d)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    active = rng.choice(d, s, replace=False)
    beta = np.zeros(d)
    beta[active] = rng.choice([-1.0, 1.0], s) * rng.uniform(.5, 1.5, s)
    beta /= np.linalg.norm(beta)
    arrays = {"beta": beta, "informative": np.isin(np.arange(d), active),
              "interaction_features": active[:3]}
    for j, (split, n) in enumerate([("train", cfg["n_train"]),
                                    ("val", cfg["n_val"]), ("test", cfg["n_test"])]):
        r = np.random.default_rng(np.random.SeedSequence([seed, 100 + j]))
        X = r.normal(size=(n, d))
        signal = X @ beta
        signal += .2 * np.sin(X[:, active[0]] * X[:, active[1]])
        signal += .15 * (X[:, active[2]] ** 2 - 1)
        arrays["X_" + split] = X
        arrays["y_" + split] = signal + .1 * r.normal(size=n)
    return arrays


def fit_seed(cfg, seed, path):
    a = make_data(cfg, seed)
    scaler = StandardScaler().fit(a["X_train"])
    for split in ("train", "val", "test"):
        a["X_" + split] = scaler.transform(a["X_" + split])
    y_mean = float(a["y_train"].mean())
    candidates = []
    best, best_score = None, -np.inf
    t0 = time.perf_counter()
    for gs in cfg["gamma_scales"]:
        for alpha in cfg["alphas"]:
            model = KernelRidge(kernel="rbf", gamma=gs / cfg["d"], alpha=alpha)
            model.fit(a["X_train"], a["y_train"] - y_mean)
            score = float(r2_score(a["y_val"], model.predict(a["X_val"]) + y_mean))
            candidates.append(dict(gamma_scale=gs, alpha=alpha, validation_r2=score))
            if score > best_score:
                best, best_score = model, score
    test_prediction = best.predict(a["X_test"]) + y_mean
    train_prediction = best.predict(a["X_train"]) + y_mean
    diag = dict(seed=seed, d=cfg["d"], n_informative=int(a["informative"].sum()),
                gamma=float(best.gamma), alpha=float(best.alpha),
                validation_r2=best_score,
                train_r2=float(r2_score(a["y_train"], train_prediction)),
                test_r2=float(r2_score(a["y_test"], test_prediction)),
                fit_seconds=time.perf_counter() - t0, candidates=candidates)
    diag["weak_prediction_warning"] = diag["test_r2"] < .5
    atomic_npz(path, **a, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
               dual_coef=best.dual_coef_, gamma=best.gamma, alpha=best.alpha,
               intercept=y_mean, test_prediction=test_prediction,
               metadata=np.array(json.dumps(diag)))
    return diag


def load_fit(path):
    with np.load(path, allow_pickle=False) as z:
        a = {key: z[key] for key in z.files}
    model = SimpleNamespace(X_fit_=a["X_train"], dual_coef_=a["dual_coef"],
                            gamma=float(a["gamma"]), intercept_=float(a["intercept"]))
    return model, a


def factor_blocks(model, x, background, size=128):
    """Generate positive product factors without materializing every background pair."""
    for b in background:
        for start in range(0, len(model.X_fit_), size):
            centers = model.X_fit_[start:start + size]
            U = np.exp(-model.gamma * (centers - x) ** 2)
            Ut = np.exp(-model.gamma * (centers - b) ** 2)
            yield U - Ut, Ut, model.dual_coef_[start:start + size] / len(background)


class MaskGame:
    """Fast shared value-function evaluator given to both official SHAP baselines.

    Features are coalition indicators. The zero-indicator background represents
    the complete empirical interventional expectation, NOT a zero-feature baseline.
    """
    def __init__(self, model, x, background):
        self.d = len(x)
        n = len(model.X_fit_)
        self.delta = np.empty((self.d, n * len(background)))
        self.base = np.empty(n * len(background))
        present = -model.gamma * (model.X_fit_ - x) ** 2
        for j, b in enumerate(background):
            absent = -model.gamma * (model.X_fit_ - b) ** 2
            self.delta[:, j*n:(j+1)*n] = (present - absent).T
            self.base[j*n:(j+1)*n] = absent.sum(axis=1)
        self.w = np.tile(model.dual_coef_ / len(background), len(background))
        self.intercept = model.intercept_
        self.calls = 0

    def __call__(self, masks):
        masks = np.atleast_2d(np.asarray(masks, dtype=np.float64))
        if masks.shape[1] != self.d or np.any((masks != 0) & (masks != 1)):
            raise ValueError("Coalition masks must be binary.")
        self.calls += len(masks)
        out = np.empty(len(masks))
        block = max(1, (32 << 20) // (8 * len(self.w)))
        for start in range(0, len(masks), block):
            values = masks[start:start + block] @ self.delta
            values += self.base
            np.exp(values, out=values)
            out[start:start + block] = values @ self.w + self.intercept
        return out


def independent_reference(model, x, background, nodes):
    """SciPy nodes and shared log-products; no package quadrature or budget code."""
    t, w = roots_legendre(nodes)
    t, w = (t + 1) / 2, w / 2
    result = np.zeros(len(x))
    # Construct factors independently of factor_blocks and the package.
    for b in background:
        for start in range(0, len(model.X_fit_), 128):
            centers = model.X_fit_[start:start + 128]
            u = np.exp(-model.gamma * np.square(centers - x))
            v = np.exp(-model.gamma * np.square(centers - b))
            delta = u - v
            coef = model.dual_coef_[start:start + 128] / len(background)
            for q in range(0, nodes, 4):
                T = v[None] * (1 - t[q:q+4, None, None]) + u[None] * t[q:q+4, None, None]
                if np.any(T <= 0):
                    raise ValueError("Independent reference requires positive factors.")
                prod = np.exp(np.sum(np.log(T), axis=2))
                result += np.einsum("q,qr,qri,ri,r->i", w[q:q+4], prod, 1/T,
                                    delta, coef, optimize=True)
    return result


def quad(model, x, background, eps=None, exact=False):
    t0 = time.perf_counter()
    summary = GameSummary(d=len(x))
    report = None
    if not exact:
        for K, Ut, weights in factor_blocks(model, x, background):
            summary.update(K, Ut, weights)
        report = budget_from_summary(summary, eps)
    nodes = (len(x) + 1) // 2 if exact else report.m_q
    budget_seconds = time.perf_counter() - t0
    values = np.zeros(len(x))
    engine = ProductGamesShapleyNumpy()
    for K, Ut, weights in factor_blocks(model, x, background):
        values += weights @ engine.phi_matrix_prefix_scan(K, nodes, Ut, node_block=2)
    return values, dict(seconds=time.perf_counter()-t0, budget_seconds=budget_seconds,
                        m_q=int(nodes), bound=0.0 if exact else float(report.bound),
                        a_max=None if exact else float(report.scale),
                        lambda_max=None if exact else float(report.lambda_max),
                        implementation="QuadraSHAP NumPy prefix-suffix; component/node blocking")


def sampler(model, x, background, method, budget, random_seed):
    import shap
    warm_start = time.perf_counter()
    # Compile SHAP's generic Numba masking kernels once outside the measured call.
    # Same float64 mask types, no experiment data, labels, reference, or true support.
    toy = lambda X: np.asarray(X).sum(axis=1)
    if method == "permutation":
        warm = shap.PermutationExplainer(toy, np.zeros((1, 4)), seed=1)
        warm(np.ones((1, 4)), max_evals=9, silent=True)
    else:
        warm = shap.KernelExplainer(toy, np.zeros((1, 4)))
        warm.shap_values(np.ones((1, 4)), nsamples=14, l1_reg=0.0, silent=True)
    warmup_seconds = time.perf_counter()-warm_start
    np.random.seed(random_seed)  # KernelExplainer 0.48 uses the global NumPy generator.
    t0 = time.perf_counter()
    game = MaskGame(model, x, background)
    setup_seconds = time.perf_counter()-t0
    zeros, ones = np.zeros((1, len(x))), np.ones((1, len(x)))
    if method == "kernel":
        explainer = shap.KernelExplainer(game, zeros, link="identity")
        phi = explainer.shap_values(ones, nsamples=budget, l1_reg=0.0, silent=True)
        implementation = "shap.KernelExplainer; l1_reg=0; identity link; shared empirical game"
    elif method == "permutation":
        explainer = shap.PermutationExplainer(game, zeros, seed=random_seed)
        phi = explainer(ones, max_evals=budget * (2 * len(x) + 1), silent=True).values
        implementation = "shap.PermutationExplainer; antithetic; shared empirical game"
    else:
        raise ValueError(method)
    return np.asarray(phi).reshape(-1), dict(seconds=time.perf_counter()-t0,
            setup_seconds=setup_seconds, warmup_seconds=warmup_seconds, coalition_evaluations=game.calls,
            m_q=None, bound=None, implementation=implementation, random_seed=random_seed)


def recovery_metrics(score, informative, tie_seed=42):
    score, informative = np.asarray(score), np.asarray(informative, dtype=bool)
    s = int(informative.sum())
    if not (0 < s < len(score)) or not np.isfinite(score).all():
        raise ValueError("Recovery needs finite scores and both feature classes.")
    tie = np.random.default_rng(tie_seed).random(len(score))
    top = np.lexsort((tie, -score))[:s]
    return dict(average_precision=float(average_precision_score(informative, score)),
                precision_at_s=float(informative[top].mean()),
                auroc=float(roc_auc_score(informative, score)))


def jobs(cfg, seed):
    for inst in range(cfg["n_instances"]):
        yield dict(seed=seed, instance=inst, method="reference", budget=0, repeat=0)
        yield dict(seed=seed, instance=inst, method="exact", budget=0, repeat=0)
        for eps in cfg["epsilons"]:
            yield dict(seed=seed, instance=inst, method="quad", budget=eps, repeat=0)
        for rep in range(cfg["sampler_repeats"]):
            for mult in cfg["kernel_multipliers"]:
                yield dict(seed=seed, instance=inst, method="kernel",
                           budget=mult * cfg["d"] + 128, repeat=rep)
            for pairs in cfg["permutations"]:
                yield dict(seed=seed, instance=inst, method="permutation", budget=pairs, repeat=rep)


def job_name(job):
    budget = f"{job['budget']:g}".replace(".", "p")
    return f"i{job['instance']:03d}_{job['method']}_{budget}_r{job['repeat']}"


def worker(output, job):
    cfg = json_read(output / "config.json")
    seed_dir = output / f"seed_{job['seed']}"
    stem = seed_dir / "jobs" / job_name(job)
    try:
        model, a = load_fit(seed_dir / "fit.npz")
        x = a["X_test"][job["instance"]]
        bg = a["X_train"][:cfg["n_background"]]
        t0 = time.perf_counter()
        if job["method"] == "reference":
            m = (cfg["d"] + 1)//2
            phi = independent_reference(model, x, bg, m)
            check = independent_reference(model, x, bg, m+3)
            gap = float(np.max(np.abs(phi-check)))
            # Check the efficient coalition evaluator against direct masked predictions.
            game = MaskGame(model, x, bg)
            rng = np.random.default_rng(np.random.SeedSequence([job["seed"], job["instance"], 531]))
            masks = rng.integers(0, 2, size=(10, cfg["d"]))
            fast = game(masks)
            direct = np.array([np.mean(rbf_kernel(np.where(mask, x, bg), model.X_fit_,
                                  gamma=model.gamma) @ model.dual_coef_ + model.intercept_)
                               for mask in masks])
            value_gap = float(np.max(np.abs(fast-direct)))
            endpoints = game(np.array([np.zeros(cfg["d"]), np.ones(cfg["d"])]))
            efficiency_gap = float(abs(phi.sum() - (endpoints[1]-endpoints[0])))
            info = dict(seconds=time.perf_counter()-t0, reference_nodes=m,
                        reference_check_nodes=m+3, reference_discrepancy=gap,
                        game_prediction_discrepancy=value_gap, efficiency_residual=efficiency_gap,
                        implementation="SciPy exact-degree GL, independent log-products")
            if max(gap, value_gap, efficiency_gap) > cfg["reference_tolerance"]:
                raise ArithmeticError(f"Reference validation failed: {info}")
        elif job["method"] in ("quad", "exact"):
            phi, info = quad(model, x, bg, eps=job["budget"], exact=job["method"] == "exact")
        else:
            random_seed = int(np.random.SeedSequence(
                [job["seed"], job["instance"], job["repeat"],
                 71 if job["method"] == "kernel" else 83]).generate_state(1)[0])
            phi, info = sampler(model, x, bg, job["method"], job["budget"], random_seed)
        if phi.shape != (cfg["d"],) or not np.isfinite(phi).all():
            raise ArithmeticError("Nonfinite or incorrectly shaped attribution vector.")
        if job["method"] != "reference":
            ref_path = seed_dir / "jobs" / (job_name(dict(job, method="reference", budget=0, repeat=0)) + ".npz")
            with np.load(ref_path, allow_pickle=False) as z:
                ref = z["phi"]
            err = float(np.max(np.abs(phi-ref)))
            info.update(max_abs_error=err,
                        relative_l2_error=float(np.linalg.norm(phi-ref)/max(np.linalg.norm(ref), 1e-300)))
            if job["method"] == "quad":
                info["target_met"] = bool(err <= job["budget"])
            if job["method"] == "exact" and err > cfg["reference_tolerance"]:
                raise ArithmeticError(f"Package exact rule disagrees with independent reference: {err}")
        info.update(job, status="ok", fit_sha256=digest(seed_dir / "fit.npz"))
        atomic_npz(stem.with_suffix(".npz"), phi=phi, metadata=np.array(json.dumps(info)))
        atomic_json(stem.with_suffix(".json"), dict(info, result_sha256=digest(stem.with_suffix(".npz"))))
    except Exception as exc:
        atomic_json(stem.with_suffix(".json"), dict(job, status="failed",
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
        raise


def configuration(profile, threads, timeout):
    return dict(PROFILES[profile], version=VERSION, profile=profile, threads=threads,
                timeout_seconds=timeout, reference_timeout_seconds=max(1800, timeout),
                informative_fraction=.30, gamma_scales=[.1, 1.0], alphas=[.01, .1],
                epsilons=[1e-1, 1e-2, 1e-3, 1e-6], reference_tolerance=1e-9,
                prediction_warning_r2=.5, sampler_warmup="untimed four-feature additive toy")


def prepare_output(output, cfg):
    import shap
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__, shap=shap.__version__)
    identity = dict(versions=versions, machine=platform.machine(), platform=platform.platform(),
                    source_hashes=source_hashes())
    if (output / "config.json").exists():
        if json_read(output / "config.json") != cfg:
            raise ValueError("Configuration differs from the checkpoint. Choose a new --output directory.")
        old = json_read(output / "provenance.json")
        for key in identity:
            if old[key] != identity[key]:
                raise ValueError(f"Resume refused: changed {key}. Use a new --output directory.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("New output directory must be empty.")
        atomic_json(output / "config.json", cfg)
        atomic_json(output / "provenance.json", dict(identity, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                     executable=sys.executable, threadpools=__import__("threadpoolctl").threadpool_info()))
        for relative in identity["source_hashes"]:
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / relative).read_bytes())
        for relative in ("experiments/exp3_recovery_analysis.py", "experiments/README_exp3.md"):
            source = REPO / relative
            if source.exists():
                target = output / "source_snapshot" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())


def collect(output, cfg, retry_failed=False):
    # A file lock prevents two local runs from racing on the same checkpoints.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another collector is already using this output directory.")
        prepare_output(output, cfg)
        total = sum(1 for seed in cfg["seeds"] for _ in jobs(cfg, seed))
        done = 0
        for seed in cfg["seeds"]:
            sd = output / f"seed_{seed}"
            fit_path = sd / "fit.npz"
            if not fit_path.exists():
                print(f"Fitting seed {seed} (validation-only hyperparameter selection)", flush=True)
                diag = fit_seed(cfg, seed, fit_path)
                atomic_json(sd / "fit.json", dict(diag, sha256=digest(fit_path)))
            else:
                if not (sd / "fit.json").exists():
                    # Finish the interrupted commit of an already saved model.
                    _, saved = load_fit(fit_path)
                    atomic_json(sd / "fit.json", dict(json.loads(str(saved["metadata"])), sha256=digest(fit_path)))
                if digest(fit_path) != json_read(sd / "fit.json")["sha256"]:
                    raise ValueError(f"Damaged model checkpoint: {fit_path}")
                diag = json_read(sd / "fit.json")
            print(f"Seed {seed}: test R2={diag['test_r2']:.3f}; informative={diag['n_informative']}", flush=True)
            for job in jobs(cfg, seed):
                stem = sd / "jobs" / job_name(job)
                record = stem.with_suffix(".json")
                if record.exists():
                    info = json_read(record)
                    if info["status"] == "ok":
                        if digest(stem.with_suffix(".npz")) != info["result_sha256"]:
                            raise ValueError(f"Damaged result checkpoint: {stem}")
                        done += 1
                        continue
                    if not retry_failed:
                        done += 1
                        continue
                ref = sd / "jobs" / (job_name(dict(job, method="reference", budget=0, repeat=0))+".json")
                if job["method"] != "reference" and (not ref.exists() or json_read(ref)["status"] != "ok"):
                    atomic_json(record, dict(job, status="blocked", error="Reference not validated"))
                    done += 1
                    continue
                print(f"[{done+1}/{total}] seed={seed} {job_name(job)}", flush=True)
                stem.parent.mkdir(parents=True, exist_ok=True)
                timeout = cfg["reference_timeout_seconds"] if job["method"] in ("reference", "exact") else cfg["timeout_seconds"]
                command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                           "--worker", json.dumps(job), "--threads", str(cfg["threads"])]
                with stem.with_suffix(".log").open("w") as log:
                    try:
                        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                        if run.returncode and not record.exists():
                            atomic_json(record, dict(job, status="failed", error=f"Worker exit {run.returncode}"))
                    except subprocess.TimeoutExpired:
                        atomic_json(record, dict(job, status="timeout", timeout_seconds=timeout,
                                    error="Worker wall-time cap includes startup; not a measured successful runtime."))
                info = json_read(record)
                print(f"  {info['status']}" + (f", {info['seconds']:.3f}s" if "seconds" in info else ""), flush=True)
                done += 1
        from exp3_recovery_analysis import analyze
        result = analyze(output)
        if not result["collection_complete"] or not result["numerical_checks_passed"]:
            raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=PROFILES, default="pilot")
    ap.add_argument("--quick", action="store_true", help="Separate tiny pipeline check, never publication evidence.")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180, help="Wall-time cap per sampling/approximation worker.")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--replot", action="store_true", help="Analyze stored vectors only; no fitting or explanations.")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.threads < 1 or args.timeout < 1:
        ap.error("threads and timeout must be positive")
    profile = "quick" if args.quick else args.profile
    output = (args.output or REPO / "experiments/results/exp3_recovery_v2" / profile).resolve()
    # Limit both this process and workers, without changing global machine settings.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.worker:
            worker(output, json.loads(args.worker))
        elif args.replot:
            from exp3_recovery_analysis import analyze
            analyze(output)
        else:
            collect(output, configuration(profile, args.threads, args.timeout), args.retry_failed)


if __name__ == "__main__":
    main()
