"""Replot Experiment 3 v2 from saved vectors; never launches numerical jobs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(output):
    from exp3_synthetic_recovery import (atomic_json, digest, job_name, jobs,
                                          json_read, recovery_metrics)
    output = Path(output)
    cfg = json_read(output / "config.json")
    groups = defaultdict(list)
    rows, failures, fits = [], [], []
    fit_hashes = {}
    expected = [job for seed in cfg["seeds"] for job in jobs(cfg, seed)]
    targets_failed = []
    for seed in cfg["seeds"]:
        fit_json = output / f"seed_{seed}/fit.json"
        if fit_json.exists():
            info = json_read(fit_json)
            fit_hashes[seed] = digest(fit_json.with_suffix(".npz"))
            if fit_hashes[seed] != info["sha256"]:
                raise ValueError(f"Corrupt model: {fit_json}")
            fits.append(info)
    for job in expected:
        stem = output / f"seed_{job['seed']}/jobs" / job_name(job)
        path = stem.with_suffix(".json")
        if not path.exists():
            failures.append(dict(job, status="missing"))
            continue
        record = json_read(path)
        if record["status"] != "ok":
            failures.append(record)
            continue
        if digest(stem.with_suffix(".npz")) != record["result_sha256"]:
            raise ValueError(f"Corrupt vector: {stem}")
        if record["fit_sha256"] != fit_hashes[job["seed"]]:
            raise ValueError(f"Model/result mismatch: {stem}")
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            phi = z["phi"]
        rows.append(record)
        if record.get("target_met") is False:
            targets_failed.append(job)
        groups[(job["seed"], job["method"], job["budget"], job["repeat"])].append((record, phi))
    per_fit = []
    for (seed, method, budget, rep), values in sorted(groups.items()):
        if len(values) != cfg["n_instances"]:
            continue  # A subset of explained inputs is not a valid global recovery score.
        with np.load(output / f"seed_{seed}/fit.npz", allow_pickle=False) as z:
            informative = z["informative"]
        score = np.mean(np.abs([phi for _, phi in values]), axis=0)
        rec = dict(seed=seed, method=method, budget=budget, repeat=rep,
                   n_instances=len(values), **recovery_metrics(score, informative, tie_seed=seed+471))
        rec["seconds"] = float(np.median([r["seconds"] for r, _ in values]))
        if method != "reference":
            rec["max_abs_error"] = float(np.mean([r["max_abs_error"] for r, _ in values]))
            rec["worst_abs_error"] = float(max(r["max_abs_error"] for r, _ in values))
            rec["relative_l2_error"] = float(np.mean([r["relative_l2_error"] for r, _ in values]))
        if method in ("quad", "exact"):
            rec["m_q"] = float(np.median([r["m_q"] for r, _ in values]))
        per_fit.append(rec)
    # First average repeated sampler runs within each fit, then summarize independent fits.
    by_seed = defaultdict(list)
    for rec in per_fit:
        by_seed[(rec["seed"], rec["method"], rec["budget"])].append(rec)
    metrics = ("average_precision", "precision_at_s", "auroc", "seconds",
               "max_abs_error", "relative_l2_error", "m_q")
    seed_means = []
    for (seed, method, budget), reps in sorted(by_seed.items()):
        required = cfg["sampler_repeats"] if method in ("kernel", "permutation") else 1
        if len(reps) != required:
            continue
        rec = dict(seed=seed, method=method, budget=budget, repeats=required)
        for key in metrics:
            if key in reps[0]:
                rec[key] = float(np.mean([r[key] for r in reps]))
        seed_means.append(rec)
    summary = []
    conditions = sorted({(job["method"], job["budget"]) for job in expected})
    for method, budget in conditions:
        sel = [r for r in seed_means if (r["method"], r["budget"]) == (method, budget)]
        rec = dict(method=method, budget=budget, n_fits=len(sel),
                   complete=len(sel) == len(cfg["seeds"]))
        for key in metrics:
            vals = [r[key] for r in sel if key in r]
            if vals:
                rec[key] = float(np.mean(vals))
                rec[key+"_min"] = float(min(vals))
                rec[key+"_max"] = float(max(vals))
        summary.append(rec)
    write_csv(output / "per_instance.csv", rows)
    write_csv(output / "failures.csv", failures)
    write_csv(output / "recovery_per_fit_repeat.csv", per_fit)
    write_csv(output / "seed_means.csv", seed_means)
    write_csv(output / "summary.csv", summary)
    counts = dict(expected_jobs=len(expected), successful_jobs=len(rows),
                  unsuccessful_jobs=len(failures), tolerance_failures=targets_failed,
                  collection_complete=not failures,
                  numerical_checks_passed=not targets_failed and not failures,
                  publication_run=cfg["profile"] == "full")
    atomic_json(output / "completion.json", counts)
    atomic_json(output / "analysis_provenance.json", dict(analysis_sha256=digest(__file__),
                collection_source_hash=json_read(output / "provenance.json")["source_hashes"]["experiments/exp3_synthetic_recovery.py"]))
    text = ["# Experiment 3: synthetic recovery and estimator fidelity", "",
            f"Profile: **{cfg['profile']}**. " + ("Publication-size collection." if cfg["profile"] == "full" else
                "**PILOT ONLY: not publication evidence.**"), "",
            f"Successful jobs: {len(rows)}/{len(expected)}; failures/timeouts/missing: {len(failures)}.",
            f"Quadrature target failures: {len(targets_failed)}.", "",
            f"d={cfg['d']}; informative={round(cfg['d']*.3)}; training seeds={cfg['seeds']}; "
            f"{cfg['n_instances']} explained test inputs per fit; {cfg['sampler_repeats']} sampler repeats.",
            "Recovery uses mean absolute attribution over the fixed explained inputs of each fit.",
            "Sampler repeats are averaged within a fit before averaging fits. Error is the mean over inputs "
            "of maximum absolute feature error. Timing is the median per input within each fit/repeat.",
            "Plotted bars span fit-level minimum to maximum, not confidence intervals.",
            "No incomplete input set, missing sampler repeat, or incomplete collection of fits contributes a plotted point.",
            "Training, reference construction, process startup, imports and a four-feature SHAP compiler warmup are excluded from estimator timing. "
            "Estimator-specific game construction, budget selection, sampling/solving, and integration are included.",
            "The fast coalition evaluator is checked against actual masked KRR predictions. "
            "Both SHAP baselines use the official installed package, with no fallback. KernelSHAP uses l1_reg=0.",
            "Reference Shapley recovery is not an upper bound on generating-feature recovery.",
            "The certificate covers quadrature error, not floating-point or population-background error.", "",
            "## Model checks", "", "| Seed | Validation R2 | Test R2 | Gamma*d | Alpha |",
            "|---|---:|---:|---:|---:|"]
    for r in fits:
        text.append(f"| {r['seed']} | {r['validation_r2']:.4f} | {r['test_r2']:.4f} | "
                    f"{r['gamma']*cfg['d']:.3g} | {r['alpha']:.3g} |")
    text += ["", "## Results", "", "| Method | Budget | Complete fits | Seconds | Average precision | Precision@s | Mean max error |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        fmt = lambda k: f"{r[k]:.5g}" if k in r else "--"
        text.append(f"| {r['method']} | {r['budget']:g} | {r['n_fits']}/{len(cfg['seeds'])} | "
                    f"{fmt('seconds')} | {fmt('average_precision')} | {fmt('precision_at_s')} | {fmt('max_abs_error')} |")
    if failures:
        text += ["", "## Incomplete conditions", "", "See failures.csv and individual job logs; failures are not dropped from denominators."]
    (output / "REPORT.md").write_text("\n".join(text)+"\n")
    plot(output, cfg, summary)
    print(json.dumps(counts, indent=2), flush=True)
    return counts


def plot(output, cfg, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), layout="constrained")
    styles = {"quad": ("#0072B2", "o", "QuadraSHAP certified"),
              "kernel": ("#D55E00", "s", "KernelSHAP"),
              "permutation": ("#009E73", "^", "PermutationSHAP")}
    handles = []
    for method, (color, marker, label) in styles.items():
        sel = sorted([r for r in summary if r["method"] == method and r["complete"]],
                     key=lambda r: r["seconds"])
        for ax, metric in zip(axes, ("max_abs_error", "average_precision")):
            if not sel:
                continue
            x, y = np.array([r["seconds"] for r in sel]), np.array([r[metric] for r in sel])
            xerr = np.array([[r["seconds"]-r["seconds_min"] for r in sel],
                             [r["seconds_max"]-r["seconds"] for r in sel]])
            yerr = np.array([[r[metric]-r[metric+"_min"] for r in sel],
                             [r[metric+"_max"]-r[metric] for r in sel]])
            h = ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, marker=marker,
                            linewidth=1.3, markersize=4, capsize=2, label=label)
            if ax is axes[0]:
                handles.append(h)
    reference = [r for r in summary if r["method"] == "reference" and r["complete"]]
    if reference:
        r = reference[0]
        h = axes[1].axhline(r["average_precision"], color="0.35", linestyle="--", linewidth=1.2,
                            label="Reference-Shapley recovery")
        handles.append(h)
        axes[1].axhspan(r["average_precision_min"], r["average_precision_max"], color="0.5", alpha=.10)
    exact = [r for r in summary if r["method"] == "exact" and r["complete"]]
    if exact:
        r = exact[0]
        h, = axes[1].plot(r["seconds"], r["average_precision"], "D", color="0.4", ms=5, label="QuadraSHAP exact")
        handles.append(h)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Time per explanation (s)")
        ax.grid(alpha=.2, which="major")
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean maximum absolute error")
    axes[0].set_title("Attribution fidelity")
    axes[1].set_ylabel("Average precision")
    axes[1].set_ylim(0, 1.025)
    axes[1].set_title("Informative-feature recovery")
    n_fits = len(cfg["seeds"])
    fit_label = "fit" if n_fits == 1 else "fits"
    title = f"d = {cfg['d']} | 30% informative | {n_fits} {fit_label} x {cfg['n_instances']} inputs"
    if cfg["profile"] != "full":
        title = "PILOT ONLY - " + title
    fig.suptitle(title, fontsize=9)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    folder = output / "figures"
    (folder / "backup").mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "recovery_tradeoff.pdf")
    fig.savefig(folder / "backup/recovery_tradeoff.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    analyze(ap.parse_args().output)
