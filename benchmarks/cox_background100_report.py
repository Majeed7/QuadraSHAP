"""Validate all eighty B=100 cases and write plain-text results; never send email.

Each approximation is compared with its own device/precision full-degree
output for the same B=100 game. No partial final report is supported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

from benchmarks.cox_background30_table import COHORTS, numeric, read_csv, read_json, require, write_csv
from benchmarks.cox_tolerance_rerun_report import ROOT, close, digest

OUT = ROOT / "benchmarks/results/cox_background100"
B30 = ROOT / "benchmarks/results/cox_background30_gpu"
B30_TABLE = B30 / "tolerance_rerun/cpu_precision_timing/cpu_exact_degree/table.csv"
MODES = ("exact", "1e-1", "1e-2", "1e-3")
DEVICES = {"gpu": ("metal", "float32", "GPU FP32"), "cpu": ("cpu", "float64", "CPU FP64")}


def key_of(row):
    return row["dataset"], int(row["patient"]), row["mode"]


def validate_inputs(folder):
    design, old = read_json(folder / "design.json"), read_json(B30 / "design.json")
    hashes = {}
    for dataset, spec in COHORTS.items():
        details = design[dataset]
        require(int(details["background_size"]) == 100 and int(details["selection_seed"]) == 42,
                f"Incorrect background protocol: {dataset}")
        require(int(details["d"]) == spec["d"] and int(details["n_train"]) == spec["n_train"],
                f"Cohort dimensions differ: {dataset}")
        require(len(details["patient_ids"]) == len(set(details["patient_ids"])) == 5,
                f"Expected five distinct patients: {dataset}")
        require(details["patient_ids"] == old[dataset]["patient_ids"], f"Patients differ from B=30: {dataset}")
        path = folder / dataset / "inputs.npz"
        hashes[dataset] = digest(path)
        require(hashes[dataset] == details["prepared_inputs_sha256"], f"Inputs changed: {dataset}")
        with np.load(path, allow_pickle=False) as current, np.load(B30 / dataset / "inputs.npz", allow_pickle=False) as prior:
            require(current["background"].shape == (100, spec["d"]), f"Incorrect background array: {dataset}")
            require(np.isfinite(current["background"]).all(), f"Nonfinite background: {dataset}")
            for field in ("beta", "X", "patient_ids"):
                require(np.array_equal(current[field], prior[field]), f"{field} differs from B=30: {dataset}")
            require(np.array_equal(current["background"][:30], prior["background"]),
                    f"The first thirty backgrounds differ from B=30: {dataset}")
    return design, hashes


def validate_record(row, record):
    """CSV is a summary; completed JSON case records are authoritative."""
    floats = {"eps", "seconds", "warmup_seconds", "bound", "budget_seconds", "checkpoint_io_seconds"}
    integers = {"patient", "B", "d", "m_q", "block_size", "node_block", "attempts"}
    fields = {"dataset", "patient_id", "mode", "core_dtype", "memory_budget", "attribution_file",
              "attribution_sha256"} | floats | integers
    for field in fields:
        require(field in row and field in record, f"Missing case field: {field}")
        if field == "eps" and row["mode"] == "exact":
            require(row[field] == "" and record[field] is None, "Exact tolerance must be empty")
        elif field in floats:
            close(row[field], record[field], f"CSV and case record disagree: {field}")
        elif field in integers:
            require(int(row[field]) == int(record[field]), f"CSV and case record disagree: {field}")
        else:
            require(str(row[field]) == str(record[field]), f"CSV and case record disagree: {field}")


def validate(folder):
    """Read only; reject incomplete suites before checking numerical outputs."""
    for device in DEVICES:
        path = folder / device / "status.json"
        require(path.exists(), f"Incomplete B=100 experiment: {device} status is missing")
        status = read_json(path)
        require(status.get("phase") == "complete" and int(status.get("cases", 0)) == 40
                and int(status.get("measured_calls", 0)) == 40,
                f"Incomplete B=100 experiment: {device} requires forty completed cases")
    design, inputs = validate_inputs(folder)
    original_env = read_json(B30 / "environment.json")
    design_hash, rule_hash = digest(folder / "design.json"), digest(folder / "rule_setup.json")
    rule_metadata = read_json(folder / "rule_setup.json")
    expected = {(d, p, m) for d in COHORTS for p in range(5) for m in MODES}
    cases, envs, sources = {}, {}, {}
    for device, (backend, dtype, _) in DEVICES.items():
        base = folder / device
        env, provenance = read_json(base / "environment.json"), read_json(base / "provenance.json")
        require(env["backend"].lower() == backend and not env["jit_disabled"], f"Wrong backend: {device}")
        require(env["work_dtype"] == env["core_dtype"] == dtype and env["host_accumulation_dtype"] == "float64",
                f"Wrong work/core/host dtype: {device}")
        require(bool(env["jax_enable_x64"]) == (device == "cpu"), f"Wrong x64 configuration: {device}")
        require(env["source_sha256"] == original_env["source_sha256"] == provenance["source_sha256"],
                f"Numerical sources differ from B=30: {device}")
        require(env["packages"] == original_env["packages"], f"Package versions differ from B=30: {device}")
        require(env["extra_source_sha256"] == provenance["extra_source_sha256"], f"Extra source hashes differ: {device}")
        require(provenance["input_sha256"] == inputs and provenance["design_sha256"] == design_hash,
                f"Input/design provenance mismatch: {device}")
        require(provenance["rule_setup_sha256"] == rule_hash, f"Rule setup provenance mismatch: {device}")
        require(provenance["background_size"] == 100 and provenance["distinct_patients_per_dataset"] == 5
                and provenance["measured_calls_per_patient_method"] == 1, f"Wrong protocol: {device}")
        require(sorted(provenance["tolerances"]) == [0.001, 0.01, 0.1], f"Wrong tolerances: {device}")
        for group in (env["source_sha256"], env["extra_source_sha256"]):
            for path, sha in group.items():
                require(digest(ROOT / path) == sha, f"Measured source changed: {path}")
        smoke = read_json(base / "stream_validation/validation.json")
        require(smoke["bitwise_equal_to_public_api"] and smoke["bitwise_resume_equal"]
                and smoke["core_dtype"] == dtype, f"Stream/resume validation failed: {device}")
        rows = read_csv(base / "summary.csv")
        require(len(rows) == 40, f"Missing summary cases: {device}")
        indexed = {}
        for row in rows:
            key = key_of(row)
            require(key in expected and key not in indexed, f"Unexpected/duplicate case: {device}, {key}")
            dataset, patient, mode = key
            record = read_json(base / "cases" / f"{dataset}_p{patient}_{mode}.json")
            validate_record(row, record)
            require(int(row["B"]) == 100 and int(row["d"]) == COHORTS[dataset]["d"], f"Wrong case dimensions: {key}")
            require(row["patient_id"] == str(design[dataset]["patient_ids"][patient]), f"Wrong case patient: {key}")
            require(row["core_dtype"] == dtype and row["memory_budget"] == "8GB" and int(row["block_size"]) == 1,
                    f"Wrong case work settings: {key}")
            require(0 < int(row["node_block"]) <= int(row["m_q"]) and int(row["attempts"]) >= 1,
                    f"Invalid node block or attempt count: {key}")
            numeric(row["seconds"], "runtime", positive=True)
            for field in ("bound", "warmup_seconds", "budget_seconds", "checkpoint_io_seconds"):
                numeric(row[field], field)
            relative = Path(row["attribution_file"])
            path = (base / relative).resolve()
            require(not relative.is_absolute() and path.is_relative_to(base.resolve()), "Attribution path escapes device folder")
            require(digest(path) == row["attribution_sha256"], f"Attribution hash changed: {key}")
            phi = np.load(path, allow_pickle=False)
            require(phi.dtype == np.float64 and phi.shape == (COHORTS[dataset]["d"],) and np.isfinite(phi).all(),
                    f"Invalid attribution output: {key}")
            if mode == "exact":
                require(int(row["m_q"]) == (int(row["d"]) + 1) // 2 and float(row["bound"]) == 0
                        and float(row["budget_seconds"]) == 0, f"Incorrect full-degree rule: {key}")
                checkpoint_dir = base / "checkpoints" / dataset / f"patient_{patient}"
                progress = read_json(checkpoint_dir / "exact_progress.json")
                require(progress["phase"] == "complete" and progress["completed_fraction"] == 1.0,
                        f"Exact checkpoint incomplete: {key}")
                manifest = progress["manifest"]
                require(manifest["B"] == 100 and manifest["m_q"] == int(row["m_q"])
                        and manifest["input_sha256"] == inputs[dataset] and manifest["core_dtype"] == dtype,
                        f"Checkpoint targets another game/precision: {key}")
                require(manifest["dataset"] == dataset and int(manifest["patient"]) == patient
                        and manifest["patient_id"] == row["patient_id"]
                        and manifest["rule_sha256"] == rule_metadata[dataset]["source_sha256"],
                        f"Checkpoint patient or quadrature rule differs: {key}")
                for field in ("source_sha256", "extra_source_sha256", "packages", "thread_environment", "backend"):
                    require(manifest[field] == env[field], f"Checkpoint environment mismatch: {field}, {key}")
                require(progress["block_plan"]["node_block"] == int(row["node_block"]), f"Checkpoint block mismatch: {key}")
                require(int(progress["attempts"]) == int(row["attempts"]), f"Checkpoint attempt count differs: {key}")
                difference = float(row["seconds"]) - float(progress["integration_compute_seconds"])
                require(-1e-9 <= difference < 1, f"Checkpoint and recorded compute times disagree: {key}")
                with np.load(checkpoint_dir / "exact_checkpoint.npz", allow_pickle=False) as checkpoint:
                    metadata = json.loads(str(checkpoint["metadata"]))
                    require(metadata["phase"] == "complete" and metadata["completed_fraction"] == 1.0,
                            f"Binary checkpoint incomplete: {key}")
                    require(np.array_equal(phi, checkpoint["phi"]), f"Exact output differs from completed checkpoint: {key}")
            else:
                require(float(row["eps"]) == float(mode) and float(row["bound"]) <= float(mode),
                        f"Analytical bound exceeds requested tolerance: {key}")
                require(int(row["m_q"]) < (int(row["d"]) + 1) // 2 and float(row["checkpoint_io_seconds"]) == 0,
                        f"Unexpected approximate execution: {key}")
            indexed[key] = {**row, "path": path}
        require(set(indexed) == expected, f"Missing device configurations: {device}")
        cases[device], envs[device] = indexed, env
        sources[device] = {name: digest(base / name) for name in
                           ("summary.csv", "status.json", "environment.json", "provenance.json")}
    require(envs["gpu"]["source_sha256"] == envs["cpu"]["source_sha256"]
            and envs["gpu"]["packages"] == envs["cpu"]["packages"], "GPU/CPU numerical environments differ")
    errors = []
    for dataset in COHORTS:
        for patient in range(5):
            reference = {dev: np.load(cases[dev][(dataset, patient, "exact")]["path"], allow_pickle=False) for dev in DEVICES}
            for mode in MODES[1:]:
                key = dataset, patient, mode
                g, c = cases["gpu"][key], cases["cpu"][key]
                require(int(g["m_q"]) == int(c["m_q"]), f"GPU/CPU node counts differ: {key}")
                close(g["bound"], c["bound"], f"GPU/CPU analytical bounds differ: {key}")
                result = {"dataset": dataset, "patient": patient, "patient_id": g["patient_id"],
                          "B": 100, "d": int(g["d"]), "mode": mode, "eps": float(mode),
                          "m_q": int(g["m_q"]), "bound": float(g["bound"])}
                for device in DEVICES:
                    delta = np.load(cases[device][key]["path"], allow_pickle=False) - reference[device]
                    value = float(np.max(np.abs(delta)))
                    result[f"delta_{device}_vs_own_degree_exact"] = value
                    result[f"{device}_relative_l2_vs_own_degree_exact"] = float(np.linalg.norm(delta) / np.linalg.norm(reference[device]))
                    result[f"{device}_within_eps"] = value <= float(mode)
                errors.append(result)
    return cases, design, errors, sources


def aggregate(cases, errors):
    indexed = {(r["dataset"], r["patient"], r["mode"]): r for r in errors}
    rows = []
    for dataset in COHORTS:
        for mode in MODES:
            group = [cases["gpu"][(dataset, p, mode)] for p in range(5)]
            row = {"dataset": dataset, "method": mode, "B": 100, "patients": 5, "calls_per_patient": 1,
                   "nodes_min": min(int(r["m_q"]) for r in group), "nodes_max": max(int(r["m_q"]) for r in group),
                   "C_max": 0.0 if mode == "exact" else max(float(r["bound"]) for r in group)}
            for device, (_, dtype, label) in DEVICES.items():
                selected = [cases[device][(dataset, p, mode)] for p in range(5)]
                times = [float(r["seconds"]) for r in selected]
                row.update({f"{device}_seconds_mean": statistics.mean(times),
                            f"{device}_seconds_sample_sd": statistics.stdev(times),
                            f"delta_{device}_vs_own_degree_exact": 0.0 if mode == "exact" else max(
                                indexed[(dataset, p, mode)][f"delta_{device}_vs_own_degree_exact"] for p in range(5)),
                            f"{device}_pass": "" if mode == "exact" else sum(
                                indexed[(dataset, p, mode)][f"{device}_within_eps"] for p in range(5)),
                            f"{device}_core_dtype": dtype,
                            f"{device}_checkpoint_io_seconds_total": sum(float(r["checkpoint_io_seconds"]) for r in selected),
                            f"{device}_attempts_max": max(int(r["attempts"]) for r in selected),
                            f"{device}_error_reference": f"Same B=100 patient, {label}, full degree-exact quadrature"})
            rows.append(row)
    return rows


def supporting_data(folder, cases):
    validation = read_json(B30_TABLE.parent / "report_validation.json")
    require(validation["same_device_precision_full_degree_references"] and validation["cpu_degree_exact_cases"] == 10,
            "B=30 comparison does not have completed matching full-degree references")
    require(digest(B30_TABLE) == validation["report_sha256"]["table.csv"], "Verified B=30 table changed")
    prior_rows = read_csv(B30_TABLE)
    prior = {(r["dataset"], r["method"]): r for r in prior_rows}
    require(len(prior_rows) == len(prior) == 8 and set(prior) == {(d, m) for d in COHORTS for m in MODES},
            "B=30 comparison table has missing/duplicate rows")
    require(all(int(r["B"]) == 30 and int(r["patients"]) == 5 and int(r["calls_per_patient"]) == 1 for r in prior_rows),
            "B=30 comparison uses a different protocol")
    rules, exact_seconds, small_ms = read_json(folder / "rule_setup.json"), {}, {}
    for dataset, spec in COHORTS.items():
        row = rules[dataset]
        require(row["exact_nodes"] == (spec["d"] + 1) // 2 and digest(row["source"]) == row["source_sha256"],
                f"Cached exact rule changed: {dataset}")
        exact_seconds[dataset] = numeric(row["original_construction_seconds"], "historical rule construction", positive=True)
    for device in DEVICES:
        rows = read_csv(folder / device / "small_rule_timings.csv")
        needed = {(key[0], int(row["m_q"])) for key, row in cases[device].items() if key[2] != "exact"}
        require(len(rows) == len(needed) and {(r["dataset"], int(r["m_q"])) for r in rows} == needed,
                f"Small-rule setup measurements incomplete: {device}")
        values = []
        for row in rows:
            require(int(row["repeats"]) == 5, "Expected five fresh constructions per rule")
            lo, med, hi = [numeric(row[f], f, positive=True) for f in ("min_seconds", "median_seconds", "max_seconds")]
            require(lo <= med <= hi, "Inconsistent small-rule construction times")
            values.append(1000 * med)
        small_ms[device] = (min(values), max(values))
    return prior, exact_seconds, small_ms


def plain_table(headers, rows):
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    def line(row):
        return "  ".join(str(value).ljust(width) for value, width in zip(row, widths)).rstrip()
    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(row) for row in rows)])


def build(folder):
    cases, design, errors, sources = validate(folder)
    rows = aggregate(cases, errors)
    prior, exact_setup, small_ms = supporting_data(folder, cases)
    indexed = {(r["dataset"], r["method"]): r for r in rows}
    passes = {dev: sum(r[f"{dev}_pass"] for r in rows if r["method"] != "exact") for dev in DEVICES}
    text = ["QuadraSHAP: 100-background Cox timing and error-check results", "",
            "The 100-background experiments are complete: both datasets, five distinct held-out patients per "
            "dataset, exact quadrature and tolerances 1e-1, 1e-2 and 1e-3, on GPU FP32 and CPU FP64.", "",
            f"All selected rules satisfy their analytical bounds C <= epsilon. Against the matching full-degree "
            f"output on the same device and precision, GPU FP32 passes {passes['gpu']}/30 patient-tolerance "
            f"cases and CPU FP64 passes {passes['cpu']}/30.", "",
            "Times are seconds, mean +/- sample standard deviation across five patients, one measured call "
            "per patient and method. Delta is the maximum absolute difference over all features and all five "
            "patients from their own-device, own-precision B=100 degree-exact output. Pass counts patients "
            "whose maximum difference is <= epsilon. C_max is the largest analytical quadrature bound across "
            "patients. Exact-row Delta=0 is self-comparison, not zero floating-point error.", ""]
    for dataset, spec in COHORTS.items():
        text += [f"{spec['label']}: {spec['n_train']} training patients, {spec['d']:,} features", ""]
        for dev, (_, _, label) in DEVICES.items():
            display = []
            for mode in MODES:
                r = indexed[(dataset, mode)]
                lo, hi = r["nodes_min"], r["nodes_max"]
                display.append(["Exact" if mode == "exact" else mode, f"{lo:,}" if lo == hi else f"{lo:,}-{hi:,}",
                                f"{r['C_max']:.3g}", f"{r[f'{dev}_seconds_mean']:,.3f} +/- {r[f'{dev}_seconds_sample_sd']:.3f}",
                                f"{r[f'delta_{dev}_vs_own_degree_exact']:.3g}", "--" if mode == "exact" else f"{r[f'{dev}_pass']}/5"])
            text += [label, plain_table(["Method", "Nodes", "C_max", "Time (s)", "Delta", "Pass"], display), ""]
    text += ["Comparison with 30 backgrounds", "",
             "The model and five patients are unchanged, and the first thirty B=100 rows are exactly the "
             "previous background sample. Increasing B changes the empirical background game and its Shapley "
             "values. These comparisons contrast numerical discrepancies within each game, not errors "
             "relative to one shared target. A larger background does not guarantee less FP32 rounding "
             "error; changes should not be described as an unconditional benefit of averaging.", ""]
    for dataset, spec in COHORTS.items():
        text.append(spec["label"] + ": B=30 -> B=100")
        for mode in MODES[1:]:
            old, new = prior[(dataset, mode)], indexed[(dataset, mode)]
            text.append(f"  {mode}: GPU Delta {float(old['delta_gpu_vs_gpu_fp32_degree_exact']):.3g} -> "
                        f"{new['delta_gpu_vs_own_degree_exact']:.3g}, pass {old['gpu_pass']}/5 -> {new['gpu_pass']}/5; "
                        f"CPU Delta {float(old['delta_cpu_vs_cpu_fp64_degree_exact']):.3g} -> "
                        f"{new['delta_cpu_vs_own_degree_exact']:.3g}, pass {old['cpu_pass']}/5 -> {new['cpu_pass']}/5.")
        text.append("")
    text += ["Timing scope and interpretation", "",
             "All B=100 attributions and timings, including both full-degree references, are newly measured. "
             "The quadrature rules are cached. Approximate timings are synchronized API wall times including "
             "automatic node selection, data transfers and host accumulation, after an excluded warm-up "
             "for each distinct dataset/node-count configuration. Exact timings accumulate active "
             "checkpointed integration, preserving the public API's factor/block order; they exclude checkpoint "
             "I/O, initial output allocations, resume loading, downtime and lost uncheckpointed work. One "
             "real full-shape kernel-block warm-up and three pilot calls are excluded from exact timing; "
             "this is not a full-explanation warm-up. Model fitting and rule construction are excluded throughout.", "",
             "The Apple M4 Pro runs use an 8 GiB planning budget and one background row per block. GPU cores "
             "use FP32, CPU cores use FP64, and both accumulate on the host in FP64. CPU/GPU runtime ratios "
             "therefore change precision as well as device. Standard deviations reflect five-patient variation, "
             "not repeated-run uncertainty or statistical significance.", "",
             f"Historical CPU construction of the cached exact rules: {exact_setup['gse24080']:.2f} s (GSE24080), "
             f"{exact_setup['tcga_lgg_methylation']:.2f} s (TCGA LGG). These costs were not paid again per patient. "
             f"Fresh approximate-rule CPU construction: {small_ms['gpu'][0]:.3f}-{small_ms['gpu'][1]:.3f} ms "
             f"during the GPU suite; {small_ms['cpu'][0]:.3f}-{small_ms['cpu'][1]:.3f} ms during the CPU suite "
             "(ranges of medians over five constructions per dataset/node count). Caching rules does not "
             "eliminate patient/background-dependent integrand evaluations.", "",
             "The analytical certificate bounds quadrature error in real arithmetic for the fixed fitted "
             "relative-hazard model and chosen 100-row empirical background. It excludes rounding, "
             "fitted-model uncertainty and population-background approximation. Agreement with a rounded "
             "degree-exact reference is an empirical consistency check, not an absolute floating-point error "
             "certificate. CPU FP64 accuracy must be paired with CPU runtime, not assigned to GPU FP32 timing.", "",
             "Full-precision tables, individual errors, timings, attributions and provenance are saved in "
             "benchmarks/results/cox_background100.", ""]
    report = "\n".join(text)
    require("1e-6" not in report, "Obsolete tolerance in report")
    # All writes occur after both suites, every attribution, and supporting data validate.
    write_csv(folder / "table.csv", rows)
    write_csv(folder / "case_errors.csv", errors)
    (folder / "results.txt").write_text(report)
    validation = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                  "background_size": 100, "distinct_patients_per_dataset": 5,
                  "cases_per_device": 40, "total_cases": 80, "full_degree_exact_cases_per_device": 10,
                  "paired_approximation_cases_per_device": 30, "gpu_passes": passes["gpu"], "cpu_passes": passes["cpu"],
                  "same_device_precision_full_degree_references": True,
                  "all_case_records_attribution_hashes_and_exact_checkpoints_verified": True,
                  "first_30_backgrounds_equal_previous_study": True,
                  "previous_B30_table": str(B30_TABLE), "previous_B30_table_sha256": digest(B30_TABLE),
                  "source_reports_sha256": sources,
                  "report_sha256": {name: digest(folder / name) for name in ("table.csv", "case_errors.csv", "results.txt")},
                  "generator_sha256": digest(__file__), "email_sent_by_generator": False}
    (folder / "report_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
    print(f"Validated all eighty B=100 cases and wrote plain-text results to {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT)
    build(parser.parse_args().folder)
