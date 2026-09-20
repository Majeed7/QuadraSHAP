"""Validate the saved study and audit every stricter-tolerance target failure."""
from __future__ import annotations
import argparse
from decimal import Decimal as D, localcontext
import hashlib
import json
from pathlib import Path
import numpy as np
from exp8_multimodel_budget import write_csv, write_json
from exp8_precision_audit import decimal_quad


def main(root):
    rows=json.loads((root/"validated_records.json").read_text())
    original=json.loads((root/"records.json").read_text())
    assert len(rows)==len(original)==16400
    assert len(list(root.glob("*/fit.json")))==175
    assert len(list(root.glob("*/instance*.npz")))==4100
    assert all(not json.loads(p.read_text())["convergence_warnings"] for p in root.glob("*/fit.json"))
    assert len({(r["family"],r["regime"],r["d"],r["seed"],r["instance"],r["value_function"],r["epsilon"]) for r in rows})==len(rows)
    main_rows=[r for r in rows if r["epsilon"]==1e-6]
    assert all(r["target_met"] and r["reference_resolved"] for r in main_rows)
    assert all(1<=r["m_certified"]<=r["exact_threshold"] for r in rows)
    assert all(r["bound_at_certified"]<=r["epsilon"] for r in rows)
    assert all(r["m_observed"] is not None and 1<=r["m_observed"]<=r["m_certified"] for r in main_rows)
    assert all(np.isfinite(r["A_max"]) and np.isfinite(r["lambda_max"]) for r in rows)
    tests=json.loads((root/"reference_tests.json").read_text())
    audit=json.loads((root/"precision_audit.json").read_text())
    assert len(tests)==29 and len(audit)==24 and all(v["below_bound"] for v in audit)
    failure_rows=[]
    for r in rows:
        if r["target_met"] or r["epsilon"]!=1e-9:
            continue
        folder=root/f"{r['regime']}_{r['family']}_d{r['d']}_s{r['seed']}"
        hp=json.loads((folder/(Path(r["npz"]).stem+"_hp.json")).read_text())
        a=np.load(folder/r["npz"])
        at=int(np.flatnonzero(a["ms"]==r["m_certified"])[0])
        with localcontext() as context:
            context.prec=50
            reference=[D(v) for v in hp["phi_reference"]]
            errors=[abs(D(float(v))-ref) for v,ref in zip(a["phi"][at],reference)]
            i=int(np.argmax([float(v) for v in errors]))
            q=decimal_quad(a["K"],a["Ut"],a["w"],i,r["m_certified"])
            truncation=abs(q-reference[i])
            rounding=abs(D(float(a["phi"][at,i]))-q)
            assert truncation<=D(r["bound_at_certified"])
            failure_rows.append(dict(family=r["family"],value_function=r["value_function"],seed=r["seed"],instance=r["instance"],
                                     d=r["d"],coordinate=i,m_q=r["m_certified"],epsilon=1e-9,
                                     float64_error=float(errors[i]),quadrature_error_50digit=float(truncation),
                                     rounding_error=float(rounding),certified_bound=r["bound_at_certified"]))
        print(f"checked stricter-target failure {r['family']} seed={r['seed']} instance={r['instance']} {r['value_function']}",flush=True)
    assert len(failure_rows)==18
    write_json(root/"strict_tolerance_failure_audit.json",failure_rows)
    write_csv(root/"strict_tolerance_failure_audit.csv",failure_rows)
    validation=dict(n_fits=175,n_explanation_cases=4100,n_tolerance_records=16400,
                    reference_checks_passed=29,selected_high_precision_checks_passed=24,
                    stricter_target_failures_audited=18,convergence_warnings=0,
                    main_target_met=4100,main_target_failures=0,main_reference_flags_remaining=0,
                    max_main_error_at_certified=max(r["error_at_certified"] for r in main_rows),
                    max_certified_d1000=max(r["m_certified"] for r in main_rows if r["d"]==1000),
                    existing_repo_tests="10 passed, JAX_ENABLE_X64=true JAX_PLATFORMS=cpu; production explicitly NumPy float64")
    write_json(root/"validation.json",validation)
    print(validation,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    main(Path(parser.parse_args().output))
