"""Resolve every main-tolerance reference flag, without dropping any instance.

Recompute full exact-threshold GL attribution vectors with 50 decimal digits,
including nodes/weights, and compare every saved float64 attribution directly
to those decimal values. Raw records stay untouched. Refined records and
diagnostics are separate. This is high-precision numerical validation, not an
interval proof about floating point. Only originally flagged main-epsilon
instances need this additional computation.
"""
from __future__ import annotations
import argparse
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import numpy as np
from exp8_precision_audit import rule
from exp8_multimodel_budget import write_json, write_csv


def full_reference(K, Ut, w, m):
    ts, qs = rule(m)
    total = [D(0)] * K.shape[1]
    for kr, ar, wr in zip(K, Ut, w):
        ks, absent = [D(float(v)) for v in kr], [D(float(v)) for v in ar]
        weight = D(float(wr))
        for t, q in zip(ts, qs):
            factors = [a + t*k for a,k in zip(absent,ks)]
            product = D(1)
            for factor in factors:
                product *= factor
            weighted = weight*q*product
            for i in range(len(total)):
                total[i] += weighted*ks[i]/factors[i]
    return total


def refine(root):
    records_file=root/"records.json"
    records=(json.loads(records_file.read_text()) if records_file.exists() else
             [r for f in root.glob("*/records.json") for r in json.loads(f.read_text())])
    main_flagged=[r for r in records if r["epsilon"]==1e-6 and not r["reference_resolved"]]
    candidates=main_flagged+[r for r in records if r["epsilon"]==1e-9 and not r["target_met"]]
    flagged=list({(r["family"],r["regime"],r["d"],r["seed"],r["instance"],r["value_function"]):r
                  for r in candidates}.values())
    for r in flagged:
        case=f"{r['regime']}_{r['family']}_d{r['d']}_s{r['seed']}"
        folder=root/case
        output=folder/(Path(r["npz"]).stem+"_hp.json")
        if output.exists():
            continue
        a=np.load(folder/r["npz"])
        if len(a["w"])>8:
            raise ValueError("Unexpected flagged kernel instance: retain and investigate")
        with localcontext() as context:
            context.prec=50
            ref=full_reference(a["K"],a["Ut"],a["w"],r["exact_threshold"])
            errors=[float(max(abs(D(float(v))-ref[i]) for i,v in enumerate(phi))) for phi in a["phi"]]
            before=float(max(abs(D(float(v))-ref[i]) for i,v in enumerate(a["phi_reference"])))
            corrected=dict(precision_digits=50,reference_nodes=r["exact_threshold"],
                           method="decimal exact-degree GL rule, including 50-digit nodes and weights",
                           phi_reference=[str(v) for v in ref], ms=a["ms"].tolist(),
                           observed_error=errors, original_reference_max_error=before)
        write_json(output,corrected)
        print(f"refined {case} {r['npz']}: old reference max error={before:.3e}",flush=True)
    updated=[]
    for row in records:
        r=dict(row)
        r.update(original_m_observed=row["m_observed"],original_error_at_certified=row["error_at_certified"],
                 reference_method="float64 independent references",reference_refined=False)
        case=f"{r['regime']}_{r['family']}_d{r['d']}_s{r['seed']}"
        f=root/case/(Path(r["npz"]).stem+"_hp.json")
        if f.exists():
            hp=json.loads(f.read_text())
            ms=np.asarray(hp["ms"])
            errors=np.asarray(hp["observed_error"])
            at=int(np.flatnonzero(ms==r["m_certified"])[0])
            passing=ms[errors<=r["epsilon"]]
            r.update(m_observed=int(passing[0]) if len(passing) else None,
                     error_at_certified=float(errors[at]),target_met=bool(errors[at]<=r["epsilon"]),
                     reference_method="50-digit exact-degree GL",reference_refined=True,
                     reference_resolved=True)
        updated.append(r)
    write_json(root/"validated_records.json",updated)
    write_csv(root/"validated_records.csv",updated)
    diagnostics=dict(main_flagged_instances=len(main_flagged),refined_instances=len(flagged),total_records=len(updated),
                     main_target_failures=sum(not r["target_met"] for r in updated if r["epsilon"]==1e-6),
                     changed_main_budgets=[{k:r[k] for k in ("family","regime","d","seed","instance","value_function","original_m_observed","m_observed")}
                                           for r in updated if r["epsilon"]==1e-6 and r["m_observed"]!=r["original_m_observed"]])
    write_json(root/"reference_refinement_summary.json",diagnostics)
    print(diagnostics,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    refine(Path(p.parse_args().output))
