"""50-decimal-digit audit of saved experiment-8 boundary/floor cases.

Selects difficult cases by a documented diagnostic, never removes them from
the study. Recomputes both the rule (Newton-refined Legendre roots) and the
independent Bernstein reference. Only the selected attribution coordinate is
audited, explicitly recorded; this is not interval certification of all outputs.
"""
from __future__ import annotations
import argparse
from decimal import Decimal as D, localcontext
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
from exp8_multimodel_budget import write_json, write_csv


def legendre(n, x):
    a, b = D(1), x
    for k in range(1, n):
        a, b = b, ((2*k+1)*x*b-k*a)/(k+1)
    return b, n*(x*b-a)/(x*x-1)


@lru_cache(None)
def rule(m):
    ts, ws = [], []
    for value in roots_legendre(m)[0]:
        x = D(float(value))
        for _ in range(20):
            p, dp = legendre(m, x)
            step = p/dp
            x -= step
            if abs(step) < D("1e-46"):
                break
        _, dp = legendre(m, x)
        ts.append((1+x)/2)
        ws.append(1/((1-x*x)*dp*dp))
    return ts, ws


def decimal_reference(K, Ut, w, i):
    total = D(0)
    for kr, ar, wr in zip(K, Ut, w):
        b = [D(1)]
        for j in range(len(kr)):
            if j == i:
                continue
            lo, hi = D(float(ar[j])), D(float(ar[j])) + D(float(kr[j]))
            n = len(b)
            out = [D(0)] * (n+1)
            for k, bk in enumerate(b):
                out[k] += (n-k)*lo*bk/n
                out[k+1] += (k+1)*hi*bk/n
            b = out
        total += D(float(wr)) * D(float(kr[i])) * sum(b) / len(b)
    return total


def decimal_quad(K, Ut, w, i, m):
    ts, ws = rule(m)
    total = D(0)
    for kr, ar, wr in zip(K, Ut, w):
        ks = [D(float(v)) for v in kr]
        absent = [D(float(v)) for v in ar]
        value = D(0)
        for t, q in zip(ts, ws):
            product = D(1)
            for j in range(len(ks)):
                if j != i:
                    product *= absent[j] + t*ks[j]
            value += q*product
        total += D(float(wr))*ks[i]*value
    return total


def audit(output):
    record_file = output / "records.json"
    records = (json.loads(record_file.read_text()) if record_file.exists() else
               [r for f in output.glob("*/records.json") for r in json.loads(f.read_text())])
    rows = []
    for family in ("poisson", "logistic", "naive_bayes"):
        for vf in ("baseline", "interventional"):
            candidates = [r for r in records if r["family"] == family and r["regime"] == "fixed_feature"
                          and r["d"] == 1000 and r["epsilon"] == 1e-6 and r["value_function"] == vf]
            if not candidates:
                continue
            r = max(candidates, key=lambda v: v["reference_disagreement"])
            case = f"{r['regime']}_{family}_d{r['d']}_s{r['seed']}"
            arrays = np.load(output / case / r["npz"])
            K, Ut, w = (arrays[k] for k in ("K", "Ut", "w"))
            ms = arrays["ms"]
            # Audit the feature with the largest discrepancy at the certified budget.
            at = int(np.flatnonzero(ms == r["m_certified"])[0])
            i = int(np.argmax(abs(arrays["phi"][at] - arrays["phi_reference"])))
            with localcontext() as context:
                context.prec = 50
                ref = decimal_reference(K, Ut, w, i)
                test_ms = sorted(set([max(1, (r["m_observed"] or 1)-1), r["m_observed"] or 1,
                                      r["m_certified"], min(int(ms[-2]), r["m_certified"]+5)]))
                for m in test_ms:
                    phi = decimal_quad(K, Ut, w, i, m)
                    idx = int(np.flatnonzero(ms == m)[0])
                    error = abs(phi-ref)
                    rows.append(dict(family=family, value_function=vf, d=1000, seed=r["seed"],
                                     instance=r["instance"], coordinate=i, m_q=m,
                                     phi_reference_decimal=str(ref), quadrature_error_decimal=str(error),
                                     quadrature_error=float(error),
                                     saved_reference_error=float(abs(D(float(arrays['phi_reference'][i]))-ref)),
                                     implementation_rounding_error=float(abs(D(float(arrays['phi'][idx,i]))-phi)),
                                     certified_bound=float(arrays["certified_bound"][idx]),
                                     below_bound=bool(error <= D(float(arrays["certified_bound"][idx])))))
            print(f"audited {family} {vf} seed={r['seed']} instance={r['instance']} feature={i}", flush=True)
            write_json(output / "precision_audit.json", rows)
            write_csv(output / "precision_audit.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    audit(Path(p.parse_args().output))
