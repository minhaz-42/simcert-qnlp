"""Does the chi*-vs-entanglement anti-correlation reach the audited models?

    python scripts/check_chistar_faithfulness.py

Regenerates the numbers quoted in the RESOLVED section of
docs/positive-control-study.md. The structural half of that argument (every audited model
reads a weight-1 Pauli, so the control's exact-cancellation mode cannot occur) is pinned
separately by tests/test_readout_weight.py.

The pathology, as found on the candidate positive control: chi*=1 certified while the
chi=1 product surrogate has tiny state fidelity (1.6%). That decouples chi* from the
quantity it is supposed to track. This scans EVERY stored run for the same signature.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

rows = []
for f in sorted(glob.glob(str(REPO / "results" / "metrics" / "*.json"))):
    d = json.loads(Path(f).read_text())
    c = d.get("certificate")
    if not c:
        continue
    fid = (c.get("fidelity_by_chi") or {}).get("1")
    agr = (c.get("agreement_by_chi") or {}).get("1")
    cs = (c.get("chi_star") or {}).get("tau_gen")
    if fid is None:
        continue
    rows.append(dict(model=c["model"], dataset=c["dataset"],
                     audit=(d.get("config", {}) or {}).get("audit_name"),
                     nq=((d.get("config", {}) or {}).get("model") or {}).get("n_qubits"),
                     seed=(d.get("config", {}) or {}).get("seed"),
                     fid=float(fid), agr=(float(agr) if agr is not None else None),
                     chistar=cs, ent=c.get("entropy_mean")))

print(f"scanned {len(rows)} runs with a recorded chi=1 fidelity\n")

# The pathological signature: chi* == 1 (certified simulable at a product state) while the
# product surrogate is nowhere near the true state.
LOW = 0.5
bad = [r for r in rows if r["chistar"] == 1 and r["fid"] < LOW]
print(f"runs certifying chi*=1 with chi=1 fidelity < {LOW}: {len(bad)}")
for r in bad[:40]:
    print(f"  {r['model']}/{r['dataset']} n={r['nq']} seed={r['seed']} "
          f"F={r['fid']:.4f} agree={r['agr']} S={r['ent']}")

print("\n--- fidelity@chi=1 by model/dataset, alongside certified chi* ---")
hdr = f"{'model/data':<22}{'n':>4}{'runs':>6}{'F@1 min':>9}{'F@1 mean':>10}{'chi* set':>12}{'agree@1 mean':>14}"
print(hdr); print("-"*len(hdr))
from collections import defaultdict
g = defaultdict(list)
for r in rows:
    g[(r["model"], r["dataset"], r["nq"])].append(r)
for k in sorted(g, key=lambda k: (k[0], k[1], k[2] or 0)):
    v = g[k]
    fids = [x["fid"] for x in v]
    cs = sorted({str(x["chistar"]) for x in v})
    ag = [x["agr"] for x in v if x["agr"] is not None]
    print(f"{k[0]+'/'+k[1]:<22}{str(k[2]):>4}{len(v):>6}{min(fids):>9.4f}{np.mean(fids):>10.4f}"
          f"{'/'.join(cs):>12}{(np.mean(ag) if ag else float('nan')):>14.4f}")

# Correlation across all runs: if chi* tracked fidelity we expect chi*=1 runs to have
# HIGH fidelity and chi*>1 runs LOW. Anti-correlation would be the reverse.
one = [r["fid"] for r in rows if r["chistar"] == 1]
more = [r["fid"] for r in rows if r["chistar"] not in (1, None)]
print(f"\nmean F@chi=1 for runs certified chi*=1  : {np.mean(one):.4f}  (n={len(one)})")
if more:
    print(f"mean F@chi=1 for runs certified chi*>1  : {np.mean(more):.4f}  (n={len(more)})")
    print("expected if chi* is faithful: the chi*=1 group has the HIGHER fidelity")
    print("VERDICT:", "faithful (no anti-correlation)" if np.mean(one) > np.mean(more)
          else "ANTI-CORRELATED -- same pathology as the control")
