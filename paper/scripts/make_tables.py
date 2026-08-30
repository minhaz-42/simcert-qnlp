"""Generate paper/tables/results.tex from committed results/metrics/*.json.

Aggregates across seeds (mean +/- std) per (model, dataset). Every number in the
paper is produced here from stored results -- none is typed by hand.

Run selection matters as much as the aggregation. results/metrics accumulates every
run ever stored, which includes the chi*-vs-n scaling sweeps (a different audit config
at several qubit counts) and superseded runs from earlier code revisions. Averaging
those into a row would report runs at n=4,6,8,10 as though they were seeds of one
configuration. ``select_runs`` therefore keeps only the default-audit runs at the
qubit count of the most recent run for that (model, dataset), and keeps one run per
seed, the newest.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "tables" / "results.tex"


def _fmt(vals, prec=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{prec}f}"
    return f"{np.mean(vals):.{prec}f}$\\pm${np.std(vals):.{prec}f}"


def _fmt_p(vals):
    """Mean McNemar p-value across seeds. A large value means the product-state
    surrogate is not significantly different from the full model."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    m = float(np.mean(vals))
    return r"$<$0.001" if m < 0.001 else f"{m:.3f}"


def _fmt_d(vals):
    """Mean number of discordant pairs behind the McNemar p-value. Zero means the
    truncated surrogate reproduced every single test prediction, which is a stronger
    statement than a large p-value obtained from a balanced discordant split."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    m = float(np.mean(vals))
    return "0" if m == 0 else f"{m:.1f}"


def select_runs(files):
    """Group stored runs into one comparable set of seeds per (model, dataset).

    Drops the scaling-sweep audits, pins each row to the qubit count and audit config of
    its most recent run, and de-duplicates seeds by keeping the newest file. Returns
    {(model, dataset): [certificate, ...]}.
    """
    recs = []
    for f in sorted(files, key=os.path.getmtime):  # oldest -> newest
        d = json.load(open(f))
        cfg = d.get("config", {})
        if cfg.get("audit_name") == "scaling":
            continue  # belongs to the chi*-vs-n figure, not the per-seed certificate table
        c = d["certificate"]
        recs.append({
            "key": (c["model"], c["dataset"]),
            "variant": ((cfg.get("model") or {}).get("n_qubits"), cfg.get("audit_name")),
            "seed": cfg.get("seed"),
            "cert": c,
        })

    groups = defaultdict(list)
    for r in recs:
        groups[r["key"]].append(r)

    out = {}
    for key, rs in groups.items():
        current = rs[-1]["variant"]  # newest run defines the current configuration
        rs = [r for r in rs if r["variant"] == current]
        by_seed = {}
        for r in rs:  # ascending mtime, so the last write per seed wins
            by_seed[r["seed"]] = r["cert"]
        out[key] = [by_seed[s] for s in sorted(by_seed, key=lambda x: (x is None, x))]
    return out


def main():
    groups = select_runs(glob.glob(str(REPO / "results" / "metrics" / "*.json")))

    lines = [
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Model & Data & seeds & Full acc & Acc@$\chi{=}1$ & $p_{\mathrm{McN}}^{\chi=1}$ & "
        r"$d^{\chi=1}$ & $\chi^\star$ & $\bar{S}$ (nats) & $\Delta A_{\mathrm{ent}}$ \\",
        r"\midrule",
    ]
    for (model, dataset), certs in sorted(groups.items()):
        n = len(certs)
        full = [c["full_accuracy"] for c in certs]
        acc1 = [c["accuracy_by_chi"].get("1") for c in certs]
        mcn = [(c.get("mcnemar_by_chi") or {}).get("1") or {} for c in certs]
        pmcn = [m.get("p_value") for m in mcn]
        dmcn = [m.get("n_discordant") for m in mcn]
        cstar = [c["chi_star"].get("tau_gen") for c in certs]
        cstar_txt = (
            "/".join(sorted({str(x) for x in cstar})) if any(x is not None for x in cstar) else "full"
        )
        ent = [c["entropy_mean"] for c in certs]
        dent = [c["delta_ent"] for c in certs]
        lines.append(
            f"\\texttt{{{model.replace('_', chr(92) + '_')}}} & {dataset.replace('_', chr(92)+'_')} & {n} & "
            f"{_fmt(full)} & {_fmt(acc1)} & {_fmt_p(pmcn)} & {_fmt_d(dmcn)} & {cstar_txt} & "
            f"{_fmt(ent)} & {_fmt(dent)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
