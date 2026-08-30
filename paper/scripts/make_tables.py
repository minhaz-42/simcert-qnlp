"""Generate paper/tables/results.tex from committed results/metrics/*.json.

Aggregates across seeds (mean +/- std) per (model, dataset). Every number in the
paper is produced here from stored results -- none is typed by hand.
"""

from __future__ import annotations

import glob
import json
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


def main():
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(REPO / "results" / "metrics" / "*.json"))):
        d = json.load(open(f))
        c = d["certificate"]
        groups[(c["model"], c["dataset"])].append(c)

    lines = [
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Model & Data & seeds & Full acc & Acc@$\chi{=}1$ & $p_{\mathrm{McN}}^{\chi=1}$ & "
        r"$\chi^\star$ & $\bar{S}$ (nats) & $\Delta A_{\mathrm{ent}}$ \\",
        r"\midrule",
    ]
    for (model, dataset), certs in sorted(groups.items()):
        n = len(certs)
        full = [c["full_accuracy"] for c in certs]
        acc1 = [c["accuracy_by_chi"].get("1") for c in certs]
        pmcn = [(c.get("mcnemar_p_by_chi") or {}).get("1") for c in certs]
        cstar = [c["chi_star"].get("tau_gen") for c in certs]
        cstar_txt = (
            "/".join(sorted({str(x) for x in cstar})) if any(x is not None for x in cstar) else "full"
        )
        ent = [c["entropy_mean"] for c in certs]
        dent = [c["delta_ent"] for c in certs]
        lines.append(
            f"\\texttt{{{model.replace('_', chr(92) + '_')}}} & {dataset.replace('_', chr(92)+'_')} & {n} & "
            f"{_fmt(full)} & {_fmt(acc1)} & {_fmt_p(pmcn)} & {cstar_txt} & {_fmt(ent)} & {_fmt(dent)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
