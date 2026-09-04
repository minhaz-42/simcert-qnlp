"""Reproduction-gap table: our audited full-accuracy vs the papers' published numbers.

Only models we can compare to a published number on a given dataset are shown; the
synthetic 'mc' is excluded (it is our stand-in, not the papers' data). Generated from
committed results/metrics/*.json.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from simcert.io_results import select_runs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "tables" / "reproduction.tex"

# Published test accuracies from the papers (see references/ + docs/implementation-plan.md)
PUBLISHED = {
    ("qsann", "mc_real"): 1.000, ("qsann", "rp"): 0.677,
    ("qmsan", "mc_real"): 1.000, ("qmsan", "rp"): 0.756,
    ("discocat", "mc_real"): 0.798, ("discocat", "rp"): 0.723,
    ("vqc_text", "sst2"): None,  # our reference model; no external number
}


def main():
    # Same selection rule as the certificate table, so the two never disagree on how
    # many seeds a row has.
    groups = {k: [d["certificate"]["full_accuracy"] for d in v]
              for k, v in select_runs(glob.glob(str(REPO / "results" / "metrics" / "*.json"))).items()}

    rows = [r"\begin{tabular}{llccc}", r"\toprule",
            r"Model & Data & seeds & ours (test) & published \\", r"\midrule"]
    printed = ["model         data      seeds  ours          published  gap"]
    for (model, dataset), accs in sorted(groups.items()):
        pub = PUBLISHED.get((model, dataset))
        if pub is None and (model, dataset) not in PUBLISHED:
            continue  # skip synthetic mc etc.
        ours = f"{np.mean(accs):.3f}$\\pm${np.std(accs):.3f}"
        pub_s = f"{pub:.3f}" if pub is not None else "n/a"
        rows.append(f"\\texttt{{{model.replace('_', chr(92)+'_')}}} & {dataset.replace('_', chr(92)+'_')} & "
                    f"{len(accs)} & {ours} & {pub_s} \\\\")
        gap = f"{np.mean(accs)-pub:+.3f}" if pub is not None else "n/a"
        printed.append(f"{model:<13} {dataset:<9} {len(accs):>4}   "
                       f"{np.mean(accs):.3f}+/-{np.std(accs):.3f}   "
                       f"{pub_s:>8}   {gap}")
    rows += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n")
    print("\n".join(printed))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
