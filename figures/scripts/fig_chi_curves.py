"""Headline figure: accuracy retained vs MPS bond dimension chi.

Reads every committed results/metrics/*.json, groups runs by (model, dataset), and plots
the mean accuracy-vs-chi curve with a +/-1 std band across seeds. Regenerates
deterministically from stored results (no retraining).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"


def main():
    groups = defaultdict(list)
    for f in sorted(METRICS.glob("*.json")):
        d = json.loads(f.read_text())
        if "details" in d and "certificate" in d:
            c = d["certificate"]
            groups[(c["model"], c["dataset"])].append(d)
    if not groups:
        print("no results in results/metrics/ yet — run the audit first")
        return

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for (model, dataset), runs in sorted(groups.items()):
        # union of finite chi values across the group's runs
        chis = sorted({int(k) for d in runs for k in d["details"]["accuracy_by_chi"] if k != "full"})
        curves = []
        for d in runs:
            acc = d["details"]["accuracy_by_chi"]
            curves.append([acc.get(str(c), acc.get(c)) for c in chis])
        arr = np.array(curves, dtype=float)
        mean, std = arr.mean(0), arr.std(0)
        verdicts = {d["certificate"]["verdict"] for d in runs}
        vshort = "/".join(sorted(v.split("_")[0][:4] for v in verdicts))
        label = f"{model} · {dataset} (n={len(runs)}, {vshort})"
        line = ax.plot(chis, mean, marker="o", label=label)[0]
        ax.fill_between(chis, mean - std, mean + std, alpha=0.15, color=line.get_color())

    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"MPS bond dimension $\chi$ (log$_2$)")
    ax.set_ylabel("test accuracy retained (mean $\\pm$ std)")
    ax.set_title("Was the quantum load-bearing? Accuracy retained vs bond dimension")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"chi_curves.{ext}", dpi=150)
    n_runs = sum(len(v) for v in groups.values())
    print(f"wrote {OUT / 'chi_curves.pdf'} and .png  ({len(groups)} groups, {n_runs} runs)")


if __name__ == "__main__":
    main()
