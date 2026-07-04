"""Accuracy retained vs MPS bond dimension chi, aggregated by (model, dataset),
mean +/- std over seeds. Color = model, line style + marker = dataset (grayscale-safe)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_rc, style_of  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"


def main():
    apply_rc()
    groups = defaultdict(list)
    for f in sorted(METRICS.glob("*.json")):
        d = json.loads(f.read_text())
        if "details" in d and "certificate" in d:
            c = d["certificate"]
            # skip the scaling sweep here (its own figure); keep one n per model
            if "16" in d["details"]["accuracy_by_chi"] and c["model"] == "vqc_text" and c["dataset"] == "sst2":
                continue
            groups[(c["model"], c["dataset"])].append(d)
    if not groups:
        print("no results yet")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (model, dataset), runs in sorted(groups.items()):
        chis = sorted({int(k) for d in runs for k in d["details"]["accuracy_by_chi"] if k != "full"})
        arr = np.array([[d["details"]["accuracy_by_chi"].get(str(c)) for c in chis] for d in runs], float)
        mean, std = arr.mean(0), arr.std(0)
        st = style_of(model, dataset)
        ax.plot(chis, mean, label=f"{model} / {dataset}", **st)
        ax.fill_between(chis, mean - std, mean + std, color=st["color"], alpha=0.12, linewidth=0)

    ax.axvline(1, color="0.4", lw=1.0, ls=(0, (1, 1)))
    ax.text(1.02, ax.get_ylim()[0] + 0.02, r"$\chi{=}1$: product state", fontsize=8.5, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"MPS bond dimension  $\chi$   (log$_2$ scale)")
    ax.set_ylabel("test accuracy retained")
    ax.set_title("A classical surrogate recovers the predictions at small $\\chi$")
    ax.legend(ncol=2, framealpha=0.9, loc="center right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"chi_curves.{ext}")
    print(f"wrote chi_curves.pdf/.png ({sum(len(v) for v in groups.values())} runs)")


if __name__ == "__main__":
    main()
