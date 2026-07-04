"""Test accuracy retained vs MPS bond dimension chi, one line per (model, dataset),
mean over seeds. Styled after Shin (2024) Fig.: thin solid lines, small markers, distinct
saturated colors, a compact boxed legend inside the axes, and no shaded confidence bands.
Color = model, marker shape = dataset (so series stay distinguishable in grayscale)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import MARKER, PALETTE, apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"

DATA_LABEL = {"mc": "MC", "mc_real": "MC-real", "rp": "RP", "sst2": "SST-2"}
MODEL_LABEL = {
    "vqc_text": "vqc_text",
    "qsann": "qsann",
    "qmsan": "qmsan",
    "claqs": "claqs",
    "discocat": "discocat",
}


def main():
    apply_rc()
    groups = defaultdict(list)
    for f in sorted(METRICS.glob("*.json")):
        d = json.loads(f.read_text())
        if "details" not in d or "certificate" not in d:
            continue
        c = d["certificate"]
        if "16" in d["details"]["accuracy_by_chi"] and c["model"] == "vqc_text" and c["dataset"] == "sst2":
            continue  # the scaling sweep has its own figure
        groups[(c["model"], c["dataset"])].append(d)
    if not groups:
        print("no results yet")
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    # order series by descending accuracy at the largest chi so the legend reads top-to-bottom
    def top_acc(runs):
        chis = sorted({int(k) for d in runs for k in d["details"]["accuracy_by_chi"] if k != "full"})
        vals = [d["details"]["accuracy_by_chi"].get(str(chis[-1])) for d in runs]
        return np.mean([v for v in vals if v is not None])

    ordered = sorted(groups.items(), key=lambda kv: -top_acc(kv[1]))

    for (model, dataset), runs in ordered:
        chis = sorted({int(k) for d in runs for k in d["details"]["accuracy_by_chi"] if k != "full"})
        arr = np.array([[d["details"]["accuracy_by_chi"].get(str(c)) for c in chis] for d in runs], float)
        mean = np.nanmean(arr, 0)
        ax.plot(
            chis, mean,
            color=PALETTE.get(model, "#000000"),
            marker=MARKER.get(dataset, "o"),
            linestyle="-", linewidth=1.5,
            markersize=5.5, markeredgecolor="white", markeredgewidth=0.6,
            label=f"{MODEL_LABEL.get(model, model)} · {DATA_LABEL.get(dataset, dataset)}",
            zorder=3,
        )

    # chi* = 1 reference marker
    ax.axvline(1, color="0.6", lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.text(1.05, 0.335, r"$\chi^\star{=}1$ product state", fontsize=8, color="0.45")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_ylim(0.30, 1.05)
    ax.set_xlim(0.9, 17.5)
    ax.set_xlabel(r"MPS bond dimension $\chi$")
    ax.set_ylabel("test accuracy retained")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)

    leg = ax.legend(
        loc="center left", bbox_to_anchor=(1.015, 0.5),
        fontsize=8.5, frameon=True, handlelength=1.9, labelspacing=0.4,
        borderpad=0.6, title="model · dataset",
    )
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.6)
    leg.get_title().set_fontsize(8.5)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"chi_curves.{ext}", bbox_inches="tight")
    print(f"wrote chi_curves.pdf/.png ({sum(len(v) for v in groups.values())} runs, {len(groups)} series)")


if __name__ == "__main__":
    main()
