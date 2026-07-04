"""Two-panel bond-dimension figure, one curve per (model, dataset), averaged over seeds.

(a) test accuracy retained vs chi: flat down to chi=1 means the decision is bond-cheap.
(b) state fidelity F(chi) between the truncated and full circuit: this climbs from low values
    toward 1, so the trained states genuinely carry entanglement.

Read together the panels are the thesis: the states are entangled (b rises) yet the predictions
do not need that entanglement (a is flat). Styled after Shin (2024): thin solid lines, small
markers, distinct saturated colors, no shaded confidence bands. Color = model, marker = dataset."""

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


def _mean_curve(runs, key):
    chis = sorted({int(k) for d in runs for k in d["details"][key] if k != "full"})
    arr = np.array([[d["details"][key].get(str(c)) for c in chis] for d in runs], float)
    return chis, np.nanmean(arr, 0)


def _series_style(model, dataset):
    return dict(
        color=PALETTE.get(model, "#000000"),
        marker=MARKER.get(dataset, "o"),
        linestyle="-", linewidth=1.6,
        markersize=5.5, markeredgecolor="white", markeredgewidth=0.6,
        label=f"{model} · {DATA_LABEL.get(dataset, dataset)}",
        zorder=3,
    )


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

    # stable order: worst chi=1 fidelity first, so the legend reads by "how far from product"
    def chi1_fid(runs):
        _, f = _mean_curve(runs, "fidelity_by_chi")
        return f[0]

    ordered = sorted(groups.items(), key=lambda kv: chi1_fid(kv[1]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.7), sharex=True)

    for (model, dataset), runs in ordered:
        st = _series_style(model, dataset)
        chis, acc = _mean_curve(runs, "accuracy_by_chi")
        axA.plot(chis, acc, **st)
        chis, fid = _mean_curve(runs, "fidelity_by_chi")
        axB.plot(chis, fid, **{k: v for k, v in st.items() if k != "label"})

    # product-state guide on both panels
    for ax in (axA, axB):
        ax.axvspan(0.9, 1.0, color="0.85", alpha=0.35, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlim(0.9, 17.5)
        ax.set_xlabel(r"MPS bond dimension $\chi$")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)

    axA.set_ylim(0.30, 1.05)
    axA.set_ylabel("test accuracy retained")
    axA.set_title(r"(a) predictions recovered at $\chi{=}1$", fontsize=11.5)
    axA.text(1.06, 0.335, "product\nstate", fontsize=7.5, color="0.4", va="bottom")

    axB.set_ylim(0.13, 1.03)
    axB.set_ylabel(r"state fidelity $F(\chi)$ to full circuit")
    axB.set_title(r"(b) state fidelity climbs with $\chi$", fontsize=11.5)
    axB.axhline(1.0, color="0.6", lw=0.8, ls=(0, (1, 2)), zorder=1)

    # one shared legend beneath both panels
    handles, labels = axA.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower center", ncol=6, fontsize=8.4,
        frameon=True, handlelength=1.9, columnspacing=1.3, labelspacing=0.4,
        borderpad=0.6, bbox_to_anchor=(0.5, -0.01),
    )
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.28, wspace=0.17)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"chi_curves.{ext}")
    print(f"wrote chi_curves.pdf/.png ({sum(len(v) for v in groups.values())} runs, {len(groups)} series)")


if __name__ == "__main__":
    main()
