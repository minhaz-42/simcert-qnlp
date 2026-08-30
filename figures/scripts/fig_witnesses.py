"""Two-panel witness figure on MC (all five models):
(a) mean bipartite entanglement entropy per model  -> entanglement IS present;
(b) full accuracy vs accuracy of the product-state (chi=1) surrogate -> predictions are
    retained at chi=1 for every model except discocat (its cups need chi=2).
Together: entanglement present, but (mostly) not load-bearing for the decision."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from simcert.io_results import select_runs  # noqa: E402
from _style import PALETTE, apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"
ORDER = ["vqc_text", "qsann", "qmsan", "claqs", "discocat"]


def main():
    apply_rc()
    ent, full, chi1 = defaultdict(list), defaultdict(list), defaultdict(list)
    # Shared selection so a seed superseded by a re-audit is not counted twice.
    for (m, dataset), runs in select_runs(METRICS.glob("*.json")).items():
        if dataset != "mc":
            continue
        for d in runs:
            c = d["certificate"]
            if c.get("entropy_mean") is not None:
                ent[m].append(c["entropy_mean"])
            full[m].append(c["full_accuracy"])
            chi1[m].append(c["accuracy_by_chi"].get("1"))
    models = [m for m in ORDER if m in full]
    x = np.arange(len(models))
    colors = [PALETTE.get(m, "#444") for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # (a) entanglement entropy per model
    means = [np.mean(ent[m]) for m in models]
    errs = [np.std(ent[m]) for m in models]
    ax1.bar(x, means, yerr=errs, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=15)
    ax1.set_ylabel(r"mean entanglement entropy $\bar S$ (nats)")
    ax1.set_title("(a) the trained circuits carry entanglement")
    # clear the error bar, not just the bar: the cap sits at mu+sigma, so a fixed offset
    # from mu draws the label straight through the whisker whenever sigma exceeds it
    for xi, mu, sd in zip(x, means, errs):
        ax1.text(xi, mu + sd + 0.04, f"{mu:.2f}", ha="center", fontsize=9)
    ax1.set_ylim(0, max(m + e for m, e in zip(means, errs)) * 1.16)

    # (b) full accuracy vs chi=1 accuracy
    w = 0.38
    fmean = [np.mean(full[m]) for m in models]
    c1mean = [np.mean(chi1[m]) for m in models]
    ax2.bar(x - w / 2, fmean, w, label="full model", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax2.bar(x + w / 2, c1mean, w, label=r"$\chi{=}1$ surrogate (product state)",
            color="#DD8452", edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=15)
    ax2.set_ylabel("MC test accuracy")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("(b) a product state recovers the predictions")
    ax2.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.annotate("cups are\nload-bearing", xy=(len(models) - 1 + w / 2, c1mean[-1]),
                 xytext=(len(models) - 1.9, 0.30), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.3"))

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"witnesses.{ext}")
    print("wrote witnesses.pdf/.png")


if __name__ == "__main__":
    main()
