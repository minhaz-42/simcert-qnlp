"""Reproduction-gap figure: our audited accuracy vs the published number, grouped bars,
for the (model, dataset) pairs with a published value. MC reproduces closely; RP runs below
(the fragile 74-example task), which we report honestly."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"

PUBLISHED = {
    ("discocat", "mc_real"): 0.798, ("discocat", "rp"): 0.723,
    ("qsann", "rp"): 0.677, ("qmsan", "rp"): 0.756,
}


def main():
    apply_rc()
    ours = defaultdict(list)
    for f in sorted(METRICS.glob("*.json")):
        d = json.loads(f.read_text())
        c = d["certificate"]
        key = (c["model"], c["dataset"])
        if key in PUBLISHED:
            ours[key].append(c["full_accuracy"])
    keys = [k for k in PUBLISHED if k in ours]
    labels = [f"{m}\n{ds}" for (m, ds) in keys]
    om = [np.mean(ours[k]) for k in keys]
    oe = [np.std(ours[k]) for k in keys]
    pub = [PUBLISHED[k] for k in keys]

    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.bar(x - w / 2, om, w, yerr=oe, capsize=4, label="ours (this audit)",
           color="#009E73", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, pub, w, label="published", color="#8172B3", edgecolor="black", linewidth=0.6)
    for xi, (o, p) in enumerate(zip(om, pub)):
        ax.text(xi, max(o, p) + 0.02, f"{o - p:+.2f}", ha="center", fontsize=9, color="0.25")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_title("Reproduction gap (numbers show ours minus published)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"repro.{ext}")
    print("wrote repro.pdf/.png")


if __name__ == "__main__":
    main()
