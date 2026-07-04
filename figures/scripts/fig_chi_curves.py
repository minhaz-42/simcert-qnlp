"""Headline figure: accuracy retained vs MPS bond dimension chi.

Reads every committed results/metrics/*.json and plots one accuracy-vs-log2(chi)
curve per (model, dataset), with the full-chi accuracy line and the chi* marker.
Regenerates deterministically from stored results (no retraining).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"


def _load():
    runs = []
    for f in sorted(METRICS.glob("*.json")):
        d = json.loads(f.read_text())
        if "details" in d and "certificate" in d:
            runs.append(d)
    return runs


def main():
    runs = _load()
    if not runs:
        print("no results in results/metrics/ yet — run the audit first")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d in runs:
        det = d["details"]
        cert = d["certificate"]
        acc = det["accuracy_by_chi"]
        finite = sorted(int(k) for k in acc if k != "full")
        xs = finite
        ys = [acc[str(c)] if str(c) in acc else acc[c] for c in finite]
        label = f"{cert['model']} / {cert['dataset']} ({cert['verdict']})"
        ax.plot(xs, ys, marker="o", label=label)
        full_acc = cert["full_accuracy"]
        ax.axhline(full_acc, ls="--", lw=0.8, alpha=0.5)
        cstar = cert["chi_star"].get("tau_gen")
        if cstar:
            ax.axvline(cstar, ls=":", lw=0.9, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"MPS bond dimension $\chi$ (log$_2$)")
    ax.set_ylabel("test accuracy retained")
    ax.set_title("Accuracy retained vs bond dimension (was the quantum load-bearing?)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"chi_curves.{ext}", dpi=150)
    print(f"wrote {OUT / 'chi_curves.pdf'} and .png  ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
