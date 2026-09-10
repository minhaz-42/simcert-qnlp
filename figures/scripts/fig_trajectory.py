"""Does the simulability verdict depend on how well the model learned?

The standing objection to a chi*=1 verdict is that the audited model underperforms its
published accuracy, so perhaps the circuit is bond-cheap only because it never learned
anything. This figure audits along the training path instead of only at the end.

Panel (a) shows test accuracy climbing epoch by epoch. Panel (b) plots chi* against that
same accuracy: if the objection held, chi* would rise as the model gets better. Written
from results/trajectory/*.json, produced by scripts/trajectory.py.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (  # noqa: E402
    INK_MUTED,
    MARKER,
    MODELS,
    PALETTE,
    REFERENCE,
    apply_rc,
    composite_legend,
    style_of,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "results" / "trajectory"
OUT = REPO / "figures"


def _load():
    """Group trajectory runs by (model, dataset); each entry is a list over seeds."""
    runs = defaultdict(list)
    for f in sorted(TRAJ.glob("*.json")):
        d = json.loads(f.read_text())
        runs[(d["model"], d["dataset"])].append(d)
    return runs


def main():
    apply_rc()
    runs = _load()
    if not runs:
        print("no trajectory runs yet — run scripts/trajectory.py first")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- (a) accuracy vs epoch, mean over seeds with a min-max band ---------
    for (model, dataset), seeds in sorted(runs.items()):
        epochs = sorted({p["epoch"] for d in seeds for p in d["points"]})
        by_epoch = {e: [] for e in epochs}
        for d in seeds:
            for p in d["points"]:
                by_epoch[p["epoch"]].append(p["test_accuracy"])
        mean = np.array([np.mean(by_epoch[e]) for e in epochs])
        lo = np.array([np.min(by_epoch[e]) for e in epochs])
        hi = np.array([np.max(by_epoch[e]) for e in epochs])
        st = style_of(model, dataset)
        st["markersize"] = 5
        st["markevery"] = max(1, len(epochs) // 8)
        ax1.plot(epochs, mean, **st)
        ax1.fill_between(epochs, lo, hi, color=st["color"], alpha=0.13, linewidth=0)

    ax1.axhline(0.5, color=REFERENCE, lw=1.4, ls=":", zorder=0)
    ax1.annotate("chance", (ax1.get_xlim()[1], 0.5), xytext=(-4, 4),
                 textcoords="offset points", fontsize=8, color=INK_MUTED, ha="right")
    ax1.set_xlabel("training epoch")
    ax1.set_ylabel("test accuracy")
    # Not "the models learn": on RP they do not, and a referee would see it in the panel.
    # What the panel shows is a wide accuracy range to audit against, which is what (b) needs.
    ax1.set_title("(a) MC runs reach 1.000; RP runs never leave chance")
    composite_legend(
        ax1,
        sorted({m for m, _ in runs}),
        sorted({d for _, d in runs}),
        loc="lower right", ncol=2, fontsize=7.5,
    )

    # ---- (b) chi* against the accuracy reached at that point ---------------
    # Every snapshot of every run is one point. chi* is integer-valued and the points
    # pile up, so the count at each level is stated rather than left to the ink.
    total, at_one, unrecovered = 0, 0, 0
    for (model, dataset), seeds in sorted(runs.items()):
        xs, ys = [], []
        for d in seeds:
            for p in d["points"]:
                total += 1
                cs = p["chi_star"]
                if cs is None:  # no finite chi reached the tolerance
                    unrecovered += 1
                    continue
                at_one += cs == 1
                xs.append(p["test_accuracy"])
                ys.append(cs)
        if not xs:
            continue
        ax2.scatter(xs, ys, s=46, alpha=0.55,
                    color=PALETTE.get(model, PALETTE["_default"]),
                    marker=MARKER.get(dataset, "o"),
                    edgecolor="white", linewidth=0.8, zorder=3)

    ax2.set_yscale("log", base=2)
    ax2.set_yticks([1, 2, 4, 8])
    ax2.set_yticklabels(["1", "2", "4", "8"])
    ax2.set_ylim(0.6, 10)
    ax2.set_xlabel("test accuracy at that point in training")
    ax2.set_ylabel(r"bond dimension $\chi^\star$ to recover predictions")
    ax2.set_title(r"(b) $\chi^\star$ does not track accuracy")
    frac = at_one / max(1, total - unrecovered)
    note = (f"{at_one}/{total - unrecovered} audited snapshots at "
            rf"$\chi^\star{{=}}1$ ({frac:.0%})")
    if unrecovered:
        note += f"\n{unrecovered} not recovered at any finite $\\chi$"
    ax2.annotate(note, (0.03, 0.94), xycoords="axes fraction", fontsize=8.5,
                 color=INK_MUTED, va="top")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"trajectory.{ext}")
    print("wrote trajectory.pdf/.png")

    for (model, dataset), seeds in sorted(runs.items()):
        accs = [p["test_accuracy"] for d in seeds for p in d["points"]]
        stars = sorted({str(p["chi_star"]) for d in seeds for p in d["points"]})
        print(f"  {model}/{dataset}: {len(seeds)} seeds, accuracy "
              f"{min(accs):.3f}-{max(accs):.3f}, chi* in {{{', '.join(stars)}}}")


if __name__ == "__main__":
    main()
