"""Reproduction gap: our audited accuracy against the published number.

Every seed is drawn, not just a mean and a whisker, because the spread is the point. On
RP the seed distribution is wide enough that the published value falls inside it for two
of the three models, so a bare mean-minus-published number misreads as a systematic
shortfall when it is largely seed variance on a 31-example test set.

Two things this script got wrong before and that are fixed here:

  * It globbed results/metrics directly and averaged EVERY stored run, including
    superseded duplicates of the same seed. That put 23 runs into the discocat/RP bar
    where the shared selection rule keeps 20, and the figure reported a gap of -0.17
    against the table's -0.158 for the same quantity. It now uses ``select_runs``, the
    same rule the tables use, so the figure and Table 3 cannot disagree.
  * It hard-coded its own two colours instead of taking the shared pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import INK_MUTED, PAIR, apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from simcert.io_results import select_runs  # noqa: E402

METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"

PUBLISHED = {
    ("discocat", "mc_real"): 0.798, ("discocat", "rp"): 0.723,
    ("qsann", "rp"): 0.677, ("qmsan", "rp"): 0.756,
}
LABEL = {"mc_real": "MC-real", "rp": "RP"}


def main():
    apply_rc()
    selected = select_runs(METRICS.glob("*.json"))
    ours = {k: [d["certificate"]["full_accuracy"] for d in v]
            for k, v in selected.items() if k in PUBLISHED}

    keys = [k for k in PUBLISHED if k in ours]
    x = np.arange(len(keys))
    w = 0.30

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, k in enumerate(keys):
        accs = np.asarray(ours[k])
        pub = PUBLISHED[k]
        mean = float(accs.mean())

        ax.bar(i - w / 2, mean, w, color=PAIR[0], edgecolor="white", linewidth=0.8,
               label="ours, seed mean" if i == 0 else None, zorder=2)
        ax.bar(i + w / 2, pub, w, color=PAIR[1], edgecolor="white", linewidth=0.8,
               label="published" if i == 0 else None, zorder=2)

        # Every seed, jittered, so the reader sees the distribution the mean came from.
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-w * 0.32, w * 0.32, size=accs.size)
        ax.scatter(np.full(accs.size, i - w / 2) + jitter, accs, s=13,
                   facecolor="white", edgecolor=INK_MUTED, linewidth=0.7, zorder=4,
                   label="individual seeds" if i == 0 else None)

        # Does the published value sit inside our seed range?
        lo, hi = float(accs.min()), float(accs.max())
        inside = lo <= pub <= hi
        # Both notes go ABOVE the bars: on top of a filled bar they had almost no
        # contrast, and the legend was landing on the right-hand gap label.
        ax.annotate(f"{mean - pub:+.3f}", (i, max(mean, pub) + 0.10), ha="center",
                    fontsize=9.5, color=INK_MUTED, weight="bold")
        ax.annotate("published inside\nour seed range" if inside
                    else "published above\nour best seed",
                    (i, max(mean, pub) + 0.035), ha="center", fontsize=7.2,
                    color=INK_MUTED if inside else "#c2382f")

    ax.axhline(0.5, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.annotate("chance", (len(keys) - 0.62, 0.5), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\mathtt{{{m.replace('_', chr(92) + '_')}}}$\n{LABEL.get(d, d)}"
                        for (m, d) in keys])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.13)
    ax.set_title("Reproduction gap, with the seed distribution behind each mean")
    # Below the axes: every in-axes corner is occupied by a bar or an annotation.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.5,
              frameon=False)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"repro.{ext}")
    print("wrote repro.pdf/.png")
    for k in keys:
        a = np.asarray(ours[k])
        pub = PUBLISHED[k]
        print(f"  {k[0]}/{k[1]}: n={a.size} mean={a.mean():.3f} "
              f"range=[{a.min():.3f},{a.max():.3f}] pub={pub:.3f} "
              f"gap={a.mean() - pub:+.3f} inside={a.min() <= pub <= a.max()}")


if __name__ == "__main__":
    main()
