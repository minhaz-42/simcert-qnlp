"""Positive control: chi* tracks a dialled entanglement resource.

Answers the objection that the audit may only ever report chi*=1. One family of states
with a single integer knob k, where the bond dimension a classical surrogate provably
needs is 2^k, run through the same audit code as every model in the paper.

Panel (a) is the calibration curve: measured chi* against k, with the exact bound 2^k for
reference. Panel (b) shows why each point lands where it does, by plotting retained
accuracy against chi for every k: the decision sits at chance until chi reaches the
threshold, then snaps to 1.0. The step is exact, not statistical.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import INK_MUTED, MODELS, REFERENCE, apply_rc, seq  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures"

_spec = importlib.util.spec_from_file_location(
    "positive_control_kofm", REPO / "scripts" / "positive_control_kofm.py"
)
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

M = 4
CHIS = [1, 2, 4, 8, 16]


def main():
    apply_rc()
    rows = [pc.run_point(k, M, n_examples=128) for k in range(M + 1)]
    ks = [r["k"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # (a) calibration curve: measured chi* against the exact bound.
    exact = [r["exact_bond"] for r in rows]
    meas = [r["chi_star"] if r["chi_star"] is not None else r["exact_bond"] for r in rows]
    ax1.plot(ks, exact, color=REFERENCE, ls="--", marker="s", lw=1.8, markersize=7,
             label=r"exact bond bound $2^{k}$")
    ax1.plot(ks, meas, color=MODELS["vqc_text"], marker="o", lw=2.4, markersize=9,
             markeredgecolor="white", markeredgewidth=0.9,
             label=r"measured $\chi^\star$ (this audit)")
    for k, cs in zip(ks, meas):
        ax1.annotate(str(cs), (k, cs), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color=MODELS["vqc_text"])
    ax1.set_yscale("log", base=2)
    ax1.set_yticks([1, 2, 4, 8, 16])
    ax1.set_yticklabels(["1", "2", "4", "8", "16"])
    ax1.set_xticks(ks)
    ax1.set_xlabel(r"entangled pairs $k$ carrying the decision")
    ax1.set_ylabel(r"bond dimension $\chi^\star$")
    ax1.set_title(r"(a) $\chi^\star$ rises with the dialled resource")
    ax1.legend(loc="upper left", fontsize=8.5)

    # (b) the mechanism: accuracy against chi, one ordered curve per k.
    for i, r in enumerate(rows):
        ys = [r["acc_by_chi"][c] for c in CHIS]
        ax2.plot(CHIS, ys, color=seq(i, len(rows)), marker="o", lw=2.0, markersize=7,
                 markeredgecolor="white", markeredgewidth=0.9, label=fr"$k={r['k']}$")
    ax2.axhline(0.5, color=INK_MUTED, ls=":", lw=1.2)
    ax2.annotate("chance", (CHIS[0], 0.5), xytext=(3, 5), textcoords="offset points",
                 fontsize=8, color=INK_MUTED)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(CHIS)
    ax2.set_xticklabels([str(c) for c in CHIS])
    ax2.set_ylim(0.42, 1.06)
    ax2.set_xlabel(r"bond dimension $\chi$")
    ax2.set_ylabel("accuracy retained")
    ax2.set_title(r"(b) the decision snaps on at $\chi = \chi^\star$")
    # The rising curves occupy the lower right; the mid-left band is the only clear space.
    ax2.legend(loc="center left", fontsize=8.5, ncol=1, framealpha=0.95)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"control.{ext}")
    print("wrote control.pdf/.png")
    for r in rows:
        print(f"  k={r['k']} exact=2^{r['k']}={r['exact_bond']:>3} "
              f"chi*={r['chi_star']} S={r['entropy']:.3f} F@1={r['fid_by_chi'][1]:.4f}")


if __name__ == "__main__":
    main()
