"""chi*-vs-n scaling: does the bond dimension needed to recover predictions grow with qubit
count, or stay flat (= classically simulable)? Distinct color + marker per n; grayscale-safe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"

# Okabe-Ito, one per qubit count
NCOLORS = {4: "#0072B2", 6: "#009E73", 8: "#E69F00", 10: "#D55E00"}
NMARKERS = {4: "o", 6: "s", 8: "^", 10: "D"}


def _exact(n):
    return 2 ** (n // 2)


def main():
    apply_rc()
    runs = {}
    for f in sorted(METRICS.glob("sst2__vqc_text__*.json")):
        d = json.loads(f.read_text())
        acc = d.get("details", {}).get("accuracy_by_chi", {})
        if "16" not in acc:  # scaling sweep only
            continue
        runs[int(d["config"]["model"]["n_qubits"])] = d
    if not runs:
        print("no scaling runs yet")
        return
    ns = sorted(runs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # (a) accuracy vs chi, one line per n
    for n in ns:
        acc = runs[n]["details"]["accuracy_by_chi"]
        chis = sorted(int(k) for k in acc if k != "full")
        ys = [acc[str(c)] for c in chis]
        ax1.plot(chis, ys, color=NCOLORS.get(n, "0.2"), marker=NMARKERS.get(n, "o"),
                 linewidth=2.2, markersize=7, markeredgecolor="white",
                 label=fr"$n={n}$ (exact $\chi={_exact(n)}$)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"bond dimension $\chi$  (log$_2$)")
    ax1.set_ylabel("SST-2 test accuracy retained")
    ax1.set_title("(a) accuracy is flat in $\\chi$ at every $n$")
    ax1.legend()

    # (b) chi* vs n against the exact bound
    chi_star = [runs[n]["certificate"]["chi_star"].get("tau_gen") or _exact(n) for n in ns]
    exact = [_exact(n) for n in ns]
    ax2.plot(ns, exact, color="0.45", marker="s", ls="--", linewidth=2.0, markersize=8,
             label=r"exact-MPS bound $2^{n/2}$")
    ax2.plot(ns, chi_star, color="#D55E00", marker="o", linewidth=2.6, markersize=9,
             markeredgecolor="white", label=r"measured $\chi^\star$")
    ax2.fill_between(ns, 0.6, chi_star, color="#D55E00", alpha=0.10, linewidth=0)
    for n, cs in zip(ns, chi_star):
        ax2.annotate(str(cs), (n, cs), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=10, color="#D55E00")
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("number of qubits  $n$")
    ax2.set_ylabel(r"bond dimension $\chi^\star$ to recover predictions")
    ax2.set_title(r"(b) $\chi^\star$ stays flat while the exact bound explodes")
    ax2.set_xticks(ns)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"scaling.{ext}")
    print("wrote scaling.pdf/.png")
    for n in ns:
        cs = runs[n]["certificate"]["chi_star"].get("tau_gen")
        print(f"  n={n}: chi*={cs}, exact={_exact(n)}, full_acc={runs[n]['certificate']['full_accuracy']:.3f}")


if __name__ == "__main__":
    main()
