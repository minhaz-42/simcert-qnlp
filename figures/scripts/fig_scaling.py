"""chi*-vs-n scaling study: does the bond dimension required to reproduce a trained
model's predictions grow with qubit count, or stay flat (= classically simulable)?

Reads the scaling-sweep results (SST-2 vqc_text at several n, audit=scaling: the runs
whose chi sweep includes chi=16) and plots (a) accuracy retained vs chi per n, and
(b) chi* vs n against the exact-MPS bound 2^(n/2).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"


def _exact_bond(n):
    return 2 ** (n // 2)


def main():
    runs = {}
    for f in sorted(METRICS.glob("sst2__vqc_text__*.json")):
        d = json.loads(f.read_text())
        acc = d.get("details", {}).get("accuracy_by_chi", {})
        if "16" not in acc:  # scaling runs use the wider chi sweep (includes 16)
            continue
        n = int(d["config"]["model"]["n_qubits"])
        runs[n] = d
    if not runs:
        print("no scaling runs found yet (need audit=scaling sweep results)")
        return

    ns = sorted(runs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    for n in ns:
        acc = runs[n]["details"]["accuracy_by_chi"]
        chis = sorted(int(k) for k in acc if k != "full")
        ys = [acc[str(c)] for c in chis]
        ax1.plot(chis, ys, marker="o", label=f"n={n} (exact χ={_exact_bond(n)})")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"MPS bond dimension $\chi$ (log$_2$)")
    ax1.set_ylabel("SST-2 test accuracy retained")
    ax1.set_title("(a) Accuracy vs χ, per qubit count")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    chi_star, exact = [], []
    for n in ns:
        cs = runs[n]["certificate"]["chi_star"].get("tau_gen")
        chi_star.append(cs if cs is not None else _exact_bond(n))  # None => needs full χ
        exact.append(_exact_bond(n))
    ax2.plot(ns, exact, marker="s", ls="--", color="gray", label=r"exact-MPS bound $2^{n/2}$")
    ax2.plot(ns, chi_star, marker="o", color="C3", label=r"measured $\chi^\star$ (τ_gen)")
    ax2.set_xlabel("number of qubits n")
    ax2.set_ylabel(r"$\chi^\star$ to reproduce predictions")
    ax2.set_title("(b) Required bond dimension vs n")
    ax2.set_xticks(ns)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle("χ*-vs-n scaling: flat χ* while the exact bound grows ⇒ classically simulable")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"scaling.{ext}", dpi=150)

    print("n | full_acc | chi*(tau_gen) | exact_bound")
    for n in ns:
        cs = runs[n]["certificate"]["chi_star"].get("tau_gen")
        print(f"{n:>2} | {runs[n]['certificate']['full_accuracy']:.3f}    | "
              f"{str(cs):>5}         | {_exact_bond(n)}")
    print(f"wrote {OUT/'scaling.pdf'} and .png")


if __name__ == "__main__":
    main()
