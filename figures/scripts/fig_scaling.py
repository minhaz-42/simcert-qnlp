"""chi*-vs-n scaling: does the bond dimension needed to recover predictions grow with qubit
count, or stay flat (= classically simulable)? Panel (a) shows accuracy is flat in chi at every n
for the reference model; panel (b) shows chi* staying flat for BOTH the reference model and an
attention model (QSANN) while the exact-MPS bound explodes. Distinct color + marker; grayscale-safe."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"

# Okabe-Ito, one per qubit count (panel a)
NCOLORS = {4: "#0072B2", 6: "#009E73", 8: "#E69F00", 10: "#D55E00"}
NMARKERS = {4: "o", 6: "s", 8: "^", 10: "D"}
# panel (b): one style per model
MODEL_STYLE = {
    "vqc_text": dict(color="#D55E00", marker="o", label=r"measured $\chi^\star$, \texttt{vqc\_text}"),
    "qsann": dict(color="#0072B2", marker="^", label=r"measured $\chi^\star$, \texttt{qsann}"),
}


def _exact(n):
    return 2 ** (n // 2)


def _collect(model):
    """Return {n_qubits: run} for the scaling sweep of a model (chi sweep reaching 16)."""
    runs = {}
    for f in sorted(METRICS.glob(f"sst2__{model}__*.json")):
        d = json.loads(f.read_text())
        acc = d.get("details", {}).get("accuracy_by_chi", {})
        if "16" not in acc:  # scaling sweep only
            continue
        runs[int(d["config"]["model"]["n_qubits"])] = d
    return runs


def main():
    apply_rc()
    scaling = {m: _collect(m) for m in ("vqc_text", "qsann")}
    scaling = {m: r for m, r in scaling.items() if r}
    if "vqc_text" not in scaling:
        print("no vqc_text scaling runs yet")
        return
    ref = scaling["vqc_text"]
    ns_ref = sorted(ref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # (a) accuracy vs chi, one line per n, for the reference model
    for n in ns_ref:
        acc = ref[n]["details"]["accuracy_by_chi"]
        chis = sorted(int(k) for k in acc if k != "full")
        ys = [acc[str(c)] for c in chis]
        ax1.plot(chis, ys, color=NCOLORS.get(n, "0.2"), marker=NMARKERS.get(n, "o"),
                 linewidth=2.2, markersize=7, markeredgecolor="white",
                 label=fr"$n={n}$ (exact $\chi={_exact(n)}$)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"bond dimension $\chi$  (log$_2$)")
    ax1.set_ylabel("SST-2 test accuracy retained")
    ax1.set_title(r"(a) \texttt{vqc\_text}: accuracy flat in $\chi$ at every $n$")
    ax1.legend()

    # (b) chi* vs n against the exact bound, for every model with a scaling sweep
    all_ns = sorted({n for r in scaling.values() for n in r})
    exact = [_exact(n) for n in all_ns]
    ax2.plot(all_ns, exact, color="0.45", marker="s", ls="--", linewidth=2.0, markersize=8,
             label=r"exact-MPS bound $2^{n/2}$")
    for model, runs in scaling.items():
        st = MODEL_STYLE.get(model, dict(color="0.2", marker="o", label=model))
        ns = sorted(runs)
        cstar = [runs[n]["certificate"]["chi_star"].get("tau_gen") or _exact(n) for n in ns]
        ax2.plot(ns, cstar, color=st["color"], marker=st["marker"], linewidth=2.6, markersize=9,
                 markeredgecolor="white", label=st["label"])
        for n, cs in zip(ns, cstar):
            ax2.annotate(str(cs), (n, cs), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, color=st["color"])
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("number of qubits  $n$")
    ax2.set_ylabel(r"bond dimension $\chi^\star$ to recover predictions")
    ax2.set_title(r"(b) $\chi^\star$ stays flat while the exact bound explodes")
    ax2.set_xticks(all_ns)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"scaling.{ext}")
    print("wrote scaling.pdf/.png")
    for model, runs in scaling.items():
        for n in sorted(runs):
            cs = runs[n]["certificate"]["chi_star"].get("tau_gen")
            print(f"  {model} n={n}: chi*={cs}, exact={_exact(n)}, "
                  f"full_acc={runs[n]['certificate']['full_accuracy']:.3f}")


if __name__ == "__main__":
    main()
