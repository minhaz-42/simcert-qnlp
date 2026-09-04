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
from _style import MODELS, REFERENCE, apply_rc, seq  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "results" / "metrics"
OUT = REPO / "figures"

# Panel (a) colours a QUBIT COUNT, which is an ORDERED quantity, so it takes the
# sequential ramp and not a categorical slot. The previous version hard-coded four
# categorical colours and let every larger n fall through to the same dark grey, which
# silently rendered n=12, n=14 and n=16 as three indistinguishable curves.
NMARKERS = {4: "o", 6: "s", 8: "^", 10: "D", 12: "v", 14: "P", 16: "X"}
# Panel (b) colours a MODEL, which is identity, so it takes the shared categorical slots.
MODEL_STYLE = {
    "vqc_text": dict(color=MODELS["vqc_text"], marker="o",
                     label=r"measured $\chi^\star$, $\mathtt{vqc\_text}$"),
    "qsann": dict(color=MODELS["qsann"], marker="^",
                  label=r"measured $\chi^\star$, $\mathtt{qsann}$"),
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
    for i, n in enumerate(ns_ref):
        acc = ref[n]["details"]["accuracy_by_chi"]
        chis = sorted(int(k) for k in acc if k != "full")
        ys = [acc[str(c)] for c in chis]
        ax1.plot(chis, ys, color=seq(i, len(ns_ref)), marker=NMARKERS.get(n, "o"),
                 linewidth=2.0, markersize=7, markeredgecolor="white", markeredgewidth=0.9,
                 label=fr"$n={n}$ (exact $\chi={_exact(n)}$)")
    ax1.axvline(1, color="#dcdbd4", lw=6, zorder=0)
    ax1.annotate("product\nstate", (1, ax1.get_ylim()[0]), xytext=(3, 3),
                 textcoords="offset points", fontsize=8, color="#52514e", va="bottom")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"bond dimension $\chi$  (log$_2$)")
    ax1.set_ylabel("SST-2 test accuracy retained")
    # "Flat" would be wrong and a referee would see it: the curves are NON-MONOTONE in
    # chi. What is exactly true, and what chi*=1 rests on, is that accuracy at chi=1
    # equals the untruncated accuracy at every n; the excursions sit at intermediate chi.
    ax1.set_title(r"(a) $\mathtt{vqc\_text}$: $\chi{=}1$ matches the full model at every $n$")
    ax1.legend(loc="lower center", ncol=3, fontsize=7.5, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.02))

    # (b) chi* vs n against the exact bound, for every model with a scaling sweep
    all_ns = sorted({n for r in scaling.values() for n in r})
    exact = [_exact(n) for n in all_ns]
    ax2.plot(all_ns, exact, color=REFERENCE, marker="s", ls="--", linewidth=1.8,
             markersize=7, label=r"exact-MPS bound $2^{n/2}$")
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
