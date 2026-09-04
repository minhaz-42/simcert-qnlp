"""Positive control: a k-of-m entanglement dial that SimCert's own audit must track.

Every verdict in the paper is CLASSICALLY_SIMULABLE (bar discocat at chi*=2), so a
referee can fairly ask whether the instrument is capable of reporting anything else.
This is the calibration curve that answers it: one family of states with a single integer
knob k, where the bond dimension a classical surrogate provably needs is 2^k, audited
through the same code path (`run_chi_sweep`, `chi_star`) the paper's models go through.

DESIGN

n = 2m qubits paired by SHIFT: pair j joins wire j and wire j+m. Pair j is entangled for
j < k and a product state for j >= k. The half-way cut is straddled by exactly the k
entangled pairs, so the Schmidt rank there is 2^k and no MPS of bond dimension below 2^k
can represent the state. The knob moves physical resource ONLY: n, wire order, readout,
readout weight, and the distribution of the full model's own readout values are all held
fixed across k. That is what the earlier candidate design failed to do -- its ladder
varied wire labelling rather than resource, and its verdict came from an algebraic
cancellation rather than from entanglement (docs/positive-control-study.md).

  pair j entangled : (|00> + s_j|11>)/sqrt(2)   <X X> = s_j,  S = ln 2
  pair j product   : |+> (x) (|0> + s_j|1>)/sqrt(2)   <X X> = s_j,  S = 0

Both rows give the same correlator s_j, so the full model computes the same function of
the same readout at every k. Only the entanglement behind that readout changes.

READOUT. The straddling-pair correlators <X_j X_{j+m}>, weight 2. Every part of that
choice is forced:

  * Weight 1 cannot work. A Bell pair's single-wire marginal is maximally mixed, so
    <X_j> = 0 whether or not the pair is entangled; no weight-1 readout can make
    entanglement load-bearing for any decision.
  * Weight n is unsound. A maximal-weight Pauli maps every computational-basis branch
    onto a different one, so it cancels to exactly zero on any branch-subset state
    regardless of fidelity. That is the trap the earlier design fell into, and it is what
    tests/test_readout_weight.py now guards the shipped models against.
  * X, not Z. A Z-correlator is diagonal, so a single retained branch reproduces it
    exactly: <Z Z> = +1 for a Bell pair AND for |00>. Diagonal correlators are blind to
    entanglement. <X X> senses coherence -- chi=1 truncation sends it 1 -> 0 on a Bell
    pair while preserving it exactly on |++>.

LABEL. The parity of the correlators, i.e. the sign of their product. Parity, not a sum:
it makes every pair load-bearing, because a single destroyed correlator collapses the
whole product. Under a sum, the surviving product pairs agree with the full sum often
enough by chance that accuracy never resolves a clean chi*.

ACCEPTANCE TEST (from the soundness lens; both must hold or this is not a control)
  * k = 0 (separable) must give chi* = 1.
  * k = m (maximal entanglement) must give chi* > 1.

Run:  python scripts/positive_control_kofm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simcert.audit.certificate import chi_star  # noqa: E402
from simcert.audit.entanglement import max_bipartite_entropy  # noqa: E402
from simcert.audit.mps_truncation import run_chi_sweep  # noqa: E402

INV_SQRT2 = 1.0 / np.sqrt(2.0)
DEAD = 1e-9  # a product this small counts as a destroyed readout, not a weak signal


def _pair_state(entangled: bool, sign: float) -> np.ndarray:
    """One pair's two-qubit state. Both branches have <X (x) X> = sign."""
    if entangled:
        return np.array([INV_SQRT2, 0.0, 0.0, sign * INV_SQRT2])  # (|00> + s|11>)/sqrt2
    plus = np.array([INV_SQRT2, INV_SQRT2])                       # |+>
    other = np.array([INV_SQRT2, sign * INV_SQRT2])               # |+> or |->
    return np.kron(plus, other)


def build_state(k: int, m: int, signs: np.ndarray) -> np.ndarray:
    """Assemble the n=2m state with pair j on wires (j, j+m).

    Pairs are built adjacent then transposed into shift order. np.transpose places source
    axis ``axes[i]`` at position i, so the permutation is ``order`` itself; using its
    inverse silently builds a different pairing whose straddling-pair count, and hence
    entropy, saturates well below k.
    """
    n = 2 * m
    psi = np.array([1.0])
    for j in range(m):
        psi = np.kron(psi, _pair_state(j < k, float(signs[j])))
    psi = psi.reshape((2,) * n)
    order = [2 * j for j in range(m)] + [2 * j + 1 for j in range(m)]
    psi = np.transpose(psi, axes=order).reshape(-1)
    return psi / np.linalg.norm(psi)


def correlator_obs(m: int) -> list[dict[int, str]]:
    """The weight-2 straddling-pair correlators <X_j X_{j+m}>."""
    return [{j: "X", j + m: "X"} for j in range(m)]


def decide(expvals: np.ndarray) -> int:
    """Parity of the correlators. A collapsed product breaks to class 1 deterministically."""
    prod = float(np.prod(np.asarray(expvals, dtype=float)))
    if abs(prod) < DEAD:
        return 1
    return int(prod >= 0.0)


def _balanced_signs(rng, m: int, want_class: int) -> np.ndarray:
    """Sign pattern whose parity gives ``want_class``.

    Classes are balanced exactly rather than sampled, so a destroyed readout scores
    exactly 0.5 instead of whatever the class balance happened to be. Without this the
    dead-zone tie-break lands on one fixed class and its "accuracy" at small chi is just
    the class frequency: luck, dressed as signal. That is the inflation the soundness lens
    caught in the earlier candidate control, and it is worth not repeating.
    """
    signs = rng.choice([-1.0, 1.0], size=m)
    target = 1.0 if want_class == 1 else -1.0
    if float(np.prod(signs)) != target:
        signs[-1] *= -1.0
    return signs


def run_point(k: int, m: int, n_examples: int = 256, seed: int = 0) -> dict:
    """Audit the k-of-m family at one k, through the repo's own chi sweep."""
    n = 2 * m
    rng = np.random.default_rng(seed + 1000 * k)
    obs = correlator_obs(m)
    chi_values = [1, 2, 4, 8, 16, 32, None]
    finite = [c for c in chi_values if c is not None]

    hits = dict.fromkeys(finite, 0)
    fids: dict[int, list[float]] = {c: [] for c in finite}
    ent: list[float] = []
    labels: list[int] = []

    for i in range(n_examples):
        signs = _balanced_signs(rng, m, i % 2)
        psi = build_state(k, m, signs)
        sweep = run_chi_sweep(None, n, obs, chi_values, full_state=psi)

        truth = decide(np.asarray(sweep["full_expvals"]))
        labels.append(truth)
        ent.append(max_bipartite_entropy(psi, n))

        for c in finite:
            rec = sweep["sweep"].get(str(c)) or sweep["sweep"].get(c)
            hits[c] += int(decide(np.asarray(rec["expvals"])) == truth)
            fids[c].append(float(rec["fidelity"]))

    # The label is the full model's own decision, so full accuracy is 1 by construction.
    full_acc = 1.0
    acc_by_chi = {c: hits[c] / n_examples for c in finite}
    # tau = 0: there is no generalization gap to absorb, the label IS the model's output.
    cs = chi_star(acc_by_chi, full_acc, tau=0.0)
    return dict(
        k=k, n=n, exact_bond=2**k, chi_star=cs, full_acc=full_acc,
        acc_by_chi=acc_by_chi,
        fid_by_chi={c: float(np.mean(fids[c])) for c in finite},
        entropy=float(np.mean(ent)),
        class_balance=float(np.mean(labels)),
    )


def main() -> int:
    m = 4  # n = 8 qubits, so k runs 0..4 and the exact bound 2^k runs 1..16
    results = [run_point(k, m) for k in range(m + 1)]

    print(f"k-of-m entanglement dial, n={2 * m} qubits, m={m} straddling pairs")
    print("readout: weight-2 correlators <X_j X_{j+m}>; label: their parity\n")
    cols = [1, 2, 4, 8, 16]
    hdr = (f"{'k':>2}{'2^k':>6}{'chi*':>7}{'S (nats)':>10}"
           + "".join(f"{'acc@' + str(c):>9}" for c in cols)
           + f"{'F@1':>8}{'bal':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        cs = "None" if r["chi_star"] is None else str(r["chi_star"])
        print(f"{r['k']:>2}{r['exact_bond']:>6}{cs:>7}{r['entropy']:>10.3f}"
              + "".join(f"{r['acc_by_chi'][c]:>9.3f}" for c in cols)
              + f"{r['fid_by_chi'][1]:>8.4f}{r['class_balance']:>7.2f}")

    lo, hi = results[0], results[-1]
    t_lo = lo["chi_star"] == 1
    t_hi = hi["chi_star"] is None or hi["chi_star"] > 1
    exact = all(r["chi_star"] == r["exact_bond"] for r in results)
    seq = [r["chi_star"] for r in results]
    monotone = all(
        (a is not None and b is not None and a <= b) or b is None
        for a, b in zip(seq, seq[1:])
    )

    print("\n--- acceptance test ---")
    print(f"  k=0 (separable) gives chi*=1        : {lo['chi_star']}  "
          f"{'PASS' if t_lo else 'FAIL'}")
    print(f"  k={m} (maximal)   gives chi*>1        : {hi['chi_star']}  "
          f"{'PASS' if t_hi else 'FAIL'}")
    print(f"  chi* non-decreasing in k            : {'PASS' if monotone else 'FAIL'}")
    print(f"  chi* == 2^k exactly at every k      : {'PASS' if exact else 'no'}")
    ok = t_lo and t_hi and monotone
    print(f"\nCONTROL {'VALID' if ok else 'INVALID'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
