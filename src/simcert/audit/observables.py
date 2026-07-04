"""Pauli-observable readouts, evaluated directly on a (possibly truncated) dense state.

Observables are represented as plain dicts mapping ``wire -> 'X'|'Y'|'Z'`` (wires
absent are Identity). This lets us compute expectation values of local Pauli
readouts in ``O(2**n)`` time/memory straight from a statevector — no ``2**n x 2**n``
operator matrix is ever materialised — so the audit scales to ~20 qubits on the M1.

Wire convention matches PennyLane's ``qml.state()``: wire 0 is the most-significant
qubit, i.e. amplitude index ``sum_w bit_w * 2**(n-1-w)``.
"""

from __future__ import annotations

import numpy as np

_PAULI_LABELS = {"I", "X", "Y", "Z"}


def apply_pauli(psi: np.ndarray, n_qubits: int, paulis: dict[int, str]) -> np.ndarray:
    """Return ``P|psi>`` for a tensor-product Pauli ``P`` given by ``paulis``."""
    t = np.asarray(psi, dtype=complex).reshape([2] * n_qubits)
    for wire, label in paulis.items():
        p = label.upper()
        if p not in _PAULI_LABELS:
            raise ValueError(f"unknown Pauli {label!r} on wire {wire}")
        if not 0 <= wire < n_qubits:
            raise ValueError(f"wire {wire} out of range for {n_qubits} qubits")
        if p == "I":
            continue
        if p == "Z":
            t = t.copy()
            idx = [slice(None)] * n_qubits
            idx[wire] = 1
            t[tuple(idx)] *= -1
        elif p == "X":
            t = np.flip(t, axis=wire).copy()
        elif p == "Y":
            # Y|0> = i|1>, Y|1> = -i|0>.  After flipping axis `wire`,
            # index 0 holds the old |1> amplitude and index 1 the old |0>.
            t = np.flip(t, axis=wire).copy()
            idx0 = [slice(None)] * n_qubits
            idx0[wire] = 0
            idx1 = [slice(None)] * n_qubits
            idx1[wire] = 1
            t[tuple(idx0)] *= -1j
            t[tuple(idx1)] *= 1j
    return t.reshape(-1)


def pauli_expval(psi: np.ndarray, n_qubits: int, paulis: dict[int, str]) -> float:
    """Expectation value ``<psi|P|psi> / <psi|psi>`` (defensively renormalised)."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    denom = np.vdot(psi, psi).real
    if denom == 0.0:
        return 0.0
    p_psi = apply_pauli(psi, n_qubits, paulis)
    return float(np.real(np.vdot(psi, p_psi)) / denom)


def single_qubit_z(wires: list[int]) -> list[dict[int, str]]:
    """A standard readout: one single-qubit ``<Z_w>`` observable per wire."""
    return [{w: "Z"} for w in wires]


def to_pennylane_observable(paulis: dict[int, str]):
    """Convert a Pauli-dict to a PennyLane observable (for device cross-checks)."""
    import pennylane as qml

    ctor = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    ops = [ctor[label.upper()](w) for w, label in sorted(paulis.items()) if label.upper() != "I"]
    if not ops:
        return qml.Identity(0)
    obs = ops[0]
    for o in ops[1:]:
        obs = obs @ o
    return obs
