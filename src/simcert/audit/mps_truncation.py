"""Axis A — the headline: MPS bond-dimension truncation of a *trained* state.

Primary route (exact, testable): compute the exact statevector of the trained
circuit, then form its optimal bond-dimension-chi MPS approximation by explicit
left-to-right sequential SVD truncation and contract back to a dense vector. This
*is* the operational definition of "can a chi-bounded classical MPS reproduce this
model's predictions?" and it maps 1:1 onto Shin/Teo/Jeong (every VQML function is a
constrained-coefficient MPS).

For circuit widths beyond dense reach (>~24 qubits) use `mps_expvals_default_tensor`,
a quimb-backed cross-check; QNLP circuits in this project are <=~20 qubits, so the
explicit route carries the reported numbers and `default.tensor` only validates it.
"""

from __future__ import annotations

import numpy as np

from .fidelity import normalize, state_fidelity
from .observables import pauli_expval, to_pennylane_observable


def exact_statevector(qfunc, n_qubits: int, backend: str = "default.qubit") -> np.ndarray:
    """Dense statevector of a trained circuit. ``qfunc`` applies gates (no measurement)."""
    import pennylane as qml

    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qfunc()
        return qml.state()

    return np.asarray(circuit(), dtype=complex).reshape(-1)


def _contract_mps(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract a list of MPS site tensors (each shape (Dl, 2, Dr)) to a dense vector."""
    first = tensors[0]
    dl, d, dr = first.shape
    acc = first.reshape(dl * d, dr)  # dl == 1 for the left boundary
    for t in tensors[1:]:
        tl, td, tr = t.shape
        acc = acc.reshape(-1, tl) @ t.reshape(tl, td * tr)
        acc = acc.reshape(-1, tr)
    return acc.reshape(-1)


def truncate_state_to_bond_dim(
    psi: np.ndarray,
    n_qubits: int,
    chi: int | None,
    cutoff: float = 1e-12,
    renormalize: bool = True,
) -> np.ndarray:
    """Optimal bond-dimension-``chi`` MPS approximation of ``psi``, returned dense.

    ``chi=None`` means no bond cap (exact up to ``cutoff``). Singular values below
    ``cutoff`` are always discarded; at least one is kept per bond.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if n_qubits <= 1:
        return normalize(psi) if renormalize else psi

    tensors: list[np.ndarray] = []
    residual = psi.reshape(1, -1)
    bond = 1
    for _ in range(n_qubits - 1):
        residual = residual.reshape(bond * 2, -1)
        u, s, vh = np.linalg.svd(residual, full_matrices=False)
        keep = int(np.sum(s > cutoff))
        if chi is not None:
            keep = min(keep, int(chi))
        keep = max(keep, 1)
        u = u[:, :keep]
        s = s[:keep]
        vh = vh[:keep, :]
        tensors.append(u.reshape(bond, 2, keep))
        residual = s[:, None] * vh  # == diag(s) @ vh
        bond = keep
    tensors.append(residual.reshape(bond, 2, 1))

    out = _contract_mps(tensors)
    return normalize(out) if renormalize else out


def exact_bond_dim(n_qubits: int) -> int:
    """Bond dimension at which an MPS represents *any* n-qubit state exactly."""
    return 2 ** (n_qubits // 2)


def run_chi_sweep(
    qfunc,
    n_qubits: int,
    observables: list[dict[int, str]],
    chi_values,
    cutoff: float = 1e-12,
    backend: str = "default.qubit",
    full_state: np.ndarray | None = None,
) -> dict:
    """Sweep bond dimension and record readout expectation values + fidelity.

    ``observables`` are Pauli-dicts (see ``observables.py``). ``chi_values`` may
    contain ``None`` to denote the untruncated (full) simulation. Deterministic.
    Returns ``{n_qubits, full_state, full_expvals, sweep: {chi_key: {...}}}``.
    """
    psi_full = full_state if full_state is not None else exact_statevector(qfunc, n_qubits, backend)
    psi_full = normalize(psi_full)
    ev_full = np.array([pauli_expval(psi_full, n_qubits, o) for o in observables])

    exact_chi = exact_bond_dim(n_qubits)
    sweep: dict = {}
    for chi in chi_values:
        if chi is None or int(chi) >= exact_chi:
            psi = psi_full
            fid = 1.0
        else:
            psi = truncate_state_to_bond_dim(psi_full, n_qubits, int(chi), cutoff)
            fid = state_fidelity(psi_full, psi)
        ev = np.array([pauli_expval(psi, n_qubits, o) for o in observables])
        sweep[_chi_key(chi)] = {"chi": chi, "expvals": ev, "fidelity": fid}

    return {
        "n_qubits": n_qubits,
        "full_state": psi_full,
        "full_expvals": ev_full,
        "sweep": sweep,
    }


def _chi_key(chi):
    return "full" if chi is None else int(chi)


def mps_expvals_default_tensor(
    qfunc, n_qubits: int, observables: list[dict[int, str]], chi: int, cutoff: float = 1e-12
) -> np.ndarray:
    """Scalability cross-check: readout expvals via PennyLane's quimb-backed MPS device."""
    import pennylane as qml

    dev = qml.device(
        "default.tensor", wires=n_qubits, method="mps", max_bond_dim=int(chi), cutoff=cutoff
    )
    obs = [to_pennylane_observable(o) for o in observables]

    @qml.qnode(dev)
    def circuit():
        qfunc()
        return [qml.expval(o) for o in obs]

    return np.asarray(circuit(), dtype=float).reshape(-1)
