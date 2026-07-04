"""Mixed-state helpers for the purification audit path (QMSAN).

The query/key objects are *reduced* density matrices: we truncate the pure n-qubit
purification to MPS bond dimension chi (the standard pure-state path), then partial-trace
out the last ``n-keep`` qubits. Attention is the Hilbert-Schmidt overlap tr(rho.sigma).
"""

from __future__ import annotations

import numpy as np


def reduced_density_matrix(psi: np.ndarray, n_qubits: int, keep: int) -> np.ndarray:
    """rho_A = Tr_B |psi><psi|, keeping the first ``keep`` qubits (A), tracing the rest (B)."""
    if not 1 <= keep < n_qubits:
        raise ValueError(f"keep must be in [1, {n_qubits - 1}], got {keep}")
    m = np.asarray(psi, dtype=complex).reshape(2**keep, 2 ** (n_qubits - keep))
    return m @ m.conj().T  # (2^keep, 2^keep), trace 1 for normalised psi


def hs_overlap(rho: np.ndarray, sigma: np.ndarray) -> float:
    """tr(rho . sigma) -- the QMSAN mixed-state attention coefficient (>=0 for valid states)."""
    return float(np.real(np.trace(rho @ sigma)))
