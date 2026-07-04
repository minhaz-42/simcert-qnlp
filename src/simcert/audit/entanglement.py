"""Entanglement witnesses (Axis C) computed from a dense statevector.

Bipartite von-Neumann entropy across the contiguous cut ``[0:cut] | [cut:n]`` and
the Schmidt rank (which lower-bounds the exact MPS bond dimension). Low entropy is
an *a-priori* certificate that a small chi suffices, corroborating the truncation
sweep from the state side.
"""

from __future__ import annotations

import numpy as np


def schmidt_values(psi: np.ndarray, n_qubits: int, cut: int) -> np.ndarray:
    """Singular values across the bipartition of the first ``cut`` qubits."""
    if not 1 <= cut <= n_qubits - 1:
        raise ValueError(f"cut must be in [1, {n_qubits - 1}], got {cut}")
    mat = np.asarray(psi, dtype=complex).reshape(2**cut, 2 ** (n_qubits - cut))
    return np.linalg.svd(mat, compute_uv=False)


def bipartite_entropy(
    psi: np.ndarray, n_qubits: int, cut: int, base: float | None = None, tol: float = 1e-15
) -> float:
    """Von-Neumann entanglement entropy across ``cut`` (nats unless ``base`` given)."""
    s = schmidt_values(psi, n_qubits, cut)
    p = s**2
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > tol]
    ent = float(-np.sum(p * np.log(p)))
    if base is not None:
        ent /= np.log(base)
    return ent


def max_bipartite_entropy(psi: np.ndarray, n_qubits: int, base: float | None = None) -> float:
    """Max entanglement entropy over all contiguous bipartitions."""
    if n_qubits < 2:
        return 0.0
    return max(bipartite_entropy(psi, n_qubits, c, base) for c in range(1, n_qubits))


def schmidt_rank(psi: np.ndarray, n_qubits: int, cut: int, rtol: float = 1e-10) -> int:
    """Numerical Schmidt rank across ``cut`` (singular values above ``rtol * s_max``)."""
    s = schmidt_values(psi, n_qubits, cut)
    if s.size == 0 or s.max() == 0:
        return 0
    return int(np.sum(s > rtol * s.max()))


def required_bond_dim(psi: np.ndarray, n_qubits: int, rtol: float = 1e-10) -> int:
    """Smallest bond dimension for an *exact* MPS = max Schmidt rank over all cuts."""
    if n_qubits < 2:
        return 1
    return max(schmidt_rank(psi, n_qubits, c, rtol) for c in range(1, n_qubits))
