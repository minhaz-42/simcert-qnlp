"""Axis D -- geometric difference g_CQ between a quantum and a classical kernel.

Huang et al., "Power of data in QML" (Nat. Commun. 12:2631, arXiv:2011.01938):
g_CQ = sqrt( || sqrt(K_Q) K_C^{-1} sqrt(K_Q) ||_inf ), with both Gram matrices normalised to
trace N. g_CQ <= sqrt(N); g_CQ << sqrt(N) means the classical kernel can reproduce whatever
the quantum kernel does (no quantum advantage possible from the data geometry). A necessary
condition for advantage is g_CQ ~ sqrt(N).
"""

from __future__ import annotations

import numpy as np


def _normalise_trace(k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    tr = np.trace(k)
    return n * k / tr if tr > 0 else k


def fidelity_kernel(states: np.ndarray) -> np.ndarray:
    """|<psi_i|psi_j>|^2 Gram matrix from a stack of (normalised) statevectors (N x dim)."""
    states = np.asarray(states, dtype=complex)
    g = states.conj() @ states.T
    return np.abs(g) ** 2


def rbf_kernel(x: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Classical RBF Gram matrix on rows of ``x`` (gamma defaults to 1/n_features)."""
    x = np.asarray(x, dtype=float)
    if gamma is None:
        gamma = 1.0 / max(x.shape[1], 1)
    sq = np.sum(x**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def geometric_difference(k_quantum: np.ndarray, k_classical: np.ndarray, reg: float = 1e-6) -> float:
    """g_CQ between a quantum and a classical kernel (both symmetric PSD, same N)."""
    n = k_quantum.shape[0]
    kq = _normalise_trace(k_quantum)
    kc = _normalise_trace(k_classical)
    # sqrt of PSD kq via symmetric eigendecomposition
    w, v = np.linalg.eigh(kq)
    sqrt_kq = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    kc_inv = np.linalg.inv(kc + reg * np.eye(n))
    m = sqrt_kq @ kc_inv @ sqrt_kq
    spectral = np.linalg.norm((m + m.conj().T) / 2, 2)  # symmetrise for numerical safety
    return float(np.sqrt(max(spectral, 0.0)))
