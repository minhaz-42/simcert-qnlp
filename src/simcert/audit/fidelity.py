"""State fidelity and prediction-agreement helpers."""

from __future__ import annotations

import numpy as np


def normalize(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(psi)
    return psi if nrm == 0 else psi / nrm


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-state fidelity ``|<a|b>|^2`` (inputs need not be normalised)."""
    a = np.asarray(a, dtype=complex).reshape(-1)
    b = np.asarray(b, dtype=complex).reshape(-1)
    num = np.abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / den) if den > 0 else 0.0


def prediction_agreement(y1: np.ndarray, y2: np.ndarray) -> float:
    """Fraction of predictions on which the two arrays agree (pi(chi))."""
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if y1.shape != y2.shape:
        raise ValueError(f"shape mismatch: {y1.shape} vs {y2.shape}")
    return float(np.mean(y1 == y2)) if y1.size else 1.0
