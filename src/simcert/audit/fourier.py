"""Axis E -- data-reuploading Fourier degree (Schuld, Sweke, Meyer, PRA 103, 032430).

A variational model's output as a function of a single scalar encoding angle is a
truncated Fourier series whose accessible frequencies are set by the encoding. A low
effective degree means a low-order classical Fourier / random-feature surrogate reproduces
the model, an independent dequantization witness alongside the MPS-bond-dimension axis.
"""

from __future__ import annotations

import numpy as np


def effective_degree(f, n_points: int = 128, rel_threshold: float = 0.02) -> int:
    """Highest Fourier harmonic with non-negligible amplitude in ``f`` over one period.

    ``f`` maps a scalar angle in [0, 2*pi) to a real model output. The DC (constant)
    term is ignored; the returned degree is the largest harmonic index whose magnitude
    exceeds ``rel_threshold`` times the largest non-DC magnitude.
    """
    xs = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    ys = np.array([float(f(x)) for x in xs], dtype=float)
    mag = np.abs(np.fft.rfft(ys))
    if mag.size <= 1:
        return 0
    ac = mag[1:]  # drop DC
    if ac.max() <= 0:
        return 0
    significant = np.where(ac > rel_threshold * ac.max())[0]
    return int(significant.max() + 1) if significant.size else 0


def fourier_spectrum(f, n_points: int = 128) -> np.ndarray:
    """Magnitude spectrum (including DC) of ``f`` over one period -- for plotting/inspection."""
    xs = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    ys = np.array([float(f(x)) for x in xs], dtype=float)
    return np.abs(np.fft.rfft(ys))
