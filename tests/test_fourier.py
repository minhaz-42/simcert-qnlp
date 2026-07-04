"""Fourier effective-degree utility on functions with known spectra."""

import numpy as np

from simcert.audit.fourier import effective_degree


def test_pure_harmonics():
    assert effective_degree(lambda x: np.cos(3 * x)) == 3
    assert effective_degree(lambda x: np.sin(5 * x)) == 5


def test_constant_is_degree_zero():
    assert effective_degree(lambda x: 0.7) == 0


def test_sum_of_harmonics_takes_highest():
    assert effective_degree(lambda x: np.cos(x) + 0.5 * np.cos(4 * x)) == 4


def test_small_harmonic_below_threshold_ignored():
    # a tiny degree-9 component below the relative threshold should not count
    f = lambda x: np.cos(2 * x) + 1e-4 * np.cos(9 * x)
    assert effective_degree(f, rel_threshold=0.02) == 2
