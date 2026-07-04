"""Pauli readout expectation values on states with known answers (NumPy-only)."""

import numpy as np

from simcert.audit.observables import apply_pauli, pauli_expval


def test_z_on_computational_basis():
    zero = np.array([1, 0], dtype=complex)  # |0>
    one = np.array([0, 1], dtype=complex)  # |1>
    assert pauli_expval(zero, 1, {0: "Z"}) == 1.0
    assert pauli_expval(one, 1, {0: "Z"}) == -1.0


def test_x_and_y_on_plus_state():
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>
    assert np.isclose(pauli_expval(plus, 1, {0: "X"}), 1.0)
    assert np.isclose(pauli_expval(plus, 1, {0: "Z"}), 0.0)
    assert np.isclose(pauli_expval(plus, 1, {0: "Y"}), 0.0)


def test_y_on_i_eigenstate():
    # |+i> = (|0> + i|1>)/sqrt2 has <Y> = +1
    plus_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    assert np.isclose(pauli_expval(plus_i, 1, {0: "Y"}), 1.0)


def test_bell_state_correlations():
    # (|00> + |11>)/sqrt2 : <Z0>=0, <Z1>=0, <Z0 Z1>=+1
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.isclose(pauli_expval(bell, 2, {0: "Z"}), 0.0)
    assert np.isclose(pauli_expval(bell, 2, {1: "Z"}), 0.0)
    assert np.isclose(pauli_expval(bell, 2, {0: "Z", 1: "Z"}), 1.0)
    # X0 X1 also +1 for this Bell state
    assert np.isclose(pauli_expval(bell, 2, {0: "X", 1: "X"}), 1.0)


def test_apply_pauli_is_involutive():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    for word in ({0: "X"}, {1: "Y"}, {2: "Z"}, {0: "X", 2: "Z"}):
        twice = apply_pauli(apply_pauli(psi, 3, word), 3, word)
        assert np.allclose(twice, psi, atol=1e-12)
