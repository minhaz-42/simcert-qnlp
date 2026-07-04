"""Fidelity + prediction-agreement identities."""

import numpy as np
import pytest

from simcert.audit.fidelity import prediction_agreement, state_fidelity


def test_fidelity_identity_and_orthogonality():
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    assert np.isclose(state_fidelity(zero, zero), 1.0)
    assert np.isclose(state_fidelity(zero, one), 0.0)


def test_fidelity_is_global_phase_invariant():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    assert np.isclose(state_fidelity(psi, np.exp(1j * 1.234) * psi), 1.0)


def test_fidelity_handles_unnormalized_inputs():
    a = np.array([2, 0], dtype=complex)  # unnormalised
    b = np.array([0, 5], dtype=complex)
    assert np.isclose(state_fidelity(a, a), 1.0)
    assert np.isclose(state_fidelity(a, b), 0.0)


def test_prediction_agreement():
    assert prediction_agreement([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert prediction_agreement([], []) == 1.0
    with pytest.raises(ValueError):
        prediction_agreement([0, 1], [0, 1, 1])
