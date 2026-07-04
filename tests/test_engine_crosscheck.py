"""Cross-check the explicit NumPy engine against PennyLane's simulators.

(1) default.qubit statevector == lightning.qubit statevector.
(2) PennyLane default.tensor (quimb MPS) at full bond dim == our explicit-SVD readout.
This validates that the NumPy audit engine agrees with an independent MPS implementation.
"""

import numpy as np
import pytest

from helpers import ghz_qfunc, random_entangling_qfunc
from simcert.audit.fidelity import state_fidelity
from simcert.audit.mps_truncation import (
    exact_statevector,
    mps_expvals_default_tensor,
    run_chi_sweep,
)
from simcert.audit.observables import single_qubit_z


def test_default_qubit_matches_lightning():
    n = 5
    qf = random_entangling_qfunc(n, layers=3, seed=7)
    psi_default = exact_statevector(qf, n, backend="default.qubit")
    try:
        psi_light = exact_statevector(qf, n, backend="lightning.qubit")
    except Exception as e:  # lightning present in env; skip only if genuinely unavailable
        pytest.skip(f"lightning.qubit unavailable: {e}")
    assert np.isclose(state_fidelity(psi_default, psi_light), 1.0, atol=1e-9)


def test_default_tensor_full_bond_matches_explicit_engine():
    n = 5
    qf = random_entangling_qfunc(n, layers=3, seed=8)
    obs = single_qubit_z(range(n))
    ref = run_chi_sweep(qf, n, obs, chi_values=[None])["full_expvals"]
    full_chi = 2 ** n  # comfortably exact
    mps = mps_expvals_default_tensor(qf, n, obs, chi=full_chi)
    assert np.allclose(ref, mps, atol=1e-6)


def test_default_tensor_ghz_readout():
    n = 4
    obs = single_qubit_z([0, 1])
    mps = mps_expvals_default_tensor(ghz_qfunc(n), n, obs, chi=2 ** n)
    assert np.allclose(mps, [0.0, 0.0], atol=1e-6)  # GHZ single-qubit <Z> = 0
