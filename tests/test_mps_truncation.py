"""The headline axis: MPS bond-dimension truncation behaves correctly.

GHZ  -> chi=1 crushes fidelity to ~0.5 (best product approx), chi>=2 is exact.
Product state -> chi=1 is lossless.
Random entangled circuit -> fidelity(chi) is non-decreasing and reaches 1 at full chi.
"""

import numpy as np
import pytest

from helpers import ghz_qfunc, product_qfunc, random_entangling_qfunc
from simcert.audit.mps_truncation import (
    exact_statevector,
    run_chi_sweep,
    truncate_state_to_bond_dim,
)
from simcert.audit.observables import single_qubit_z


def test_ghz_chi1_fidelity_half_and_chi2_exact():
    n = 4
    psi = exact_statevector(ghz_qfunc(n), n)
    res = run_chi_sweep(ghz_qfunc(n), n, single_qubit_z(range(n)), chi_values=[1, 2, None], full_state=psi)
    assert np.isclose(res["sweep"][1]["fidelity"], 0.5, atol=1e-9)
    assert np.isclose(res["sweep"][2]["fidelity"], 1.0, atol=1e-9)  # GHZ needs only bond dim 2
    assert np.isclose(res["sweep"]["full"]["fidelity"], 1.0, atol=1e-12)


def test_ghz_readout_flips_under_chi1():
    n = 4
    res = run_chi_sweep(ghz_qfunc(n), n, single_qubit_z([0]), chi_values=[1, None])
    # Full GHZ: <Z0> = 0 (equal |0000> and |1111>). chi=1 collapses to a product branch: |<Z0>| = 1.
    assert np.isclose(res["full_expvals"][0], 0.0, atol=1e-9)
    assert np.isclose(abs(res["sweep"][1]["expvals"][0]), 1.0, atol=1e-9)


def test_product_state_chi1_lossless():
    angles = [0.3, 0.7, 1.1]
    n = len(angles)
    res = run_chi_sweep(product_qfunc(angles), n, single_qubit_z(range(n)), chi_values=[1, None])
    assert np.isclose(res["sweep"][1]["fidelity"], 1.0, atol=1e-10)
    # readout preserved exactly at chi=1
    assert np.allclose(res["sweep"][1]["expvals"], res["full_expvals"], atol=1e-10)


def test_fidelity_monotonic_in_chi():
    n = 6
    psi = exact_statevector(random_entangling_qfunc(n, layers=3, seed=1), n)
    chis = [1, 2, 4, 8, None]
    res = run_chi_sweep(
        random_entangling_qfunc(n, layers=3, seed=1), n, single_qubit_z(range(n)),
        chi_values=chis, full_state=psi,
    )
    fids = [res["sweep"][c if c is not None else "full"]["fidelity"] for c in chis]
    assert all(b >= a - 1e-9 for a, b in zip(fids, fids[1:]))  # non-decreasing
    assert np.isclose(fids[-1], 1.0, atol=1e-12)  # full chi is exact


def test_truncate_full_bond_is_identity():
    n = 5
    psi = exact_statevector(random_entangling_qfunc(n, seed=2), n)
    exact = truncate_state_to_bond_dim(psi, n, chi=2 ** (n // 2))  # exact bond dim
    from simcert.audit.fidelity import state_fidelity

    assert np.isclose(state_fidelity(psi, exact), 1.0, atol=1e-10)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_truncated_state_is_normalized(n):
    psi = exact_statevector(random_entangling_qfunc(n, seed=3), n)
    t = truncate_state_to_bond_dim(psi, n, chi=1)
    assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-10)
