"""Entanglement witnesses on known states (Axis C)."""

import numpy as np

from helpers import ghz_qfunc, product_qfunc, random_entangling_qfunc
from simcert.audit.entanglement import (
    bipartite_entropy,
    max_bipartite_entropy,
    required_bond_dim,
    schmidt_rank,
)
from simcert.audit.mps_truncation import exact_statevector


def test_ghz_entropy_is_ln2_everywhere():
    n = 4
    psi = exact_statevector(ghz_qfunc(n), n)
    for cut in range(1, n):
        assert np.isclose(bipartite_entropy(psi, n, cut), np.log(2), atol=1e-9)
        assert schmidt_rank(psi, n, cut) == 2
    assert np.isclose(max_bipartite_entropy(psi, n, base=2), 1.0, atol=1e-9)  # 1 bit
    assert required_bond_dim(psi, n) == 2


def test_product_state_zero_entropy():
    angles = [0.3, 0.7, 1.1, 1.9]
    n = len(angles)
    psi = exact_statevector(product_qfunc(angles), n)
    assert np.isclose(max_bipartite_entropy(psi, n), 0.0, atol=1e-9)
    assert required_bond_dim(psi, n) == 1


def test_random_circuit_has_entanglement():
    n = 6
    psi = exact_statevector(random_entangling_qfunc(n, layers=3, seed=5), n)
    assert max_bipartite_entropy(psi, n) > 0.1
    assert required_bond_dim(psi, n) > 1
