"""Geometric-difference g_CQ on kernels with known relationships."""

import numpy as np

from simcert.audit.kernels import fidelity_kernel, geometric_difference, rbf_kernel


def test_identical_kernels_give_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3))
    k = rbf_kernel(x)
    # g between a kernel and itself is 1 (<< sqrt(N)) -> classically reproducible
    assert np.isclose(geometric_difference(k, k), 1.0, atol=1e-4)


def test_finite_and_positive():
    # g_CQ ~ sqrt(N) is the advantage *signal*; with a regulariser on a near-singular
    # classical kernel it can slightly exceed sqrt(N), so we only check it is finite & >0.
    rng = np.random.default_rng(1)
    kq = fidelity_kernel(_random_states(rng, 10, 4))
    kc = rbf_kernel(rng.normal(size=(10, 4)))
    g = geometric_difference(kq, kc)
    assert np.isfinite(g) and g > 0.0


def test_fidelity_kernel_diagonal_is_one():
    rng = np.random.default_rng(2)
    kq = fidelity_kernel(_random_states(rng, 6, 3))
    assert np.allclose(np.diag(kq), 1.0, atol=1e-9)


def _random_states(rng, n, n_qubits):
    dim = 2**n_qubits
    s = rng.normal(size=(n, dim)) + 1j * rng.normal(size=(n, dim))
    return s / np.linalg.norm(s, axis=1, keepdims=True)
