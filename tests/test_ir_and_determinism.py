"""Circuit IR round-trip + audit determinism + run-hash keying."""

import numpy as np
import pytest

from helpers import random_entangling_qfunc
from simcert.audit.mps_truncation import exact_statevector, run_chi_sweep
from simcert.audit.observables import single_qubit_z
from simcert.circuits.ir import BoundCircuit
from simcert.io_results import result_path, run_hash


def test_bound_circuit_json_roundtrip_preserves_int_wire_keys():
    bc = BoundCircuit(n_qubits=3, qasm="// dummy", observables=[{0: "Z"}, {1: "Z", 2: "X"}], label=1)
    bc2 = BoundCircuit.from_json(bc.to_json())
    assert bc2.observables == [{0: "Z"}, {1: "Z", 2: "X"}]
    assert all(isinstance(w, int) for o in bc2.observables for w in o)
    assert bc2.label == 1 and bc2.n_qubits == 3


def test_qasm_roundtrip_reproduces_state():
    import pennylane as qml

    from simcert.circuits.io import qfunc_to_qasm

    n = 3

    def qf():
        qml.Hadamard(wires=0)
        qml.RY(0.7, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(1.1, wires=2)
        qml.CNOT(wires=[1, 2])

    psi_before = exact_statevector(qf, n)
    qasm = qfunc_to_qasm(qf, n)
    bc = BoundCircuit(n_qubits=n, qasm=qasm, observables=single_qubit_z(range(n)))
    psi_after = exact_statevector(bc.qfunc(), n)

    from simcert.audit.fidelity import state_fidelity

    assert np.isclose(state_fidelity(psi_before, psi_after), 1.0, atol=1e-9)


def test_audit_is_deterministic():
    n = 5
    qf = random_entangling_qfunc(n, layers=3, seed=11)
    obs = single_qubit_z(range(n))
    r1 = run_chi_sweep(qf, n, obs, chi_values=[1, 2, 4, None])
    r2 = run_chi_sweep(qf, n, obs, chi_values=[1, 2, 4, None])
    for c in (1, 2, 4, "full"):
        assert np.array_equal(r1["sweep"][c]["expvals"], r2["sweep"][c]["expvals"])
        assert r1["sweep"][c]["fidelity"] == r2["sweep"][c]["fidelity"]


def test_run_hash_stable_and_sensitive():
    cfg_a = {"model": "qsann", "dataset": "sst2", "seed": 1, "paths": {"results": "/x"}}
    cfg_b = {"seed": 1, "dataset": "sst2", "model": "qsann", "paths": {"results": "/y"}}  # reordered + volatile path differs
    assert run_hash(cfg_a) == run_hash(cfg_b)  # path dropped, key order irrelevant
    cfg_c = {"model": "qsann", "dataset": "sst2", "seed": 2}
    assert run_hash(cfg_a) != run_hash(cfg_c)  # scientific param changed


def test_result_path_format():
    p = result_path("results", "sst2", "qsann", "abc123def456")
    assert p.name == "sst2__qsann__abc123def456.json"
    assert p.parent.name == "metrics"
