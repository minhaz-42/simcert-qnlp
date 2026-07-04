"""Serialisation for the circuit IR + a PennyLane->OpenQASM helper for producers."""

from __future__ import annotations

import json
from pathlib import Path

from .ir import BoundCircuit


def qfunc_to_qasm(qfunc, n_qubits: int) -> str:
    """Serialise a PennyLane quantum function to OpenQASM 2.0 (gates only).

    Used on the producer side after a model is trained, to freeze each trained
    circuit into the portable IR the audit harness consumes.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qfunc()
        return qml.state()

    # gates only: no terminal measurements, no observable basis-rotation gates
    return qml.to_openqasm(circuit, wires=range(n_qubits), measure_all=False, rotations=False)()


def save_bound_circuit(bc: BoundCircuit, path: str | Path) -> None:
    Path(path).write_text(json.dumps(bc.to_json(), indent=2))


def load_bound_circuit(path: str | Path) -> BoundCircuit:
    return BoundCircuit.from_json(json.loads(Path(path).read_text()))
