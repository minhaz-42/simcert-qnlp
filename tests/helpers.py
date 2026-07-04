"""Reference circuits with known simulability properties, for the audit tests."""

from __future__ import annotations

import numpy as np


def ghz_qfunc(n: int):
    """GHZ state: maximally entangled, Schmidt rank 2 across every cut."""
    import pennylane as qml

    def qf():
        qml.Hadamard(wires=0)
        for i in range(1, n):
            qml.CNOT(wires=[0, i])

    return qf


def product_qfunc(angles):
    """A pure product state (only local rotations): Schmidt rank 1, chi=1 lossless."""
    import pennylane as qml

    def qf():
        for i, a in enumerate(angles):
            qml.RY(a, wires=i)

    return qf


def random_entangling_qfunc(n: int, layers: int = 3, seed: int = 0):
    """A generically entangled circuit (RX/RY/RZ + CNOT ladders)."""
    import pennylane as qml

    rng = np.random.default_rng(seed)
    params = rng.uniform(0, 2 * np.pi, size=(layers, n, 3))

    def qf():
        for layer in range(layers):
            for i in range(n):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])

    return qf
