"""``BoundCircuit`` — a trained, parameter-bound circuit in a framework-agnostic IR.

Every model exports its per-example circuits to this IR (OpenQASM gates + a Pauli
readout spec). The audit harness consumes *only* this IR, so it never needs to know
whether lambeq, PennyLane, or Qiskit produced the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoundCircuit:
    n_qubits: int
    qasm: str  # OpenQASM 2.0, state-preparation only (no measurement instructions)
    observables: list[dict[int, str]] = field(default_factory=list)  # readout Pauli-dicts
    label: int | None = None  # ground-truth label for this example (optional)
    meta: dict = field(default_factory=dict)

    def qfunc(self):
        """Return a PennyLane quantum function that applies the circuit's gates."""
        import pennylane as qml

        loaded = qml.from_qasm(self.qasm)

        def _apply():
            loaded()

        return _apply

    def to_json(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "qasm": self.qasm,
            # JSON keys must be strings; wires are stored as str and restored to int
            "observables": [{str(w): p for w, p in o.items()} for o in self.observables],
            "label": self.label,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> "BoundCircuit":
        obs = [{int(w): p for w, p in o.items()} for o in d.get("observables", [])]
        return cls(
            n_qubits=int(d["n_qubits"]),
            qasm=d["qasm"],
            observables=obs,
            label=d.get("label"),
            meta=d.get("meta", {}),
        )
