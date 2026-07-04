"""``vqc_text`` — a compact variational quantum text classifier.

The first real, auditable zoo member and the pipeline-validation model: a bag-of-
embeddings sentence feature is angle-encoded (with data re-uploading) into an
``n_qubits`` circuit, a trainable ansatz is applied, and class is read from ``<Z_0>``.
It is deliberately small (<=~6 qubits, few layers) so it trains in minutes on the M1
CPU/MPS, yet it exercises the entire train -> export(QASM) -> audit -> certificate flow
and is a legitimate "simple PQC" reference point in the paper's zoo.

Label convention: class 1 <-> <Z_0> = -1, class 0 <-> <Z_0> = +1, so the trained
decision rule is exactly ``sign(<Z_0>)`` and the audit stays framework-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..circuits.io import qfunc_to_qasm
from ..circuits.ir import BoundCircuit
from ..data.loaders import TextExample, build_vocab
from ..registry import register
from .base import QNLPModel, TrainReport


def _apply_ops(qml, features, theta, n_qubits: int, n_layers: int, entangling: bool = True):
    """Apply the VQC gate sequence. Works with torch tensors or python floats.

    ``entangling=False`` drops the CNOT ring -> a product-state ("separable") twin,
    which is exactly the entanglement-removal ablation (Axis B, Bowles et al.).
    """
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(features[i], wires=i)  # data re-uploading
        for i in range(n_qubits):
            qml.RY(theta[layer][i][0], wires=i)
            qml.RZ(theta[layer][i][1], wires=i)
        if entangling:
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])


@register("vqc_text")
class VQCTextModel(QNLPModel):
    is_mixed_state = False

    def __init__(self):
        self.n_qubits = 4
        self.n_layers = 2
        self.entangling = True
        self.vocab: dict[str, int] = {}
        self._torch = None
        self.embedding = None
        self.theta = None
        self._published_accuracy = None

    # ---- construction -------------------------------------------------------
    def build(self, cfg, vocab: dict[str, int]) -> None:
        import torch

        self._torch = torch
        self.n_qubits = int(getattr(cfg, "n_qubits", 4))
        self.n_layers = int(getattr(cfg, "n_layers", 2))
        self.entangling = bool(getattr(cfg, "entangling", True))
        self.vocab = vocab
        self._published_accuracy = getattr(cfg, "published_accuracy", None)
        g = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)))
        # float64 throughout: PennyLane statevector sims return float64 expectations.
        self.embedding = torch.nn.Parameter(
            0.3 * torch.randn(len(vocab), self.n_qubits, generator=g, dtype=torch.float64)
        )
        self.theta = torch.nn.Parameter(
            0.1 * torch.randn(self.n_layers, self.n_qubits, 2, generator=g, dtype=torch.float64)
        )

    def _qnode(self):
        import pennylane as qml

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(features, theta):
            _apply_ops(qml, features, theta, self.n_qubits, self.n_layers, self.entangling)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _features(self, ex: TextExample):
        torch = self._torch
        idx = [self.vocab.get(t, 0) for t in ex.tokens] or [0]
        return self.embedding[torch.tensor(idx)].mean(dim=0)

    # ---- training -----------------------------------------------------------
    def fit(self, train, val, cfg) -> TrainReport:
        torch = self._torch
        circuit = self._qnode()
        opt = torch.optim.Adam([self.embedding, self.theta], lr=float(getattr(cfg, "lr", 0.05)))
        epochs = int(getattr(cfg, "epochs", 30))
        eps = 1e-6
        for _ in range(epochs):
            opt.zero_grad()
            probs, ys = [], []
            for ex in train:
                z = circuit(self._features(ex), self.theta)
                probs.append((1 - z) / 2)  # P(class 1)
                ys.append(float(ex.label))
            p = torch.stack(probs).clamp(eps, 1 - eps)
            y = torch.tensor(ys, dtype=p.dtype)
            loss = torch.nn.functional.binary_cross_entropy(p, y)
            loss.backward()
            opt.step()
        return TrainReport(
            train_accuracy=self._accuracy(train),
            val_accuracy=self._accuracy(val),
            published_accuracy=self._published_accuracy,
            n_params=int(self.embedding.numel() + self.theta.numel()),
            epochs=epochs,
        )

    def _z_values(self, batch) -> np.ndarray:
        torch = self._torch
        circuit = self._qnode()
        with torch.no_grad():
            return np.array([float(circuit(self._features(ex), self.theta)) for ex in batch])

    def _accuracy(self, batch) -> float:
        preds = self.predict(batch)
        y = np.array([ex.label for ex in batch])
        return float(np.mean(preds == y)) if len(y) else 0.0

    # ---- inference / export -------------------------------------------------
    def predict(self, batch) -> np.ndarray:
        return np.array([self.decision_from_expvals(np.array([z])) for z in self._z_values(batch)])

    def export_circuits(self, batch) -> list[BoundCircuit]:
        torch = self._torch
        theta_np = self.theta.detach().cpu().numpy()
        circuits = []
        for ex in batch:
            with torch.no_grad():
                feats = self._features(ex).cpu().numpy()
            qf = self._make_bound_qfunc(feats, theta_np)
            qasm = qfunc_to_qasm(qf, self.n_qubits)
            circuits.append(
                BoundCircuit(
                    n_qubits=self.n_qubits,
                    qasm=qasm,
                    observables=[{0: "Z"}],
                    label=int(ex.label),
                    meta={"text": ex.text},
                )
            )
        return circuits

    def _make_bound_qfunc(self, feats_np, theta_np):
        import pennylane as qml

        n_qubits, n_layers = self.n_qubits, self.n_layers

        entangling = self.entangling

        def qf():
            _apply_ops(
                qml,
                [float(x) for x in feats_np],
                [[[float(theta_np[l, i, k]) for k in range(2)] for i in range(n_qubits)]
                 for l in range(n_layers)],
                n_qubits,
                n_layers,
                entangling,
            )

        return qf

    # ---- persistence --------------------------------------------------------
    def save(self, path) -> None:
        torch = self._torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({"embedding": self.embedding, "theta": self.theta}, p / "weights.pt")
        (p / "meta.json").write_text(
            json.dumps(
                {
                    "n_qubits": self.n_qubits,
                    "n_layers": self.n_layers,
                    "entangling": self.entangling,
                    "vocab": self.vocab,
                    "published_accuracy": self._published_accuracy,
                }
            )
        )

    @classmethod
    def load(cls, path) -> "VQCTextModel":
        import torch

        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        m = cls()
        m._torch = torch
        m.n_qubits = meta["n_qubits"]
        m.n_layers = meta["n_layers"]
        m.entangling = meta.get("entangling", True)
        m.vocab = meta["vocab"]
        m._published_accuracy = meta.get("published_accuracy")
        w = torch.load(p / "weights.pt")
        m.embedding = w["embedding"]
        m.theta = w["theta"]
        return m


def make_vocab_for(examples: list[TextExample]) -> dict[str, int]:
    return build_vocab(examples)
