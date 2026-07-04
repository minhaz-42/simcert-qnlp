"""QMSAN -- Quantum Mixed-State Self-Attention Network (Fu Chen et al.; arXiv:2403.02871).

Per token, three trainable data-reuploading circuits (query/key/value). Query and key pure
states are partial-traced into MIXED states rho_q, sigma_k; attention is the Hilbert-Schmidt
overlap alpha_{s,j} = tr(rho_q . sigma_k) (paper estimates this with a mixed-state SWAP test).
Value is read out as per-qubit <Z>. Classical residual + mean-pool + sigmoid head.

This is the model that exercises the SimCert **mixed-state / purification audit path**: we
truncate the pure n-qubit purification to MPS bond dimension chi, then take its partial trace
to form rho/sigma -- so the audit probes whether the entanglement between the kept and
traced-out subsystems (i.e. the *mixed-ness* the attention relies on) is load-bearing.

Variant implemented: QMSAN-NP (no positional encoding). Documented deviations: word->x_s
embedding is unspecified in the paper (we use a trainable dim-n embedding); init follows
Algorithm 1 (theta ~ N(0,0.01)); linear-chain IsingZZ entangler (paper's "R" topology).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..audit.mixed import hs_overlap, reduced_density_matrix
from ..circuits.io import qfunc_to_qasm
from ..circuits.ir import BoundCircuit
from ..data.loaders import TextExample, build_vocab
from ..registry import register
from .base import QNLPModel, TrainReport


def _prep(qml, x, theta, n, layers, entangling=True):
    """Data-reuploading embedding: Rx(x) -> L x [IsingZZ chain, Ry layer] -> Rx(x)."""
    for i in range(n):
        qml.RX(x[i], wires=i)
    idx = 0
    for _ in range(layers):
        for i in range(n - 1):
            if entangling:
                qml.IsingZZ(theta[idx], wires=[i, i + 1])
            idx += 1  # advance regardless so theta layout is identical for the separable twin
        for i in range(n):
            qml.RY(theta[idx], wires=i); idx += 1
    for i in range(n):
        qml.RX(x[i], wires=i)  # reupload


@register("qmsan")
class QMSAN(QNLPModel):
    is_mixed_state = True

    def __init__(self):
        self.n_qubits = 2
        self.layers = 1
        self.entangling = True
        self.vocab: dict[str, int] = {}
        self._torch = None
        self.embedding = self.theta_q = self.theta_k = self.theta_v = self.w = self.b = None
        self._published_accuracy = None

    @property
    def keep(self) -> int:
        return self.n_qubits // 2

    @property
    def _theta_len(self) -> int:
        return self.layers * (2 * self.n_qubits - 1)

    # ---- construction -------------------------------------------------------
    def build(self, cfg, vocab) -> None:
        import torch

        self._torch = torch
        self.n_qubits = int(getattr(cfg, "n_qubits", 2))
        self.layers = int(getattr(cfg, "layers", 1))
        self.entangling = bool(getattr(cfg, "entangling", True))
        self.vocab = vocab
        self._published_accuracy = getattr(cfg, "published_accuracy", None)
        if self.n_qubits < 2:
            raise ValueError("QMSAN needs n_qubits >= 2 (partial trace requires >=1 traced qubit)")
        g = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)))
        emb_std = float(getattr(cfg, "emb_std", 0.3))
        mk = lambda std, *s: torch.nn.Parameter(std * torch.randn(*s, generator=g, dtype=torch.float64))
        self.embedding = mk(emb_std, len(vocab), self.n_qubits)
        self.theta_q = mk(0.01, self._theta_len)
        self.theta_k = mk(0.01, self._theta_len)
        self.theta_v = mk(0.01, self._theta_len)
        self.w = mk(0.01, self.n_qubits)
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self._build_qnodes()

    def _build_qnodes(self):
        import pennylane as qml

        dev = qml.device("default.qubit", wires=self.n_qubits)
        n, L, ent, keep = self.n_qubits, self.layers, self.entangling, self.keep

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def dm(x, theta):
            _prep(qml, x, theta, n, L, ent)
            return qml.density_matrix(wires=range(keep))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def val(x, theta):
            _prep(qml, x, theta, n, L, ent)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        self._dm_qnode, self._val_qnode = dm, val

    def _params(self):
        return [self.embedding, self.theta_q, self.theta_k, self.theta_v, self.w, self.b]

    def _x_torch(self, token):
        return self.embedding[self.vocab.get(token, 0)]

    # ---- forward / head -----------------------------------------------------
    def _forward_logit(self, example):
        torch = self._torch
        toks = example.tokens or ["<unk>"]
        x = [self._x_torch(t) for t in toks]
        rho = [self._dm_qnode(xi, self.theta_q) for xi in x]
        sig = [self._dm_qnode(xi, self.theta_k) for xi in x]
        v = [torch.stack(list(self._val_qnode(xi, self.theta_v))) for xi in x]
        s_n = len(x)
        rows = []
        for s in range(s_n):
            a = torch.stack([torch.clamp(torch.real(torch.trace(rho[s] @ sig[j])), min=1e-9)
                             for j in range(s_n)])
            a = a / a.sum()
            rows.append(x[s] + sum(a[j] * v[j] for j in range(s_n)))
        m = torch.stack(rows).mean(dim=0)
        return self.w @ m + self.b

    def fit(self, train, val, cfg) -> TrainReport:
        torch = self._torch
        opt = torch.optim.Adam(self._params(), lr=float(getattr(cfg, "lr", 0.05)))
        epochs = int(getattr(cfg, "epochs", 40))
        for _ in range(epochs):
            opt.zero_grad()
            logits = torch.stack([self._forward_logit(ex) for ex in train])
            probs = torch.sigmoid(logits)
            y = torch.tensor([float(ex.label) for ex in train], dtype=probs.dtype)
            loss = ((probs - y) ** 2).mean()
            loss.backward()
            opt.step()
        return TrainReport(
            train_accuracy=self._accuracy(train),
            val_accuracy=self._accuracy(val),
            published_accuracy=self._published_accuracy,
            n_params=int(sum(p.numel() for p in self._params())),
            epochs=epochs,
        )

    def _accuracy(self, batch) -> float:
        if not batch:
            return 0.0
        return float(np.mean(self.predict(batch) == np.array([ex.label for ex in batch])))

    def predict(self, batch) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            return np.array([int(torch.sigmoid(self._forward_logit(ex)) >= 0.5) for ex in batch])

    # ---- the auditable object (per-token q/k/v; q/k use the mixed-state path) ----
    def _units_for(self, example) -> list[BoundCircuit]:
        torch = self._torch
        toks = example.tokens or ["<unk>"]
        units = []
        with torch.no_grad():
            thq, thk, thv = (t.cpu().numpy() for t in (self.theta_q, self.theta_k, self.theta_v))
            for ti, tok in enumerate(toks):
                x = self._x_torch(tok).cpu().numpy()
                for role, theta, obs in (
                    ("q", thq, []),  # q/k: audited via the reduced density matrix (uses state)
                    ("k", thk, []),
                    ("v", thv, [{i: "Z"} for i in range(self.n_qubits)]),
                ):
                    qf = self._make_qfunc(x, theta)
                    units.append(BoundCircuit(
                        n_qubits=self.n_qubits, qasm=qfunc_to_qasm(qf, self.n_qubits),
                        observables=obs, label=int(example.label),
                        meta={"token": ti, "role": role, "text": tok},
                    ))
        return units

    def audit_units(self, example) -> list[BoundCircuit]:
        return self._units_for(example)

    def export_circuits(self, batch) -> list[BoundCircuit]:
        return [u for ex in batch for u in self._units_for(ex)]

    def _make_qfunc(self, x_np, theta_np):
        import pennylane as qml

        n, L, ent = self.n_qubits, self.layers, self.entangling
        x = [float(v) for v in x_np]
        th = [float(v) for v in theta_np]

        def qf():
            _prep(qml, x, th, n, L, ent)

        return qf

    def compose(self, example, units: list[dict]) -> int:
        """Mixed-state GPQSA head: rho_q/sigma_k from the (truncated) states, tr(rho.sigma)."""
        toks = example.tokens or ["<unk>"]
        s_n = len(toks)
        n, keep = self.n_qubits, self.keep
        rho = [reduced_density_matrix(units[3 * t]["state"], n, keep) for t in range(s_n)]
        sig = [reduced_density_matrix(units[3 * t + 1]["state"], n, keep) for t in range(s_n)]
        v = [np.asarray(units[3 * t + 2]["expvals"], dtype=float) for t in range(s_n)]
        x = [self._x_torch(t).detach().cpu().numpy() for t in toks]
        w = self.w.detach().cpu().numpy()
        b = float(self.b.detach().cpu())
        rows = []
        for s in range(s_n):
            a = np.array([max(hs_overlap(rho[s], sig[j]), 1e-9) for j in range(s_n)])
            a = a / a.sum()
            rows.append(x[s] + sum(a[j] * v[j] for j in range(s_n)))
        m = np.mean(rows, axis=0)
        logit = float(w @ m + b)
        return int(1.0 / (1.0 + math.exp(-logit)) >= 0.5)

    # ---- persistence --------------------------------------------------------
    def save(self, path) -> None:
        torch = self._torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({k: getattr(self, k) for k in
                    ("embedding", "theta_q", "theta_k", "theta_v", "w", "b")}, p / "weights.pt")
        (p / "meta.json").write_text(json.dumps({
            "n_qubits": self.n_qubits, "layers": self.layers, "entangling": self.entangling,
            "vocab": self.vocab, "published_accuracy": self._published_accuracy,
        }))

    @classmethod
    def load(cls, path) -> "QMSAN":
        import torch

        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        m = cls()
        m._torch = torch
        m.n_qubits, m.layers = meta["n_qubits"], meta["layers"]
        m.entangling = meta.get("entangling", True)
        m.vocab = meta["vocab"]
        m._published_accuracy = meta.get("published_accuracy")
        for k, v in torch.load(p / "weights.pt").items():
            setattr(m, k, v)
        m._build_qnodes()
        return m


def make_vocab_for(examples: list[TextExample]) -> dict[str, int]:
    return build_vocab(examples)
