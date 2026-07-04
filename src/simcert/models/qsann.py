"""QSANN -- Quantum Self-Attention Neural Network (Li, Zhao, Wang; arXiv:2205.05625).

Faithful reimplementation of Gaussian-Projected Quantum Self-Attention (GPQSA):

  per word s:  |psi_s> = U_enc(x_s) H^{⊗n} |0>            (angle encoding, d = n(D_enc+2) angles)
  query/key:   zq_s = <Z_0> of U_q|psi_s>,  zk_s = <Z_0> of U_k|psi_s>   (scalars)
  value:       o_s  = [<P>] of U_v|psi_s|,  P in {Z_i,X_i,Y_i}           (d-vector)
  attention:   alpha_{s,j} = exp(-(zq_s - zk_j)^2)         (CLASSICAL)
  output:      y_s = x_s + sum_j alpha~_{s,j} o_j ; mean-pool; sigmoid(w.m + b)

The quantum part is only state-prep + measurement; all cross-token interaction is
classical -> the auditable object is PER-TOKEN (3 circuits/token: q, k, v), and the
classical head is the `compose` function. At n=2 (MC) the exact MPS bond dim is 2, so the
chi-truncation audit is near-lossless here -- the discriminating signal lives in the other
axes and in scaling n up (see docs/implementation-plan.md).

Deviations from the paper (documented): (1) the word->x_s embedding is unspecified in the
paper; we use a trainable dim-d embedding. (2) value observables use the single-qubit
{Z_i,X_i,Y_i} set (exact when D_enc=1, i.e. d=3n); D_enc>1 needs the paper's under-specified
two-qubit set and is not yet supported. (3) optimizer LR differs from Table I.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..circuits.io import qfunc_to_qasm
from ..circuits.ir import BoundCircuit
from ..data.loaders import TextExample, build_vocab
from ..registry import register
from .base import QNLPModel, TrainReport


def _template(qml, angles, n: int, depth: int, entangling: bool = True):
    """Shared strongly-entangled template: Rx-all, Ry-all, then depth x [CNOT chain, Ry-all].

    Consumes n*(depth+2) angles from `angles` (torch tensor or numpy/float sequence)."""
    idx = 0
    for i in range(n):
        qml.RX(angles[idx], wires=i); idx += 1
    for i in range(n):
        qml.RY(angles[idx], wires=i); idx += 1
    for _ in range(depth):
        if entangling:
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
        for i in range(n):
            qml.RY(angles[idx], wires=i); idx += 1


def _prep(qml, x, theta_role, n, d_enc, d_qkv, entangling=True):
    for i in range(n):
        qml.Hadamard(wires=i)
    _template(qml, x, n, d_enc, entangling)          # encoder (data angles)
    _template(qml, theta_role, n, d_qkv, entangling)  # trainable q/k/v ansatz


@register("qsann")
class QSANN(QNLPModel):
    is_mixed_state = False

    def __init__(self):
        self.n_qubits = 2
        self.d_enc = 1
        self.d_qkv = 1
        self.entangling = True
        self.vocab: dict[str, int] = {}
        self._torch = None
        self.embedding = self.theta_q = self.theta_k = self.theta_v = self.w = self.b = None
        self._published_accuracy = None

    # ---- structural helpers -------------------------------------------------
    @property
    def d(self) -> int:  # feature dimension = #encoder angles
        return self.n_qubits * (self.d_enc + 2)

    def _value_obs_dicts(self) -> list[dict[int, str]]:
        return [{i: p} for i in range(self.n_qubits) for p in ("Z", "X", "Y")]

    def _value_obs_pl(self):
        import pennylane as qml

        ctor = {"Z": qml.PauliZ, "X": qml.PauliX, "Y": qml.PauliY}
        return [ctor[p](i) for i in range(self.n_qubits) for p in ("Z", "X", "Y")]

    # ---- construction -------------------------------------------------------
    def build(self, cfg, vocab) -> None:
        import torch

        self._torch = torch
        self.n_qubits = int(getattr(cfg, "n_qubits", 2))
        self.d_enc = int(getattr(cfg, "d_enc", 1))
        self.d_qkv = int(getattr(cfg, "d_qkv", 1))
        self.entangling = bool(getattr(cfg, "entangling", True))
        self.vocab = vocab
        self._published_accuracy = getattr(cfg, "published_accuracy", None)
        if 3 * self.n_qubits != self.d:
            raise ValueError(
                f"value-observable set ({3 * self.n_qubits}) must match d={self.d}; "
                "only D_enc=1 is supported (see module docstring)."
            )
        g = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)))
        std = 0.01  # paper: N(0, 0.01)
        p = self.n_qubits * (self.d_qkv + 2)
        mk = lambda *s: torch.nn.Parameter(std * torch.randn(*s, generator=g, dtype=torch.float64))
        self.embedding = mk(len(vocab), self.d)
        self.theta_q, self.theta_k, self.theta_v = mk(p), mk(p), mk(p)
        self.w = mk(self.d)
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self._build_qnodes()

    def _build_qnodes(self):
        import pennylane as qml

        dev = qml.device("default.qubit", wires=self.n_qubits)
        n, de, dq, ent = self.n_qubits, self.d_enc, self.d_qkv, self.entangling
        value_obs = self._value_obs_pl()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def scalar(x, theta):
            _prep(qml, x, theta, n, de, dq, ent)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def value(x, theta):
            _prep(qml, x, theta, n, de, dq, ent)
            return [qml.expval(o) for o in value_obs]

        self._scalar_qnode, self._value_qnode = scalar, value

    def _params(self):
        return [self.embedding, self.theta_q, self.theta_k, self.theta_v, self.w, self.b]

    def _x_torch(self, token):
        return self.embedding[self.vocab.get(token, 0)]

    # ---- forward / head -----------------------------------------------------
    def _forward_logit(self, example):
        torch = self._torch
        toks = example.tokens or ["<unk>"]
        x = [self._x_torch(t) for t in toks]
        zq = [self._scalar_qnode(xi, self.theta_q) for xi in x]
        zk = [self._scalar_qnode(xi, self.theta_k) for xi in x]
        o = [torch.stack(list(self._value_qnode(xi, self.theta_v))) for xi in x]
        s_n = len(x)
        rows = []
        for s in range(s_n):
            a = torch.stack([torch.exp(-((zq[s] - zk[j]) ** 2)) for j in range(s_n)])
            a = a / a.sum()
            y = x[s] + sum(a[j] * o[j] for j in range(s_n))
            rows.append(y)
        m = torch.stack(rows).mean(dim=0)
        return self.w @ m + self.b

    def fit(self, train, val, cfg) -> TrainReport:
        torch = self._torch
        opt = torch.optim.Adam(self._params(), lr=float(getattr(cfg, "lr", 0.05)))
        epochs = int(getattr(cfg, "epochs", 40))
        lam, gam = float(getattr(cfg, "lam", 0.0)), float(getattr(cfg, "gam", 0.0))
        for _ in range(epochs):
            opt.zero_grad()
            logits = torch.stack([self._forward_logit(ex) for ex in train])
            probs = torch.sigmoid(logits)
            y = torch.tensor([float(ex.label) for ex in train], dtype=probs.dtype)
            loss = ((probs - y) ** 2).mean()
            if lam:
                loss = loss + lam / self.d * (self.w @ self.w)
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
        preds = self.predict(batch)
        y = np.array([ex.label for ex in batch])
        return float(np.mean(preds == y))

    def predict(self, batch) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            return np.array([int(torch.sigmoid(self._forward_logit(ex)) >= 0.5) for ex in batch])

    # ---- the auditable object -----------------------------------------------
    def _units_for(self, example) -> list[BoundCircuit]:
        torch = self._torch
        toks = example.tokens or ["<unk>"]
        units = []
        with torch.no_grad():
            thq = self.theta_q.cpu().numpy()
            thk = self.theta_k.cpu().numpy()
            thv = self.theta_v.cpu().numpy()
            for ti, tok in enumerate(toks):
                x = self._x_torch(tok).cpu().numpy()
                for role, theta, obs in (
                    ("q", thq, [{0: "Z"}]),
                    ("k", thk, [{0: "Z"}]),
                    ("v", thv, self._value_obs_dicts()),
                ):
                    qf = self._make_qfunc(x, theta)
                    units.append(
                        BoundCircuit(
                            n_qubits=self.n_qubits,
                            qasm=qfunc_to_qasm(qf, self.n_qubits),
                            observables=obs,
                            label=int(example.label),
                            meta={"token": ti, "role": role, "text": tok},
                        )
                    )
        return units

    def audit_units(self, example) -> list[BoundCircuit]:
        return self._units_for(example)

    def export_circuits(self, batch) -> list[BoundCircuit]:
        return [u for ex in batch for u in self._units_for(ex)]

    def _make_qfunc(self, x_np, theta_np):
        import pennylane as qml

        n, de, dq, ent = self.n_qubits, self.d_enc, self.d_qkv, self.entangling
        x = [float(v) for v in x_np]
        th = [float(v) for v in theta_np]

        def qf():
            _prep(qml, x, th, n, de, dq, ent)

        return qf

    def compose(self, example, units: list[dict]) -> int:
        """Classical GPQSA head over the (possibly truncated) per-token expvals."""
        toks = example.tokens or ["<unk>"]
        s_n = len(toks)
        ev = [np.asarray(u["expvals"], dtype=float) for u in units]
        zq = [float(ev[3 * t][0]) for t in range(s_n)]
        zk = [float(ev[3 * t + 1][0]) for t in range(s_n)]
        o = [ev[3 * t + 2] for t in range(s_n)]
        x = [self._x_torch(t).detach().cpu().numpy() for t in toks]
        w = self.w.detach().cpu().numpy()
        b = float(self.b.detach().cpu())
        rows = []
        for s in range(s_n):
            a = np.array([math.exp(-((zq[s] - zk[j]) ** 2)) for j in range(s_n)])
            a = a / a.sum()
            rows.append(x[s] + sum(a[j] * o[j] for j in range(s_n)))
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
            "n_qubits": self.n_qubits, "d_enc": self.d_enc, "d_qkv": self.d_qkv,
            "entangling": self.entangling, "vocab": self.vocab,
            "published_accuracy": self._published_accuracy,
        }))

    @classmethod
    def load(cls, path) -> "QSANN":
        import torch

        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        m = cls()
        m._torch = torch
        m.n_qubits, m.d_enc, m.d_qkv = meta["n_qubits"], meta["d_enc"], meta["d_qkv"]
        m.entangling = meta.get("entangling", True)
        m.vocab = meta["vocab"]
        m._published_accuracy = meta.get("published_accuracy")
        w = torch.load(p / "weights.pt")
        for k, v in w.items():
            setattr(m, k, v)
        m._build_qnodes()
        return m


def make_vocab_for(examples: list[TextExample]) -> dict[str, int]:
    return build_vocab(examples)
