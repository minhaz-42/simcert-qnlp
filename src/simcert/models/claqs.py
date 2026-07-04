"""CLAQS -- Compact Learnable All-Quantum Token Mixer (Chen et al.; arXiv:2510.06532).

q=8 data qubits. Per token: ansatz-14 embedding unitary U_j (angles = W_E . token_embedding).
Tokens are mixed by a learnable, L1-normalised COMPLEX linear-combination-of-unitaries
M = sum_j b_bar_j U_j, passed through a learnable QSVT polynomial P_c(M) = sum_k c_k M^k, then
a window feed-forward unitary U_FF; the 8-qubit state is read out via 24 XYZ Pauli
expectations into a small MLP.

This is the ONLY model where MPS-bond-dimension truncation is non-trivial (q=8 => exact
bond dim 16), so it is the interesting audit regime. The auditable object is the window
statevector psi = normalise(U_FF . P_c(M) . |0^8>) -- a matrix polynomial, not a gate
sequence, so it uses the audit's precomputed-`state` path (no QASM IR).

IMPORTANT (compute honesty): the paper trains on full SST-2/IMDB with a GPU (RTX A5000) to
91.64%/87.08%. On an 8 GB M1 (no GPU) we run a REDUCED-SCALE demo (small subset, short
windows, few epochs); the contribution here is demonstrating the 8-qubit audit, NOT
reproducing the published accuracy. Deviations documented; hyperparameters the paper left
unspecified are chosen and noted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..circuits.ir import BoundCircuit
from ..data.loaders import build_vocab
from ..registry import register
from .base import QNLPModel, TrainReport


def _ansatz14(qml, angles, q, layers):
    """Sim et al. (2019) circuit-14: per layer RY-all, CRX ring fwd, RY-all, CRX ring bwd."""
    idx = 0
    for _ in range(layers):
        for i in range(q):
            qml.RY(angles[idx], wires=i); idx += 1
        for i in range(q):
            qml.CRX(angles[idx], wires=[i, (i + 1) % q]); idx += 1
        for i in range(q):
            qml.RY(angles[idx], wires=i); idx += 1
        for i in range(q):
            qml.CRX(angles[idx], wires=[(i + 1) % q, i]); idx += 1


def _pauli_expvals_torch(torch, psi, q):
    """<X_i>,<Y_i>,<Z_i> for i in range(q) from a torch complex statevector (differentiable)."""
    t = psi.reshape([2] * q)
    out = []
    for i in range(q):
        # Z_i
        zt = t.clone()
        zt = zt.transpose(0, i)
        zt[1] = -zt[1]
        zt = zt.transpose(0, i)
        # X_i: flip along axis i
        xt = torch.flip(t, dims=[i])
        # Y_i: flip then index0 *= -1j, index1 *= +1j
        yt = torch.flip(t, dims=[i]).transpose(0, i).clone()
        yt[0] = yt[0] * (-1j)
        yt[1] = yt[1] * (1j)
        yt = yt.transpose(0, i)
        flat = t.reshape(-1)
        for op in (xt, yt, zt):
            out.append(torch.real(torch.vdot(flat, op.reshape(-1))))
    return torch.stack(out)  # order: per qubit [X,Y,Z]


def _pauli_expvals_np(psi, q):
    from ..audit.observables import pauli_expval

    obs = [{i: p} for i in range(q) for p in ("X", "Y", "Z")]
    return np.array([pauli_expval(psi, q, o) for o in obs])


@register("claqs")
class CLAQS(QNLPModel):
    is_mixed_state = False

    def __init__(self):
        self.q = 8
        self.layers = 1
        self.degree = 5
        self.d_e = 8
        self.max_tokens = 6
        self.vocab: dict[str, int] = {}
        self._torch = None
        self._published_accuracy = None

    @property
    def n_angles(self) -> int:
        return 4 * self.q * self.layers

    def build(self, cfg, vocab) -> None:
        import torch

        self._torch = torch
        self.q = int(getattr(cfg, "q", 8))
        self.layers = int(getattr(cfg, "layers", 1))
        self.degree = int(getattr(cfg, "degree", 5))
        self.d_e = int(getattr(cfg, "d_e", 8))
        self.max_tokens = int(getattr(cfg, "max_tokens", 6))
        self.vocab = vocab
        self._published_accuracy = getattr(cfg, "published_accuracy", None)
        g = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)))
        f64 = torch.float64
        rn = lambda *s: torch.randn(*s, generator=g, dtype=f64)
        self.embedding = torch.nn.Parameter(0.3 * rn(len(vocab), self.d_e))
        self.W_E = torch.nn.Parameter(0.3 * rn(self.d_e, self.n_angles))
        self.b_re = torch.nn.Parameter(0.1 + 0.05 * rn(self.max_tokens))
        self.b_im = torch.nn.Parameter(0.05 * rn(self.max_tokens))
        self.c = torch.nn.Parameter(0.1 * rn(self.degree + 1))
        self.phi = torch.nn.Parameter(0.1 * rn(self.n_angles))  # U_FF angles
        h = 16
        self.mlp1 = torch.nn.Parameter(0.2 * rn(3 * self.q, h))
        self.mlp1b = torch.nn.Parameter(torch.zeros(h, dtype=f64))
        self.mlp2 = torch.nn.Parameter(0.2 * rn(h, 2))
        self.mlp2b = torch.nn.Parameter(torch.zeros(2, dtype=f64))
        self._build_matrix_fn()

    def _build_matrix_fn(self):
        import pennylane as qml

        q, L = self.q, self.layers

        def unitary(angles):
            # qml.matrix(qfunc, wire_order) returns a callable -> invoke it to get the matrix
            return qml.matrix(lambda: _ansatz14(qml, angles, q, L), wire_order=range(q))()

        self._unitary = unitary

    def _params(self):
        return [self.embedding, self.W_E, self.b_re, self.b_im, self.c, self.phi,
                self.mlp1, self.mlp1b, self.mlp2, self.mlp2b]

    def _window_state(self, example):
        """psi = normalise(U_FF . P_c(M) . |0^8>) as a torch complex vector."""
        torch = self._torch
        toks = (example.tokens or ["<unk>"])[: self.max_tokens]
        idx = torch.tensor([self.vocab.get(t, 0) for t in toks])
        theta = self.embedding[idx] @ self.W_E  # (n_tok, n_angles)
        U = [self._unitary(theta[t]).to(torch.complex128) for t in range(len(toks))]
        b = torch.complex(self.b_re[: len(toks)], self.b_im[: len(toks)])
        b = b / b.abs().sum()  # L1 normalise
        dim = 2**self.q
        v = torch.zeros(dim, dtype=torch.complex128)
        v[0] = 1.0
        acc = self.c[0].to(torch.complex128) * v
        for k in range(1, self.degree + 1):
            v = sum(b[t] * (U[t] @ v) for t in range(len(toks)))  # M v
            acc = acc + self.c[k].to(torch.complex128) * v
        u_ff = self._unitary(self.phi).to(torch.complex128)
        psi = u_ff @ acc
        nrm = torch.linalg.norm(psi)
        return psi / nrm if nrm > 0 else psi

    def _logits_from_expvals(self, ev):
        torch = self._torch
        h = torch.tanh(ev @ self.mlp1 + self.mlp1b)
        return h @ self.mlp2 + self.mlp2b

    def _forward_logits(self, example):
        torch = self._torch
        psi = self._window_state(example)
        ev = _pauli_expvals_torch(torch, psi, self.q)
        return self._logits_from_expvals(ev)

    def fit(self, train, val, cfg) -> TrainReport:
        torch = self._torch
        opt = torch.optim.AdamW(self._params(), lr=float(getattr(cfg, "lr", 0.05)))
        epochs = int(getattr(cfg, "epochs", 12))
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = torch.stack([self._forward_logits(ex) for ex in train])
            y = torch.tensor([ex.label for ex in train])
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
        return TrainReport(
            train_accuracy=self._accuracy(train), val_accuracy=self._accuracy(val),
            published_accuracy=self._published_accuracy,
            n_params=int(sum(p.numel() for p in self._params())), epochs=epochs,
        )

    def _accuracy(self, batch) -> float:
        if not batch:
            return 0.0
        return float(np.mean(self.predict(batch) == np.array([ex.label for ex in batch])))

    def predict(self, batch) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            return np.array([int(torch.argmax(self._forward_logits(ex))) for ex in batch])

    # ---- audit: whole-window statevector (precomputed-state path) ----
    def _state_np(self, example) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            return self._window_state(example).cpu().numpy()

    def audit_units(self, example) -> list[BoundCircuit]:
        obs = [{i: p} for i in range(self.q) for p in ("X", "Y", "Z")]
        return [BoundCircuit(n_qubits=self.q, qasm="", observables=obs,
                             label=int(example.label), state=self._state_np(example))]

    def export_circuits(self, batch) -> list[BoundCircuit]:
        return [self.audit_units(ex)[0] for ex in batch]

    def compose(self, example, units: list[dict]) -> int:
        torch = self._torch
        ev = torch.tensor(np.asarray(units[0]["expvals"], dtype=float), dtype=torch.float64)
        with torch.no_grad():
            return int(torch.argmax(self._logits_from_expvals(ev)))

    def save(self, path) -> None:
        torch = self._torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({k: getattr(self, k) for k in
                    ("embedding", "W_E", "b_re", "b_im", "c", "phi",
                     "mlp1", "mlp1b", "mlp2", "mlp2b")}, p / "weights.pt")
        (p / "meta.json").write_text(json.dumps({
            "q": self.q, "layers": self.layers, "degree": self.degree, "d_e": self.d_e,
            "max_tokens": self.max_tokens, "vocab": self.vocab,
            "published_accuracy": self._published_accuracy}))

    @classmethod
    def load(cls, path) -> "CLAQS":
        import torch

        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        m = cls()
        m._torch = torch
        for k in ("q", "layers", "degree", "d_e", "max_tokens", "vocab"):
            setattr(m, k, meta[k])
        m._published_accuracy = meta.get("published_accuracy")
        for k, v in torch.load(p / "weights.pt").items():
            setattr(m, k, v)
        m._build_matrix_fn()
        return m


def make_vocab_for(examples):
    return build_vocab(examples)
