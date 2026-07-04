"""DisCoCat consumer (qnlp env) -- audits the circuits exported by the lambeq producer.

DisCoCat is the native one-circuit-per-sentence case. This model does not train in the
qnlp env; it loads the producer's trained OpenQASM circuits + readout metadata
(post-selected qubits, open wire) and runs the pure-state audit: truncate the whole-
sentence state to bond dimension chi, post-select the cup qubits on |0>, renormalise, and
read the open wire's Born rule. Run the producer first:

    conda run -n qnlp-lambeq python -m simcert.producers.discocat_producer --seed <s>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audit.mps_truncation import exact_statevector
from ..circuits.ir import BoundCircuit
from ..registry import register
from .base import QNLPModel, TrainReport

REPO = Path(__file__).resolve().parents[3]


def _postselect_readout(state: np.ndarray, post_select, open_wire) -> int:
    """Post-select `post_select` wires on |0>, renormalise, argmax the open wire's Born probs."""
    n = int(round(np.log2(len(state))))
    t = np.asarray(state, dtype=complex).reshape([2] * n)
    idx = [slice(None)] * n
    for w in post_select:
        idx[w] = 0
    sub = t[tuple(idx)].reshape(-1)  # amplitudes over the open wire(s)
    p = np.abs(sub) ** 2
    s = p.sum()
    if s <= 0:
        return 0
    return int(np.argmax(p / s))


@register("discocat")
class DisCoCat(QNLPModel):
    is_mixed_state = False

    def __init__(self):
        import json  # noqa: F401  (kept local; consumer is import-light)

        self.records: dict[str, dict] = {}
        self._manifest: dict = {}

    def build(self, cfg, vocab) -> None:
        import json

        seed = int(getattr(cfg, "seed", 1))
        root = Path(getattr(cfg, "artifact_dir", REPO / "results" / "circuits" / "discocat"))
        if not root.is_absolute():
            root = REPO / root
        adir = root / f"seed{seed}"
        self._manifest = json.loads((adir / "manifest.json").read_text())
        self.records = {}
        for split in ("train", "val", "test"):
            for rec in json.loads((adir / f"{split}.json").read_text()):
                self.records[rec["text"]] = rec

    def fit(self, train, val, cfg) -> TrainReport:
        m = self._manifest  # already trained by the producer; report its numbers
        return TrainReport(
            train_accuracy=m.get("train_acc", 0.0),
            val_accuracy=m.get("val_acc", 0.0),
            published_accuracy=m.get("published_accuracy"),
            n_params=m.get("n_params"),
            extra={"reader": m.get("reader"), "note": "trained by lambeq producer"},
        )

    def _rec(self, ex) -> dict:
        return self.records[ex.text]

    def _bc(self, rec: dict) -> BoundCircuit:
        return BoundCircuit(
            n_qubits=rec["n_qubits"], qasm=rec["qasm"], observables=[], label=rec["label"],
            meta={"post_select": rec["post_select"], "open_wire": rec["open_wire"], "text": rec["text"]},
        )

    def audit_units(self, example) -> list[BoundCircuit]:
        return [self._bc(self._rec(example))]

    def export_circuits(self, batch) -> list[BoundCircuit]:
        return [self._bc(self._rec(ex)) for ex in batch]

    def compose(self, example, units: list[dict]) -> int:
        rec = self._rec(example)
        return _postselect_readout(units[0]["state"], rec["post_select"], rec["open_wire"])

    def predict(self, batch) -> np.ndarray:
        preds = []
        for ex in batch:
            rec = self._rec(ex)
            psi = exact_statevector(self._bc(rec).qfunc(), rec["n_qubits"])
            preds.append(_postselect_readout(psi, rec["post_select"], rec["open_wire"]))
        return np.array(preds)

    def save(self, path) -> None:  # trained artifacts live under results/circuits/discocat/
        pass

    @classmethod
    def load(cls, path) -> "DisCoCat":
        raise NotImplementedError("DisCoCat loads producer artifacts in build(), not from a ckpt")
