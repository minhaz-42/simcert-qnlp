"""The pluggable model-zoo interface — the linchpin abstraction.

``export_circuits()`` is what makes the audit model-agnostic: after training, every
model emits per-example parameter-bound circuits as ``BoundCircuit`` (OpenQASM +
readout spec). The audit harness only ever sees that IR. Classical baselines
implement ``predict`` only (no circuit) and subclass ``BaselineModel``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..circuits.ir import BoundCircuit


@dataclass
class TrainReport:
    """Returned by ``fit`` — used for the reproduction-gap table vs published numbers."""

    train_accuracy: float
    val_accuracy: float
    published_accuracy: float | None = None
    n_params: int | None = None
    epochs: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def reproduction_gap(self) -> float | None:
        if self.published_accuracy is None:
            return None
        return self.published_accuracy - self.val_accuracy


class QNLPModel(ABC):
    """Common API every quantum model in the zoo must implement."""

    name: str = "abstract"
    is_mixed_state: bool = False  # if True, audit uses the purification / MPDO path

    @abstractmethod
    def build(self, cfg, vocab) -> None:
        """Instantiate architecture from a config + dataset vocabulary."""

    @abstractmethod
    def fit(self, train, val, cfg) -> TrainReport:
        """Train to convergence; return a report incl. published-accuracy for the gap table."""

    @abstractmethod
    def predict(self, batch) -> np.ndarray:
        """Predicted labels for a batch of examples (uses the model's trained readout)."""

    @abstractmethod
    def export_circuits(self, batch) -> list[BoundCircuit]:
        """Per-example trained, parameter-bound circuits for the audit harness."""

    def decision_from_expvals(self, expvals: np.ndarray) -> int:
        """Map readout expectation values to a class label using the trained head.

        The audit recomputes ``expvals`` under a truncated device, then calls this to
        get the truncated prediction. Override if the head is non-trivial; the default
        is a sign/argmax readout suitable for the common single-observable case.
        """
        expvals = np.asarray(expvals).reshape(-1)
        if expvals.size == 1:
            return int(expvals[0] < 0)  # <Z> >= 0 -> class 0, else class 1
        return int(np.argmax(expvals))

    @abstractmethod
    def save(self, path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path) -> "QNLPModel": ...


class BaselineModel(ABC):
    """Classical baseline — accuracy floor only, no circuit to audit."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, train, val, cfg) -> TrainReport: ...

    @abstractmethod
    def predict(self, batch) -> np.ndarray: ...
