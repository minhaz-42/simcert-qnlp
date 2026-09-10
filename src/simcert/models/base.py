"""The pluggable model-zoo interface — the linchpin abstraction.

``export_circuits()`` is what makes the audit model-agnostic: after training, every
model emits per-example parameter-bound circuits as ``BoundCircuit`` (OpenQASM +
readout spec). The audit harness only ever sees that IR. Classical baselines
implement ``predict`` only (no circuit) and subclass ``BaselineModel``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

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

    # ---- the two-level auditable object ------------------------------------
    # An example decomposes into atomic auditable *units* (per-token / per-window
    # circuits) plus a *composition* that maps their (possibly truncated) expvals to a
    # label. Single-circuit models are the degenerate one-unit case.
    def audit_units(self, example) -> list[BoundCircuit]:
        """Atomic auditable circuits for one example (default: the single circuit)."""
        return self.export_circuits([example])

    def compose(self, example, units: list[dict]) -> int:
        """Map the units' audit info to one label (default: single-unit head).

        ``units`` is a list aligned with ``audit_units``; each element is a dict
        ``{"expvals": np.ndarray, "state": np.ndarray, "n_qubits": int}`` for the
        current bond dimension. Pure-state models read ``expvals``; mixed-state
        models (QMSAN) form reduced density matrices from ``state``.
        """
        return self.decision_from_expvals(units[0]["expvals"])

    # ---- optional mid-training snapshots -----------------------------------
    # Set cfg.snapshot_dir to have fit() checkpoint itself as it trains, so the audit
    # can be re-run along the training path. That answers the obvious objection to a
    # simulability verdict on an under-performing model: does chi* depend on how well
    # the model actually learned? No-op unless snapshot_dir is set.
    def _snapshot_hook(self, epoch: int, cfg) -> None:
        snapshot_dir = getattr(cfg, "snapshot_dir", None)
        if not snapshot_dir:
            return
        every = int(getattr(cfg, "snapshot_every", 1))
        if epoch % every:
            return
        self.save(Path(snapshot_dir) / f"epoch{epoch:04d}")

    # ---- optional best-on-validation selection ------------------------------
    # These models otherwise train a fixed number of epochs and keep the LAST weights,
    # so a run whose validation accuracy peaks early is evaluated well past its best.
    # Off by default: turning it on changes trained weights, so every stored result
    # would have to be regenerated. Set cfg.select_best_val to enable.
    def _best_val_hook(self, epoch: int, cfg, val) -> None:
        if not getattr(cfg, "select_best_val", False) or not val:
            return
        acc = self._accuracy(val)
        if acc > getattr(self, "_best_val_acc", -1.0):
            self._best_val_acc = acc
            self._best_val_epoch = epoch
            self._best_val_state = [p.detach().clone() for p in self._params()]

    def _restore_best_val(self, cfg) -> None:
        """Restore the weights remembered by _best_val_hook. Call after the epoch loop."""
        state = getattr(self, "_best_val_state", None)
        if not getattr(cfg, "select_best_val", False) or state is None:
            return
        import torch

        with torch.no_grad():
            for param, best in zip(self._params(), state):
                param.copy_(best)

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
