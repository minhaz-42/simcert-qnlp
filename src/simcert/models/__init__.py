"""The pluggable QNLP model zoo.

Each model implements the ``QNLPModel`` interface so the *same* audit runs on all
of them. Concrete models (DisCoCat/lambeq, QSANN, QMSAN, CLAQS, hybrid quantum-BERT)
are added in milestones M1/M4; this package ships the interface first (harness-first).
"""

from .base import BaselineModel, QNLPModel, TrainReport

__all__ = ["QNLPModel", "BaselineModel", "TrainReport"]
