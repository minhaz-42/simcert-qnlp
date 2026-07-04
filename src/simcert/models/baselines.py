"""Classical baselines — the accuracy floor (Bowles-style out-of-the-box classifiers)."""

from __future__ import annotations

import numpy as np

from ..data.loaders import TextExample, build_vocab
from ..registry import register
from .base import BaselineModel, TrainReport


def _bow_matrix(examples, vocab):
    x = np.zeros((len(examples), len(vocab)), dtype=float)
    for r, ex in enumerate(examples):
        for tok in ex.tokens:
            x[r, vocab.get(tok, 0)] += 1.0
    return x


@register("logreg_bow")
class LogRegBoW(BaselineModel):
    """Logistic regression on bag-of-words counts."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.clf = None

    def fit(self, train, val, cfg) -> TrainReport:
        from sklearn.linear_model import LogisticRegression

        self.vocab = build_vocab(train)
        xt = _bow_matrix(train, self.vocab)
        yt = np.array([ex.label for ex in train])
        self.clf = LogisticRegression(max_iter=1000, C=float(getattr(cfg, "C", 1.0)))
        self.clf.fit(xt, yt)
        train_acc = float(self.clf.score(xt, yt))
        val_acc = (
            float(self.clf.score(_bow_matrix(val, self.vocab), [ex.label for ex in val]))
            if len(val)
            else 0.0
        )
        return TrainReport(
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            n_params=int(self.clf.coef_.size + self.clf.intercept_.size),
        )

    def predict(self, batch) -> np.ndarray:
        return self.clf.predict(_bow_matrix(batch, self.vocab))
