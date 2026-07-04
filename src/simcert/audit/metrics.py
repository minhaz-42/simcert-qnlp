"""Classification metrics and bootstrap confidence intervals (NumPy-only)."""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_ci(y_true, y_pred, metric=accuracy, n_boot: int = 1000, ci: float = 0.95, seed: int = 0):
    """Return ``(point_estimate, lo, hi)`` for a metric via example-level bootstrap."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y_true[idx], y_pred[idx])
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return (float(metric(y_true, y_pred)), lo, hi)
