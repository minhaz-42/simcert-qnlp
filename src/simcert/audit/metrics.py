"""Classification metrics, bootstrap confidence intervals, and paired significance
tests (NumPy-only, no SciPy dependency)."""

from __future__ import annotations

import math

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


def mcnemar(y_true, pred_full, pred_trunc) -> dict:
    """Exact McNemar test comparing the full model against a truncated surrogate.

    Contingency is built on per-example correctness against ``y_true``:
      b = full correct, truncated wrong;  c = full wrong, truncated correct.
    Returns the discordant counts and the two-sided exact-binomial p-value under
    H0 (the two classifiers are equally accurate). A large p-value means truncation
    does not significantly change the decisions, which is the simulability signal; a
    small p-value flags a surrogate that is significantly worse than the full model.
    """
    yt = np.asarray(y_true)
    full_correct = np.asarray(pred_full) == yt
    trunc_correct = np.asarray(pred_trunc) == yt
    b = int(np.sum(full_correct & ~trunc_correct))
    c = int(np.sum(~full_correct & trunc_correct))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p)}
