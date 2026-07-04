"""Synthetic MC-style meaning-classification dataset (food vs IT).

The canonical QNLP benchmark (Lorenz et al. 2021; used by lambeq and the QSANN paper)
is a ~130-sentence 2-class "meaning classification" set: cooking vs computing. We
generate a faithful, fully self-contained stand-in from subject-verb-object templates
so the pipeline needs no network. When the ``qnlp-lambeq`` env is built, the *real* MC
dataset (shipped with lambeq) can be swapped in behind the same loader interface.

Label convention: 0 = FOOD/cooking, 1 = IT/computing.
"""

from __future__ import annotations

import random

_FOOD = {
    "subj": ["chef", "cook", "baker", "waiter", "person"],
    "verb": ["cooks", "prepares", "serves", "bakes", "tastes"],
    "obj": ["meal", "dinner", "bread", "soup", "sauce"],
}
_IT = {
    "subj": ["programmer", "developer", "engineer", "admin", "person"],
    "verb": ["writes", "runs", "debugs", "compiles", "deploys"],
    "obj": ["software", "program", "code", "application", "script"],
}


def _all_sentences(bank: dict[str, list[str]]) -> list[str]:
    out = []
    for s in bank["subj"]:
        for v in bank["verb"]:
            for o in bank["obj"]:
                out.append(f"{s} {v} {o}")
    return out


def generate(n_per_class: int = 65, seed: int = 0) -> list[tuple[str, int]]:
    """Return a shuffled list of ``(sentence, label)`` (balanced across the two classes)."""
    rng = random.Random(seed)
    food = _all_sentences(_FOOD)
    it = _all_sentences(_IT)
    rng.shuffle(food)
    rng.shuffle(it)
    n = min(n_per_class, len(food), len(it))
    data = [(s, 0) for s in food[:n]] + [(s, 1) for s in it[:n]]
    rng.shuffle(data)
    return data
