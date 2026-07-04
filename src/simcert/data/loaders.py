"""Unified dataset loading + vocabulary construction with seed-stable splits."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TextExample:
    text: str
    label: int

    @property
    def tokens(self) -> list[str]:
        return self.text.lower().split()


@dataclass
class Dataset:
    name: str
    train: list[TextExample]
    val: list[TextExample]
    test: list[TextExample]
    n_classes: int

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_classes": self.n_classes,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
        }


def _split(pairs, val_frac, test_frac, seed):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = pairs[:n_test]
    val = pairs[n_test : n_test + n_val]
    train = pairs[n_test + n_val :]
    to_ex = lambda seq: [TextExample(t, int(l)) for t, l in seq]
    return to_ex(train), to_ex(val), to_ex(test)


def load_dataset(
    name: str, seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2, **kwargs
) -> Dataset:
    """Load a dataset by name. Currently: ``mc`` (synthetic meaning classification)."""
    if name == "mc":
        from .datasets import mc

        pairs = mc.generate(n_per_class=kwargs.get("n_per_class", 65), seed=seed)
        train, val, test = _split(pairs, val_frac, test_frac, seed)
        return Dataset(name=name, train=train, val=val, test=test, n_classes=2)
    if name == "sst2":
        from .datasets import sst2

        train_pairs, test_pairs = sst2.load_splits(
            n_train=kwargs.get("n_train", 600),
            n_test=kwargs.get("n_test", 200),
            max_tokens=kwargs.get("max_tokens", 16),
            seed=seed,
        )
        # dataset provides its own train/test; carve val out of train (test_frac ignored)
        train, val, _ = _split(train_pairs, val_frac, 0.0, seed)
        test = [TextExample(t, int(l)) for t, l in test_pairs]
        return Dataset(name=name, train=train, val=val, test=test, n_classes=2)
    raise ValueError(f"unknown dataset {name!r} (available: 'mc', 'sst2')")


def build_vocab(examples: list[TextExample], min_freq: int = 1) -> dict[str, int]:
    """Word -> index vocabulary (index 0 reserved for <unk>)."""
    from collections import Counter

    counts = Counter(tok for ex in examples for tok in ex.tokens)
    vocab = {"<unk>": 0}
    for tok, c in sorted(counts.items()):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab
