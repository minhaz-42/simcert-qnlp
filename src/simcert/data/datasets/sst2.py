"""SST-2 (Stanford Sentiment Treebank, binary) via the HuggingFace datasets-server API.

A real, non-separable sentiment benchmark (the headline dataset in CLAQS and a standard
QNLP task). We download a small balanced subset at simulator scale (matching the ~1000-
example subsets the QSANN/QMSAN papers use), cache it locally, and cap sequence length.

Label convention: 0 = negative, 1 = positive.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

_API = "https://datasets-server.huggingface.co/rows"
_DATASET = "stanfordnlp/sst2"


def _fetch_split(split: str, n: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    offset = 0
    while len(rows) < n:
        length = min(100, n - len(rows))  # API caps length at 100
        url = f"{_API}?dataset={_DATASET}&config=default&split={split}&offset={offset}&length={length}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item["row"]
            rows.append((row["sentence"].strip(), int(row["label"])))
        offset += length
    return rows


def _cache(cache_dir: str, n_train_pool: int, n_test_pool: int) -> dict:
    path = Path(cache_dir) / "sst2_cache.json"
    if path.exists():
        obj = json.loads(path.read_text())
        if len(obj.get("train", [])) >= n_train_pool and len(obj.get("validation", [])) >= n_test_pool:
            return obj
    obj = {
        "train": _fetch_split("train", n_train_pool),
        "validation": _fetch_split("validation", n_test_pool),  # has gold labels; test does not
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return obj


def _prep(pairs, n, max_tokens, rng) -> list[tuple[str, int]]:
    by = {0: [], 1: []}
    for text, label in pairs:
        toks = text.split()
        if not toks:
            continue
        by[label].append((" ".join(toks[:max_tokens]), label))
    per = n // 2
    out = []
    for label in (0, 1):
        rng.shuffle(by[label])
        out += by[label][:per]
    rng.shuffle(out)
    return out


def load_splits(n_train: int = 600, n_test: int = 200, max_tokens: int = 16, seed: int = 0,
                cache_dir: str = "data"):
    """Return ``(train_pairs, test_pairs)`` as balanced, token-capped subsets."""
    rng = random.Random(seed)
    obj = _cache(cache_dir, max(2 * n_train, 1200), max(2 * n_test, 872))
    train = _prep(obj["train"], n_train, max_tokens, rng)
    test = _prep(obj["validation"], n_test, max_tokens, rng)
    return train, test
