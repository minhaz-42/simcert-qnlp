"""Real MC / RP datasets from Lorenz et al. 2021 (Quantinuum resources repo).

The canonical QNLP benchmarks used by QSANN, QMSAN, and DisCoCat. MC = meaning
classification (cooking vs IT), RP = relative-pronoun (subject vs object relative clause);
RP is the discriminating task where the models' published accuracies actually differ.

Line format: ``<label>\\t<word>_<POS> <word>_<POS> ...``. We strip the POS tags (our
bag-of-token models don't use them; the DisCoCat producer re-parses with cups_reader).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE = "https://raw.githubusercontent.com/Quantinuum/qnlp_lorenz_etal_2021_resources/main/datasets"


def _fetch(name: str, cache_dir: str) -> list[tuple[str, int]]:
    path = Path(cache_dir) / f"lorenz_{name}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{_BASE}/{name}.txt", timeout=60) as r:
            path.write_bytes(r.read())
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        label = int(parts[0])
        words = [t.rsplit("_", 1)[0] for t in parts[1:]]  # drop _POS tags
        out.append((" ".join(words), label))
    return out


def mc(cache_dir: str = "data") -> dict[str, list[tuple[str, int]]]:
    return {s: _fetch(f"mc_{s}_data", cache_dir) for s in ("train", "dev", "test")}


def rp(cache_dir: str = "data") -> dict[str, list[tuple[str, int]]]:
    return {s: _fetch(f"rp_{s}_data", cache_dir) for s in ("train", "test")}
