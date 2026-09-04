"""Render the DisCoCat string diagrams that Figure 1 embeds.

Runs in the LAMBEQ env, not the audit env, because it needs lambeq's drawing code:

    conda run -n qnlp-lambeq python scripts/render_diagram_objects.py

Kept separate from figures/scripts/fig_objects.py for exactly that reason. The two envs
cannot coexist (lambeq pins pennylane<0.37 while the truncation device needs a modern
one), so the figure's two halves are produced by two commands and committed as images.

The sentences are read from the datasets rather than hard-coded, so a diagram can never
drift from a sentence the corpus does not actually contain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lambeq import cups_reader  # noqa: E402

from simcert.data.loaders import load_dataset  # noqa: E402

OUT = REPO / "figures" / "objects"

# Only the sentence Figure 1 quotes. Rendering diagrams nothing includes just leaves
# unused images in the repo.
WANTED = [
    ("mc", "mc", "engineer writes program"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rc = 0
    for tag, dsname, sentence in WANTED:
        ds = load_dataset(dsname, seed=1, val_frac=0.2, test_frac=0.2)
        corpus = {e.text for e in list(ds.train) + list(ds.val) + list(ds.test)}
        if sentence not in corpus:
            print(f"FAIL {tag}: {sentence!r} is not in the {dsname} corpus")
            rc = 1
            continue
        diagram = cups_reader.sentence2diagram(sentence)
        path = OUT / f"diagram_{tag}.png"
        diagram.draw(draw_as_pregroup=True, path=str(path), show=False, figsize=(6, 3))
        print(f"wrote {path.relative_to(REPO)}  ({sentence!r} from {dsname})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
