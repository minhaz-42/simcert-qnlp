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

# Only sentences the paper actually quotes. Rendering diagrams nothing includes just
# leaves unused images in the repo. The RP item is the one the method figure uses: its
# relative pronoun is where the grammar cups do work that a bag of tokens cannot copy.
WANTED = [
    ("mc", "mc", "engineer writes program"),
    ("rp", "rp", "person that teacher teach"),
]



def _trim_whitespace(path, pad: int = 6) -> None:
    """Crop the white margin matplotlib leaves around the diagram.

    Without this the figure has to be scaled up to make the word boxes legible, which then
    wastes a third of the column on empty canvas.
    """
    from PIL import Image, ImageChops

    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img, bg).getbbox()
    if box is None:
        return
    left, top, right, bottom = box
    img.crop((max(0, left - pad), max(0, top - pad),
              min(img.width, right + pad), min(img.height, bottom + pad))).save(path)


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
        _trim_whitespace(path)
        print(f"wrote {path.relative_to(REPO)}  ({sentence!r} from {dsname})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
