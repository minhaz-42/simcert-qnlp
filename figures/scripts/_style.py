"""Shared plotting style for every figure in the paper.

One module so the figures read as one system: a model keeps its colour across every
panel it appears in, a dataset keeps its marker, and nothing is encoded by colour alone.

THREE COLOUR JOBS, THREE RULES. Getting these confused is what produced the defect this
module replaces: the scaling figure coloured its qubit counts from the categorical cycle,
which ran out at seven series and silently painted n=12, n=14 and n=16 the same shade.

  * IDENTITY (which model)      -> CATEGORICAL. Fixed hue order, never cycled. Five
                                   models, five slots, assigned once in MODELS.
  * MAGNITUDE (which qubit count, which k) -> SEQUENTIAL. One hue, light to dark, because
                                   these values are ORDERED and a reader should be able to
                                   see the order in the ink. Use ``seq()``.
  * CONTRAST (ours vs published, full vs surrogate) -> a two-colour PAIR, ``PAIR``.

Twelve (model, dataset) series exceed what any categorical palette can carry, so identity
is COMPOSITE: colour carries the model, marker and dash carry the dataset, and the legend
is split into those two halves rather than listing twelve combinations.

The categorical order below is the validated default palette. On the adjacent pairlist,
which is the right one for lines and bars, its worst adjacent colour-vision-deficiency
separation is dE 9.1 and its worst normal-vision separation dE 19.6, both above their
gates. Three slots sit below 3:1 contrast against white, which is why every figure that
uses them also carries direct value labels and is accompanied by the results table.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Categorical slots, in fixed order. A sixth model takes slot 6, never a generated hue.
_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]

MODELS = {
    "vqc_text": _SLOTS[0],   # blue
    "qsann": _SLOTS[1],      # orange
    "qmsan": _SLOTS[2],      # aqua
    "claqs": _SLOTS[3],      # yellow
    "discocat": _SLOTS[4],   # magenta
}
PALETTE = {**MODELS, "_default": "#52514e"}

# Dataset -> marker and dash. Carries identity alongside colour, so the figures survive
# grayscale printing and colour-vision deficiency.
MARKER = {"mc": "o", "mc_real": "s", "rp": "^", "sst2": "D"}
LINESTYLE = {"mc": "-", "mc_real": "--", "rp": "-.", "sst2": ":"}

# Two-colour pair for before/after comparisons. Slots 1 and 2, the best-separated pair.
PAIR = (_SLOTS[0], _SLOTS[1])

# Recessive ink for grids, reference lines and annotation.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
REFERENCE = "#8a8983"

# Sequential ramp for ordered quantities. One hue, light to dark, lightness monotone.
_SEQ = LinearSegmentedColormap.from_list(
    "simcert_seq", ["#bcd6f2", "#2a78d6", "#123a69"]
)


def seq(i: int, n: int):
    """Colour for step ``i`` of ``n`` on the sequential ramp (ordered values only).

    Never use this for identity, and never use a categorical slot for an ordered value.
    """
    if n <= 1:
        return _SEQ(0.5)
    return _SEQ(0.12 + 0.88 * (i / (n - 1)))


def style_of(model: str, dataset: str) -> dict:
    """Line style for one (model, dataset) series: colour = model, shape = dataset."""
    return dict(
        color=PALETTE.get(model, PALETTE["_default"]),
        linestyle=LINESTYLE.get(dataset, "-"),
        marker=MARKER.get(dataset, "o"),
        linewidth=2.0,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.9,
    )


def composite_legend(ax, models, datasets, **kw):
    """Split legend for composite encoding: model colours, then dataset shapes.

    Listing every (model, dataset) pair would need twelve entries and be unreadable; this
    states the two encodings separately so the reader composes them.
    """
    handles = [Line2D([], [], color=MODELS.get(m, PALETTE["_default"]), lw=3, label=m)
               for m in models]
    handles += [Line2D([], [], color=INK_MUTED, lw=1.6, marker=MARKER.get(d, "o"),
                       linestyle=LINESTYLE.get(d, "-"), markersize=7,
                       markeredgecolor="white", label=_ds_label(d))
                for d in datasets]
    return ax.legend(handles=handles, **kw)


def _ds_label(d: str) -> str:
    return {"mc": "MC", "mc_real": "MC-real", "rp": "RP", "sst2": "SST-2"}.get(d, d)


def apply_rc():
    """Paper-facing defaults: recessive grid and axes, text in ink rather than series colour."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#dcdbd4",
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind the data, never over it
        "grid.color": "#dcdbd4",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "axes.edgecolor": "#b8b7b0",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,        # drop the box; keep only the two axes that read
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
