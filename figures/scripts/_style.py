"""Shared plotting style: colorblind-safe colors + distinct line styles + markers,
so every series is distinguishable by color AND by shape (readable in grayscale too)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette
PALETTE = {
    "vqc_text": "#0072B2",   # blue
    "qsann": "#D55E00",      # vermillion
    "qmsan": "#009E73",      # green
    "claqs": "#CC79A7",      # magenta
    "discocat": "#E69F00",   # orange
    "_default": "#000000",
}
LINESTYLE = {"mc": "-", "mc_real": "--", "rp": "-.", "sst2": ":"}
MARKER = {"mc": "o", "mc_real": "s", "rp": "^", "sst2": "D"}


def style_of(model: str, dataset: str):
    return dict(
        color=PALETTE.get(model, PALETTE["_default"]),
        linestyle=LINESTYLE.get(dataset, "-"),
        marker=MARKER.get(dataset, "o"),
        linewidth=2.2,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )


def apply_rc():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })
