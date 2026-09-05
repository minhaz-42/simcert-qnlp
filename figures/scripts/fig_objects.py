"""Render the real objects that Figures 1 and 2 embed.

These datasets are text, so there is no photograph to show. What there is, and what a
reader of a simulability paper actually wants to see, is the object the audit operates
on: the trained state itself. This script renders three kinds, all measured from real
trained models on real dataset sentences, and saves them as image files that the LaTeX
figures include.

  amplitudes_mc.pdf   the 64 amplitudes of one trained state (the audited reference model
                      is 6 qubits), full against chi=1, so the "same label at 35 percent
                      fidelity" claim is visible rather than asserted
  schmidt_all.pdf     the Schmidt spectrum at the middle cut for one trained state per
                      dataset, with the chi=1 and chi=2 truncation points marked; this is
                      the picture of why chi*=1 holds for the attention models and why
                      the compositional model needs 2

Regenerate with:  python figures/scripts/fig_objects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from _style import INK_MUTED, MODELS, PAIR, REFERENCE, apply_rc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

import importlib.util  # noqa: E402

from simcert import models  # noqa: F401,E402
from simcert.audit.mps_truncation import truncate_state_to_bond_dim  # noqa: E402
from simcert.audit.observables import pauli_expval  # noqa: E402

# One code path for building a trained state, shared with scripts/trace_example.py. It
# seeds exactly as simcert.runner does; a local copy that forgot to seed is why this
# script and Figure 1 once reported different fidelities for the same example.
_spec = importlib.util.spec_from_file_location(
    "trace_example", REPO / "scripts" / "trace_example.py"
)
_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trace)

OUT = REPO / "figures" / "objects"

# One real trained state per dataset. discocat is the model that needs chi=2, so it has to
# be in the Schmidt panel or the panel would only show the easy case.
CASES = [
    ("vqc_text", "mc", "engineer writes program", {}),
    ("qsann", "mc", None, {}),
    ("discocat", "rp", None, {}),
    ("vqc_text", "sst2", None, {}),
]


def _trained_state(model_name, dataset, text, extra, seed=1):
    got = _trace.trained_state(model_name=model_name, dataset=dataset, seed=seed,
                               text=text, extra=extra)
    return got["psi"], got["n"], got["obs"], got["example"]


def _schmidt(psi, n):
    """Singular values at the middle cut, normalised."""
    half = n // 2
    s = np.linalg.svd(psi.reshape(2**half, -1), compute_uv=False)
    return s / np.linalg.norm(s)


def amplitudes_figure():
    """The trained state's amplitudes, exact against the chi=1 product surrogate."""
    psi, n, obs, ex = _trained_state(*CASES[0][:3], CASES[0][3])
    phi = truncate_state_to_bond_dim(psi, n, 1)
    fid = float(abs(np.vdot(psi, phi)) ** 2)
    ev_f, ev_t = pauli_expval(psi, n, obs), pauli_expval(phi, n, obs)

    idx = np.arange(2**n)
    w = 0.44
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    ax.bar(idx - w / 2, np.abs(psi) ** 2, w, color=PAIR[0], linewidth=0,
           label="trained state")
    ax.bar(idx + w / 2, np.abs(phi) ** 2, w, color=PAIR[1], linewidth=0,
           label=r"$\chi{=}1$ product surrogate")
    # n=6 is 64 basis states, so per-bar binary labels are unreadable; tick the corners
    # and let the shape carry the message.
    if 2**n <= 16:
        ax.set_xticks(idx)
        ax.set_xticklabels([format(i, f"0{n}b") for i in idx], rotation=90, fontsize=6)
    else:
        ticks = [0, 2**n // 4, 2**n // 2, 3 * (2**n) // 4, 2**n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([format(t, f"0{n}b") for t in ticks], fontsize=7)
    ax.set_xlim(-1, 2**n)
    ax.set_ylabel("probability")
    ax.set_xlabel("computational basis state")
    ax.set_title(f"'{ex.text}'   $F={fid:.2f}$,   "
                 rf"$\langle Z_0\rangle$ {ev_f:+.3f} $\to$ {ev_t:+.3f},   same label",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"amplitudes_mc.{e}")
    plt.close(fig)
    print(f"wrote amplitudes_mc  (n={n}, F={fid:.4f}, ev {ev_f:+.4f} -> {ev_t:+.4f})")


def schmidt_figure():
    """Schmidt spectra of real trained states, one per case, with the chi cuts marked."""
    apply_rc()
    fig, axes = plt.subplots(1, len(CASES), figsize=(11.0, 2.6), sharey=True)
    for ax, (mdl, dsname, text, extra) in zip(axes, CASES):
        psi, n, obs, ex = _trained_state(mdl, dsname, text, extra)
        sv = _schmidt(psi, n)
        k = np.arange(1, len(sv) + 1)
        ax.bar(k, sv**2, color=MODELS.get(mdl, INK_MUTED), edgecolor="white", linewidth=0.6)
        for chi, style in ((1, "-"), (2, ":")):
            if chi < len(sv):
                ax.axvline(chi + 0.5, color=REFERENCE, ls=style, lw=1.4)
        ax.set_yscale("log")
        # discocat's spectrum falls ~30 orders after index 2, which is exactly why it
        # needs chi=2; without a floor that cliff stretches the axis until every other
        # panel is a flat band. Clip and say so in the caption.
        ax.set_ylim(1e-7, 2.0)
        ax.set_xticks(k[: min(8, len(k))])
        ax.set_xlim(0.4, min(8, len(sv)) + 0.6)
        ax.set_title(rf"$\mathtt{{{mdl.replace('_', chr(92) + '_')}}}$ on {dsname}"
                     f"\n$n={n}$, kept weight at $\\chi{{=}}1$: {sv[0]**2:.2f}", fontsize=8.5)
        ax.set_xlabel("Schmidt index")
    axes[0].set_ylabel("Schmidt weight $s_k^2$")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"schmidt_all.{e}")
    plt.close(fig)
    print("wrote schmidt_all")


if __name__ == "__main__":
    apply_rc()
    amplitudes_figure()
    schmidt_figure()
