"""Trace one real audited example through both tracks of the audit.

Produces the measured numbers quoted in Figure 1 (paper/figs/fig_pipeline.tex) and the
dataset panel of Figure 2 (paper/figs/fig_venn.tex), so neither figure carries a
hand-copied value that nothing can check.

    python scripts/trace_example.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simcert import models  # noqa: F401,E402  (populates the registry)
from simcert.audit.entanglement import max_bipartite_entropy  # noqa: E402
from simcert.audit.mps_truncation import exact_statevector, truncate_state_to_bond_dim  # noqa: E402
from simcert.audit.observables import pauli_expval  # noqa: E402
from simcert.data.loaders import build_vocab, load_dataset  # noqa: E402
from simcert.io_results import select_runs  # noqa: E402
from simcert.registry import get_model  # noqa: E402

MODEL = "vqc_text"
DATASET = "mc"
SEED = 1
TEST_INDEX = 2  # 'engineer writes program', the sentence quoted in Figure 1


def trace():
    ds = load_dataset(DATASET, seed=SEED, val_frac=0.2, test_frac=0.2)
    cfg = OmegaConf.create({"name": MODEL})
    m = get_model(MODEL)()
    m.build(cfg, build_vocab(ds.train))
    rep = m.fit(ds.train, ds.val, cfg)

    ex = ds.test[TEST_INDEX]
    unit = m.audit_units(ex)[0]
    n = unit.n_qubits
    psi = exact_statevector(unit.qfunc(), n)
    obs = unit.observables[0]

    print("FIGURE 1, the traced example")
    print(f"  model={MODEL} dataset={DATASET} seed={SEED} "
          f"train_acc={rep.train_accuracy:.3f} val_acc={rep.val_accuracy:.3f}")
    print(f"  sentence   : {ex.text!r}")
    print(f"  true label : {ex.label}")
    print(f"  n_qubits   : {n}")
    print(f"  entropy    : {max_bipartite_entropy(psi, n):.3f} nats")
    ev_full = pauli_expval(psi, n, obs)
    print(f"  full       : <Z_0> = {ev_full:+.4f}  ->  label {int(ev_full < 0)}")
    for chi in (1, 2, 4):
        phi = truncate_state_to_bond_dim(psi, n, chi)
        ev = pauli_expval(phi, n, obs)
        fid = float(abs(np.vdot(psi, phi)) ** 2)
        print(f"  chi={chi:<2}     : <Z_0> = {ev:+.4f}  ->  label {int(ev < 0)}   F={fid:.4f}")


def dataset_panel():
    """The per-dataset rows of Figure 2: real sizes, label balance and observed chi*."""
    sel = select_runs(glob.glob(str(REPO / "results" / "metrics" / "*.json")))
    print("\nFIGURE 2, the dataset panel")
    print(f"  {'dataset':<9}{'train':>6}{'val':>5}{'test':>5}{'pos%':>7}{'vocab':>7}"
          f"  chi* observed")
    for name in ("mc", "mc_real", "rp", "sst2"):
        ds = load_dataset(name, seed=SEED, val_frac=0.2, test_frac=0.2)
        allex = list(ds.train) + list(ds.val) + list(ds.test)
        pos = 100.0 * float(np.mean([e.label for e in allex]))
        vocab = len({t for e in allex for t in e.text.split()})
        stars = sorted({r["certificate"]["chi_star"].get("tau_gen")
                        for (mm, dd), runs in sel.items() if dd == name for r in runs},
                       key=lambda v: (v is None, v))
        star_txt = "/".join("full" if v is None else str(v) for v in stars)
        # SST-2 rows are clipped phrases, so prefer one that reads as a whole clause and
        # carries no LaTeX-hostile punctuation. "--" in particular would set as an en
        # dash, and the paper is typeset without dashes of any kind.
        def clean(e):
            t = e.text
            return ("--" not in t and "(" not in t and ")" not in t
                    and " - " not in t and not t.endswith(",") and len(t.split()) >= 3)
        example = next((e for e in ds.test if clean(e)), ds.test[0])
        print(f"  {name:<9}{len(ds.train):>6}{len(ds.val):>5}{len(ds.test):>5}"
              f"{pos:>7.1f}{vocab:>7}  {star_txt}")
        print(f"    example (label {example.label}): {example.text!r}")


if __name__ == "__main__":
    trace()
    dataset_panel()
