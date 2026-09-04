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
from simcert.runner import load_config  # noqa: E402
from simcert.seed import seed_everything  # noqa: E402

MODEL = "vqc_text"
DATASET = "mc"
SEED = 1
TEST_INDEX = 2  # 'engineer writes program', the sentence quoted in Figure 1


def trained_state(model_name=MODEL, dataset=DATASET, seed=SEED, text=None,
                  audit="default", extra=None):
    """Build and train the model the AUDIT actually ran, then return one example's state.

    Two things here are load-bearing and both were got wrong first time round.

    The config comes from ``load_config``, the same Hydra composition ``run_grid.py``
    uses, not from a hand-built OmegaConf dict. Building the dict by hand silently
    produced a 4-qubit reference classifier when every audited run of it is 6 qubits, so
    the traced numbers described a model the paper never audited.

    And ``seed_everything`` runs before build and fit, exactly as ``runner.main`` does.
    Without it the parameter initialisation depends on ambient RNG state and the same
    nominal example yields a different fidelity on each invocation.
    """
    argv = ["mode=both", f"model={model_name}", f"dataset={dataset}",
            f"audit={audit}", f"seed={seed}"]
    for k, v in (extra or {}).items():
        argv.append(f"model.{k}={v}")
    cfg = load_config(argv)

    seed_everything(int(cfg.seed))
    ds_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.dataset, resolve=True).items()
                 if k not in ("name", "val_frac", "test_frac")}
    ds = load_dataset(cfg.dataset_name, seed=int(cfg.seed),
                      val_frac=float(cfg.dataset.val_frac),
                      test_frac=float(cfg.dataset.get("test_frac", 0.2)), **ds_kwargs)

    m = get_model(cfg.model_name)()
    m.build(cfg.model, build_vocab(ds.train))
    rep = m.fit(ds.train, ds.val, cfg.model)

    pool = [e for e in ds.test if getattr(m, "records", None) is None or e.text in m.records]
    pool = pool or list(ds.test)
    ex = next((e for e in pool if e.text == text), None)
    if ex is None:
        ex = pool[TEST_INDEX] if (text is None and len(pool) > TEST_INDEX) else pool[0]

    unit = m.audit_units(ex)[0]
    n = unit.n_qubits
    psi = unit.state if unit.state is not None else exact_statevector(unit.qfunc(), n)
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    obs = unit.observables[0] if unit.observables else None
    return dict(psi=psi, n=n, obs=obs, example=ex, report=rep, model=cfg.model_name,
                dataset=cfg.dataset_name, cfg=cfg)


def trace():
    got = trained_state()
    rep, ex, n, psi, obs = (got["report"], got["example"], got["n"], got["psi"], got["obs"])
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
