"""Every audited model must read out a bounded-weight Pauli.

This guards a specific, measured failure mode of chi*, found while designing a positive
control (docs/positive-control-study.md). On the candidate control, chi* came out
ANTI-correlated with entanglement: a maximally entangled state was certified at chi*=1
with 1.6% state fidelity, while a near-separable state at 99.3% fidelity got chi*=None.

The cause was not entanglement, it was the readout. The control read out the
maximal-weight Pauli (weight n, one factor on every wire). A maximal-weight Pauli maps
each computational-basis branch onto a different branch, so on any state supported on a
subset of branches -- which is exactly what a low-chi MPS truncation produces -- every
term in the expectation value pairs a retained amplitude with a discarded one and the
readout cancels to exactly zero, no matter how far the truncated state is from the true
one. chi* then reports a property of the operator's alignment with the truncation basis
rather than a property of the state.

A weight-1 readout cannot do this: it depends only on a single wire's reduced density
matrix, so it cannot be driven to an exact algebraic zero by branch bookkeeping, and it
obeys |<O>_psi - <O>_phi| <= 2 sqrt(1 - F) with ||O|| = 1.

Every shipped model reads weight 1 (measured, not assumed -- see the test below), which
is why the paper's verdicts are not exposed to this. The test exists so that a future
model cannot introduce a high-weight readout without someone deciding to.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from simcert import models  # noqa: F401  (import populates the model registry)
from simcert.audit.mps_truncation import truncate_state_to_bond_dim
from simcert.audit.observables import pauli_expval
from simcert.data.loaders import build_vocab, load_dataset
from simcert.registry import get_model

# (model, dataset, extra config) for every model with a Pauli readout. discocat is absent
# on purpose: it carries no observables at all and is audited through the post-selected
# open-wire Born rule, which is a projector (norm 1) rather than a Pauli string.
PAULI_READOUT = [
    ("vqc_text", "mc", {}),
    ("qsann", "mc", {}),
    ("qmsan", "mc", {}),
    ("claqs", "mc", {"train_chunk": 1}),
]

MAX_ALLOWED_WEIGHT = 1


def _audit_units(model_name, dataset, extra):
    ds = load_dataset(dataset, seed=1, val_frac=0.2, test_frac=0.2)
    vocab = build_vocab(ds.train)
    cfg = OmegaConf.create({"name": model_name, **extra})
    m = get_model(model_name)()
    m.build(cfg, vocab)
    return m.audit_units(ds.test[0])


@pytest.mark.parametrize("model_name,dataset,extra", PAULI_READOUT)
def test_readout_is_bounded_weight(model_name, dataset, extra):
    """No audited model may read a Pauli acting on more than one wire."""
    units = _audit_units(model_name, dataset, extra)
    weights = [len(o) for u in units for o in (u.observables or [])]
    assert weights, f"{model_name} produced no observables to check"
    worst = max(weights)
    assert worst <= MAX_ALLOWED_WEIGHT, (
        f"{model_name} reads a weight-{worst} Pauli. A readout whose weight approaches "
        f"n_qubits can cancel to exactly zero under MPS truncation independently of "
        f"state fidelity, which decouples chi* from what it is meant to measure. See "
        f"this module's docstring."
    )


def test_discocat_has_no_pauli_readout():
    """discocat is exempt by construction, not by oversight; pin that so it stays true.

    Driven from the producer's own exported records rather than a freshly split dataset:
    discocat replays circuits the lambeq producer exported under the real run's split, so
    an example drawn from a different split is simply not in its manifest.
    """
    from simcert.data.loaders import TextExample

    cfg = OmegaConf.create({"name": "discocat", "seed": 1, "dataset": "rp"})
    m = get_model("discocat")()
    m.build(cfg, vocab=None)
    assert m.records, "discocat has no exported records; run the lambeq producer first"

    text = next(iter(m.records))
    units = m.audit_units(TextExample(text=text, label=m.records[text]["label"]))
    assert all(not u.observables for u in units), (
        "discocat gained a Pauli readout; it is audited via the post-selected Born rule "
        "and must be added to PAULI_READOUT with its weight checked if that changes"
    )


def test_maximal_weight_pauli_cancels_under_truncation():
    """Demonstrate the failure mode this module guards against, so it stays understood.

    A GHZ state's weight-n X readout is +-1 exactly, but the chi=1 truncation keeps a
    single branch, and X^(x)n sends that branch to the (discarded) complementary one, so
    the truncated readout is exactly zero while the weight-1 readout is not.
    """
    n = 6
    ghz = np.zeros(2**n)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)

    weight_n = {i: "X" for i in range(n)}
    weight_1 = {0: "Z"}

    exact_n = pauli_expval(ghz, n, weight_n)
    assert abs(exact_n) == pytest.approx(1.0, abs=1e-12), "GHZ should read +-1 on X^(x)n"

    phi = truncate_state_to_bond_dim(ghz, n, 1)
    fidelity = float(abs(np.vdot(ghz, phi)) ** 2)

    # The pathology: the readout is destroyed exactly, not approximately, and the size of
    # the error carries no information about how wrong the truncated state is.
    assert pauli_expval(phi, n, weight_n) == pytest.approx(0.0, abs=1e-12)

    # The protection: a weight-1 readout on the same state and same truncation obeys the
    # fidelity bound, so its error cannot be decoupled from the state's closeness.
    err_1 = abs(pauli_expval(ghz, n, weight_1) - pauli_expval(phi, n, weight_1))
    assert err_1 <= 2 * np.sqrt(max(0.0, 1 - fidelity)) + 1e-9
