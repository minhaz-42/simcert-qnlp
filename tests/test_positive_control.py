"""The positive control must keep discriminating, or it is not a control.

These are the acceptance tests the soundness lens demanded before any control is allowed
into the paper (docs/positive-control-study.md). They run the real audit path -- the same
`run_chi_sweep` and `chi_star` the paper's models go through -- on the k-of-m dial, and
assert the two endpoints plus the shape of the curve between them.

The earlier candidate design failed exactly these: it reported chi*=1 on a maximally
entangled state and chi*=None on a near-separable one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "positive_control_kofm", REPO / "scripts" / "positive_control_kofm.py"
)
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

M = 4  # n = 8 qubits, k in 0..4, exact bound 2^k in 1..16
NEX = 64  # enough to separate 0.5 from 1.0 decisively; the step is exact, not statistical


@pytest.fixture(scope="module")
def curve():
    return [pc.run_point(k, M, n_examples=NEX) for k in range(M + 1)]


def test_separable_endpoint_is_certified_simulable(curve):
    """k=0 has no entanglement, so a product surrogate must suffice."""
    assert curve[0]["chi_star"] == 1


def test_maximal_endpoint_is_not_certified_simulable(curve):
    """k=m is maximally entangled; chi*=1 here would be the earlier design's failure."""
    top = curve[-1]["chi_star"]
    assert top is None or top > 1, f"maximal entanglement certified at chi*={top}"


def test_chi_star_is_non_decreasing_in_k(curve):
    """More load-bearing entangled pairs must never need less bond dimension."""
    seq = [r["chi_star"] for r in curve]
    for a, b in zip(seq, seq[1:]):
        if b is None:
            continue
        assert a is not None and a <= b, f"chi* went backwards: {seq}"


def test_chi_star_never_exceeds_the_exact_bound(curve):
    """chi* is a decision-level quantity, so it can only be <= the state-level bound 2^k."""
    for r in curve:
        if r["chi_star"] is not None:
            assert r["chi_star"] <= r["exact_bond"], (
                f"k={r['k']}: chi*={r['chi_star']} exceeds the exact bound {r['exact_bond']}"
            )


def test_chi_star_spans_the_grid(curve):
    """The instrument must demonstrably report more than one value."""
    seen = {r["chi_star"] for r in curve}
    assert len(seen) >= 3, f"dial produced too few distinct chi* values: {seen}"
    assert max(x for x in seen if x is not None) >= 8


def test_entanglement_entropy_is_exactly_k_ln2(curve):
    """Pins the state construction: k straddling Bell pairs and not one more.

    This is the check that caught a transposed wire permutation, which silently built a
    different pairing whose entropy saturated at 2 ln 2 instead of growing with k.
    """
    for r in curve:
        assert r["entropy"] == pytest.approx(r["k"] * np.log(2), abs=1e-9)


def test_chi_one_fidelity_is_exactly_two_to_the_minus_k(curve):
    """A product surrogate keeps one of 2^k equally weighted Schmidt branches."""
    for r in curve:
        assert r["fid_by_chi"][1] == pytest.approx(2.0 ** -r["k"], abs=1e-9)


def test_destroyed_readout_scores_exactly_chance(curve):
    """Where the decision is not recovered, accuracy must be 0.5, not the class frequency.

    Classes are balanced by construction so that a dead-zone tie-break cannot inflate
    small-chi accuracy into something that looks like retained signal.
    """
    for r in curve:
        for chi, acc in sorted(r["acc_by_chi"].items()):
            if r["chi_star"] is not None and chi < r["chi_star"]:
                assert acc == pytest.approx(0.5, abs=1e-9), (
                    f"k={r['k']} chi={chi}: expected chance, got {acc}"
                )


def test_readout_is_weight_two(curve):
    """Weight 2 is forced: weight 1 cannot sense a Bell pair, weight n cancels exactly."""
    obs = pc.correlator_obs(M)
    assert obs, "no observables"
    assert {len(o) for o in obs} == {2}
    assert all(set(o.values()) == {"X"} for o in obs), (
        "correlators must be X-type; a diagonal Z-correlator is reproduced exactly by a "
        "single retained branch and is therefore blind to entanglement"
    )
