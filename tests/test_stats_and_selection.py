"""Paired significance test, run-hash identity, and the table's run-selection rule.

These cover the three pieces the certificate's statistical claims rest on: the McNemar
test itself, the fact that a checkpoint written by one mode is findable by another, and
the rule that decides which stored runs are allowed to be averaged into a table row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from simcert.audit.metrics import mcnemar
from simcert.io_results import run_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from make_tables import select_runs  # noqa: E402


def test_mcnemar_identical_predictions_is_maximally_nonsignificant():
    """Zero discordant pairs is the strongest simulability signal, not a missing one."""
    y = np.array([0, 1, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])
    r = mcnemar(y, pred, pred.copy())
    assert r["b"] == 0 and r["c"] == 0
    assert r["n_discordant"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_counts_discordant_pairs_in_the_right_direction():
    y = np.array([1, 1, 1, 1])
    full = np.array([1, 1, 1, 1])       # all correct
    trunc = np.array([1, 1, 0, 0])      # two that the surrogate gets wrong
    r = mcnemar(y, full, trunc)
    assert r["b"] == 2  # full correct, truncated wrong
    assert r["c"] == 0
    assert r["n_discordant"] == 2
    # exact binomial, two-sided, k=0, n=2 -> 2 * 0.25
    assert r["p_value"] == 0.5


def test_mcnemar_flags_a_clearly_worse_surrogate():
    n = 30
    y = np.ones(n, dtype=int)
    full = np.ones(n, dtype=int)
    trunc = np.ones(n, dtype=int)
    trunc[:12] = 0  # surrogate breaks 12 predictions, fixes none
    r = mcnemar(y, full, trunc)
    assert r["b"] == 12 and r["c"] == 0
    assert r["p_value"] < 0.001


def test_mcnemar_is_symmetric_under_swapping_the_classifiers():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    a = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    b = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    r1, r2 = mcnemar(y, a, b), mcnemar(y, b, a)
    assert r1["p_value"] == r2["p_value"]
    assert (r1["b"], r1["c"]) == (r2["c"], r2["b"])


def test_run_hash_ignores_mode_so_audit_finds_the_train_checkpoint():
    base = {"model_name": "vqc_text", "dataset_name": "mc", "seed": 1}
    hashes = {m: run_hash({**base, "mode": m}) for m in ("train", "audit", "both")}
    assert len(set(hashes.values())) == 1


def test_run_hash_still_separates_genuinely_different_configs():
    a = run_hash({"model_name": "vqc_text", "seed": 1, "mode": "both"})
    b = run_hash({"model_name": "vqc_text", "seed": 2, "mode": "both"})
    assert a != b


def _write_run(tmp_path, name, *, model, dataset, seed, n_qubits, audit_name, acc):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({
        "run_hash": name,
        "config": {"seed": seed, "audit_name": audit_name, "model": {"n_qubits": n_qubits}},
        "certificate": {"model": model, "dataset": dataset, "full_accuracy": acc},
    }))
    return p


def test_select_runs_drops_scaling_sweeps_and_dedupes_seeds(tmp_path):
    """The scaling sweep varies n at a fixed seed; averaging it as seeds is wrong."""
    files = [
        _write_run(tmp_path, "old_s1", model="vqc", dataset="sst2", seed=1,
                   n_qubits=6, audit_name="default", acc=0.10),
        _write_run(tmp_path, "scal_n4", model="vqc", dataset="sst2", seed=1,
                   n_qubits=4, audit_name="scaling", acc=0.90),
        _write_run(tmp_path, "scal_n10", model="vqc", dataset="sst2", seed=1,
                   n_qubits=10, audit_name="scaling", acc=0.90),
        _write_run(tmp_path, "new_s1", model="vqc", dataset="sst2", seed=1,
                   n_qubits=6, audit_name="default", acc=0.70),
        _write_run(tmp_path, "new_s2", model="vqc", dataset="sst2", seed=2,
                   n_qubits=6, audit_name="default", acc=0.80),
    ]
    for i, f in enumerate(files):  # force a strictly increasing mtime order
        import os
        os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))

    got = select_runs([str(f) for f in files])
    certs = got[("vqc", "sst2")]
    assert len(certs) == 2, "one run per seed, scaling sweeps excluded"
    accs = sorted(c["full_accuracy"] for c in certs)
    assert accs == [0.70, 0.80], "the newer run must supersede the older one"


def test_select_runs_pins_to_the_current_audit_config(tmp_path):
    """A row must not average runs taken under two different audit configs."""
    import os
    files = [
        _write_run(tmp_path, "a_s1", model="dc", dataset="mc", seed=1,
                   n_qubits=None, audit_name="default", acc=0.30),
        _write_run(tmp_path, "b_s1", model="dc", dataset="mc", seed=1,
                   n_qubits=None, audit_name="discocat", acc=0.95),
        _write_run(tmp_path, "b_s2", model="dc", dataset="mc", seed=2,
                   n_qubits=None, audit_name="discocat", acc=0.85),
    ]
    for i, f in enumerate(files):
        os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))
    certs = select_runs([str(f) for f in files])[("dc", "mc")]
    assert sorted(c["full_accuracy"] for c in certs) == [0.85, 0.95]


def test_select_runs_pins_to_the_current_qubit_count(tmp_path):
    """A row must not mix an old n=4 configuration with the current n=6 one."""
    import os
    files = [
        _write_run(tmp_path, "n4_s1", model="vqc", dataset="rp", seed=1,
                   n_qubits=4, audit_name="default", acc=0.50),
        _write_run(tmp_path, "n4_s2", model="vqc", dataset="rp", seed=2,
                   n_qubits=4, audit_name="default", acc=0.50),
        _write_run(tmp_path, "n6_s1", model="vqc", dataset="rp", seed=1,
                   n_qubits=6, audit_name="default", acc=0.62),
    ]
    for i, f in enumerate(files):
        os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))

    certs = select_runs([str(f) for f in files])[("vqc", "rp")]
    assert len(certs) == 1
    assert certs[0]["full_accuracy"] == 0.62
