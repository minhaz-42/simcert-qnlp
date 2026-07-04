"""Certificate scoring: chi* selection and ordinal verdict logic."""

import numpy as np

from simcert.audit.certificate import SimCert, chi_star, classify_verdict
from simcert.audit.metrics import accuracy, bootstrap_ci


def test_chi_star_picks_smallest_within_tolerance():
    acc = {1: 0.60, 2: 0.74, 4: 0.805, 8: 0.809, "full": 0.812}
    full = 0.812
    # tau = 0.01 -> first chi within 0.01 of full is chi=4 (0.805)
    assert chi_star(acc, full, tau=0.01) == 4
    # loose tau -> chi=2 qualifies (0.812-0.74=0.072)
    assert chi_star(acc, full, tau=0.08) == 2
    # strict tau -> none of the finite chis qualify -> needs full
    assert chi_star(acc, full, tau=0.001) is None


def test_verdict_classically_simulable():
    v = classify_verdict(chi_star_gen=2, delta_ent=0.005, baseline_gap=0.01, scaling_slope=0.0)
    assert v == "CLASSICALLY_SIMULABLE"


def test_verdict_quantum_resourceful():
    # never reaches full accuracy at finite chi + entanglement clearly matters
    v = classify_verdict(chi_star_gen=None, delta_ent=0.20, baseline_gap=0.15)
    assert v == "QUANTUM_RESOURCEFUL"


def test_verdict_ambiguous():
    # small chi* but classical baseline is far behind -> mixed signal
    v = classify_verdict(chi_star_gen=2, delta_ent=0.005, baseline_gap=0.20)
    assert v == "AMBIGUOUS"


def test_simcert_finalize_and_serialize():
    cert = SimCert(
        model="toy",
        dataset="mc",
        full_accuracy=0.90,
        accuracy_by_chi={1: 0.55, 2: 0.895, "full": 0.90},
        delta_ent=0.004,
        baseline_gap=0.01,
        scaling_slope=0.0,
    )
    cert.chi_star["tau_gen"] = chi_star(cert.accuracy_by_chi, cert.full_accuracy, tau=0.02)
    assert cert.chi_star["tau_gen"] == 2
    assert cert.finalize_verdict() == "CLASSICALLY_SIMULABLE"
    d = cert.to_dict()
    assert d["verdict"] == "CLASSICALLY_SIMULABLE" and d["model"] == "toy"


def test_bootstrap_ci_brackets_point_estimate():
    y_true = np.array([0, 1] * 50)
    y_pred = y_true.copy()
    y_pred[:5] = 1 - y_pred[:5]  # 5 wrong out of 100
    point, lo, hi = bootstrap_ci(y_true, y_pred, metric=accuracy, n_boot=500, seed=0)
    assert np.isclose(point, 0.95)
    assert lo <= point <= hi
