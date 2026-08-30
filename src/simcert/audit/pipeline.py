"""Run the full simulability audit for a trained model on a dataset -> SimCert.

For every example: export the trained circuit to the QASM IR, load it back, run the
bond-dimension sweep, and map each truncated readout to a label with the model's own
decision rule. Aggregate into accuracy-retained / prediction-agreement / fidelity
curves and the entanglement statistics, then score the certificate.
"""

from __future__ import annotations

import numpy as np

from .certificate import SimCert, chi_star
from .entanglement import max_bipartite_entropy
from .fidelity import prediction_agreement
from .metrics import accuracy, bootstrap_ci, mcnemar
from .mps_truncation import _chi_key, run_chi_sweep


def audit_model(
    model,
    examples,
    chi_values,
    dataset_name: str,
    cutoff: float = 1e-12,
    train_accuracy: float | None = None,
    baseline_preds=None,
    delta_ent: float | None = None,
    seed: int = 0,
    meta: dict | None = None,
) -> tuple[SimCert, dict]:
    """Audit ``model`` on ``examples``. Returns ``(certificate, details_for_figures)``."""
    labels = np.array([ex.label for ex in examples])

    keys = [_chi_key(c) for c in chi_values]
    preds_by_key: dict = {k: [] for k in keys}
    fid_by_key: dict = {k: [] for k in keys}
    full_preds: list[int] = []
    entropies: list[float] = []

    for ex in examples:
        # Each example decomposes into atomic auditable units; truncate every unit, then
        # let the model's composition map the (truncated) per-unit info to a label.
        units = model.audit_units(ex)
        sweeps = [
            run_chi_sweep(
                None if u.state is not None else u.qfunc(),
                u.n_qubits, u.observables, chi_values, cutoff, full_state=u.state,
            )
            for u in units
        ]
        nqs = [u.n_qubits for u in units]
        full_info = [
            {"expvals": s["full_expvals"], "state": s["full_state"], "n_qubits": nq}
            for s, nq in zip(sweeps, nqs)
        ]
        full_preds.append(model.compose(ex, full_info))
        entropies.append(
            float(np.mean([max_bipartite_entropy(s["full_state"], nq)
                           for s, nq in zip(sweeps, nqs)]))
        )
        for k in keys:
            info = [
                {"expvals": s["sweep"][k]["expvals"], "state": s["sweep"][k]["state"], "n_qubits": nq}
                for s, nq in zip(sweeps, nqs)
            ]
            preds_by_key[k].append(model.compose(ex, info))
            fid_by_key[k].append(float(np.mean([s["sweep"][k]["fidelity"] for s in sweeps])))

    full_preds = np.array(full_preds)
    full_accuracy = accuracy(labels, full_preds)

    accuracy_by_chi: dict = {}
    accuracy_ci_by_chi: dict = {}
    agreement_by_chi: dict = {}
    fidelity_by_chi: dict = {}
    mcnemar_p_by_chi: dict = {}
    for k in keys:
        p = np.array(preds_by_key[k])
        accuracy_by_chi[k] = accuracy(labels, p)
        _, clo, chi_ = bootstrap_ci(labels, p, metric=accuracy, n_boot=1000, seed=seed)
        accuracy_ci_by_chi[k] = [clo, chi_]
        agreement_by_chi[k] = prediction_agreement(full_preds, p)
        fidelity_by_chi[k] = float(np.mean(fid_by_key[k]))
        if isinstance(k, int):  # paired McNemar of the truncated surrogate against the full model
            mcnemar_p_by_chi[k] = mcnemar(labels, full_preds, p)["p_value"]

    # tolerances for chi*
    tau_gen = max(0.0, (train_accuracy - full_accuracy)) if train_accuracy is not None else 0.02
    _, lo, hi = bootstrap_ci(labels, full_preds, metric=accuracy, n_boot=1000, seed=seed)
    tau_ci = max((hi - lo) / 2.0, 1e-9)

    finite = sorted(k for k in accuracy_by_chi if isinstance(k, int))
    chi_star_agree = next((c for c in finite if agreement_by_chi[c] >= 0.95), None)

    cert = SimCert(
        model=getattr(model, "name", model.__class__.__name__),
        dataset=dataset_name,
        full_accuracy=full_accuracy,
        accuracy_by_chi=accuracy_by_chi,
        accuracy_ci_by_chi=accuracy_ci_by_chi,
        mcnemar_p_by_chi=mcnemar_p_by_chi,
        fidelity_by_chi=fidelity_by_chi,
        agreement_by_chi=agreement_by_chi,
        chi_star={
            "tau_gen": chi_star(accuracy_by_chi, full_accuracy, tau_gen),
            "tau_ci": chi_star(accuracy_by_chi, full_accuracy, tau_ci),
            "tau_agree": chi_star_agree,
        },
        delta_ent=delta_ent,
        entropy_mean=float(np.mean(entropies)) if entropies else None,
        entropy_p99=float(np.percentile(entropies, 99)) if entropies else None,
        baseline_gap=(full_accuracy - accuracy(labels, np.array(baseline_preds)))
        if baseline_preds is not None
        else None,
        meta={
            "n_examples": len(examples),
            "chi_values": [c for c in chi_values],
            "tau_gen": tau_gen,
            "tau_ci": tau_ci,
            "train_accuracy": train_accuracy,
            **(meta or {}),
        },
    )
    cert.finalize_verdict()

    details = {
        "chi_values": list(chi_values),
        "accuracy_by_chi": accuracy_by_chi,
        "accuracy_ci_by_chi": accuracy_ci_by_chi,
        "mcnemar_p_by_chi": mcnemar_p_by_chi,
        "fidelity_by_chi": fidelity_by_chi,
        "agreement_by_chi": agreement_by_chi,
        "full_accuracy": full_accuracy,
        "ci": [lo, hi],
    }
    return cert, details
