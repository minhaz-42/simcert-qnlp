"""The per-model simulability certificate — the paper's deliverable object.

Turns a chi-vs-accuracy curve plus the auxiliary witnesses into a compact record
and an ordinal verdict:

  CLASSICALLY_SIMULABLE : chi* small & non-growing, entanglement not load-bearing,
                          classical baseline competitive  -> quantum was NOT load-bearing
  QUANTUM_RESOURCEFUL   : accuracy collapses below full chi, chi* large/growing,
                          entanglement ablation hurts
  AMBIGUOUS             : mixed signals
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SMALL_CHI = 4  # "chi* small" threshold (config-overridable)


def chi_star(accuracy_by_chi: dict, full_accuracy: float, tau: float) -> int | None:
    """Smallest finite chi whose accuracy is within ``tau`` of the full-chi accuracy.

    ``accuracy_by_chi`` maps int chi (and optionally ``"full"``) -> accuracy.
    Returns ``None`` if no finite chi reaches the tolerance (i.e. full chi needed).
    """
    finite = sorted(k for k in accuracy_by_chi if isinstance(k, int))
    for c in finite:
        if full_accuracy - accuracy_by_chi[c] <= tau + 1e-12:
            return c
    return None


def classify_verdict(
    chi_star_gen: int | None,
    delta_ent: float | None,
    baseline_gap: float | None,
    scaling_slope: float | None = None,
    small_chi: int = SMALL_CHI,
    ent_tol: float = 0.02,
    gap_tol: float = 0.02,
    slope_tol: float = 0.05,
) -> str:
    """Ordinal verdict from the certificate axes. Thresholds are conservative defaults."""
    simulable_chi = chi_star_gen is not None and chi_star_gen <= small_chi
    ent_negligible = delta_ent is None or abs(delta_ent) <= ent_tol
    classical_ok = baseline_gap is None or baseline_gap <= gap_tol
    flat_scaling = scaling_slope is None or scaling_slope <= slope_tol

    if simulable_chi and ent_negligible and classical_ok and flat_scaling:
        return "CLASSICALLY_SIMULABLE"

    strongly_quantum = (chi_star_gen is None) or (
        chi_star_gen > small_chi and (delta_ent or 0.0) > 0.05
    )
    if strongly_quantum and not simulable_chi:
        return "QUANTUM_RESOURCEFUL"
    return "AMBIGUOUS"


@dataclass
class SimCert:
    """A per-(model, dataset) simulability certificate."""

    model: str
    dataset: str
    full_accuracy: float
    accuracy_by_chi: dict = field(default_factory=dict)  # {int chi | "full": accuracy}
    accuracy_ci_by_chi: dict = field(default_factory=dict)  # {int chi | "full": [lo, hi]} 95% bootstrap
    mcnemar_p_by_chi: dict = field(default_factory=dict)  # {int chi: p} full vs truncated (exact McNemar)
    fidelity_by_chi: dict = field(default_factory=dict)  # {int chi | "full": mean fidelity}
    agreement_by_chi: dict = field(default_factory=dict)  # {int chi | "full": pi(chi)}
    chi_star: dict = field(default_factory=dict)  # {"tau_gen"|"tau_ci"|"tau_agree": chi|None}
    scaling_slope: float | None = None  # chi* vs sequence length / n_qubits
    scaling_r2: float | None = None
    delta_ent: float | None = None  # A_full - A_separable (entanglement-removal ablation)
    entropy_mean: float | None = None
    entropy_p99: float | None = None
    gcq_ratio: float | None = None  # g_CQ / sqrt(N)   (feature-map models)
    fourier_degree: int | None = None
    baseline_gap: float | None = None  # A_full - A_best_classical
    mps_surrogate_repro: dict = field(default_factory=dict)  # {chi: reproduction accuracy}
    verdict: str = "AMBIGUOUS"
    meta: dict = field(default_factory=dict)  # seeds, lib versions, git sha, config hash

    def finalize_verdict(self, **kwargs) -> str:
        self.verdict = classify_verdict(
            self.chi_star.get("tau_gen"),
            self.delta_ent,
            self.baseline_gap,
            self.scaling_slope,
            **kwargs,
        )
        return self.verdict

    def to_dict(self) -> dict:
        return asdict(self)
