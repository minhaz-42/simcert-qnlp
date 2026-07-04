"""The SimCert audit harness — the paper's core contribution.

Written once, applied uniformly to every model via the OpenQASM circuit IR.

Design note: the *truncation math is pure NumPy* — the chi-bond-dimension MPS
approximation of a trained state is computed by explicit sequential SVD
(`mps_truncation.truncate_state_to_bond_dim`). PennyLane is used only to extract
the exact statevector from a circuit; `default.tensor` (quimb MPS) is an optional
scalability cross-check, not a correctness dependency. This keeps every reported
number provably correct and unit-testable on circuits with known answers.
"""

from .certificate import SimCert, chi_star, classify_verdict
from .entanglement import (
    bipartite_entropy,
    max_bipartite_entropy,
    required_bond_dim,
    schmidt_values,
)
from .fidelity import prediction_agreement, state_fidelity
from .metrics import accuracy, bootstrap_ci, macro_f1
from .mps_truncation import (
    exact_statevector,
    run_chi_sweep,
    truncate_state_to_bond_dim,
)
from .observables import apply_pauli, pauli_expval, single_qubit_z, to_pennylane_observable

__all__ = [
    "SimCert",
    "chi_star",
    "classify_verdict",
    "bipartite_entropy",
    "max_bipartite_entropy",
    "required_bond_dim",
    "schmidt_values",
    "prediction_agreement",
    "state_fidelity",
    "accuracy",
    "bootstrap_ci",
    "macro_f1",
    "exact_statevector",
    "run_chi_sweep",
    "truncate_state_to_bond_dim",
    "apply_pauli",
    "pauli_expval",
    "single_qubit_z",
    "to_pennylane_observable",
]
