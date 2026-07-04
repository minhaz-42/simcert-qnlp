"""End-to-end smoke test: train a tiny model -> export -> audit -> certificate."""

from types import SimpleNamespace

import pytest

import simcert.models  # noqa: F401  (populate registry)
from simcert.audit.pipeline import audit_model
from simcert.data.loaders import build_vocab, load_dataset
from simcert.registry import get_model


@pytest.mark.slow
def test_end_to_end_pipeline_smoke():
    ds = load_dataset("mc", seed=0, n_per_class=10)
    vocab = build_vocab(ds.train)
    cfg = SimpleNamespace(n_qubits=3, n_layers=1, epochs=2, lr=0.1, seed=0, published_accuracy=None)

    model = get_model("vqc_text")()
    model.build(cfg, vocab)
    report = model.fit(ds.train, ds.val, cfg)
    assert 0.0 <= report.train_accuracy <= 1.0
    assert report.n_params > 0

    cert, details = audit_model(
        model, ds.test, chi_values=[1, 2, None], dataset_name="mc",
        train_accuracy=report.train_accuracy,
    )
    assert cert.verdict in {"CLASSICALLY_SIMULABLE", "AMBIGUOUS", "QUANTUM_RESOURCEFUL"}
    assert set(details["accuracy_by_chi"]) >= {1, 2, "full"}
    assert 0.0 <= cert.full_accuracy <= 1.0
    # chi=full must retain full accuracy exactly (it is the untruncated model)
    assert details["accuracy_by_chi"]["full"] == cert.full_accuracy
