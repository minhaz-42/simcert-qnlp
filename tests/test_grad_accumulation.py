"""Chunked training must be indistinguishable from full-batch training.

Backprop runs through a statevector simulator, so the autograd graph spans the whole
training set and its cost grows with 2^n. Full-batch training exhausted a 16 GB machine
on sst2 and at n>=12, so fit() accepts a train_chunk and accumulates gradients over
chunks instead. That is only legitimate if the resulting update is the one full-batch
training would have produced, which is what these tests check, on the real fit() path
rather than on a reimplementation of each model's loss.

The last test exists because of a specific bug: train_chunk was added to qsann and claqs
but not to vqc_text, so passing model.train_chunk on the command line was accepted by the
config and silently ignored by the model, and two large runs died on memory that the
option was supposed to bound. A silently ignored knob is worse than a missing one.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from omegaconf import OmegaConf

from simcert import models  # noqa: F401  (import populates the model registry)
from simcert.data.loaders import build_vocab, load_dataset
from simcert.registry import _REGISTRY, get_model

# (model name, dataset, extra config) for every model whose fit() we can drive cheaply.
CHUNKABLE = [
    ("vqc_text", "mc", {}),
    ("qsann", "mc", {}),
    ("qmsan", "mc", {}),
    ("claqs", "mc", {}),
]


def _fit_params(model_name, dataset, chunk, extra, n_train=10, epochs=1, seed=1):
    """Fit for one step with the given chunk setting; return the resulting parameters."""
    ds = load_dataset(dataset, seed=seed, val_frac=0.2, test_frac=0.2)
    vocab = build_vocab(ds.train)
    cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
    cfg.seed, cfg.dataset, cfg.epochs = seed, dataset, epochs
    cfg.train_chunk = chunk
    for k, v in extra.items():
        cfg[k] = v
    m = get_model(model_name)()
    m.build(cfg, vocab)
    m.fit(ds.train[:n_train], ds.val[:2], cfg)
    return [p.detach().numpy().copy() for p in m._params()] if hasattr(m, "_params") \
        else [m.embedding.detach().numpy().copy(), m.theta.detach().numpy().copy()]


@pytest.mark.parametrize("model_name,dataset,extra", CHUNKABLE)
def test_chunked_fit_matches_full_batch(model_name, dataset, extra):
    """One optimizer step must land on the same parameters either way."""
    full = _fit_params(model_name, dataset, None, extra)
    # 3 deliberately does not divide 10, so the last chunk is short
    chunked = _fit_params(model_name, dataset, 3, extra)
    assert len(full) == len(chunked) and len(full) > 0
    for a, b in zip(full, chunked):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("model_name,dataset,extra", CHUNKABLE)
def test_chunk_size_one_matches_full_batch(model_name, dataset, extra):
    """chunk=1 is the setting the heaviest runs use, so it gets its own check."""
    full = _fit_params(model_name, dataset, None, extra)
    one = _fit_params(model_name, dataset, 1, extra)
    for a, b in zip(full, one):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def test_chunk_larger_than_the_training_set_is_the_full_batch_path():
    full = _fit_params("vqc_text", "mc", None, {})
    big = _fit_params("vqc_text", "mc", 10_000, {})
    for a, b in zip(full, big):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


# Models that legitimately do not need train_chunk, with the reason. Listing them
# explicitly rather than sniffing for a stub means adding a new model forces a
# deliberate choice instead of quietly inheriting an exemption.
CHUNK_EXEMPT = {
    "discocat": "does not train here; the lambeq producer trains it in the other env "
                "and fit() only reports the producer's numbers",
    "logreg_bow": "classical sklearn baseline, no autograd graph and no statevector",
}


def test_no_model_silently_ignores_train_chunk():
    """Every trainable model must actually read train_chunk in fit().

    Accepting the option in config and then ignoring it is the failure this guards, and
    it is not hypothetical: vqc_text accepted model.train_chunk and ignored it, so two
    n>=12 runs were launched believing their memory was bounded, blew past the cap, and
    were killed with nothing to indicate the knob had done nothing.
    """
    ignored = []
    for name, cls in sorted(_REGISTRY.items()):
        if name in CHUNK_EXEMPT:
            continue
        fit = getattr(cls, "fit", None)
        if fit is None:
            continue
        try:
            src = inspect.getsource(fit)
        except (OSError, TypeError):
            continue
        if "train_chunk" not in src:
            ignored.append(name)
    assert not ignored, (
        f"these models accept train_chunk in config but never read it in fit(): {ignored}. "
        "Either honour it, or add it to CHUNK_EXEMPT with a reason."
    )


def test_chunk_exempt_list_stays_honest():
    """An exemption must name a model that still exists, or it is silently protecting nothing."""
    stale = [n for n in CHUNK_EXEMPT if n not in _REGISTRY]
    assert not stale, f"CHUNK_EXEMPT names models that are no longer registered: {stale}"
