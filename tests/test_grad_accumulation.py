"""Chunked training must compute the same gradient as full-batch training.

Backprop through a per-token statevector simulator keeps the entire autograd graph
alive until backward(), so a 600-example training set exhausts memory on a laptop
even though every individual circuit is tiny. Chunking bounds that graph, but it is
only legitimate if the accumulated gradient is the one full-batch training would have
produced, so that is what these tests check, parameter by parameter.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from simcert import models  # noqa: F401  (import populates the model registry)
from simcert.data.loaders import build_vocab, load_dataset


def _grads(model_name, chunk, seed=1, n_epochs=1):
    """Train one step with the given chunk setting; return the resulting gradients."""
    from simcert.registry import get_model

    ds = load_dataset("mc", seed=seed, val_frac=0.2, test_frac=0.2)
    vocab = build_vocab(ds.train)
    cfg = OmegaConf.create({
        "n_qubits": 2, "d_enc": 1, "d_qkv": 1, "epochs": n_epochs, "lr": 0.05,
        "lam": 0.0, "gam": 0.0, "seed": seed, "dataset": "mc",
        "published_accuracy": None, "baseline": None, "train_chunk": chunk,
    })
    m = get_model(model_name)()
    m.build(cfg, vocab)

    torch = m._torch
    train = ds.train
    n = len(train)
    opt = torch.optim.Adam(m._params(), lr=0.05)
    opt.zero_grad()
    if not chunk or chunk >= n:
        logits = torch.stack([m._forward_logit(ex) for ex in train])
        probs = torch.sigmoid(logits)
        y = torch.tensor([float(ex.label) for ex in train], dtype=probs.dtype)
        ((probs - y) ** 2).mean().backward()
    else:
        for i in range(0, n, chunk):
            part = train[i:i + chunk]
            logits = torch.stack([m._forward_logit(ex) for ex in part])
            probs = torch.sigmoid(logits)
            y = torch.tensor([float(ex.label) for ex in part], dtype=probs.dtype)
            (((probs - y) ** 2).sum() / n).backward()
    return [p.grad.detach().numpy().copy() for p in m._params()], n


def test_chunked_gradient_matches_full_batch_qsann():
    full, n = _grads("qsann", chunk=None)
    chunked, _ = _grads("qsann", chunk=7)  # deliberately not a divisor of n
    assert len(full) == len(chunked)
    for a, b in zip(full, chunked):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_chunk_size_does_not_change_the_gradient():
    """Any chunking of the same data must land on the same accumulated gradient."""
    g1, n = _grads("qsann", chunk=3)
    g2, _ = _grads("qsann", chunk=16)
    for a, b in zip(g1, g2):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_chunk_larger_than_dataset_is_the_full_batch_path():
    full, n = _grads("qsann", chunk=None)
    big, _ = _grads("qsann", chunk=10_000)
    for a, b in zip(full, big):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def _claqs_grads(chunk, seed=1):
    """Same check for claqs, whose per-example graph is ~20x larger than qsann's."""
    from simcert.registry import get_model

    ds = load_dataset("mc", seed=seed, val_frac=0.2, test_frac=0.2)
    vocab = build_vocab(ds.train)
    cfg = OmegaConf.load("configs/model/claqs.yaml")
    cfg.seed, cfg.dataset, cfg.train_chunk = seed, "mc", chunk
    m = get_model("claqs")()
    m.build(cfg, vocab)

    torch = m._torch
    train = ds.train[:12]  # a slice keeps this test quick; claqs is ~0.3 s an example
    n = len(train)
    torch.optim.AdamW(m._params(), lr=0.05).zero_grad()
    if not chunk or chunk >= n:
        logits = torch.stack([m._forward_logits(ex) for ex in train])
        y = torch.tensor([ex.label for ex in train])
        torch.nn.CrossEntropyLoss()(logits, y).backward()
    else:
        lf = torch.nn.CrossEntropyLoss(reduction="sum")
        for i in range(0, n, chunk):
            part = train[i:i + chunk]
            logits = torch.stack([m._forward_logits(ex) for ex in part])
            y = torch.tensor([ex.label for ex in part])
            (lf(logits, y) / n).backward()
    return [p.grad.detach().numpy().copy() for p in m._params() if p.grad is not None]


def test_chunked_gradient_matches_full_batch_claqs():
    """chunk=1 is what claqs/sst2 actually runs, so it is the case that must match."""
    full = _claqs_grads(chunk=None)
    one = _claqs_grads(chunk=1)
    assert len(full) == len(one) and len(full) > 0
    for a, b in zip(full, one):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
