"""DisCoCat producer -- run in the qnlp-lambeq (Python 3.11) environment.

    conda run -n qnlp-lambeq python -m simcert.producers.discocat_producer --seed 1

Builds DisCoCat-style circuits with lambeq (offline cups_reader, since the Bobcat parser
model host is defunct), trains them with SPSA, then exports each *trained* per-sentence
circuit to gate-only OpenQASM + readout metadata (which qubits are post-selected on |0>,
which wire is the open readout) into results/circuits/discocat/seed<seed>/. The qnlp-env
`discocat` consumer model loads these and runs the simulability audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lambeq import AtomicType, IQPAnsatz, NumpyModel, cups_reader
from lambeq.training import BinaryCrossEntropyLoss, Dataset, QuantumTrainer, SPSAOptimizer
from pytket import Circuit, OpType
from pytket.qasm import circuit_to_qasm_str

from simcert.data.loaders import load_dataset

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "results" / "circuits" / "discocat"
PUBLISHED_MC = 0.798  # Lorenz et al. classical-sim test accuracy on the *real* MC


def _circuits(exs, ansatz):
    cs, ys = [], []
    for ex in exs:
        cs.append(ansatz(cups_reader.sentence2diagram(ex.text)))
        ys.append([1.0, 0.0] if ex.label == 0 else [0.0, 1.0])
    return cs, ys


def _export_records(circuits, exs, symbols, weights):
    recs = []
    for c, ex in zip(circuits, exs):
        bound = c.lambdify(*symbols)(*weights)
        tk = bound.to_tk()
        post = sorted(dict(tk.post_selection))
        open_wire = [q for q in range(tk.n_qubits) if q not in post]
        gates = Circuit(tk.n_qubits)
        for cmd in tk.get_commands():
            if cmd.op.type == OpType.Measure:
                continue
            gates.add_gate(cmd.op.type, cmd.op.params, [q.index[0] for q in cmd.qubits])
        recs.append({
            "text": ex.text, "label": int(ex.label), "n_qubits": tk.n_qubits,
            "post_select": post, "open_wire": open_wire,
            "qasm": circuit_to_qasm_str(gates),
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", default="mc")  # mc | mc_real | rp
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n_per_class", type=int, default=65)
    args = ap.parse_args()

    kw = {"n_per_class": args.n_per_class} if args.dataset == "mc" else {}
    ds = load_dataset(args.dataset, seed=args.seed, **kw)
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=args.n_layers)
    tr_c, tr_y = _circuits(ds.train, ansatz)
    va_c, va_y = _circuits(ds.val, ansatz)
    te_c, te_y = _circuits(ds.test, ansatz)

    model = NumpyModel.from_diagrams(tr_c + va_c + te_c)
    trainer = QuantumTrainer(
        model, loss_function=BinaryCrossEntropyLoss(), epochs=args.epochs,
        optimizer=SPSAOptimizer,
        optim_hyperparams={"a": 0.05, "c": 0.06, "A": 0.01 * args.epochs},
        seed=args.seed, verbose="suppress",
    )
    trainer.fit(Dataset(tr_c, tr_y, batch_size=len(tr_c)), Dataset(va_c, va_y, shuffle=False))

    def acc(cs, ys):
        return float((np.asarray(model(cs)).argmax(1) == np.asarray(ys).argmax(1)).mean())

    symbols = model.symbols
    weights = [float(w) for w in model.weights]
    out = OUT_ROOT / args.dataset / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for split, cs, exs in (("train", tr_c, ds.train), ("val", va_c, ds.val), ("test", te_c, ds.test)):
        (out / f"{split}.json").write_text(json.dumps(_export_records(cs, exs, symbols, weights)))
    manifest = {
        "seed": args.seed, "dataset": args.dataset, "n_layers": args.n_layers, "epochs": args.epochs,
        "n_params": len(weights), "reader": "cups", "ansatz": "IQP",
        "train_acc": acc(tr_c, tr_y), "val_acc": acc(va_c, va_y), "test_acc": acc(te_c, te_y),
        "published_accuracy": PUBLISHED_MC,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"[discocat-producer] seed={args.seed} {manifest}")


if __name__ == "__main__":
    main()
