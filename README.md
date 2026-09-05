# SimCert — A Classical-Simulability Audit of Trained QNLP Text Models

> **Research question:** when a quantum-NLP model reports an accuracy gain on text, was the
> *quantum* actually load-bearing — or does the trained circuit live in the efficiently
> classically-simulable regime the whole time?

Recent QNLP text models (DisCoCat/lambeq, quantum self-attention **QSANN** `arXiv:2205.05625`
and **QMSAN** `arXiv:2403.02871`, all-quantum token mixers **CLAQS** `arXiv:2510.06532`, and
hybrid quantum-BERT) report gains on small text datasets. This project retrains them on a
shared suite and re-simulates each **trained** circuit under **MPS bond-dimension truncation**
(χ = 1, 2, 4, …) plus an **entanglement-removal ablation** and further classical-equivalence
witnesses, reporting how much accuracy survives inside the classically-simulable regime versus
parameter/FLOP-matched classical baselines. The deliverable is a per-model **simulability
certificate**.

The audit itself has been done for tabular QML and vision QCNNs (Bowles/Ahmed/Schuld
`arXiv:2403.07059`; Cerezo et al. `arXiv:2312.09121`; Bermejo et al. `arXiv:2408.12739`) and the
theory is in place (Shin/Teo/Jeong, *Phys. Rev. Research* 6, 023218 — every VQML function is a
constrained-coefficient MPS). It has **never** been run on text/DisCoCat/quantum-attention
circuits. Venue: a non-archival abstract at **QTML 2026** (submitted) and the full archival paper to
**TMLR** (rolling, OpenReview, double-blind).

## Architecture: producer → IR → consumer

`lambeq` pins `pennylane<0.37`, but the MPS-truncation device that *is* the audit's core knob
(`default.tensor(method="mps", max_bond_dim=χ)`) needs modern PennyLane. They cannot coexist, so:

- **Producers** (per model, possibly in different envs) train and export each *trained,
  parameter-bound* circuit to **OpenQASM** + a params/observable spec.
- **One shared audit consumer** re-simulates every model identically from that IR.

This makes the audit genuinely model-agnostic — the χ-sweep / entanglement / fidelity probes are
written **once** in `src/simcert/audit/` and applied uniformly.

## Environments

Two isolated conda envs (never touch your existing envs):

| Env | Python | Role | Create |
|-----|--------|------|--------|
| `qnlp`        | 3.12 | audit harness + QSANN/QMSAN/CLAQS/QBERT | `make env-audit` |
| `qnlp-lambeq` | 3.11 | lambeq DisCoCat producer only          | `make env-lambeq` |

```bash
make env-audit          # create the audit env
make install-dev        # pip install -e . into it
make test               # prove the audit harness on known circuits (GHZ, product state, ...)
```

## Layout

```
src/simcert/audit/      # THE contribution: mps_truncation, entanglement, fidelity, certificate, ...
src/simcert/models/     # pluggable zoo: each model implements export_circuits() -> OpenQASM IR
src/simcert/circuits/   # BoundCircuit IR + QASM round-trip
src/simcert/runner.py   # Hydra entrypoint (mode=train|audit), deterministic run-hash keying
configs/                # Hydra config groups: model/ dataset/ audit/ experiment/
results/                # COMMITTED: metrics/*.json, certificates/*.json, circuits/*.qasm
figures/                # COMMITTED PDFs + scripts that regenerate them from results/
paper/                  # TMLR LaTeX sources (tmlr.sty) + QTML abstract
```

## Status

**Grid complete.** All five models are built and audited: 129/129 certificate runs on disk,
20 seeds on the RP reproduction probe for `discocat`/`qsann`/`qmsan`/`vqc_text`, and a
chi*-vs-n scaling sweep to 16 qubits for two architectures. `make reproduce` regenerates
every figure, both tables, and both PDFs from committed `results/`.

```bash
python scripts/run_grid.py --dry-run   # should report "0 to run"
make reproduce                         # figures + tables + main.pdf and main_preprint.pdf
```

Open, and not compute-bound: the positive-control design study is paused
(`docs/positive-control-study.md`). The chi*-vs-entanglement anti-correlation it turned up
has been checked and does **not** affect the audited models -- both adversarial lenses
rejected the candidate control design, so any resumption should start from the k-of-m
entanglement dial described there. See the full plan for the certificate spec, experiment
matrix, statistics, and milestones.

## Compute note

Developed on Apple Silicon laptops with no CUDA, an M1 with 8 GB and an M5 with 16 GB. Statevector simulation is CPU-only and small
(≤16-qubit state ≈ 1 MB). Heavy training (hybrid quantum-BERT, CLAQS) runs on remote GPU and
syncs exported circuits + result stubs back for local auditing.
