# SimCert — model-zoo implementation plan

Distilled from faithful deep-reads of the four model papers (see `references/`).
Full extracted specs live in the workflow transcript; this is the actionable roadmap.

## Build order (by new-capability risk, not prominence)

**DisCoCat → QSANN → QMSAN → CLAQS.** Each step reuses the previous machinery and adds
exactly one new capability, so the biggest unbuilt investment lands last.

We build **QSANN first in practice** (it stays in the existing `qnlp` env, needs no new
conda env) to stand up the per-token + composition machinery on the pure-state path;
DisCoCat slots in once the `qnlp-lambeq` env is built (it is the native one-circuit case
and ships official data/code at `github.com/CQCL/qnlp_lorenz_etal_2021_resources`).

## ⚠️ The scale caveat that shapes the science

An exact MPS of an `n`-qubit state needs bond dimension only `χ = 2^⌊n/2⌋`:

| model | qubits | exact χ | χ-truncation discriminating? |
|-------|--------|---------|------------------------------|
| DisCoCat | ~3–5 (1 open wire) | ≤4 | no — trivially simulable |
| QSANN | 2 (MC) / 4 | 2–4 | no |
| QMSAN | 2 (MC) / 4 | 2–4 | no |
| CLAQS | **8** | **16** | **yes — first real regime** |

So for the small models the χ-sweep *certifies* "trivially classically simulable" — a real
finding. The discriminating axes there are **entanglement-removal ΔA_ent, entropy S, g_CQ,
Fourier degree**, plus a **χ\*-vs-n scaling study** (scale n up on a fixed task and watch
whether the required χ grows). CLAQS is where χ-truncation itself becomes informative.

## The "auditable object" generalization (new harness capability)

Our audit assumed one circuit → one state → one decision. That holds only for DisCoCat.
General rule: an **example** decomposes into **atomic auditable units** (per-token / per-window
/ per-pair circuits) plus a **composition function** (the classical head or aggregator) that
maps the units' (possibly truncated) expectation values to one label.

| model | atomic unit | composition | path |
|-------|-------------|-------------|------|
| DisCoCat | 1 whole-sentence circuit | identity | pure |
| QSANN | per-token q/k/v circuits (3·S) | Gaussian attention → residual → mean → sigmoid | pure |
| QMSAN | per-token q/k/v (+ S² SWAP tests) | tr(ρσ) attention → residual → mean → sigmoid | **mixed / purification** |
| CLAQS | per-window 8-qubit state | MLP + sliding-window doc aggregation | pure |

Implemented as `QNLPModel.audit_units(example)` + `QNLPModel.compose(example, unit_expvals)`;
single-circuit models are the degenerate one-unit case (`compose = decision_from_expvals`).

## Per-model cards (reproduction targets)

- **QSANN** (`2205.05625`): GPQSA. Per word: `H^⊗n` → encoder template (Rx,Ry, then D_enc×[CNOT
  chain, Ry]) with `d=n(D_enc+2)` data angles; trainable U_q/U_k/U_v same template; q,k→⟨Z_0⟩,
  v→d Pauli expvals `{Z_i,X_i,Y_i}`. Attention `α=exp(-(⟨Z_q⟩-⟨Z_k⟩)²)`, row-normalize, residual
  `y_s=x_s+Σα̃·o_j`, mean-pool, sigmoid. Adam, MSE+λ,γ reg, init N(0,0.01), 9 seeds. **Targets:
  MC 100%, Yelp 84.8%, IMDb 80.3%, Amazon 84.3%; RP 67.7% (fragile).** Risk: word→x_s embedding
  unspecified (use trainable dim-d embedding, document).
- **QMSAN** (`2403.02871`): mixed-state attention `α=tr(ρ_q σ_k)` via SWAP test; sinusoidal
  quantum positional encoding via fixed Rx; IsingZZ entanglers (R/CB/AA topologies). Forces the
  **mixed-state/purification** truncation path (partial-trace last n/2 qubits). **Targets: MC
  100%, RP 75.6% (beats QSANN — headline), sentiment ~85–87%.** Init discrepancy N(0,0.1) vs
  N(0,0.01): follow Algorithm 1 (0.01).
- **CLAQS** (`2510.06532`): 8 data qubits, ansatz-14 embedding, learnable **complex L1-normalized
  LCU** mixer + **QSVT** polynomial `P_c(M)=Σc_k M^k` (deg 5) + window FFN; XYZ readout → MLP;
  sliding windows (128 SST-2 / 256 IMDB). **Targets: SST-2 91.64%, IMDB 87.08%.** Risk: wall of
  undocumented hyperparameters + no code — pin to the 326/454 attention-param budget.
- **DisCoCat** (`2102.12846`): pregroup parse → one sentence circuit; IQP ansatz for multi-qubit
  words, Euler/Rx for single-qubit, `that`=GHZ, cups=Bell post-selection. SPSA + cross-entropy
  (sim-only: Adam with same loss). **Targets (noise-free sim): MC ~79.8%, RP ~72.3%.** Official
  code + data exist. Needs the `qnlp-lambeq` producer env.
