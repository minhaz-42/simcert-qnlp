# Reference papers we follow

PDFs are in `references/pdfs/` (kept local, git-ignored to avoid repo bloat). All
verified against arXiv (titles/authors confirmed from the PDF first pages).

## The models we reimplement + audit (the "zoo")
| # | arXiv | Paper | Notes |
|---|-------|-------|-------|
| 1 | [2205.05625](https://arxiv.org/abs/2205.05625) | **QSANN** — Quantum Self-Attention Neural Networks for Text Classification (Li, Zhao, Wang) | Gaussian-projected quantum self-attention; pure-state feature map |
| 2 | [2403.02871](https://arxiv.org/abs/2403.02871) | **QMSAN** — Quantum Mixed-State Self-Attention Network (Chen et al.) | Mixed-state attention + quantum positional encoding → uses the purification/MPDO audit path |
| 3 | [2510.06532](https://arxiv.org/abs/2510.06532) | **CLAQS** — Compact Learnable All-Quantum Token Mixer (Chen et al.) | 8 data qubits, 91.64% SST-2 / 87.08% IMDB; LCU/QSVT-style mixing |
| 4 | [2102.12846](https://arxiv.org/abs/2102.12846) | **DisCoCat/lambeq** — QNLP in Practice (Lorenz et al.) | Compositional circuits; producer in the `qnlp-lambeq` env |

## The audit methodology we follow (the "how")
| arXiv | Paper | Role |
|-------|-------|------|
| [2403.07059](https://arxiv.org/abs/2403.07059) | **Bowles, Ahmed, Schuld** — Better than classical? The subtle art of benchmarking QML | ★ **Primary methodological template**: matched baselines + entanglement-removal ablation |
| [2408.12739](https://arxiv.org/abs/2408.12739) | **Bermejo et al.** — QCNNs are (Effectively) Classically Simulable | Closest sibling: audits *trained* circuits for simulability |
| [2312.09121](https://arxiv.org/abs/2312.09121) | **Cerezo et al.** — No barren plateaus ⇒ classically simulable | Theoretical prior |
| [2307.06937](https://arxiv.org/abs/2307.06937) | **Shin, Teo, Jeong** — Dequantizing QML using tensor networks | The MPS-bond-dimension resource axis (Axis A) |
| [2011.01938](https://arxiv.org/abs/2011.01938) | **Huang et al.** — Power of data in QML | Geometric difference g_CQ (Axis D) |

**Start here:** Bowles 2403.07059 (the audit template) + QSANN 2205.05625 (first model to reimplement).
