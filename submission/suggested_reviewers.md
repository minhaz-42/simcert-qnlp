# Suggested reviewers — Quantum Machine Intelligence

Chosen because each has published the specific result this paper builds on, so each can
judge whether the bond-dimension criterion is the right instrument and whether it was
applied correctly.

## 1. Maria Schuld — Xanadu Quantum Technologies
The closest methodological relative. Co-author of "Better than classical? The subtle art of
benchmarking quantum machine learning models" (arXiv:2403.07059), whose entanglement-removal
ablation this paper adopts and whose discipline of scoping every claim to the tested regime
it follows. Also authored the kernel and Fourier-spectrum readings of variational models
that supply two of this paper's corroborating witnesses.

## 2. M. Cerezo — Los Alamos National Laboratory
Author of "Does provable absence of barren plateaus imply classical simulability?" (Nature
Communications, 2025), the theoretical claim this paper turns into an operational
measurement. Best placed to say whether the chi* criterion is a fair instrument for that
question.

## 3. Jens Eisert — Freie Universität Berlin
Co-author of "Classical surrogates for quantum learning models" (Physical Review Letters
131:100803, 2023). This paper's surrogate is of the same species, specialised to a
bond-dimension-limited MPS, so he can assess whether that specialisation is sound.

## 4. Hyunseok Jeong — Seoul National University
Co-author of "Dequantizing quantum machine learning models using tensor networks" (Physical
Review Research 6:023218, 2024), the direct theoretical parent of the bond-dimension
criterion used here.

## 5. Pablo Bermejo — (affiliation per arXiv:2408.12739)
Lead author of "Quantum convolutional neural networks are (effectively) classically
simulable", the vision-domain analogue of the question asked here for language. Well placed
to compare the two audits' methodologies.

---

## Reviewers to exclude, and why

The authors of the four audited architectures should not review this manuscript. The paper
evaluates their published models and reports that a classical surrogate reproduces those
models' predictions, which is a direct conflict of interest. This covers the authors of the
DisCoCat text-classification pipeline (Lorenz et al.), the quantum self-attention networks
QSANN and QMSAN (Li et al.; Chen et al.), and the all-quantum token mixer CLAQS (Chen et
al., 2025).

This is stated not to shield the work from criticism but because their objections are better
raised in public after publication than as anonymous gatekeeping.
