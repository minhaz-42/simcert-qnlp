# Cover letter — Quantum Machine Intelligence

Dear Editors,

I am submitting "SimCert: A Classical-Simulability Audit of Trained Quantum-NLP Text
Classifiers" for consideration as a research article in *Quantum Machine Intelligence*.

Quantum natural language processing models are routinely promoted for a quantum benefit on
text classification, and that benefit is argued from accuracy. What is rarely tested is
whether a trained QNLP circuit ever leaves the efficiently classically-simulable regime.
This paper tests it directly.

SimCert re-simulates each trained circuit as a bond-dimension-chi matrix product state,
sweeps chi, and reports the smallest value at which a classical surrogate recovers the
model's own predictions. Applied to four published QNLP architectures plus a variational
reference, the answer is chi = 1: a product state. On SST-2, the one corpus here whose test
set can resolve a few accuracy points, three architectures keep their full accuracy under a
product-state surrogate over 200 test items, and a bag-of-words logistic regression matches
or beats every audited model on the same data. The required bond dimension stays at 1 as the
qubit count grows to sixteen, for two architectures independently, while an exact
representation would need 256.

Two controls keep this from being an artifact of the instrument. A positive control with a
known required bond dimension recovers that number, so the criterion is not simply printing
1. And auditing along each model's training path rather than only at its end, chi* is 1 at
378 of 392 snapshots spanning accuracies from chance to perfect, which rules out the reading
that these circuits are cheap only because they failed to learn. One audited model, the
compositional DisCoCat pipeline, genuinely needs chi = 2, which is the discriminating case.

The work fits the journal's scope on the "Quantum Computing for Artificial Intelligence"
axis, and it is deliberately constructive rather than dismissive: by naming the quantity a
genuine quantum ingredient would have to move, it gives QNLP a target and a reusable
instrument. The protocol is model-agnostic, consuming any trained circuit through a common
intermediate representation, and the full harness with every stored result is released
openly so that any QNLP model can be checked against this bar.

I have tried to be candid about scope. Everything runs at simulator scale on a laptop CPU;
the relative-pronoun benchmark turns out not to clear a majority-class predictor and I draw
no verdict from it; and the all-quantum mixer runs below its published scale, so I report
its audit rather than an accuracy reproduction. None of this settles the asymptotic
question, and the paper says so.

This manuscript is original, is not under consideration elsewhere, and has not been
published previously. I am the sole author. There are no competing interests and no funding
to declare.

Thank you for your consideration.

Tanvir Ahmed
North South University, Dhaka, Bangladesh
tanvir.ahmed32@northsouth.edu
