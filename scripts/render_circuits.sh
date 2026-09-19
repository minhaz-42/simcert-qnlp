#!/usr/bin/env bash
# Render the quantikz circuit diagrams to standalone PDFs.
#
# Springer's SNAPP compiler rejected the inline quantikz source: its quantikz differs
# from the one tectonic fetches, and `\\` immediately after \end{quantikz} inside a
# minipage raised "Missing $ inserted" there while compiling cleanly here. Rather than
# guess at which construct a remote LaTeX installation will accept, the circuits are
# rendered here once and included as images, so no submission system ever has to
# compile quantikz.
#
#   bash scripts/render_circuits.sh
#
# Writes figures/objects/circuit_*.pdf. Re-run if a circuit changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/figures/objects"
TMP="$(mktemp -d)"
mkdir -p "$OUT"
trap 'rm -rf "$TMP"' EXIT

preamble() {
  cat <<'EOF'
\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usepackage{quantikz}
\providecommand{\ket}[1]{\ensuremath{\left|#1\right\rangle}}
\begin{document}
EOF
}

render() {   # render <name> <body-file>
  local name="$1" body="$2"
  { preamble; cat "$body"; echo '\end{document}'; } > "$TMP/$name.tex"
  ( cd "$TMP" && tectonic -X compile "$name.tex" >/dev/null 2>&1 ) \
    || { echo "ERROR: $name failed to compile" >&2; exit 1; }
  cp "$TMP/$name.pdf" "$OUT/$name.pdf"
  echo "  wrote figures/objects/$name.pdf"
}

# ---- the reference ansatz (Figure: fig:circuit) ----------------------------------
cat > "$TMP/body_ref.tex" <<'EOF'
\begin{quantikz}[column sep=7pt, row sep=10pt]
\lstick{$\ket{0}$} & \gate[style={fill=orange!22}]{R_y(x_1)} & \gate[style={fill=green!25}]{R_y(\theta_1)} & \ctrl{1} & \qw      & \gate[style={fill=green!25}]{R_z(\phi_1)} & \meter{$Z$} \\
\lstick{$\ket{0}$} & \gate[style={fill=orange!22}]{R_y(x_2)} & \gate[style={fill=green!25}]{R_y(\theta_2)} & \targ{}  & \ctrl{1} & \gate[style={fill=green!25}]{R_z(\phi_2)} & \qw \\
\lstick{$\ket{0}$} & \gate[style={fill=orange!22}]{R_y(x_3)} & \gate[style={fill=green!25}]{R_y(\theta_3)} & \qw      & \targ{}  & \gate[style={fill=green!25}]{R_z(\phi_3)} & \qw
\end{quantikz}
EOF
render circuit_ref "$TMP/body_ref.tex"

# ---- QSANN ------------------------------------------------------------------------
cat > "$TMP/body_qsann.tex" <<'EOF'
\begin{quantikz}[column sep=5pt, row sep=8pt]
\lstick{$\ket{0}$} & \gate[style={fill=blue!12}]{H} & \gate[style={fill=orange!22}]{R_x} & \ctrl{1} & \gate[style={fill=green!25}]{R_y(\theta)} & \meter{$Z$} \\
\lstick{$\ket{0}$} & \gate[style={fill=blue!12}]{H} & \gate[style={fill=orange!22}]{R_x} & \targ{}  & \gate[style={fill=green!25}]{R_y(\theta)} & \qw
\end{quantikz}
EOF
render circuit_qsann "$TMP/body_qsann.tex"

# ---- QMSAN ------------------------------------------------------------------------
cat > "$TMP/body_qmsan.tex" <<'EOF'
\begin{quantikz}[column sep=5pt, row sep=8pt]
\lstick{$\ket{0}$} & \gate[style={fill=orange!22}]{R_x(x)} & \gate[2,style={fill=green!25}]{ZZ} & \gate[style={fill=green!25}]{R_y} & \gate[style={fill=orange!22}]{R_x(x)} & \meter{$\rho$} \\
\lstick{$\ket{0}$} & \gate[style={fill=orange!22}]{R_x(x)} &              & \gate[style={fill=green!25}]{R_y} & \gate[style={fill=orange!22}]{R_x(x)} & \trash{tr}
\end{quantikz}
EOF
render circuit_qmsan "$TMP/body_qmsan.tex"

# ---- CLAQS ------------------------------------------------------------------------
cat > "$TMP/body_claqs.tex" <<'EOF'
\begin{quantikz}[column sep=5pt, row sep=8pt]
\lstick{$\ket{0}$} & \gate[style={fill=green!25}]{R_y} & \ctrl{1} & \gate[style={fill=green!25}]{R_y} & \gate[2,style={fill=violet!20}]{\text{LCU}+\text{QSVT}} & \meter{$XYZ$} \\
\lstick{$\ket{0}$} & \gate[style={fill=green!25}]{R_y} & \gate[style={fill=green!25}]{R_x} & \gate[style={fill=green!25}]{R_y} &                                & \meter{$XYZ$}
\end{quantikz}
EOF
render circuit_claqs "$TMP/body_claqs.tex"

# ---- DisCoCat ---------------------------------------------------------------------
cat > "$TMP/body_discocat.tex" <<'EOF'
\begin{quantikz}[column sep=5pt, row sep=8pt]
\lstick{noun} & \gate[style={fill=green!25}]{R_x} & \ctrl{1} & \qw       & \qw \\
\lstick{verb} & \gate[2,style={fill=green!25}]{\text{IQP}} & \targ{} & \gate[style={fill=blue!12}]{H} & \meter{$0$} \\
\lstick{sent.}&            & \qw      & \qw       & \meter{$Z$}
\end{quantikz}
EOF
render circuit_discocat "$TMP/body_discocat.tex"

echo "done: 5 circuit PDFs in figures/objects/"
