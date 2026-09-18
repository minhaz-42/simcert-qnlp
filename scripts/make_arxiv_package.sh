#!/usr/bin/env bash
# Build the arXiv submission package from the current paper source.
#
# arXiv compiles the LaTeX itself, so the tarball has to be self-contained: every .sty,
# every \input, and every figure, with figure paths rewritten because arXiv puts all
# sources in one flat directory rather than the paper/ + figures/ split used here.
#
# The de-anonymised [preprint] option is applied to a staged copy; paper/main.tex is
# never edited, for the same reason build_paper.sh does it that way.
#
#   bash scripts/make_arxiv_package.sh
#
# Produces arxiv/ (the staged tree, kept for inspection) and arxiv_submission.tar.gz.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/arxiv"
cd "$ROOT"

rm -rf "$STAGE"
mkdir -p "$STAGE/figs" "$STAGE/tables" "$STAGE/appendix" "$STAGE/objects"

# ---- main source, de-anonymised, with the figure path flattened -------------------
sed -e 's/^\\usepackage{tmlr}$/\\usepackage[preprint]{tmlr}/' \
    -e 's|^\\graphicspath{{\.\./figures/}}$|\\graphicspath{{./}}|' \
    paper/main.tex > "$STAGE/main.tex"

# arXiv archives the source, comments included. Drop the whole-line comments that are
# build scaffolding about the TMLR submission: they are meaningless to a reader and
# there is no reason to publish them. Only full-line comments, never trailing ones,
# which would change spacing.
sed -i '' -E '/^%.*(TMLR|camera-ready|anonymi|double-blind|QTML)/d' "$STAGE/main.tex"

grep -q 'usepackage\[preprint\]{tmlr}' "$STAGE/main.tex" \
  || { echo "ERROR: could not apply the [preprint] option" >&2; exit 1; }
grep -q 'graphicspath{{\./}}' "$STAGE/main.tex" \
  || { echo "ERROR: could not flatten \\graphicspath" >&2; exit 1; }

# ---- everything main.tex \inputs --------------------------------------------------
cp paper/tmlr.sty paper/fancyhdr.sty paper/math_commands.tex "$STAGE/" 2>/dev/null || true
cp paper/figs/*.tex "$STAGE/figs/"
cp paper/tables/*.tex "$STAGE/tables/"
cp paper/appendix/*.tex "$STAGE/appendix/"

# ---- exactly the figures the source references, no more --------------------------
# Derived from the source rather than listed by hand: a hand-kept list is how the
# previous package ended up missing three figures.
# (a while-read loop, not mapfile: macOS ships bash 3.2)
n_figs=0
grep -rhoE 'includegraphics\[[^]]*\]\{[^}]+\}' \
  paper/main.tex paper/figs/*.tex paper/appendix/*.tex \
  | sed 's/.*{\(.*\)}/\1/' | sort -u > /tmp/_arxiv_figs.txt
while IFS= read -r f; do
  [ -n "$f" ] || continue
  src="$ROOT/figures/$f"
  [ -f "$src" ] || { echo "ERROR: referenced figure missing: figures/$f" >&2; exit 1; }
  cp "$src" "$STAGE/$f"
  n_figs=$((n_figs + 1))
done < /tmp/_arxiv_figs.txt
rm -f /tmp/_arxiv_figs.txt
echo "staged $n_figs figures"

# ---- prove it compiles standalone, the way arXiv will ----------------------------
( cd "$STAGE" && tectonic -X compile main.tex >/dev/null 2>&1 || tectonic main.tex )
pages=$(mdls -name kMDItemNumberOfPages -raw "$STAGE/main.pdf" 2>/dev/null || echo "?")
echo "arxiv/main.pdf builds standalone (${pages} pages)"

# The preprint must carry the author; the anonymous build must not. This is the
# mirror of build_paper.sh's gate, and it is the one that matters here.
if command -v pdftotext >/dev/null 2>&1; then
  txt="$(pdftotext "$STAGE/main.pdf" - 2>/dev/null || true)"
  if grep -qiE 'tanvir|north south|northsouth' <<<"$txt"; then
    echo "  de-anonymisation check: author present, as required for arXiv"
  else
    echo "ERROR: the preprint does not carry the author name" >&2; exit 1
  fi
fi

# ---- tarball: sources only, no build artefacts ----------------------------------
rm -f "$ROOT/arxiv_submission.tar.gz"
tar -czf "$ROOT/arxiv_submission.tar.gz" -C "$STAGE" \
  --exclude='*.pdf.bak' --exclude='main.aux' --exclude='main.log' --exclude='main.out' \
  --exclude='main.pdf' .
echo "wrote arxiv_submission.tar.gz ($(du -h "$ROOT/arxiv_submission.tar.gz" | cut -f1))"
echo
echo "Upload arxiv_submission.tar.gz to arXiv. Keep arxiv/main.pdf to compare against"
echo "the PDF arXiv generates; they should match."
