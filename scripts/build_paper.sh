#!/usr/bin/env bash
# Build both PDFs from one source tree.
#
# The paper ships in two forms: the anonymous main.pdf that goes to TMLR under
# double-blind review, and the de-anonymised main_preprint.pdf for arXiv. They differ
# only by tmlr.sty's [preprint] option. Toggling that by hand in main.tex risks the one
# mistake that actually matters, committing a de-anonymised file as the blind
# submission, so the toggle happens on a throwaway copy and main.tex is never edited.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper"

echo "building anonymous submission..."
tectonic -X compile main.tex >/dev/null 2>&1 || tectonic main.tex
echo "  wrote paper/main.pdf (anonymous, for TMLR)"

echo "building de-anonymised preprint..."
trap 'rm -f _preprint.tex _preprint.aux _preprint.log _preprint.out' EXIT
sed 's/^\\usepackage{tmlr}$/\\usepackage[preprint]{tmlr}/' main.tex > _preprint.tex
if ! grep -q 'usepackage\[preprint\]{tmlr}' _preprint.tex; then
  echo "ERROR: could not apply the [preprint] option; main.tex's tmlr line changed shape" >&2
  exit 1
fi
tectonic -X compile _preprint.tex >/dev/null 2>&1 || tectonic _preprint.tex
mv _preprint.pdf main_preprint.pdf
echo "  wrote paper/main_preprint.pdf (de-anonymised, for arXiv)"

# The anonymous build must not carry the author name; the preprint must.
#
# The text is extracted into a variable and matched with a here-string rather than
# piped into "grep -q". Under pipefail, grep -q exits on its first match, pdftotext
# then dies of SIGPIPE, and the pipeline reports failure even though the pattern DID
# match. That inverts the result silently, which on the main.pdf gate would mean a
# leaking blind submission reported as clean.
IDENT='tanvir|north south|northsouth'
if command -v pdftotext >/dev/null 2>&1; then
  echo "anonymity check:"
  anon_txt="$(pdftotext main.pdf - 2>/dev/null || true)"
  pre_txt="$(pdftotext main_preprint.pdf - 2>/dev/null || true)"

  if [ -z "$anon_txt" ]; then
    echo "  ERROR: could not read text from main.pdf; refusing to certify it" >&2; exit 1
  fi
  if grep -qiE "$IDENT" <<<"$anon_txt"; then
    echo "  LEAK: main.pdf names the author but is the blind submission" >&2; exit 1
  fi
  echo "  main.pdf is anonymous"

  if grep -qiE "$IDENT" <<<"$pre_txt"; then
    echo "  main_preprint.pdf is de-anonymised"
  else
    echo "  WARNING: main_preprint.pdf does not name the author; is [preprint] working?" >&2
  fi
fi
