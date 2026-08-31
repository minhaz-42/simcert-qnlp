#!/usr/bin/env bash
# Build an ANONYMIZED code snapshot for double-blind review (OpenReview supplementary,
# or to feed anonymous.4open.science). Excludes git history, the (de-anonymising) paper,
# reference PDFs, data caches, and checkpoints; scrubs author-identifying strings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/simcert"
mkdir -p "$STAGE"

# Copy code, keeping only what a reviewer needs to reproduce the audit. The two
# excluded scripts build the paper and this snapshot; both name the author on
# purpose (the anonymity gate greps for exactly those strings) and neither is
# useful to a reviewer, since paper/ is excluded anyway.
rsync -a --delete \
  --exclude='.git' --exclude='paper' --exclude='references/pdfs' \
  --exclude='data' --exclude='checkpoints' --exclude='*.egg-info' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='outputs' --exclude='multirun' --exclude='*.zip' \
  --exclude='/scripts/make_anon_snapshot.sh' --exclude='/scripts/build_paper.sh' \
  --exclude='/docs/positive-control-study.md' \
  --exclude='*.log' \
  --exclude='/arxiv' --exclude='arxiv_submission.tar.gz' \
  "$ROOT"/ "$STAGE"/

# scrub identifiers
if [ -f "$STAGE/pyproject.toml" ]; then
  sed -i '' 's/name = "tanvir"/name = "Anonymous"/' "$STAGE/pyproject.toml" 2>/dev/null || \
  sed -i    's/name = "tanvir"/name = "Anonymous"/' "$STAGE/pyproject.toml"
fi
# drop the editable-install line that names the GitHub account
if [ -f "$STAGE/envs/requirements-audit.lock.txt" ]; then
  grep -v 'github.com/minhaz-42' "$STAGE/envs/requirements-audit.lock.txt" > "$STAGE/envs/_lock" \
    && mv "$STAGE/envs/_lock" "$STAGE/envs/requirements-audit.lock.txt"
fi
# anonymized README banner
printf '\n> Anonymized code for double-blind review. Author and repository details removed.\n' \
  >> "$STAGE/README.md"

# gate on residual identifiers BEFORE writing the zip, so a leaky snapshot is never produced
echo "residual identifier check:"
if grep -rinE "tanvir|gozayaan|minhaz|palash|north.?south" "$STAGE" 2>/dev/null; then
  echo "LEAK: author identifiers survived in the staged tree; aborting without writing the zip." >&2
  exit 1
fi
echo "  clean"

OUT="$ROOT/simcert_anon_code.zip"
rm -f "$OUT"   # start fresh; zip -r appends to an existing archive and would keep stale entries
( cd "$(dirname "$STAGE")" && zip -qr "$OUT" simcert )
echo "wrote $OUT"
