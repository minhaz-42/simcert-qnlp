#!/usr/bin/env bash
# Build an ANONYMIZED code snapshot for double-blind review (OpenReview supplementary,
# or to feed anonymous.4open.science). Excludes git history, the (de-anonymising) paper,
# reference PDFs, data caches, and checkpoints; scrubs author-identifying strings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/simcert"
mkdir -p "$STAGE"

# copy code, keeping only what a reviewer needs to reproduce the audit
rsync -a --delete \
  --exclude='.git' --exclude='paper' --exclude='references/pdfs' \
  --exclude='data' --exclude='checkpoints' --exclude='*.egg-info' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='outputs' --exclude='multirun' --exclude='*.zip' --exclude='scripts' \
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

OUT="$ROOT/simcert_anon_code.zip"
( cd "$(dirname "$STAGE")" && zip -qr "$OUT" simcert )
echo "wrote $OUT"
echo "residual identifier check:"
grep -rinE "tanvir|gozayaan|minhaz|palash|north.?south" "$STAGE" 2>/dev/null || echo "  clean"
