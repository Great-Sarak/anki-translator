#!/usr/bin/env bash
# tests/e2e/smoke.sh — runnable scaffold for the manual smoke test.
#
# This automates steps 1, 2, 5, 7 (the script-able parts). Steps 3, 4, 6 are still
# human-driven — they require eyeballing the queue file and verifying Anki state.
#
# Required:
#   openclaw on PATH with an auth profile for the target model. The default
#   dispatcher shells out to `openclaw infer model run --json` — verify with:
#     openclaw infer model run --json --prompt ping --model anthropic/claude-haiku-4-5
#
# Optional env:
#   ANKI_TRANSLATOR_MODEL       — override the default model (anthropic/claude-haiku-4-5)
#   ANKI_TRANSLATOR_CONCURRENCY — classifier/tagger fan-out width (default 8;
#                                 set 1 for sequential debug runs)
#   ANKI_TRANSLATOR_URL         — defaults to Wikipedia Mitochondrion article
#   ANKI_TEST_DECK              — defaults to Myrzka::Testing

set -euo pipefail

URL="${ANKI_TRANSLATOR_URL:-https://en.wikipedia.org/wiki/Mitochondrion}"
DECK="${ANKI_TEST_DECK:-Myrzka::Testing}"
TAG="e2e-smoke-$(date +%Y-%m-%d)"

echo "=== 1. Bootstrap starter note types ==="
anki-translator bootstrap

echo
echo "=== 2. Ingest $URL ==="
INGEST_JSON=$(anki-translator ingest "$URL" --deck "$DECK" --tag "$TAG")
echo "$INGEST_JSON"
QUEUE_FILE=$(echo "$INGEST_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['queue_file'])")
echo "Queue file: $QUEUE_FILE"

echo
echo "=== 3, 4. Review the queue file manually ==="
echo "Open $QUEUE_FILE in your editor."
echo "Delete one block (presence-equals-approved). Edit one field on another block."
echo "Press Enter when done..."
read -r

echo
echo "=== 5. Commit ==="
anki-translator commit "$QUEUE_FILE"

echo
echo "=== 6. Verify in Anki ==="
echo "Run: anki-manager call findNotes 'query=tag:$TAG'"
echo "Or browse 'tag:$TAG' in Anki Desktop."
echo "Compare the count to the surviving block count in queue/committed/."

echo
echo "=== 7. Retry idempotence ==="
ARCHIVED="queue/committed/$(basename "$QUEUE_FILE")"
if [[ -f "$ARCHIVED" ]]; then
  echo "Retrying commit on archived file: $ARCHIVED"
  anki-translator commit "$ARCHIVED" || echo "(expected: no duplicates created)"
fi

echo
echo "=== Done ==="
echo "If steps 6's note count matches the surviving block count in $ARCHIVED,"
echo "and step 7 produced zero new notes, v0.1 is shippable."
