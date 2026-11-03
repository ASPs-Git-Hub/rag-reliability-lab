#!/bin/sh
# Usage: fetch_data.sh <new-empty-dir>. Downloads SQuAD 2.0 dev set and checks its sha256. Data is never executed.
set -eu
DIR="$1"
mkdir -p "$DIR"
[ -z "$(ls -A "$DIR")" ] || { echo "directory not empty" >&2; exit 1; }
curl -sSL -o "$DIR/dev-v2.0.json" https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
EXPECTED_FILE="$(dirname "$0")/SHA256.txt"
ACTUAL=$(shasum -a 256 "$DIR/dev-v2.0.json" | cut -d' ' -f1)
echo "sha256 $ACTUAL"
if [ -f "$EXPECTED_FILE" ]; then
  [ "$ACTUAL" = "$(cut -d' ' -f1 "$EXPECTED_FILE")" ] && echo "MATCHES SHA256.txt" || { echo "MISMATCH" >&2; exit 2; }
fi
