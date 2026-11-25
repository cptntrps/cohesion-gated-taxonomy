#!/usr/bin/env bash
# WORK-2283: harvest (redacts each full text in memory before writing) -> key (truncation after redaction).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$HOME/data/cohesion-gated-taxonomy/catalog_retrieval}"
rm -f "$out/tasks.raw.jsonl" "$out/catalog.raw.jsonl"
nice -n 10 python3 "$here/harvest.py" "$out"
python3 "$here/build_key.py" "$out"
