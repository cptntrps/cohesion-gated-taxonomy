#!/usr/bin/env bash
# Launch the taxonomy server. Provide DEEPSEEK_API_KEY in the environment (or adapt
# naming.py to any OpenAI-compatible endpoint). The key is never written to disk.
set -euo pipefail
cd "$(dirname "$0")"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
export PORT="${PORT:-8799}"
exec python3 server.py
