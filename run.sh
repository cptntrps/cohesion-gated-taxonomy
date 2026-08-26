#!/usr/bin/env bash
# Launch the taxonomy server with the DeepSeek key decrypted into the process env
# (never written to disk). Used by ledgar-taxonomy.service.
set -euo pipefail
export PATH="/home/gui/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd /home/gui/projects/ledgar-taxonomy-mvp
export SOPS_AGE_KEY_FILE=/home/gui/.keys/operator-drop.age
export DEEPSEEK_API_KEY=$(sops -d /home/gui/src/infrastructure-secrets/services/livingos-crons.enc.env \
  | grep '^DEEPSEEK_API_KEY=' | cut -d= -f2-)
export PORT="${PORT:-8799}"
exec python3 server.py
