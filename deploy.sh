#!/usr/bin/env bash
# deploy.sh — in place: the user unit cohesion-gated-taxonomy.service runs this checkout on
# aurora. Deploy = restart the unit with the merged code, then prove /healthz ok (the
# server loaded the corpus and a built tree covers it). Fail loud; roll back to the last
# good commit when health does not come.
set -euo pipefail
cd "$(dirname "$0")"
UNIT=cohesion-gated-taxonomy.service
PORT="${PORT:-8580}"
LAST_GOOD_FILE=".deploy-last-good"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || { echo "deploy: refusing — live checkout on '$BRANCH', not main" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "deploy: refusing — dirty tree" >&2; exit 1; }
git fetch --quiet origin main || { echo "deploy: refusing — cannot fetch origin/main" >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || {
    echo "deploy: refusing — HEAD $(git rev-parse --short HEAD) != origin/main $(git rev-parse --short origin/main)" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo "deploy: refusing — no .venv (python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .)" >&2; exit 1; }
.venv/bin/pip install -q -r requirements.txt -e . >/dev/null

healthy() {
    local i body
    for i in $(seq 1 90); do
        sleep 1
        body=$(curl -s --max-time 3 "http://127.0.0.1:${PORT}/healthz" || true)
        if printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("ok") else 1)' 2>/dev/null; then
            echo "deploy: healthy — /healthz ok after ${i}s: $body"
            return 0
        fi
    done
    return 1
}

TARGET="$(git rev-parse HEAD)"
echo "deploy: restarting $UNIT @ $(git rev-parse --short HEAD)"
systemctl --user restart "$UNIT"
if healthy; then
    printf '%s\n' "$TARGET" > "$LAST_GOOD_FILE"
    exit 0
fi
echo "deploy: FAILED health at $(git rev-parse --short HEAD)" >&2
systemctl --user status "$UNIT" --no-pager -n 20 >&2 || true
if [ -f "$LAST_GOOD_FILE" ] && [ "$(cat "$LAST_GOOD_FILE")" != "$TARGET" ]; then
    echo "deploy: rolling back to $(cut -c1-7 "$LAST_GOOD_FILE")" >&2
    git checkout --quiet "$(cat "$LAST_GOOD_FILE")"
    systemctl --user restart "$UNIT"
    healthy && { echo "deploy: rollback healthy; main checkout left detached at last-good" >&2; exit 1; }
fi
exit 1
