#!/usr/bin/env bash
# Start the real PinPoint3D Studio backend (FastAPI + MinkowskiEngine).
#
#   bash scripts/start_backend.sh            # 127.0.0.1:9502
#   HOST=0.0.0.0 PORT=9503 bash scripts/start_backend.sh
#
# Optional object recognition is disabled unless ARK_API_KEY / GEMINI_API_KEY
# is already present in the environment.  No external call is ever made just by
# starting the server.  Startup logs redact credential-like values.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$WS/interSeg3D-Studio/src/backend"
PY="$WS/env/.pixi/envs/default/bin/python"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9502}"

if [ ! -x "$PY" ]; then
  echo "Python environment missing: $PY" >&2
  echo "Run: (cd '$WS/env' && pixi install --locked)" >&2
  exit 1
fi

if [ ! -f "$BACKEND/pinpoint3d/weights/checkpoint1099.pth" ]; then
  echo "Checkpoint missing: $BACKEND/pinpoint3d/weights/checkpoint1099.pth" >&2
  echo "See README.md (weights download + sha256)." >&2
  exit 1
fi

cd "$BACKEND"
echo "Starting PinPoint3D backend on http://$HOST:$PORT (pid $$)"
exec "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT" --log-level info
