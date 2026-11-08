#!/usr/bin/env bash
# Reproducible run of every prompt configuration on the ScanNet chair.
# Usage: bash run_all.sh   (from pinpoint3d-workspace)
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$WS/../data_examples/ScanNet/scene0025_00}"
CKPT="$WS/weights/pinpoint3d_pretrained.pth"
LOGDIR="$WS/outputs/logs"
PY="$WS/env/.pixi/envs/default/bin/python"
mkdir -p "$LOGDIR"

echo "script pid=$$ started=$(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
for cfg in pos1 pos1_neg_floor1 pos1_neg_floor1_desk1 pos1_neg_fp1 pos1_neg_fp3 pos1_repeat; do
  prompt="scan0025_chair13_$cfg.json"
  [ "$cfg" = "pos1_repeat" ] && prompt="scan0025_chair13_pos1.json"
  log="$LOGDIR/$cfg.log"
  echo "=== $cfg start $(date -Is) ==="
  "$PY" "$WS/scripts/pinpoint3d_run.py" \
    --data "$DATA" --checkpoint "$CKPT" --instance 13 \
    --prompts "$WS/prompts/$prompt" \
    --out "$WS/outputs/$cfg" > "$log" 2>&1
  rc=$?
  echo "$cfg exit=$rc log=$log"
  if [ $rc -ne 0 ]; then tail -20 "$log"; fi
done
echo "script pid=$$ finished=$(date -Is)"
