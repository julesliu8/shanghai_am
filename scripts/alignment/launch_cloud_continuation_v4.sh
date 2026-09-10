#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/am_project
RUN="$ROOT/runs/phone_ctc_v4_20260910"
DATA="$ROOT/wuu_phone_ctc_v1"
# Leave ten minutes before the user-reported provider shutdown at 01:00 Shanghai.
CUTOFF=$(date -u -d '2026-09-10 16:50:00 UTC' +%s)
REMAINING=$(( CUTOFF - $(date -u +%s) ))
WALL=3600
if (( REMAINING < WALL + 240 )); then WALL=$(( REMAINING - 240 )); fi
if (( WALL < 300 )); then echo 'Insufficient time before cutoff'; exit 2; fi
mkdir -p "$RUN"
printf '%s\n' "$$" > "$RUN/launcher.pid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
set +e
timeout --signal=TERM --kill-after=120s "$(( WALL + 120 ))s" \
  "$ROOT/venv/bin/python" "$ROOT/train_phone_ctc_v4.py" \
  --train "$DATA/dataset/train.jsonl" --dev "$DATA/dataset/dev.jsonl" \
  --vocab "$DATA/dataset/vocab.json" --audio-root "$DATA" \
  --model "$ROOT/models/v3_best_step006750" --local-files-only \
  --output "$RUN" --device cuda --seed 20260911 \
  --unfreeze-last 6 --head-only-steps 0 --head-lr 3e-5 --encoder-lr 3e-6 \
  --batch-size 8 --grad-accum 2 --warmup-steps 100 --initial-eval \
  --max-steps 21750 --max-wall-seconds "$WALL" \
  --eval-every 500 --checkpoint-every 500 --patience 12 --min-delta 0.0005
RC=$?
printf '{"exit_code":%d,"finished_utc":"%s"}\n' "$RC" "$(date -u +%FT%TZ)" > "$RUN/launcher_exit.json"
exit "$RC"
