#!/usr/bin/env bash
# Run from the verified portable bundle. No credentials or provider API calls.
set -euo pipefail
PACKAGE_DIR=${1:?verified package directory required}
RUN_DIR=${2:?new run directory required}
PYTHON_BIN=${3:?Python executable required}
mkdir -p "$RUN_DIR"
test ! -e "$RUN_DIR/run_config.json"
cd "$PACKAGE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/launcher_started_utc.txt"
"$PYTHON_BIN" -m pip freeze > "$RUN_DIR/environment_freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RUN_DIR/gpu_at_start.csv"
set +e
timeout --signal=TERM --kill-after=120s 6500s "$PYTHON_BIN" -u train_phone_ctc.py \
  --train "$PACKAGE_DIR/dataset/train.jsonl" \
  --dev "$PACKAGE_DIR/dataset/dev.jsonl" \
  --vocab "$PACKAGE_DIR/dataset/vocab.json" \
  --audio-root "$PACKAGE_DIR" \
  --model "$PACKAGE_DIR/base_model" \
  --revision 3991242c806928916fff4a8c0e4f76acf661b743 \
  --local-files-only --device cuda \
  --output "$RUN_DIR" \
  --max-steps 600 --max-wall-seconds 6000 \
  --batch-size 8 --grad-accum 2 \
  --head-only-steps 100 --unfreeze-last 2 \
  --eval-every 50 --checkpoint-every 50 --patience 5
RESULT=$?
set -e
printf '%s\n' "$RESULT" > "$RUN_DIR/launcher_exit_code.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/launcher_finished_utc.txt"
# Checkpoints remain on the cloud disk for later review. Platform shutdown is
# scheduled separately by the user; killing a training process does not stop billing.
exit "$RESULT"
