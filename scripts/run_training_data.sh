#!/usr/bin/env bash
# ==============================================================================
# S10-S12: manifests -> partition -> pad  (build the ASR training data).
#   usage: scripts/run_training_data.sh [START] [END] [ONLY]
#     START END : file range for the segment manifests (END exclusive; default 1 25)
#     ONLY      : comma list of splits to pad (default train,dev,test; RP6 gate used "test")
# All stages run under the main env. Corpus-level S11 ignores the file range.
# ==============================================================================
set -euo pipefail
cd /workspace/code2
source ./setup_env.sh >/dev/null

S="${1:-1}"; E="${2:-25}"; ONLY="${3:-train,dev,test}"
PY=python3
LOG="$PWD/logs/training-data"; mkdir -p "$LOG"

run() {
  local label="$1" logf="$2"; shift 2
  local t=$SECONDS
  echo "########## $label ##########  $(date +%H:%M:%S)"
  if ! "$@" > "$logf" 2>&1; then
    echo "!!! $label FAILED (see $logf) — aborting"; tail -15 "$logf"; exit 1
  fi
  echo "   $label ok  ($((SECONDS-t))s)"
}

echo "=== training data S10-S12: files ${S}..$((E-1)), pad=[$ONLY] — $(date) ==="
run "S10 manifests" "$LOG/s10.log" $PY -m asrpipe.asr.manifests "$S" "$E"
run "S11 partition" "$LOG/s11.log" $PY -m asrpipe.asr.partition
run "S12 pad"       "$LOG/s12.log" $PY -m asrpipe.asr.pad --only "$ONLY"
echo "=== training data complete ==="
