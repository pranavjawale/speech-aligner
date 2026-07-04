#!/usr/bin/env bash
# ==============================================================================
# S2-S9 end-to-end for a file range (VAD split -> ... -> speaker segments).
#   usage: scripts/run_pipeline.sh [START] [END]   (END exclusive; default 1 25 = files 001-024)
# S3 runs under the NeMo venv (Parakeet); all other stages under the main env.
# Per-stage logs -> logs/pipeline/. Aborts on the first stage failure.
# ==============================================================================
set -euo pipefail
cd /workspace/code2
source ./setup_env.sh >/dev/null          # HF token + PYTHONPATH (config + asrpipe)

S="${1:-1}"; E="${2:-25}"
PY=python3
NEMO_PY="$PWD/.venv-nemo/bin/python"
LOG="$PWD/logs/pipeline"; mkdir -p "$LOG"

run() {                                    # run <label> <logfile> <cmd...>
  local label="$1" logf="$2"; shift 2
  local t=$SECONDS
  echo "########## $label ##########  $(date +%H:%M:%S)"
  if ! "$@" > "$logf" 2>&1; then
    echo "!!! $label FAILED (see $logf) — aborting"; tail -15 "$logf"; exit 1
  fi
  echo "   $label ok  ($((SECONDS-t))s)"
}

echo "=== pipeline S2-S9: files ${S}..$((E-1)) — $(date) ==="
run "S2 vad_split"        "$LOG/s2.log"   $PY      -m asrpipe.split.vad_split           "$S" "$E"
run "S3 lm_prepass(NeMo)" "$LOG/s3.log"   $NEMO_PY -m asrpipe.align.lm_prepass          "$S" "$E"
run "S4 align_p1"         "$LOG/s4.log"   $PY      -m asrpipe.align.align_p1            "$S" "$E"
run "S4.5 resync"         "$LOG/s45.log"  $PY      -m asrpipe.align.resync              "$S" "$E" --apply
run "S5 align_p2"         "$LOG/s5.log"   $PY      -m asrpipe.align.align_p2            "$S" "$E"
run "S6/S7 conf_trim"     "$LOG/s67.log"  $PY      -m asrpipe.align.conf_trim           "$S" "$E"
run "S8 timeline"         "$LOG/s8.log"   $PY      -m asrpipe.align.timeline            "$S" "$E"
run "S9 speaker_segments" "$LOG/s9.log"   $PY      -m asrpipe.segments.speaker_segments "$S" "$E"
echo "=== pipeline S2-S9 complete ==="
