#!/usr/bin/env bash
# ==============================================================================
# S13: NeMo WER eval of ctc-0.6b + rnnt-1.1b on a padded manifest.
#   usage: scripts/run_eval.sh [MANIFEST]
#     MANIFEST : manifest to score (default data/asr/splits-padded/test.jsonl — the RP6 gate)
# Runs under the NeMo venv. Results + summary -> data/asr/results/. Prints corpus WER.
# ==============================================================================
set -euo pipefail
cd /workspace/code2
source ./setup_env.sh >/dev/null

MANIFEST="${1:-data/asr/splits-padded/test.jsonl}"
NEMO_PY="$PWD/.venv-nemo/bin/python"
LOG="$PWD/logs/eval"; mkdir -p "$LOG"
RES="$PWD/data/asr/results"; mkdir -p "$RES"

run() {
  local label="$1" logf="$2"; shift 2
  local t=$SECONDS
  echo "########## $label ##########  $(date +%H:%M:%S)"
  if ! "$@" > "$logf" 2>&1; then
    echo "!!! $label FAILED (see $logf) — aborting"; tail -15 "$logf"; exit 1
  fi
  echo "   $label ok  ($((SECONDS-t))s)"
}

echo "=== S13 eval on $MANIFEST — $(date) ==="
run "eval ctc-0.6b"  "$LOG/s13_ctc.log"  $NEMO_PY -m asrpipe.asr.eval \
      --manifest "$MANIFEST" --model nvidia/parakeet-ctc-0.6b \
      --out "$RES/ctc06b_padded_test_6C.jsonl"  --batch-size 32
run "eval rnnt-1.1b" "$LOG/s13_rnnt.log" $NEMO_PY -m asrpipe.asr.eval \
      --manifest "$MANIFEST" --model nvidia/parakeet-rnnt-1.1b \
      --out "$RES/rnnt11b_padded_test_6C.jsonl" --batch-size 32

echo "--- corpus WER (frozen 6c: ctc 36.73% / rnnt 34.37%) ---"
grep -H "corpus_wer" "$RES"/ctc06b_padded_test_6C.jsonl.summary.json \
                     "$RES"/rnnt11b_padded_test_6C.jsonl.summary.json 2>/dev/null || true
