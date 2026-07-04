#!/usr/bin/env bash
# ==============================================================================
# Regenerate the 4 corpus reports that apply to the 6c-only pipeline.
#   usage: scripts/run_reports.sh [START] [END] [RNNT_RESULTS]
#     START END    : file range for the corpus reports (END exclusive; default 1 25)
#     RNNT_RESULTS : eval JSONL for per_speaker_wer (default data/asr/results/rnnt11b_padded_test_6C.jsonl)
# Outputs -> config.REPORTS_DIR. (indep_wer = multi-arm, N/A to 6c-only; report_docx = deferred.)
# ==============================================================================
set -euo pipefail
cd /workspace/code2
source ./setup_env.sh >/dev/null

S="${1:-1}"; E="${2:-25}"
RNNT="${3:-data/asr/results/rnnt11b_padded_test_6C.jsonl}"
PY=python3

echo "=== reports: files ${S}..$((E-1)) — $(date) ==="
$PY -m asrpipe.reports.speech_recovery  "$S" "$E"
$PY -m asrpipe.reports.conf_filter      "$S" "$E"
$PY -m asrpipe.reports.speaker_duration "$S" "$E"

if [ -f "$RNNT" ]; then
  $PY -m asrpipe.reports.per_speaker_wer "$RNNT"
else
  echo "[skip] per_speaker_wer: results file not found: $RNNT (run scripts/run_eval.sh first)"
fi
echo "=== reports complete ==="
