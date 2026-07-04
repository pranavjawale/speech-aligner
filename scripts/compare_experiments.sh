#!/usr/bin/env bash
# ==============================================================================
# Multi-experiment comparison (baseline vs fine-tuned side-by-side, N-way).
# Reads finished single-experiment metadata from docs/experiment-logs/ and emits
# an Excel-portable comparison per docs/multi-expt-comparison-metadata-format.txt.
#
#   usage: scripts/compare_experiments.sh <comparison-name> <baseline-exp> <exp> [exp ...]
#   e.g. : scripts/compare_experiments.sh 6c-test_rnnt_base-vs-ft \
#              all-speakers_parakeetrnnt1.1b_baselineexpt1 \
#              all-speakers_parakeetrnnt1.1b_finetuneexpt1
# Outputs -> docs/experiment-logs/comparisons/<comparison-name>.{csv,per_speaker.csv,metadata.json,summary.txt}
# ==============================================================================
set -euo pipefail
cd /workspace/code2
source ./setup_env.sh >/dev/null

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <comparison-name> <baseline-exp> <exp> [exp ...]" >&2; exit 2
fi
NAME="$1"; BASELINE="$2"; shift 2
python3 -m asrpipe.reports.compare_experiments --name "$NAME" --baseline "$BASELINE" --exps "$BASELINE" "$@"
