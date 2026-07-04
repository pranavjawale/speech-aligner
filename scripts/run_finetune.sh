#!/usr/bin/env bash
# ==============================================================================
# Phase 15 — fine-tune parakeet-ctc-0.6b on legal2023_38hrs (train 35.66h / dev 0.57h).
# Runs asrpipe.finetune.train_ctc (conf/ctc_finetune.yaml) under the NeMo venv, logs +
# TensorBoard under exp/finetune/. Baseline to beat: ctc-0.6b padded-test corpus WER 23.39%.
#
#   bash scripts/run_finetune.sh                 # full fine-tune
#   bash scripts/run_finetune.sh --fast-dev-run  # 1-batch smoke test (validate wiring)
#   bash scripts/run_finetune.sh --max-epochs 3  # short run
# Extra args are passed through to the trainer. TensorBoard:
#   tensorboard --logdir exp/finetune --port 6006
# ==============================================================================
set -euo pipefail
cd /workspace/code2
source setup_env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce CUDA fragmentation OOM

PY=.venv-nemo/bin/python
CFG=src/asrpipe/finetune/conf/ctc_finetune.yaml
LOG=logs/finetune; mkdir -p "$LOG"
STAMP=$(date -u +%Y%m%d_%H%M%S)

echo "[finetune] parakeet-ctc-0.6b on legal2023_38hrs  (cfg=$CFG)"
echo "[finetune] log -> $LOG/train_${STAMP}.log ; exp -> exp/finetune/"
"$PY" -m asrpipe.finetune.train_ctc --config "$CFG" "$@" 2>&1 | tee "$LOG/train_${STAMP}.log"
