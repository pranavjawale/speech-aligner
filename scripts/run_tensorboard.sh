#!/usr/bin/env bash
# Launch TensorBoard for the Phase-15 fine-tune, LIVE during training.
# exp_manager (create_tensorboard_logger=true in conf/ctc_finetune.yaml) writes event files
# under exp/finetune/<run>/ : train_loss (per step), val_loss, val_wer, and lr.
#
#   bash scripts/run_tensorboard.sh            # port 6006
#   bash scripts/run_tensorboard.sh 6007       # custom port
#
# RunPod access: expose a TCP port (pod "Connect" / port settings) mapping to this PORT,
# then open the URL RunPod gives you. (Or SSH-tunnel: ssh -L 6006:localhost:6006 <pod>.)
# --host 0.0.0.0 binds all interfaces so the exposed port reaches it.
set -euo pipefail
cd /workspace/code2
PORT="${1:-6006}"
TB=.venv-nemo/bin/tensorboard
[ -x "$TB" ] || TB=tensorboard
mkdir -p exp/finetune
echo "TensorBoard -> exp/finetune  (port $PORT, all interfaces)"
echo "  RunPod: expose TCP $PORT on the pod, then open the mapped URL."
exec "$TB" --logdir exp/finetune --host 0.0.0.0 --port "$PORT" --reload_multifile true
