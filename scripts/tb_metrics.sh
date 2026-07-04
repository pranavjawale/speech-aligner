#!/usr/bin/env bash
# Browser-free view of the fine-tune metrics — reads the TensorBoard event files and prints
# per-epoch val_wer/val_loss + the latest train_loss/lr. For when you can't reach the TB web
# UI (remote box / no port forward). Usage: bash scripts/tb_metrics.sh
set -uo pipefail
cd /workspace/code2
source setup_env.sh >/dev/null 2>&1
.venv-nemo/bin/python - <<'PY' 2>/dev/null
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
dirs = sorted(glob.glob("exp/finetune/*/version_*"))
if not dirs:
    print("no TB runs under exp/finetune yet"); raise SystemExit
d = dirs[-1]; ea = EventAccumulator(d, size_guidance={'scalars': 0}); ea.Reload()
tags = ea.Tags()["scalars"]
def last(t): return ea.Scalars(t)[-1].value if t in tags else None
print(f"run: {d}")
if "val_wer" in tags:
    vw = {s.step: s.value for s in ea.Scalars("val_wer")}
    vl = {s.step: s.value for s in ea.Scalars("val_loss")} if "val_loss" in tags else {}
    print("  epoch-end validation (monitored for best ckpt / early stop):")
    for step in sorted(vw):
        print(f"    step {step:>6}  val_wer {vw[step]*100:6.2f}%   val_loss {vl.get(step, float('nan')):.4f}")
    best = min(vw.values())
    print(f"  best val_wer so far: {best*100:.2f}%")
tl, lr, ep, gs = last("train_loss"), last("learning_rate"), last("epoch"), last("global_step")
if tl is not None:
    print(f"  latest train: epoch {ep:.0f}  step {gs:.0f}  train_loss {tl:.4f}  lr {lr:.2e}")
print("  baseline to beat (ctc-0.6b, legal2023_38hrs padded test): 23.39%")
PY
