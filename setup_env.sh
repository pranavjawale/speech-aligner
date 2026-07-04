#!/usr/bin/env bash
# ==============================================================================
# code2 / asrpipe — environment restore
# ==============================================================================
# Everything lives under /workspace (persistent). After a pod restart, only
# ~/.cache (model downloads) is lost; NeMo/whisperx re-fetch on first use.
#
#   source /workspace/code2/setup_env.sh
#
# What this does:
#   - export HF_TOKEN + write ~/.cache/huggingface/token
#   - put the asrpipe package on PYTHONPATH (until `pip install -e .` is run)
#   - point at the KenLM / pyctcdecode tooling (S3 LM side)
#   - print a status summary
#
# STATUS: reorg complete (RP0-RP6). This script only restores env vars + PYTHONPATH;
# it does not (re)create the heavy interpreters. Those already exist:
#   - main env : system python3 with requirements.txt installed (S2,S4-S12, reports)
#   - NeMo venv: .venv-nemo/ (NeMo 2.7.3 + torch 2.8, Q5) — used for S3 + S13
# ==============================================================================
# SYSTEM COMPUTE INFO  (this RunPod pod; verified 2026-07-03)
#   OS    : Ubuntu 24.04.3 LTS  (kernel 6.8.0-45-generic)
#   CPU   : AMD EPYC 7402P 24-Core  (48 vCPU, 2 threads/core, 1 socket)
#   RAM   : 251 GB total (~230 GB avail)
#   GPU   : NVIDIA RTX A4500, 20 GB VRAM  (driver 550.127.05)
#   CUDA  : 12.8 (nvcc V12.8.93)
#   Python: 3.12.3   torch: 2.8.0+cu128   NeMo: 2.7.3 (.venv-nemo)   jiwer: 3.1.0 (.venv-nemo)
#   Disk  : /workspace = 2.1 P RunPod network volume (PERSISTENT across restarts)
#   Note  : only ~/.cache (HF/NeMo model downloads) AND apt-installed system pkgs are lost on
#           restart; NeMo/whisperx from_pretrained re-fetch models automatically, and the
#           Boost self-heal below re-installs lmplz's runtime libs on demand.
# ==============================================================================
set -u

CODE2=/workspace/code2

# --- HF token (same as /workspace/setup_env.sh) ---
export HF_TOKEN="TBA"
mkdir -p ~/.cache/huggingface
echo "$HF_TOKEN" > ~/.cache/huggingface/token

# --- make `import asrpipe` work without an editable install yet ---
export PYTHONPATH="${CODE2}:${CODE2}/src:${PYTHONPATH:-}"

# --- KenLM tooling location (built by the main /workspace/setup_env.sh section 9) ---
export KENLM_DIR=/workspace/code/kenlm     # reused; not duplicated into code2

# --- S3 LM-decode deps (kenlm/pyctcdecode) in the NeMo venv — self-heal after a restart ---
# align/lm_prepass.py (S3) is the ONLY consumer and it runs under the NeMo venv. These are
# pip-installs; if they ever land in ephemeral system site-packages they vanish on restart,
# so (re)install them into the PERSISTENT NeMo venv (.venv-nemo, under /workspace) when
# missing. Guarded: a no-op once present. pyctcdecode is --no-deps on purpose — its stale
# numpy<2 pin must NOT downgrade numpy 2.4.x (needed by torch/NeMo/numba).
_nemo_py="${CODE2}/.venv-nemo/bin/python"
_pyctc=$(ls -d "${CODE2}"/.venv-nemo/lib/python*/site-packages/pyctcdecode 2>/dev/null | head -n1)
if [ -x "$_nemo_py" ] && [ -z "$_pyctc" ]; then
    echo "   [heal] S3 LM-decode deps missing in NeMo venv -> installing kenlm/pygtrie/pyctcdecode ..."
    "${CODE2}/.venv-nemo/bin/pip" install -q kenlm pygtrie >/dev/null 2>&1
    "${CODE2}/.venv-nemo/bin/pip" install -q --no-deps pyctcdecode >/dev/null 2>&1
    _pyctc=$(ls -d "${CODE2}"/.venv-nemo/lib/python*/site-packages/pyctcdecode 2>/dev/null | head -n1)
fi

# --- KenLM LM-BUILD tooling (lmplz) needs Boost 1.83 runtime libs. These are apt-installed
#     system packages that vanish on a pod restart. Only needed to BUILD an n-gram LM (lmplz/
#     build_binary), NOT to decode with one. Self-heal ONLY if lmplz is present but its libs
#     are missing (cheap ldd check; no apt on the happy path). ---
_lmplz=/workspace/code/kenlm/build/bin/lmplz
if [ -x "$_lmplz" ] && ldd "$_lmplz" 2>/dev/null | grep -q "not found"; then
    echo "   [heal] lmplz missing Boost runtime -> apt-get install libboost-{program-options,thread}1.83.0 ..."
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y --no-install-recommends libboost-program-options1.83.0 libboost-thread1.83.0 >/dev/null 2>&1 \
        || echo "   [heal] apt failed; run: apt-get install -y libboost-program-options1.83.0 libboost-thread1.83.0"
fi

echo "=============================================================="
echo " code2 / asrpipe env"
echo "   ROOT        : ${CODE2}"
echo "   PYTHONPATH  : ${CODE2} + ${CODE2}/src (config + asrpipe importable)"
echo "   HF_TOKEN    : exported + written to ~/.cache/huggingface/token"
echo "   data/wav-files -> $(readlink -f ${CODE2}/data/wav-files 2>/dev/null || echo MISSING)"
echo "   LM assets   : $(ls ${CODE2}/data/lm 2>/dev/null | tr '\n' ' ')"
echo "--------------------------------------------------------------"
echo "   main env  : system python3 (whisperx/faster-whisper/pyannote) — S2,S4-S12,reports"
echo "   NeMo venv : $([ -x ${CODE2}/.venv-nemo/bin/python ] && echo "${CODE2}/.venv-nemo (ready)" || echo "MISSING — see requirements-nemo.txt")"
echo "   S3 LM deps: $([ -n "${_pyctc:-}" ] && echo "kenlm+pyctcdecode ready (NeMo venv)" || echo "MISSING (S3 only; hyp caches present)")"
echo "   run: scripts/run_pipeline.sh [S E] | run_training_data.sh | run_eval.sh | run_reports.sh"
echo "=============================================================="
