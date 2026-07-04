"""
Central configuration for the asrpipe codebase (code2).

ONE source of truth for every path and tunable constant. 

Convention:
  - Paths are pathlib.Path, derived from ROOT (this repo) and DATA.
  - Constants are grouped by pipeline stage (S1..S13) and mirror the values
    currently used in /workspace/code (extracted verbatim, not invented).
  - The NeMo fine-tune config is the ONE thing that lives in YAML, not here
    (src/asrpipe/finetune/conf/ctc_finetune.yaml) — Hydra/OmegaConf convention.

Nothing in this module performs I/O or side effects at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# ROOTS  — everything derives from here
# ==============================================================================
# config/ lives one level under the repo root (code2/).
ROOT = Path(__file__).resolve().parent.parent          # /workspace/code2
DATA = ROOT / "data"                                    # all data under here

# ------------------------------------------------------------------ inputs
TRANSCRIPTS_DIR = DATA / "transcripts-v3"               # copied (24 .txt)
MP3_DIR         = DATA / "mp3-downloads"                # empty placeholder
WAV_DIR         = DATA / "wav-files"                    # SYMLINK -> /workspace/wav-files (Q3)

# ------------------------------------------------------------------ derived (regenerated, Q2)
CHUNKS_DIR      = DATA / "chunks"                        # VAD chunk wavs (no strategy-N)
# ALIGN_DIR is env-overridable so an ablation run can write its alignments + labels +
# timeline into an isolated tree (data/ablation/<exptTag>/...) without touching the
# canonical corpus. labels_dir/timeline_json/align_json all derive from ALIGN_DIR.
ALIGN_DIR       = Path(os.environ["ASRPIPE_ALIGN_DIR"]) if os.environ.get("ASRPIPE_ALIGN_DIR") else (DATA / "alignments")
# S5 gap-fill subchunk wavs: None -> chunks/<stem>/subchunks; env override -> isolated tree.
SUBCHUNK_DIR    = Path(os.environ["ASRPIPE_SUBCHUNK_DIR"]) if os.environ.get("ASRPIPE_SUBCHUNK_DIR") else None
MANIFEST_DIR    = DATA / "manifests"                     # chunk + speaker manifests
SEGMENTS_DIR    = DATA / "speaker-segments"              # per-speaker segment wavs
SEGMENTS_PAD_DIR= DATA / "speaker-segments-padded"

# ------------------------------------------------------------------ LM assets (copied)
LM_DIR          = DATA / "lm"
KENLM_BIN       = LM_DIR / "court_6gram.bin"             # 6-gram KenLM
UNIGRAMS_TXT    = LM_DIR / "unigrams.txt"                # 12.6k court unigram list

# ------------------------------------------------------------------ ASR / training-data track
ASR_DIR         = DATA / "asr"
ASR_MANIFEST_DIR= ASR_DIR / "manifests"
SPLITS_DIR      = ASR_DIR / "splits"                     # train/dev/test.jsonl
SPLITS_PAD_DIR  = ASR_DIR / "splits-padded"
RESULTS_DIR     = ASR_DIR / "results"
TOKENIZER_DIR   = DATA / "tokenizer"

# ------------------------------------------------------------------ misc
LOGS_DIR        = ROOT / "logs"
REPORTS_DIR     = ROOT / "reports-out"      # generated corpus analysis reports
EXPT_RESULTS_DIR   = ROOT / "results"                 # raw per-utterance CSVs (expt-metadata spec #8)
EXPERIMENT_LOGS_DIR= ROOT / "docs" / "experiment-logs" # metadata json + summary (spec #9)

def stem(file_no: int) -> str:
    """audio_file_003 for 3, etc. (canonical file stem)."""
    return f"audio_file_{file_no:03d}"

# ==============================================================================
# RUNTIME / DEVICE
# ==============================================================================
# Resolved lazily by callers; kept as strings here to avoid importing torch.
DEVICE_PREF   = "cuda"          # fall back to cpu at runtime if unavailable
HF_TOKEN      = os.environ.get("HF_TOKEN", "")          # set by setup_env.sh
SAMPLE_RATE   = 16000           # whisperx.load_audio always returns 16 kHz mono
LANGUAGE      = "en"

# ==============================================================================
# S2 — VAD SPLIT   (from split_strategy5.py VAD_PARAMS)
# ==============================================================================
VAD_ONSET        = 0.500
VAD_OFFSET       = 0.363
VAD_CHUNK_SIZE   = 30           # seconds, hard max per chunk
VAD_MIN_DUR_ON   = 0.1
VAD_MIN_DUR_OFF  = 0.1

# ==============================================================================
# S3 — LM PRE-PASS  (Parakeet-CTC + KenLM + unigrams; ctc_kenlm_prepass_parallel.py)
# ==============================================================================
CTC_LM_MODEL     = "nvidia/parakeet-ctc-0.6b"           # acoustic model for the pre-pass
KENLM_ALPHA      = 0.5          # LM weight   (pyctcdecode)
KENLM_BETA       = 1.0          # word-insertion bonus
KENLM_BEAM_SIZE  = 128
PREPASS_WORKERS  = 40           # CPU cores for decode_batch
PREPASS_GPU_BATCH = 1           # GPU forward batch (rollout used 1; keeps logits exact)

# ==============================================================================
# S4 — ALIGN PASS 1  (Whisper ASR + best-of-two match + masked wav2vec2)
#   from align_chunks_withmask.py (base) + align_chunks_withmask_lm.py
# ==============================================================================
ASR_MODEL_SIZE   = "large-v2"   # faster-whisper
ASR_COMPUTE_TYPE = "float16"
ASR_BEAM_SIZE    = 5

# sliding-window reference matcher
SEARCH_BUFFER_BACK = 600        # words of back-buffer around expected_pos
SEARCH_BUFFER_FWD  = 200        # words forward
FALLBACK_WPS       = 1.71       # words/sec fallback when speech duration unknown

# anchor-based scoring
ANCHOR_MIN_LEN     = 3
ANCHOR_MAX_LEN     = 6
ANCHOR_POS_THRESH  = 0.25
ANCHOR_ALPHA       = 0.0      # DISABLED (was 4.0) — anchor-bonus OFF in main pipeline 2026-07-02 (disablePass1Extra)
MAX_ANCHOR_BONUS   = 60       # (unused when ANCHOR_ALPHA=0: bonus is always 0; kept only as matcher cutoff slack)

# committed_pos soft floor + boundary scoring
COMMIT_THRESHOLD     = 0.35     # nd <= this => commit position
OVERLAP_PENALTY_WORD = 0        # DISABLED (was 8) — overlap-penalty OFF in main pipeline (disablePass1Extra)
BOUNDARY_K           = 10       # words at each boundary used for head/tail scoring
BOUNDARY_ALPHA       = 0.0      # DISABLED (was 1.0) — boundary-penalty OFF in main pipeline (disablePass1Extra); 0 = off

# ==============================================================================
# ABLATION SWITCHES  (env-var gated; ALL default OFF -> canonical pipeline).
# Reversible: unset the env var to restore default behavior. Output isolation is via
# ASRPIPE_ALIGN_DIR/SUBCHUNK_DIR (above). See docs/aligner-ablation-study.txt.
# ==============================================================================
def _ablate(_name: str) -> bool:
    return os.environ.get(_name, "").strip().lower() not in ("", "0", "false", "no")

ABLATE_PASS1EXTRA = _ablate("ASRPIPE_ABLATE_PASS1EXTRA")  # zero anchor/boundary/overlap penalties
ABLATE_WHISPER    = _ablate("ASRPIPE_ABLATE_WHISPER")     # CTC-only localization hyp (no best-of-two)
ABLATE_PASS2      = _ablate("ASRPIPE_ABLATE_PASS2")       # skip S5 gap-fill (labels still written)
ABLATE_TRIM       = _ablate("ASRPIPE_ABLATE_TRIM")        # skip S6/S7 trim (copy labels through)

if ABLATE_PASS1EXTRA:                                     # Exp2: effective = dist (only)
    ANCHOR_ALPHA = 0.0
    BOUNDARY_ALPHA = 0.0
    OVERLAP_PENALTY_WORD = 0

# silence masking before forced alignment
SILENCE_MASK_THRESHOLD_S = 0.8  # zero silence gaps >= this many seconds

IO_WORKERS = 8                  # ThreadPoolExecutor workers for chunk/JSON I/O

# ==============================================================================
# S5 — ALIGN PASS 2  (gap-fill; align_chunks_pass2_withmask.py)
# ==============================================================================
GAP_THRESHOLD_S = 3.0           # tail/head gap that triggers gap-fill
GAP_OVERSHOOT   = 1.10          # 10% extra ref words when filling
GAP_BACKOFF_S   = 0.10          # audio back-off before the tail-gap boundary
GAP_FORWARD_S   = 0.10          # audio forward-extend past the head-gap boundary
GAP_MIN_SUB_DUR_S = 0.5         # skip gap-fill if the sub-chunk is shorter than this

# ==============================================================================
# S6/S7 — CONFIDENCE TRIMMING  (confidence_trim_realign.py, head mirror)
# ==============================================================================
TOP_HALF_MIN         = 0.50     # adaptive quality gate: skip if top-half conf < this
CONTRAST_THRESH      = 0.30     # min avg-quality gap to trim
RIGHT_AVG_MAX        = 0.35     # fixed floor for the discarded-side guard
RIGHT_AVG_ADAPTIVE_K = 0.45     # adaptive component: K * top_half_conf
MIN_KEEP_WORDS       = 5
MIN_TRIM_WORDS       = 5

# ==============================================================================
# S9 — SPEAKER SEGMENTS / enrichment
# ==============================================================================
SEGMENT_MERGE_GAP_S  = 0.5      # merge same-speaker consecutive words if gap <= this
MIN_SEG_DUR_S        = 1.0      # drop speaker segments shorter than this
CONF_THRESHOLD       = 0.4      # canonical avg_score filter for the dataset

# ==============================================================================
# S11/S12 — PARTITION + PAD  (partition_train_dev_test.py, pad_silence_splits.py)
# ==============================================================================
SPLIT_SEED       = 42           # speaker-disjoint partition seed
SPLIT_THRESHOLD  = 0.4          # avg_score >= this at split-build time
SPLIT_DEV_MIN_S  = 30.0         # dev pool minimum (seconds)
PAD_SECONDS      = 0.5          # silence prepended AND appended per segment

# Bad-audio files excluded from the dataset — partition drops ALL their segments
# (independent of avg_score). 015: CONFIRMED bad audio (user-verified by listening,
# 2026-07-02); unrecoverable (resync re-localizes 0% — see docs/resync-fix.txt §8).
# NOT quarantined: 005/020 are recoverable localization DRIFT (fix via resync), not
# bad audio. File numbers (ints); matched via stem().
QUARANTINE_FILES = frozenset({15})

# ==============================================================================
# S13 — EVAL  (run_asr_baseline_eval.py)
# ==============================================================================
EVAL_MODEL_CTC   = "nvidia/parakeet-ctc-0.6b"           # Phase-15 fine-tune candidate
EVAL_MODEL_RNNT  = "nvidia/parakeet-rnnt-1.1b"          # independent (non-circular) judge
EVAL_DECODING    = "greedy_batch"
EVAL_BATCH_SIZE  = 32
EVAL_NUM_WORKERS = 8

# ==============================================================================
# FROZEN REFERENCE NUMBERS  (RP4/RP6 acceptance gate — must reproduce)
# ==============================================================================
FROZEN = {
    "ctc06b_padded_test_corpus_wer_6c":  0.3673,   # 36.73%
    "rnnt11b_padded_test_corpus_wer_6c": 0.3437,   # 34.37%
    "test_speakers": 20,
    "test_segments_6c": 3391,
}
