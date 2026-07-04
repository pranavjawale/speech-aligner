"""Canonical path builders (file number/stem -> artifact paths).

Single place that knows the on-disk layout, derived entirely from config. There
is NO strategy-N namespacing in code2 (6c is the only strategy, promoted to flat
paths) — one reason this is much simpler than the scattered path logic in /code.

Naming conventions preserved from /workspace/code:
  chunk wav : {stem}_chunk_{idx:06d}_start{start:06d}_end{end:06d}.wav
  manifests : {stem}_manifest.json, {stem}_speaker_manifest.json,
              {stem}_speaker_map.json, {stem}_segment_chunk_map.tsv
  per-chunk : {stem}_ctc_lm.jsonl (S3 hyp), {stem}_asr.jsonl (Whisper cache)
  timeline  : {stem}_timeline.json

Port source : scattered path logic across /code scripts (unified here)
Reorg phase : RP2
"""
from __future__ import annotations

from pathlib import Path

import config as cfg


def stem(file_no: int) -> str:
    """3 -> 'audio_file_003'."""
    return cfg.stem(file_no)


# ---------------------------------------------------------------- inputs
def source_wav(file_no: int) -> Path:
    return cfg.WAV_DIR / f"{stem(file_no)}.wav"


def transcript(file_no: int) -> Path:
    return cfg.TRANSCRIPTS_DIR / f"transcript_{file_no:03d}.txt"


# ---------------------------------------------------------------- S2 chunks
def chunk_dir(file_no: int) -> Path:
    return cfg.CHUNKS_DIR / stem(file_no)


def subchunk_dir(file_no: int) -> Path:
    """S5 gap-fill subchunk wavs. Honors cfg.SUBCHUNK_DIR (ablation isolation);
    default is chunks/<stem>/subchunks."""
    base = getattr(cfg, "SUBCHUNK_DIR", None)
    return (base / stem(file_no)) if base else (chunk_dir(file_no) / "subchunks")


def chunk_wav(file_no: int, idx: int, start_s: int, end_s: int) -> Path:
    return chunk_dir(file_no) / (
        f"{stem(file_no)}_chunk_{idx:06d}_start{start_s:06d}_end{end_s:06d}.wav"
    )


def chunk_manifest(file_no: int) -> Path:
    return cfg.MANIFEST_DIR / f"{stem(file_no)}_manifest.json"


# ---------------------------------------------------------------- S3/S4 per-chunk hyp
def ctc_lm_jsonl(file_no: int) -> Path:
    """S3 Parakeet+KenLM+unigrams hypotheses (one line per chunk)."""
    return chunk_dir(file_no) / f"{stem(file_no)}_ctc_lm.jsonl"


def asr_cache_jsonl(file_no: int) -> Path:
    """S4 Whisper-greedy ASR cache (skip re-decode on rerun)."""
    return chunk_dir(file_no) / f"{stem(file_no)}_asr.jsonl"


# ---------------------------------------------------------------- S4-S7 alignments
def align_dir(file_no: int) -> Path:
    return cfg.ALIGN_DIR / stem(file_no)


def align_json(file_no: int, chunk_file: str) -> Path:
    """Per-chunk alignment JSON. Named after the chunk wav (.wav -> .json), matching
    /code so downstream S5/S8 find them: {stem}_chunk_XXXXXX_startAAA_endBBB.json."""
    return align_dir(file_no) / chunk_file.replace(".wav", ".json")


def labels_dir(file_no: int) -> Path:
    """Audacity label files (labels / corrected1 / corrected2)."""
    return align_dir(file_no) / "labels"


# ---------------------------------------------------------------- S8 timeline
def timeline_json(file_no: int) -> Path:
    return cfg.ALIGN_DIR / f"{stem(file_no)}_timeline.json"


# ---------------------------------------------------------------- S9 speaker segments
def speaker_map(file_no: int) -> Path:
    return cfg.MANIFEST_DIR / f"{stem(file_no)}_speaker_map.json"


def speaker_manifest(file_no: int) -> Path:
    return cfg.MANIFEST_DIR / f"{stem(file_no)}_speaker_manifest.json"


def segment_chunk_map(file_no: int) -> Path:
    return cfg.MANIFEST_DIR / f"{stem(file_no)}_segment_chunk_map.tsv"


def segments_dir(file_no: int, *, padded: bool = False) -> Path:
    root = cfg.SEGMENTS_PAD_DIR if padded else cfg.SEGMENTS_DIR
    return root / stem(file_no)


# ---------------------------------------------------------------- S10-S13 asr track
def asr_manifest(kind: str, file_no: int | None = None) -> Path:
    """NeMo manifest JSONL. kind in {'chunks','segments'}; file_no=None -> the _all file.
    e.g. asr_manifest('segments', 2) -> data/asr/manifests/segments/segments_002.jsonl."""
    d = cfg.ASR_MANIFEST_DIR / kind
    return d / (f"{kind}_{file_no:03d}.jsonl" if file_no is not None else f"{kind}_all.jsonl")


def split_manifest(name: str, *, padded: bool = False) -> Path:
    """name in {'train','dev','test'}."""
    root = cfg.SPLITS_PAD_DIR if padded else cfg.SPLITS_DIR
    return root / f"{name}.jsonl"


def result_jsonl(name: str) -> Path:
    return cfg.RESULTS_DIR / f"{name}.jsonl"
