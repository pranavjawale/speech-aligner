"""S2 — VAD-based audio splitting with speech sub-segment metadata.

pyannote VAD (whisperx built-in) -> merge into <=30s chunks. For each merged
chunk, the PRE-merge speech sub-segments are extracted from the raw VAD scores
(before merge_chunks) and stored as chunk-relative ``speech_segments`` in the
manifest. Downstream (S4) uses these to zero silence gaps >= threshold before
wav2vec2 forced alignment.

Faithful port of /workspace/code/split_strategy5.py. Differences (none affect
output): config-driven paths/params (no strategy-N namespace), the shared
common.audio_io writer, and the cross-file GPU/write pipelining latency-hack is
dropped (it never changed outputs). Chunk boundaries + speech_segments are
byte-identical to the strategy-5 manifest (verified on file_002).

Manifest written to  data/manifests/{stem}_manifest.json
Chunk wavs written to data/chunks/{stem}/{stem}_chunk_...wav

Port source : split_strategy5.py
Reorg phase : RP3
"""
from __future__ import annotations

import gc
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.vads.pyannote import Binarize, Pyannote, load_vad_model

import config as cfg
from asrpipe.common import paths as P

VAD_PARAMS = {                       # kept as a dict for the manifest (parity with /code)
    "vad_onset":        cfg.VAD_ONSET,
    "vad_offset":       cfg.VAD_OFFSET,
    "chunk_size":       cfg.VAD_CHUNK_SIZE,
    "min_duration_on":  cfg.VAD_MIN_DUR_ON,
    "min_duration_off": cfg.VAD_MIN_DUR_OFF,
}


def _device() -> str:
    return "cuda" if (cfg.DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"


# ---------------------------------------------------------------- speech segments
def extract_raw_speech(vad_scores) -> list[dict]:
    """Binarize raw VAD scores WITHOUT chunk-size splitting -> pre-merge speech segs.

    max_duration=inf so long speech regions are not split (unlike merge_chunks,
    which uses max_duration=chunk_size). onset/offset only; min_duration_on/off
    default to 0.0 as in merge_chunks.
    """
    binarize = Binarize(
        onset=cfg.VAD_ONSET,
        offset=cfg.VAD_OFFSET,
        max_duration=float("inf"),
    )
    annotation = binarize(vad_scores)
    segments = [
        {"start": round(seg.start, 4), "end": round(seg.end, 4)}
        for seg in annotation.get_timeline()
    ]
    return sorted(segments, key=lambda x: x["start"])


def chunk_speech_segments(raw_speech: list[dict], chunk_start: float,
                          chunk_end: float) -> list[dict]:
    """Speech sub-segments within [chunk_start, chunk_end], chunk-relative."""
    result = []
    for seg in raw_speech:
        overlap_start = max(seg["start"], chunk_start)
        overlap_end = min(seg["end"], chunk_end)
        if overlap_end > overlap_start:
            result.append({
                "start": round(overlap_start - chunk_start, 4),
                "end":   round(overlap_end - chunk_start, 4),
            })
    return result


# ---------------------------------------------------------------- VAD + merge
def vad_and_merge(wav_path: Path, vad_pipeline):
    """Run VAD, extract raw speech segments, merge into chunks.
    Returns (audio, merged_chunks, raw_speech_segments, file_duration)."""
    stem = wav_path.stem
    audio = load_audio(str(wav_path))
    file_duration = len(audio) / SAMPLE_RATE

    waveform = torch.from_numpy(audio).unsqueeze(0)
    vad_scores = vad_pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    del waveform

    raw_speech = extract_raw_speech(vad_scores)          # BEFORE merge_chunks
    merged = Pyannote.merge_chunks(
        vad_scores,
        chunk_size=cfg.VAD_CHUNK_SIZE,
        onset=cfg.VAD_ONSET,
        offset=cfg.VAD_OFFSET,
    )
    print(f"[{stem}] {len(raw_speech)} speech segs -> {len(merged)} chunks", flush=True)
    return audio, merged, raw_speech, file_duration


def _write_chunk(path: str, data: np.ndarray) -> None:
    sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")


def split_file(file_no: int, vad_pipeline, executor: ThreadPoolExecutor) -> dict:
    """Split one file: write chunk wavs + manifest. Returns stats dict."""
    wav_path = P.source_wav(file_no)
    stem = P.stem(file_no)
    audio, merged, raw_speech, file_duration = vad_and_merge(wav_path, vad_pipeline)

    out_dir = P.chunk_dir(file_no)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_info: list[dict] = []
    futures: list[Future] = []
    for idx, seg in enumerate(merged):
        start_s, end_s = seg["start"], seg["end"]
        chunk_path = P.chunk_wav(file_no, idx + 1, int(start_s), int(end_s))
        futures.append(executor.submit(
            _write_chunk, str(chunk_path),
            audio[int(start_s * SAMPLE_RATE): int(end_s * SAMPLE_RATE)],
        ))
        chunks_info.append({
            "chunk_index":     idx + 1,
            "chunk_file":      chunk_path.name,
            "start_s":         round(start_s, 4),
            "end_s":           round(end_s, 4),
            "duration_s":      round(end_s - start_s, 4),
            "speech_segments": chunk_speech_segments(raw_speech, start_s, end_s),
        })

    for f in futures:
        f.result()

    manifest_path = P.chunk_manifest(file_no)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_file":     str(wav_path),
        "file_duration_s": round(file_duration, 4),
        "total_chunks":    len(chunks_info),
        "vad_params":      VAD_PARAMS,
        "chunks":          chunks_info,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{stem}] manifest -> {manifest_path}", flush=True)

    del audio
    gc.collect()
    return {"stem": stem, "n_chunks": len(chunks_info), "file_duration": file_duration}


def split_range(start_file: int, end_file: int) -> list[dict]:
    """Split files [start_file, end_file) (end EXCLUSIVE)."""
    device = _device()
    print(f"Split S2: files {start_file:03d}-{end_file-1:03d}  device={device}", flush=True)
    vad_pipeline = load_vad_model(
        device=device,
        vad_onset=cfg.VAD_ONSET,
        vad_offset=cfg.VAD_OFFSET,
        token=cfg.HF_TOKEN or None,
    )
    stats = []
    with ThreadPoolExecutor(max_workers=cfg.IO_WORKERS) as executor:
        for n in range(start_file, end_file):
            if not P.source_wav(n).exists():
                print(f"WARN: {P.stem(n)}.wav not found — skipping", file=sys.stderr)
                continue
            t0 = time.time()
            s = split_file(n, vad_pipeline, executor)
            s["seconds"] = round(time.time() - t0, 1)
            stats.append(s)
    total = sum(s["n_chunks"] for s in stats)
    print(f"S2 done: {len(stats)} file(s), {total} chunks", flush=True)
    return stats


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1
    split_range(start, end)


if __name__ == "__main__":
    main()
