"""Audio I/O + silence masking + padding.

  read_wav / write_wav  — thin soundfile wrappers (float32, 16 kHz mono, PCM_16).
  mask_silence          — zero VAD silence gaps >= threshold before forced align
                          (ported from align_chunks_withmask.py::mask_silence).
  pad_silence           — prepend AND append PAD_SECONDS of silence to a segment
                          (ported from data-for-asr-baseline/pad_silence_splits.py).

All sample-rate / threshold defaults come from config (SAMPLE_RATE,
SILENCE_MASK_THRESHOLD_S, PAD_SECONDS) — no magic numbers here.

Port source : align_chunks_withmask.py (mask_silence) + pad_silence_splits.py
Reorg phase : RP2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import config as cfg


def read_wav(path: str | Path) -> np.ndarray:
    """Read a wav as float32 mono. Asserts the expected sample rate."""
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != cfg.SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != expected {cfg.SAMPLE_RATE}")
    if audio.ndim > 1:            # collapse stereo -> mono (inputs are already mono)
        audio = audio.mean(axis=1)
    return audio


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    """Write float32 audio as 16-bit PCM at the configured sample rate."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, cfg.SAMPLE_RATE, subtype="PCM_16")


def mask_silence(
    audio: np.ndarray,
    speech_segments: list[dict],
    threshold_s: float | None = None,
) -> np.ndarray:
    """Return a copy of `audio` with silence gaps >= threshold_s zeroed out.

    speech_segments: chunk-relative [{start, end}] (sorted by start). Only leading,
    trailing, and inter-segment gaps exceeding the threshold are zeroed; short
    inter-word pauses are preserved so wav2vec2 keeps its boundary cues.
    """
    if threshold_s is None:
        threshold_s = cfg.SILENCE_MASK_THRESHOLD_S
    if not speech_segments:
        return audio                              # no VAD metadata -> pass through

    sr = cfg.SAMPLE_RATE
    masked = audio.copy()
    chunk_dur_s = len(audio) / sr

    # leading silence before first speech segment
    if speech_segments[0]["start"] >= threshold_s:
        masked[: int(speech_segments[0]["start"] * sr)] = 0.0

    # gaps between consecutive speech segments
    for i in range(len(speech_segments) - 1):
        gap_start_s = speech_segments[i]["end"]
        gap_end_s = speech_segments[i + 1]["start"]
        if gap_end_s - gap_start_s >= threshold_s:
            masked[int(gap_start_s * sr): int(gap_end_s * sr)] = 0.0

    # trailing silence after last speech segment
    if chunk_dur_s - speech_segments[-1]["end"] >= threshold_s:
        masked[int(speech_segments[-1]["end"] * sr):] = 0.0

    return masked


def pad_silence(audio: np.ndarray, pad_s: float | None = None) -> np.ndarray:
    """Prepend AND append `pad_s` seconds of silence (default config.PAD_SECONDS)."""
    if pad_s is None:
        pad_s = cfg.PAD_SECONDS
    pad_n = int(round(pad_s * cfg.SAMPLE_RATE))
    sil = np.zeros(pad_n, dtype=np.float32)
    return np.concatenate([sil, audio.astype(np.float32), sil])
