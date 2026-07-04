"""S12 — silence-padded variant of the splits.

Prepend AND append PAD_SECONDS of silence to every train/dev/test segment wav
(originals untouched), producing a second version of the data (padding gave ~0.2pp
lower baseline WER). For each split manifest: pad each wav -> {wav_root}/{file}/{name},
emit a padded manifest row (audio_filepath -> padded path, duration += 2*pad, plus
pad_s and orig_audio_filepath; all other fields kept).

Faithful port of pad_silence_splits.py, using the shared audio_io padding helper.

Port source : pad_silence_splits.py
Reorg phase : RP4
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

import config as cfg
from asrpipe.common.audio_io import pad_silence


def run(splits_dir=None, out_splits=None, wav_root=None, pad=None,
        only=("train", "dev", "test"), workers=None) -> None:
    splits_dir = Path(splits_dir) if splits_dir else cfg.SPLITS_DIR
    out_splits = Path(out_splits) if out_splits else cfg.SPLITS_PAD_DIR
    wav_root = Path(wav_root) if wav_root else cfg.SEGMENTS_PAD_DIR
    pad = cfg.PAD_SECONDS if pad is None else pad
    workers = workers or min(16, os.cpu_count() or 4)
    out_splits.mkdir(parents=True, exist_ok=True)

    def pad_one(row):
        src = row["audio_filepath"]
        dst_dir = wav_root / row["file"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(src).name
        audio, _ = sf.read(src, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(str(dst), pad_silence(audio, pad), cfg.SAMPLE_RATE, subtype="PCM_16")
        new = dict(row)
        new["orig_audio_filepath"] = src
        new["audio_filepath"] = str(dst)
        new["duration"] = round(float(row["duration"]) + 2 * pad, 3)
        new["pad_s"] = pad
        return new

    for split in only:
        man = splits_dir / f"{split}.jsonl"
        if not man.exists():
            print(f"  skip {split}: no manifest {man}", flush=True)
            continue
        rows = [json.loads(l) for l in man.open() if l.strip()]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            padded = list(ex.map(pad_one, rows))
        with (out_splits / f"{split}.jsonl").open("w") as f:
            for r in padded:
                f.write(json.dumps(r) + "\n")
        dur = sum(r["duration"] for r in padded) / 3600
        print(f"{split}: {len(padded)} segs padded (+{pad*2:.1f}s each) -> "
              f"{out_splits / (split + '.jsonl')}  ({dur:.2f}h)", flush=True)
    print(f"padded wavs under: {wav_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out-splits", default=None)
    ap.add_argument("--wav-root", default=None)
    ap.add_argument("--pad", type=float, default=None)
    ap.add_argument("--only", default="train,dev,test")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(a.splits, a.out_splits, a.wav_root, a.pad,
        tuple(s.strip() for s in a.only.split(",")), a.workers)


if __name__ == "__main__":
    main()
