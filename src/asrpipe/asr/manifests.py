"""S10 — build NeMo manifest JSONL files from the pipeline outputs.

Two evaluation units:
  chunks/   one row per VAD chunk; text = corrected2 timeline words grouped by
            chunk_index (apples-to-apples with the per-chunk WER table).
  segments/ one row per speaker segment; text = the segment's words, carrying
            speaker/seg_index/avg_score/min_score inline (segments feed S11 partition).

Row: {audio_filepath, duration, text, file, unit, [chunk_index | speaker/seg_index/
      avg_score/min_score]}. Empty-reference rows are skipped (WER undefined).

audio_filepath points at code2 data paths (so eval reads code2 wavs) — this is the
one field that legitimately differs from /code's manifests. Faithful port of
build_asr_baseline_manifests.py otherwise.

Port source : build_asr_baseline_manifests.py
Reorg phase : RP4
"""
from __future__ import annotations

import argparse
import json

import config as cfg
from asrpipe.common import paths as P


def join_words(words) -> str:
    return " ".join(w["word"].strip() for w in words if w.get("word", "").strip())


def build_chunk_rows(file_no: int):
    """One row per chunk: text = corrected2 timeline words grouped by chunk_index."""
    stem = P.stem(file_no)
    tl_path = P.timeline_json(file_no)
    if not tl_path.exists():
        print(f"  [chunks] no timeline for {stem}; skipping", flush=True)
        return [], 0
    timeline = json.loads(tl_path.read_text())
    by_chunk: dict = {}
    for w in timeline:
        by_chunk.setdefault(w["chunk_index"], []).append(w)

    rows, skipped = [], 0
    for jf in sorted(P.align_dir(file_no).glob("*chunk_*.json")):
        d = json.loads(jf.read_text())
        ci = d["chunk_index"]
        text = join_words(by_chunk.get(ci, []))
        wav = P.chunk_dir(file_no) / d["chunk_file"]
        if not text or not wav.exists():
            skipped += 1
            continue
        rows.append({
            "audio_filepath": str(wav),
            "duration": round(float(d["chunk_end_s"]) - float(d["chunk_start_s"]), 3),
            "text": text, "file": stem, "unit": "chunk", "chunk_index": ci,
        })
    return rows, skipped


def build_segment_rows(file_no: int):
    """One row per speaker segment, from the speaker manifest."""
    stem = P.stem(file_no)
    man_path = P.speaker_manifest(file_no)
    if not man_path.exists():
        print(f"  [segments] no speaker manifest for {stem}; skipping", flush=True)
        return [], 0
    man = json.loads(man_path.read_text())
    segs = man["segments"] if isinstance(man, dict) else man
    seg_dir = P.segments_dir(file_no)
    rows, skipped = [], 0
    for s in segs:
        text = join_words(s["words"])
        wav = seg_dir / s["audio_file"]
        if not text or not wav.exists():
            skipped += 1
            continue
        rows.append({
            "audio_filepath": str(wav),
            "duration": round(float(s["duration"]), 3),
            "text": text, "file": stem, "unit": "segment",
            "speaker": s["speaker_id"], "seg_index": s["seg_index"],
            "avg_score": s.get("avg_score"), "min_score": s.get("min_score"),
        })
    return rows, skipped


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def run_range(start: int, end: int) -> None:
    quarantined = getattr(cfg, "QUARANTINE_FILES", set())
    all_chunks, all_segs = [], []
    for n in range(start, end):
        if n in quarantined:
            print(f"{P.stem(n)}: QUARANTINED (config.QUARANTINE_FILES) — skipped", flush=True)
            continue
        crows, cskip = build_chunk_rows(n)
        srows, sskip = build_segment_rows(n)
        write_jsonl(P.asr_manifest("chunks", n), crows)
        write_jsonl(P.asr_manifest("segments", n), srows)
        all_chunks += crows
        all_segs += srows
        nspk = len({r["speaker"] for r in srows})
        print(f"{P.stem(n)}: chunks {len(crows)} (skip {cskip})  "
              f"segments {len(srows)} (skip {sskip})  speakers {nspk}", flush=True)
    write_jsonl(P.asr_manifest("chunks"), all_chunks)
    write_jsonl(P.asr_manifest("segments"), all_segs)
    dur_c = sum(r["duration"] for r in all_chunks) / 3600
    dur_s = sum(r["duration"] for r in all_segs) / 3600
    print(f"\nTOTAL: chunks {len(all_chunks)} rows ({dur_c:.2f}h)  "
          f"segments {len(all_segs)} rows ({dur_s:.2f}h)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)   # exclusive
    a = ap.parse_args()
    run_range(a.start, a.end)


if __name__ == "__main__":
    main()
