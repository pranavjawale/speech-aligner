"""S9 — speaker-wise audio segmentation from the word timeline.

  1. Parse the reference transcript -> per-word speaker map (ref_word_idx -> speaker).
  2. Join the timeline on ref_word_idx -> speaker-annotated words.
  3. Group consecutive same-speaker words into segments (split on speaker change or
     gap > SEGMENT_MERGE_GAP_S; drop segments shorter than MIN_SEG_DUR_S).
  4. Splice segment audio from the chunk WAVs; write per-segment WAVs.
  5. Write the speaker manifest with native enrichment: source_chunks, avg_score,
     min_score, scored_words.

Faithful port of /workspace/code/build_speaker_segments.py. NOTE: clean_ref here is
deliberately DISTINCT from textnorm (it strips <noise>/<...> tags and does not touch
spk_ labels, which parse_transcript_speakers handles line-wise) — ported verbatim so
the ref_word_idx join with the timeline is preserved exactly. The strategy-N
namespacing branch is dropped (code2 has a single canonical path).

Outputs: data/speaker-segments/{stem}/*.wav + data/manifests/{stem}_speaker_manifest.json

Port source : build_speaker_segments.py
Reorg phase : RP3
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

import config as cfg
from asrpipe.common import paths as P

SR = cfg.SAMPLE_RATE
N_WORKERS = int(os.environ.get("SPKSEG_WORKERS", "0")) or min(16, (os.cpu_count() or 4))

_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA_APOS = re.compile(r"[^a-zA-Z']")
_EDGE_APOS = re.compile(r"(?<!\w)'|'(?!\w)")
_WS = re.compile(r"\s+")
_SPK_LINE = re.compile(r"^spk_\w+:$")


def clean_ref(text: str) -> str:
    """Strip <noise>/<...> tags, keep within-word apostrophes, lowercase.
    Distinct from textnorm — must match /code's speaker-map tokenization exactly."""
    text = _TAG.sub(" ", text)
    text = _NON_ALPHA_APOS.sub(" ", text)
    text = _EDGE_APOS.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def parse_transcript_speakers(path) -> list[dict]:
    """Return [{ref_word_idx, speaker_id, word}] for the full transcript."""
    speaker_map = []
    global_idx = 0
    current_spk = None
    current_lines: list[str] = []

    def flush():
        nonlocal global_idx
        if current_spk is not None and current_lines:
            for w in clean_ref(" ".join(current_lines)).split():
                speaker_map.append({"ref_word_idx": global_idx,
                                    "speaker_id": current_spk, "word": w})
                global_idx += 1

    for raw_line in open(path, encoding="utf-8-sig"):
        line = raw_line.rstrip("\r\n").strip()
        if _SPK_LINE.match(line):
            flush()
            current_spk = line[:-1]
            current_lines = []
        elif line:
            current_lines.append(line)
    flush()
    return speaker_map


def annotate_timeline(timeline, speaker_map) -> list[dict]:
    idx_to_speaker = {e["ref_word_idx"]: e["speaker_id"] for e in speaker_map}
    return [{**w, "speaker_id": idx_to_speaker.get(w["ref_word_idx"], "UNKNOWN")}
            for w in timeline]


def group_segments(annotated) -> list[dict]:
    """Group consecutive same-speaker words; split on speaker change or gap>gap_s;
    drop segments shorter than MIN_SEG_DUR_S."""
    if not annotated:
        return []
    raw_segs = []
    cur = [annotated[0]]
    for w in annotated[1:]:
        prev = cur[-1]
        gap = w["start_abs"] - prev["end_abs"]
        if w["speaker_id"] == cur[0]["speaker_id"] and gap <= cfg.SEGMENT_MERGE_GAP_S:
            cur.append(w)
        else:
            raw_segs.append(cur)
            cur = [w]
    raw_segs.append(cur)

    segments = []
    seg_num = 0
    for words in raw_segs:
        dur = words[-1]["end_abs"] - words[0]["start_abs"]
        if dur < cfg.MIN_SEG_DUR_S:
            continue
        segments.append({
            "seg_index": seg_num, "speaker_id": words[0]["speaker_id"],
            "start_abs": words[0]["start_abs"], "end_abs": words[-1]["end_abs"],
            "duration": round(dur, 4), "word_count": len(words), "words": words,
        })
        seg_num += 1
    return segments


def build_chunk_index(file_no) -> list[dict]:
    manifest = json.loads(P.chunk_manifest(file_no).read_text())
    chunk_dir = P.chunk_dir(file_no)
    chunks = [{"chunk_index": c.get("chunk_index"), "chunk_file": c["chunk_file"],
               "start_s": c["start_s"], "end_s": c["end_s"],
               "wav_path": chunk_dir / c["chunk_file"]} for c in manifest["chunks"]]
    return sorted(chunks, key=lambda x: x["start_s"])


def extract_segment_audio(seg_start, seg_end, chunk_index) -> np.ndarray:
    """Splice audio for absolute [seg_start, seg_end] from the chunk WAVs
    (chunk WAV sample 0 = chunk start_s)."""
    pieces = []
    for c in chunk_index:
        if c["start_s"] >= seg_end or c["end_s"] <= seg_start:
            continue
        local_start = max(0.0, seg_start - c["start_s"])
        local_end = min(c["end_s"] - c["start_s"], seg_end - c["start_s"])
        if local_end <= local_start:
            continue
        try:
            audio, _ = sf.read(str(c["wav_path"]), dtype="float32",
                               start=int(local_start * SR), stop=int(local_end * SR))
            pieces.append(audio)
        except Exception as e:
            print(f"  [audio error] {c['wav_path'].name}: {e}", flush=True)
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)


def process_file(file_no: int) -> None:
    stem = P.stem(file_no)
    print(f"\n[{stem}] speaker segmentation", flush=True)

    timeline_path = P.timeline_json(file_no)
    if not timeline_path.exists():
        print(f"ERROR: {timeline_path} not found. Run S8 timeline first.", flush=True)
        return
    timeline = json.loads(timeline_path.read_text())

    spk_map = parse_transcript_speakers(P.transcript(file_no))
    annotated = annotate_timeline(timeline, spk_map)
    segments = group_segments(annotated)
    chunk_index = build_chunk_index(file_no)
    print(f"  timeline={len(timeline)} spk_words={len(spk_map)} segments={len(segments)}", flush=True)

    out_dir = P.segments_dir(file_no)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = list(out_dir.glob("*.wav"))
    for f in stale:
        f.unlink()

    def process_segment(seg):
        spk = seg["speaker_id"]
        fname = (f"{stem}_{spk}_seg_{seg['seg_index']:06d}"
                 f"_start{int(seg['start_abs']):06d}_end{int(seg['end_abs']):06d}.wav")
        audio = extract_segment_audio(seg["start_abs"], seg["end_abs"], chunk_index)
        if len(audio) == 0:
            return None
        sf.write(str(out_dir / fname), audio, SR)
        src_chunks = [
            {"chunk_index": c["chunk_index"], "chunk_file": c["chunk_file"],
             "chunk_start_s": round(c["start_s"], 4),
             "overlap_start_abs": round(max(seg["start_abs"], c["start_s"]), 4),
             "overlap_end_abs": round(min(seg["end_abs"], c["end_s"]), 4),
             "overlap_start_in_chunk": round(max(seg["start_abs"], c["start_s"]) - c["start_s"], 4),
             "overlap_end_in_chunk": round(min(seg["end_abs"], c["end_s"]) - c["start_s"], 4)}
            for c in chunk_index
            if not (c["start_s"] >= seg["end_abs"] or c["end_s"] <= seg["start_abs"])
        ]
        scores = [w["score"] for w in seg["words"] if w.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 4) if scores else None
        min_score = round(min(scores), 4) if scores else None
        return {
            "seg_index": seg["seg_index"], "speaker_id": seg["speaker_id"],
            "audio_file": fname, "start_abs": seg["start_abs"], "end_abs": seg["end_abs"],
            "duration": seg["duration"], "word_count": seg["word_count"],
            "avg_score": avg_score, "min_score": min_score, "scored_words": len(scores),
            "source_chunks": src_chunks,
            "words": [{"word": w["word"], "start_abs": w["start_abs"],
                       "end_abs": w["end_abs"], "score": w["score"]} for w in seg["words"]],
        }

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(process_segment, segments))
    manifest_segs = sorted((r for r in results if r is not None), key=lambda r: r["seg_index"])

    spk_stats: dict[str, dict] = {}
    for s in manifest_segs:
        st = spk_stats.setdefault(s["speaker_id"],
                                  {"segments": 0, "total_duration_s": 0.0, "total_words": 0})
        st["segments"] += 1
        st["total_duration_s"] += s["duration"]
        st["total_words"] += s["word_count"]

    manifest = {
        "source_file": f"{stem}.wav", "file_num": file_no,
        "chunk_strategy": "chunk-strategy-6c",
        "total_segments": len(manifest_segs),
        "max_gap_s": cfg.SEGMENT_MERGE_GAP_S, "min_seg_dur_s": cfg.MIN_SEG_DUR_S,
        "speaker_stats": {k: {**v, "total_duration_s": round(v["total_duration_s"], 2)}
                          for k, v in sorted(spk_stats.items())},
        "segments": manifest_segs,
    }
    P.speaker_manifest(file_no).write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {len(manifest_segs)} segment wavs + speaker manifest", flush=True)


def run_range(start: int, end: int) -> None:
    for file_no in range(start, end):
        process_file(file_no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=None)
    a = ap.parse_args()
    end = a.end if a.end is not None else a.start + 1
    run_range(a.start, end)


if __name__ == "__main__":
    main()
