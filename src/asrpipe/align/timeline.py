"""S8 — build the per-file word timeline with absolute timestamps.

For each chunk, reads the final corrected2 label words (falls back corrected1 ->
labels), converts chunk-relative times to absolute (chunk_start + t), and assigns
each word a ref_word_idx = ref_match.word_start_idx + head_trim_offset + i, where
head_trim_offset = len(corrected1) - len(corrected2) (words the head-trim removed).

Output: data/alignments/{stem}_timeline.json
  [{word, start_abs, end_abs, score, ref_word_idx, chunk_index}]  (sorted by start)

Faithful port of /workspace/code/build_timeline.py (pure; no GPU).

Port source : build_timeline.py
Reorg phase : RP3
"""
from __future__ import annotations

import argparse
import json
import statistics

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.labels import parse_label_file


def build_timeline(file_no: int) -> list[dict]:
    stem = P.stem(file_no)
    aln_dir = P.align_dir(file_no)
    label_dir = P.labels_dir(file_no)
    json_files = sorted(aln_dir.glob("*chunk_*.json"))
    if not json_files:
        print(f"[{stem}] no alignment JSONs.", flush=True)
        return []

    timeline = []
    no_c2 = no_c1 = 0
    for jf in json_files:
        data = json.loads(jf.read_text())
        chunk_start = data["chunk_start_s"]
        chunk_idx = data["chunk_index"]
        ms = data["ref_match"]["word_start_idx"]
        chunk_stem = jf.stem

        c1_path = label_dir / f"{chunk_stem}_labels_corrected1.txt"
        c2_path = label_dir / f"{chunk_stem}_labels_corrected2.txt"
        lb_path = label_dir / f"{chunk_stem}_labels.txt"

        if c1_path.exists():
            c1_words = parse_label_file(c1_path)
        elif lb_path.exists():
            c1_words = parse_label_file(lb_path)
            no_c1 += 1
        else:
            c1_words = []
            no_c1 += 1

        if c2_path.exists():
            c2_words = parse_label_file(c2_path)
        else:
            c2_words = c1_words
            no_c2 += 1

        head_trim = max(0, len(c1_words) - len(c2_words))
        effective_start = ms + head_trim
        for i, w in enumerate(c2_words):
            timeline.append({
                "word": w["word"],
                "start_abs": round(chunk_start + w["start"], 4),
                "end_abs": round(chunk_start + w["end"], 4),
                "score": w["score"],
                "ref_word_idx": effective_start + i,
                "chunk_index": chunk_idx,
            })

    timeline.sort(key=lambda x: x["start_abs"])
    if no_c2:
        print(f"  note: {no_c2} chunks fell back from corrected2", flush=True)
    if no_c1:
        print(f"  note: {no_c1} chunks fell back from corrected1", flush=True)
    return timeline


def run_file(file_no: int) -> None:
    stem = P.stem(file_no)
    timeline = build_timeline(file_no)
    if not timeline:
        return
    out_path = P.timeline_json(file_no)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(timeline, indent=2))
    scores = [w["score"] for w in timeline if w["score"] is not None]
    print(f"[{stem}] timeline: {len(timeline)} words  "
          f"{timeline[0]['start_abs']:.1f}-{timeline[-1]['end_abs']:.1f}s  "
          f"conf median={statistics.median(scores):.3f} -> {out_path.name}"
          if scores else f"[{stem}] timeline: {len(timeline)} words", flush=True)


def run_range(start: int, end: int) -> None:
    for file_no in range(start, end):
        if P.align_dir(file_no).exists():
            run_file(file_no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=None)
    a = ap.parse_args()
    end = a.end if a.end is not None else a.start + 1
    run_range(a.start, end)


if __name__ == "__main__":
    main()
