"""Export Audacity labels + enrich manifests with chunk-relative source-chunk times.

Three products, all derived from existing data (speaker manifest + timeline +
transcript speaker map + chunk manifest). No GPU, no re-alignment.

  (1) enrich_manifest  — add chunk_start_s + overlap_{start,end}_in_chunk to each
      segment's source_chunks, IN PLACE (S9 also stores these natively going forward).
  (2) segment_labels   — one Audacity label file per speaker-segment WAV, with
      SEGMENT-RELATIVE word times + confidence:
        data/speaker-segments/<stem>/labels/<segment>.txt
  (3) chunk_speakerseg — one combined Audacity label file per chunk (overlapping
      labels in a single track):
        data/alignments/<stem>/labels/<chunk>_speakerseg.txt
      containing BOTH:
        * per-word labels   :  start  end  word (score)
        * per-speaker labels :  start  end  [spk_XXXX] words-of-that-speaker
      all CHUNK-RELATIVE, sorted by start.

Usage: python -m asrpipe.segments.export_labels 1 25 [--no-enrich] [--no-seg] [--no-chunk]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.labels import write_labels
from asrpipe.segments.speaker_segments import parse_transcript_speakers, annotate_timeline


def _chunk_meta(file_no):
    """chunk_index -> chunk manifest entry (has start_s, end_s, chunk_file)."""
    chunks = json.loads(P.chunk_manifest(file_no).read_text())["chunks"]
    return {c["chunk_index"]: c for c in chunks}


def _timeline_path(file_no):
    return cfg.ALIGN_DIR / f"{P.stem(file_no)}_timeline.json"


def _fmt(st, en, label):
    return f"{st:.3f}\t{en:.3f}\t{label}"


# --------------------------------------------------------------------- (1)
def enrich_manifest(file_no) -> int:
    mp = P.speaker_manifest(file_no)
    if not mp.exists():
        return 0
    man = json.loads(mp.read_text())
    cm = _chunk_meta(file_no)
    n = 0
    for s in man.get("segments", []):
        for sc in s.get("source_chunks", []):
            cs = cm.get(sc["chunk_index"], {}).get("start_s")
            if cs is None:
                continue
            sc["chunk_start_s"] = round(cs, 4)
            sc["overlap_start_in_chunk"] = round(sc["overlap_start_abs"] - cs, 4)
            sc["overlap_end_in_chunk"] = round(sc["overlap_end_abs"] - cs, 4)
            n += 1
    mp.write_text(json.dumps(man, indent=1))
    return n


# --------------------------------------------------------------------- (2)
def segment_labels(file_no) -> int:
    mp = P.speaker_manifest(file_no)
    if not mp.exists():
        return 0
    man = json.loads(mp.read_text())
    ldir = P.segments_dir(file_no) / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in man.get("segments", []):
        seg0 = s["start_abs"]
        words = [{"start": round(w["start_abs"] - seg0, 3),
                  "end": round(w["end_abs"] - seg0, 3),
                  "word": w["word"], "score": w.get("score")}
                 for w in s["words"]]
        write_labels(words, ldir / f"{Path(s['audio_file']).stem}.txt")
        n += 1
    return n


# --------------------------------------------------------------------- (3)
def chunk_speakerseg(file_no) -> int:
    tl = _timeline_path(file_no)
    if not tl.exists():
        return 0
    annotated = annotate_timeline(json.loads(tl.read_text()),
                                  parse_transcript_speakers(P.transcript(file_no)))
    cm = _chunk_meta(file_no)
    ldir = P.labels_dir(file_no)
    ldir.mkdir(parents=True, exist_ok=True)
    by_chunk = {}
    for w in annotated:
        by_chunk.setdefault(w["chunk_index"], []).append(w)

    n = 0
    for ci, ws in by_chunk.items():
        c = cm.get(ci)
        if not c:
            continue
        cs = c["start_s"]
        ws = sorted(ws, key=lambda w: w["start_abs"])
        rows = []
        # per-word labels: word (score)
        for w in ws:
            sc = w.get("score")
            lab = w["word"] + (f" ({sc:.3f})" if sc is not None else "")
            rows.append((w["start_abs"] - cs, w["end_abs"] - cs, lab))
        # per-speaker labels: [spk] words  (contiguous same-speaker runs)
        run = [ws[0]]
        for w in ws[1:]:
            if w["speaker_id"] == run[-1]["speaker_id"]:
                run.append(w)
            else:
                rows.append(_spk_row(run, cs)); run = [w]
        rows.append(_spk_row(run, cs))
        rows.sort(key=lambda r: (r[0], r[1]))
        (ldir / f"{Path(c['chunk_file']).stem}_speakerseg.txt").write_text(
            "\n".join(_fmt(st, en, lab) for st, en, lab in rows) + ("\n" if rows else ""))
        n += 1
    return n


def _spk_row(run, cs):
    return (run[0]["start_abs"] - cs, run[-1]["end_abs"] - cs,
            f"[{run[0]['speaker_id']}] " + " ".join(w["word"] for w in run))


def run_range(start, end, do_enrich=True, do_seg=True, do_chunk=True):
    for n in range(start, end):
        if not P.speaker_manifest(n).exists():
            continue
        e = enrich_manifest(n) if do_enrich else 0
        sg = segment_labels(n) if do_seg else 0
        ck = chunk_speakerseg(n) if do_chunk else 0
        print(f"{P.stem(n)}: enriched {e} source-chunks | {sg} segment labels | "
              f"{ck} chunk _speakerseg files", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--no-chunk", action="store_true")
    a = ap.parse_args()
    run_range(a.start, a.end, not a.no_enrich, not a.no_seg, not a.no_chunk)


if __name__ == "__main__":
    main()
