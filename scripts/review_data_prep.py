#!/usr/bin/env python3
"""review-data-prep — chunk↔speaker word-count mapping for manual review.

For the requested audio files and speaker ids, emit one file per speaker under
data/review/ listing every chunk that speaker spoke in and how many words they
spoke in that chunk (tab-separated).

Source of truth: data/manifests/audio_file_NNN_speaker_manifest.json (S9 output).
Chunks are speaker-agnostic, so the mapping is derived from each segment's
speaker_id + source_chunks + per-word timestamps. A segment that spans more than
one chunk has its words attributed to a chunk by each word's timestamp.

Usage:
  python scripts/review_data_prep.py --audio-files 2 20 --speakers spk_1012 spk_1001
  python scripts/review_data_prep.py --audio-files 1-24 --speakers spk_1019 --out data/review

Accepts audio file ids as ints (2), zero-padded (002), stems (audio_file_002),
or ranges (1-24). Speaker ids as given (spk_1012).

Standalone: inserts the repo root on sys.path so `import config` works without
sourcing setup_env.sh.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
import config as cfg  # noqa: E402


def parse_file_ids(tokens: list[str]) -> list[int]:
    """'2' '002' 'audio_file_002' '1-24' -> sorted unique ints."""
    ids: set[int] = set()
    for tok in tokens:
        t = str(tok).lower().replace("audio_file_", "").strip()
        if "-" in t:
            lo, hi = t.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(t))
    return sorted(ids)


def _chunk_of_word(word: dict, source_chunks: list[dict]) -> tuple[str, int]:
    """Attribute a word to (chunk_file, chunk_index) by its timestamp."""
    if len(source_chunks) == 1:
        sc = source_chunks[0]
        return sc["chunk_file"], sc["chunk_index"]
    t = word.get("start_abs")
    for sc in source_chunks:                       # word inside a chunk's overlap window
        if sc["overlap_start_abs"] <= t <= sc["overlap_end_abs"]:
            return sc["chunk_file"], sc["chunk_index"]
    sc = min(source_chunks,                         # fallback: nearest window edge
             key=lambda s: min(abs(t - s["overlap_start_abs"]),
                               abs(t - s["overlap_end_abs"])))
    return sc["chunk_file"], sc["chunk_index"]


def build_counts(file_ids: list[int], speakers: set[str]):
    """-> counts[speaker][chunk_file] = words ; and chunk_file -> (file_no, chunk_index)."""
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    chunk_meta: dict[str, tuple[int, int]] = {}
    missing = []
    for fno in file_ids:
        mpath = cfg.MANIFEST_DIR / f"{cfg.stem(fno)}_speaker_manifest.json"
        if not mpath.exists():
            missing.append(mpath.name)
            continue
        manifest = json.loads(mpath.read_text())
        for seg in manifest.get("segments", []):
            spk = seg["speaker_id"]
            if spk not in speakers:
                continue
            scs = seg["source_chunks"]
            for w in seg["words"]:
                cf, ci = _chunk_of_word(w, scs)
                counts[spk][cf] += 1
                chunk_meta[cf] = (fno, ci)
    return counts, chunk_meta, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-files", nargs="+", required=True,
                    help="file ids: 2 002 audio_file_002 or a range like 1-24")
    ap.add_argument("--speakers", nargs="+", required=True, help="speaker ids, e.g. spk_1012")
    ap.add_argument("--out", default=None, help="output dir (default: data/review)")
    a = ap.parse_args()

    file_ids = parse_file_ids(a.audio_files)
    speakers = set(a.speakers)
    out_dir = Path(a.out) if a.out else (cfg.DATA / "review")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts, chunk_meta, missing = build_counts(file_ids, speakers)
    if missing:
        print(f"[warn] no speaker manifest for: {', '.join(missing)}", file=sys.stderr)

    print(f"files={file_ids}  speakers={sorted(speakers)}  -> {out_dir}/")
    for spk in sorted(speakers):
        rows = sorted(counts.get(spk, {}).items(),
                      key=lambda kv: chunk_meta[kv[0]])       # (file_no, chunk_index)
        out_path = out_dir / f"{spk}.tsv"
        with out_path.open("w") as f:
            f.write("audio_file\tchunk_file\twords\n")
            for cf, n in rows:
                fno, _ = chunk_meta[cf]
                f.write(f"{cfg.stem(fno)}\t{cf}\t{n}\n")
        total_w = sum(n for _, n in rows)
        print(f"  {spk}: {len(rows)} chunks, {total_w} words -> {out_path}")


if __name__ == "__main__":
    main()
