#!/usr/bin/env python3
"""speaker-wise WER table for an experiment, at a confidence-score threshold.

Input : an experiment metadata json (from asrpipe.asr.eval --exp-name). Reads its
        raw_results_csv (per-utterance rows with avg_score) so the score threshold
        can be applied — the metadata's own per_speaker block is UNfiltered.
Filter: keep segments with avg_score >= threshold (default 0.4, the dataset filter).
Output: data/review/<stem>.tsv with 4 tab-separated columns
          speaker_id   split(train/dev/test)   wer   duration_min
        aggregated per GLOBAL speaker_id (splits are assigned per speaker_id).

Usage:
  python scripts/speaker_wer_table.py \
     --metadata docs/experiment-logs/all-speakers_parakeetrnnt1.1b_baselineexpt1.metadata.json \
     --threshold 0.4
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "src"))
import config as cfg  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="experiment metadata json")
    ap.add_argument("--threshold", type=float, default=0.4, help="min avg_score (default 0.4)")
    ap.add_argument("--assignment", default=str(cfg.SPLITS_DIR / "speaker_assignment.json"))
    ap.add_argument("--out", default=None, help="output tsv (default data/review/<stem>.tsv)")
    a = ap.parse_args()

    meta = json.loads(Path(a.metadata).read_text())
    csv_path = meta.get("raw_results_csv")
    if not csv_path or not Path(csv_path).exists():
        sys.exit(f"raw_results_csv missing/unreadable in metadata: {csv_path}")
    assign = json.loads(Path(a.assignment).read_text())
    split_of = {spk: v["split"] if isinstance(v, dict) else v for spk, v in assign.items()}

    # aggregate per global speaker over segments with avg_score >= threshold
    edits = collections.Counter()
    refw = collections.Counter()
    dur = collections.Counter()
    kept = dropped = 0
    for r in csv.DictReader(open(csv_path)):
        try:
            score = float(r["avg_score"]) if r["avg_score"] not in ("", "None") else None
        except ValueError:
            score = None
        if score is None or score < a.threshold:
            dropped += 1
            continue
        kept += 1
        spk = r["speaker"]
        rw = r["ref_norm"].split()
        hw = r["hyp_norm"].split()
        if rw:
            edits[spk] += Levenshtein.distance(rw, hw)
            refw[spk] += len(rw)
        dur[spk] += float(r["duration"] or 0)

    out = Path(a.out) if a.out else (cfg.DATA / "review" / f"{Path(a.metadata).stem.replace('.metadata','')}_speaker_wer_conf{a.threshold}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spk in dur:
        wer = (edits[spk] / refw[spk]) if refw[spk] else None
        rows.append((spk, split_of.get(spk, "unassigned"), wer, dur[spk] / 60.0))
    rows.sort(key=lambda x: (x[2] if x[2] is not None else -1), reverse=True)   # worst WER first

    with out.open("w") as f:
        f.write("speaker_id\tset\twer\tduration_min\n")
        for spk, sp, wer, dmin in rows:
            f.write(f"{spk}\t{sp}\t{'' if wer is None else f'{wer:.4f}'}\t{dmin:.2f}\n")

    n_split = collections.Counter(sp for _, sp, _, _ in rows)
    print(f"threshold avg_score >= {a.threshold}: kept {kept} / dropped {dropped} segments")
    print(f"speakers: {len(rows)}  {dict(n_split)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
