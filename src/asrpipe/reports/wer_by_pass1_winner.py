"""Corpus WER of speaker segments, split by which ASR won their source pass-1 chunk.

Pipeline lineage this reconstructs:
  pass-1 localizes each VAD chunk using the best of two hypotheses
  (whisper vs parakeet-ctc+kenlm). The winning hyp's localization -> FA -> the
  speaker segments that fall inside that chunk's audio time window. So every
  scored segment inherits a "winner" (whisper / parakeet+kenlm / tie / single_option)
  from its source chunk. This tool buckets segments by that winner and reports
  corpus WER (sum edits / sum ref words), the same estimator asrpipe.asr.eval uses.

Winner per chunk is recomputed by replaying match_best_option in manifest order
(reproducing committed_pos), identical to the {stem}_hyp_winner_stats.json logic.
The winner *category* is defined by the per-option nds (nds[0]=whisper vs
nds[1]=parakeet), so it is independent of the tie-break rule (< vs <=).

Segment -> chunk is by max time-overlap: a segment's absolute [start,end]
(parsed from its wav filename) lies within its source chunk's [start_s,end_s].

Usage (main env, CPU-only):
  python3 -m asrpipe.reports.wer_by_pass1_winner --exp-name all-speakers_parakeetrnnt1.1b_disablePass1Extra
Reads results/<exp>.csv (per-utterance ref_norm/hyp_norm) + the on-disk pass-1
alignments. Writes results/<exp>.wer_by_pass1_winner.csv and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

import config as cfg
from asrpipe.common import paths as P

WINNERS = ["whisper", "parakeet+kenlm", "single_option"]
_SEG_TIME = re.compile(r"start(\d+)_end(\d+)")


def chunk_winners(file_no: int):
    """{chunk_index: (start_s, end_s, winner)} from the CHOSEN pass-1 hyp vs the cached options.

    Options are [whisper, parakeet+kenlm] (deduped, whisper first). The chunk's align JSON
    records asr_hypothesis = the option actually used. So:
      chosen == opts[0] -> 'whisper'         (whisper hyp used; includes score-ties, which the
                                              pipeline kept as whisper -> their segments used whisper)
      chosen == opts[1] -> 'parakeet+kenlm'  (parakeet hyp used)
      len(opts) < 2     -> 'single_option'   (both ASRs agreed / one empty -> no contest)
    O(1) per chunk (no matcher replay). Returns None if the file has no alignments."""
    adir = P.align_dir(file_no)
    jfs = sorted(glob.glob(f"{adir}/*_chunk_*.json"))
    if not jfs:
        return None
    cache = {}
    cpath = P.asr_cache_jsonl(file_no)
    if cpath.exists():
        for l in cpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                cache[r["chunk_index"]] = r["hyp_options"]
    out = {}
    for jf in jfs:
        d = json.load(open(jf))
        ci = d["chunk_index"]
        chosen = d.get("asr_hypothesis") or ""
        opts = cache.get(ci, [chosen])
        if len(opts) < 2:
            win = "single_option"
        elif chosen == opts[1]:
            win = "parakeet+kenlm"
        else:                                       # chosen == opts[0] (whisper), incl. ties
            win = "whisper"
        out[ci] = (float(d["chunk_start_s"]), float(d["chunk_end_s"]), win)
    return out


def _map_segment(seg_start, seg_end, chunks):
    """Return (winner, chunk_index) of the chunk with max time-overlap (nearest-center fallback).
    chunks: list of (start_s, end_s, winner, chunk_index)."""
    best, best_ov = None, 0.0
    for cs, ce, win, ci in chunks:
        ov = min(seg_end, ce) - max(seg_start, cs)
        if ov > best_ov:
            best_ov, best = ov, (win, ci)
    if best is not None:
        return best
    if not chunks:
        return (None, None)
    mid = 0.5 * (seg_start + seg_end)             # no overlap (gap) -> nearest chunk center
    cs, ce, win, ci = min(chunks, key=lambda c: abs(0.5 * (c[0] + c[1]) - mid))
    return (win, ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True, help="experiment name -> results/<exp>.csv")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=25, help="exclusive")
    a = ap.parse_args()

    csv_path = cfg.EXPT_RESULTS_DIR / f"{a.exp_name}.csv"
    rows = list(csv.DictReader(csv_path.open()))
    print(f"loaded {len(rows)} utterances from {csv_path}")

    # winners + chunk intervals per file
    quarantined = set(getattr(cfg, "QUARANTINE_FILES", frozenset()))
    winners_by_file, nchunks = {}, defaultdict(int)
    for fno in range(a.start, a.end):
        if fno in quarantined:                     # e.g. file_015 — excluded from the corpus
            continue
        cw = chunk_winners(fno)
        if cw is None:
            continue
        winners_by_file[fno] = [(cs, ce, w, ci) for ci, (cs, ce, w) in cw.items()]
        for _, _, w in cw.values():
            nchunks[w] += 1
    print("pass-1 chunk winners (replayed): " +
          "  ".join(f"{w}={nchunks[w]}" for w in WINNERS) +
          f"  total={sum(nchunks.values())}")

    # bucket segments -> corpus WER (+ collect per-segment detail for example queries)
    agg = {w: [0, 0, 0, 0.0] for w in WINNERS}     # [sum_edits, sum_ref_words, n_segs, dur_s]
    unmapped = empty_ref = 0
    tot_d = tot_n = 0
    detail = []
    for r in rows:
        fno = int(r["file"].rsplit("_", 1)[1])
        chunks = winners_by_file.get(fno)
        m = _SEG_TIME.search(Path(r["audio_filepath"]).name)
        if chunks is None or m is None:
            unmapped += 1
            continue
        ss, se = float(m.group(1)), float(m.group(2))
        win, src_ci = _map_segment(ss, se, chunks)
        if win is None:
            unmapped += 1
            continue
        rw = (r.get("ref_norm") or "").split()
        hw = (r.get("hyp_norm") or "").split()
        dur = float(r.get("duration", 0) or 0)
        agg[win][3] += dur
        agg[win][2] += 1
        seg_wer = None
        if not rw:                                 # empty-ref: not WER-scored (matches eval)
            empty_ref += 1
        else:
            d = Levenshtein.distance(rw, hw)
            agg[win][0] += d
            agg[win][1] += len(rw)
            tot_d += d
            tot_n += len(rw)
            seg_wer = d / len(rw)
        detail.append({
            "file": r["file"], "source_chunk_index": src_ci, "pass1_winner": win,
            "seg_start_s": int(ss), "seg_end_s": int(se), "speaker": r.get("speaker"),
            "seg_index": r.get("seg_index"), "duration": round(dur, 3),
            "wer": ("" if seg_wer is None else round(seg_wer, 4)),
            "ref_words": len(rw), "avg_score": r.get("avg_score"),
            "ref_norm": r.get("ref_norm", ""), "hyp_norm": r.get("hyp_norm", ""),
            "audio_filepath": r["audio_filepath"],
        })

    # ---- report ----
    print(f"\nunmapped segments: {unmapped}   empty-ref (unscored): {empty_ref}")
    print(f"overall corpus WER (validation, all buckets): {100*tot_d/tot_n:.2f}%  "
          f"(sum edits {tot_d} / ref words {tot_n})\n")
    hdr = f"{'winner':<16}{'chunks':>8}{'segments':>10}{'ref_words':>11}{'edits':>9}{'corpus_wer':>12}{'dur_min':>9}{'%ref_w':>8}"
    print(hdr)
    print("-" * len(hdr))
    out_rows = []
    for w in WINNERS:
        e, n, ns, dur = agg[w]
        cwer = 100 * e / n if n else 0.0
        pct = 100 * n / tot_n if tot_n else 0.0
        print(f"{w:<16}{nchunks[w]:>8}{ns:>10}{n:>11}{e:>9}{cwer:>11.2f}%{dur/60:>9.1f}{pct:>7.1f}%")
        out_rows.append({"pass1_winner": w, "n_chunks": nchunks[w], "n_segments": ns,
                         "ref_words": n, "edits": e, "corpus_wer_pct": round(cwer, 2),
                         "duration_min": round(dur / 60, 2), "pct_of_ref_words": round(pct, 1)})
    print("-" * len(hdr))
    e_all = sum(agg[w][0] for w in WINNERS)
    n_all = sum(agg[w][1] for w in WINNERS)
    ns_all = sum(agg[w][2] for w in WINNERS)
    d_all = sum(agg[w][3] for w in WINNERS)
    print(f"{'TOTAL':<16}{sum(nchunks.values()):>8}{ns_all:>10}{n_all:>11}{e_all:>9}"
          f"{100*e_all/n_all:>11.2f}%{d_all/60:>9.1f}{100.0:>7.1f}%")

    out_path = cfg.EXPT_RESULTS_DIR / f"{a.exp_name}.wer_by_pass1_winner.csv"
    with out_path.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)
        wtr.writerow({"pass1_winner": "TOTAL", "n_chunks": sum(nchunks.values()),
                      "n_segments": ns_all, "ref_words": n_all, "edits": e_all,
                      "corpus_wer_pct": round(100 * e_all / n_all, 2),
                      "duration_min": round(d_all / 60, 2), "pct_of_ref_words": 100.0})
    print(f"\nCSV -> {out_path}")

    # per-segment detail (queryable: filter by pass1_winner, sort by wer, pick examples)
    det_path = cfg.EXPT_RESULTS_DIR / f"{a.exp_name}.segment_pass1_winner.csv"
    with det_path.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        wtr.writeheader()
        wtr.writerows(detail)
    print(f"per-segment detail -> {det_path}  ({len(detail)} rows)")


if __name__ == "__main__":
    main()
