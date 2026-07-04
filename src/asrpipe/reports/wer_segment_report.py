"""Per-segment WER breakdown CSV + a 5-variant summary WER report (spec: docs/wer-spec.txt).

Companion to wer_error_analysis.py. Reads results/<exp>.csv (asrpipe.asr.eval per-utterance
output: normalized ref_norm/hyp_norm + avg_score) and, via jiwer word alignment, writes:

  results/<exp>.wer_segment_report.csv   one row per segment:
    file, speaker, seg_index, ref, hyp,
    C, S, D, I, T          (# correct, substitutions, deletions, insertions, total ref words)
    avg_score              (mean per-word alignment confidence for the segment)
    ref_number_words       (# number-related words in the reference)
    wrong_number_words     (# of those not correctly recognized -> substituted or deleted)

  results/<exp>.wer_variants_report.txt  corpus WER (sum edits / sum ref words) under 5 rules:
    1. usual WER
    2. ignore substitutions that differ only by a trailing 's' (plural / article-vs-articles)
    3. ignore reference sentences with only 1 word
    4. ignore 1-word refs  +  ignore plural (s) substitutions
    5. keep only refs that contain a number, then ignore 1-word refs + plural substitutions

A "number-related" word is a cardinal/ordinal word (one, two, hundred, thousand, first,
nineteenth, lakh, crore, ...) OR any token containing a digit. Word-level Levenshtein == S+D+I,
so variant 1 reproduces the experiment's reported corpus WER (a built-in check).

Run under the NeMo venv (jiwer 3.x), CPU-only:
  source setup_env.sh
  .venv-nemo/bin/python -m asrpipe.reports.wer_segment_report \
      --exp-name all-speakers_parakeetrnnt1.1b_baselineexpt1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import jiwer

try:
    import config as cfg
    RESULTS = Path(cfg.EXPT_RESULTS_DIR)
except Exception:
    RESULTS = Path("results")

try:                                    # for the source-chunk column (segment -> chunk by time)
    from asrpipe.common import paths as P
except Exception:
    P = None

_SEG_TIME = re.compile(r"start(\d+)_end(\d+)")

_CARDINAL = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion", "lakh", "crore", "dozen",
    "hundreds", "thousands", "millions", "billions", "lakhs", "crores", "dozens",
}
_ORDINAL = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth", "millionth",
}
_NUMBER_WORDS = _CARDINAL | _ORDINAL
_HAS_DIGIT = re.compile(r"\d")


def is_number_word(w: str) -> bool:
    return w in _NUMBER_WORDS or bool(_HAS_DIGIT.search(w))


def is_plural_sub(r: str, h: str) -> bool:
    """Substitution differing only by a trailing 's' (article<->articles, lord<->lords)."""
    return r != h and (r + "s" == h or h + "s" == r)


def build_chunk_maps(start=1, end=25):
    """{file_no: [(start_s, end_s, chunk_name), ...]} from the (config-independent) chunk
    manifests, for mapping each speaker segment back to its source pass-1 chunk by time."""
    maps = {}
    if P is None:
        return maps
    for fno in range(start, end):
        cm = P.chunk_manifest(fno)
        if not cm.exists():
            continue
        maps[fno] = [(float(c["start_s"]), float(c["end_s"]), Path(c["chunk_file"]).stem)
                     for c in json.loads(cm.read_text())["chunks"]]
    return maps


def source_chunk(seg_start, seg_end, chunks):
    """Name of the chunk with max time-overlap (a segment lies within its source chunk)."""
    best, best_ov = "", 0.0
    for cs, ce, name in chunks:
        ov = min(seg_end, ce) - max(seg_start, cs)
        if ov > best_ov:
            best_ov, best = ov, name
    if best or not chunks:
        return best
    mid = 0.5 * (seg_start + seg_end)                # gap fallback: nearest chunk center
    return min(chunks, key=lambda c: abs(0.5 * (c[0] + c[1]) - mid))[2]


def _chunk_stats(chunks, ref_w, hyp_w):
    """(C, S, D, I, sub_pairs, ref_correct) from one segment's jiwer alignment chunks."""
    C = S = D = I = 0
    sub_pairs = []
    ref_correct = [False] * len(ref_w)
    for ch in chunks:
        if ch.type == "equal":
            C += ch.ref_end_idx - ch.ref_start_idx
            for i in range(ch.ref_start_idx, ch.ref_end_idx):
                ref_correct[i] = True
        elif ch.type == "substitute":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                S += 1
                sub_pairs.append((ref_w[ch.ref_start_idx + k], hyp_w[ch.hyp_start_idx + k]))
        elif ch.type == "delete":
            D += ch.ref_end_idx - ch.ref_start_idx
        elif ch.type == "insert":
            I += ch.hyp_end_idx - ch.hyp_start_idx
    return C, S, D, I, sub_pairs, ref_correct


def _finalize(rec, seg, rw, C, S, D, I, sub_pairs, ref_correct):
    num_pos = [i for i, w in enumerate(rw) if is_number_word(w)]
    plural = sum(1 for rr, hh in sub_pairs if is_plural_sub(rr, hh))
    wrong_num = sum(1 for i in num_pos if not ref_correct[i])
    T = len(rw)
    rec.update(C=C, S=S, D=D, I=I, T=T,
               segment_wer=round((S + D + I) / T, 4), subst_ratio=round(S / T, 4),
               deletion_ratio=round(D / T, 4), insert_ratio=round(I / T, 4),
               ref_number_words=len(num_pos), wrong_number_words=wrong_num)
    seg.append({"T": T, "edits": S + D + I, "plural": plural, "has_num": bool(num_pos)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True, help="reads results/<exp>.csv")
    a = ap.parse_args()

    src = RESULTS / f"{a.exp_name}.csv"
    rows = list(csv.DictReader(src.open()))
    print(f"loaded {len(rows)} utterances from {src}")

    chunk_maps = build_chunk_maps()
    seg = []           # per-segment aggregates for the variant report
    out_rows = []      # per-segment CSV rows
    # jiwer errors on empty strings -> batch only both-non-empty pairs; handle the rest inline.
    b_ref, b_hyp, b_ctx = [], [], []               # b_ctx: (rec, rw) to fill from the batch
    for r in rows:
        rw = (r.get("ref_norm") or "").split()
        hw = (r.get("hyp_norm") or "").split()
        seg_file = Path(r.get("audio_filepath", "")).stem
        src_chunk = ""
        m = _SEG_TIME.search(seg_file)
        try:
            chunks = chunk_maps.get(int(r["file"].rsplit("_", 1)[1]))
        except (ValueError, KeyError, AttributeError):
            chunks = None
        if chunks and m:
            src_chunk = source_chunk(float(m.group(1)), float(m.group(2)), chunks)
        rec = {"ref": r.get("ref_norm", ""), "hyp": r.get("hyp_norm", ""),
               "segment_file": seg_file, "source_chunk": src_chunk,
               "avg_score": r.get("avg_score")}
        out_rows.append(rec)
        if not rw:                                  # empty ref -> not WER-scored (ratios undefined)
            rec.update(C=0, S=0, D=0, I=0, T=0, segment_wer="", subst_ratio="",
                       deletion_ratio="", insert_ratio="", ref_number_words=0,
                       wrong_number_words=0)
            continue
        if not hw:                                  # empty hyp -> every ref word deleted
            _finalize(rec, seg, rw, 0, 0, len(rw), 0, [], [False] * len(rw))
            continue
        b_ref.append(r["ref_norm"]); b_hyp.append(r["hyp_norm"]); b_ctx.append((rec, rw))

    out = jiwer.process_words(b_ref, b_hyp)
    for chunks, ref_w, hyp_w, (rec, rw) in zip(out.alignments, out.references, out.hypotheses, b_ctx):
        C, S, D, I, sub_pairs, ref_correct = _chunk_stats(chunks, ref_w, hyp_w)
        _finalize(rec, seg, rw, C, S, D, I, sub_pairs, ref_correct)

    # ---- per-segment CSV ----
    RESULTS.mkdir(exist_ok=True)
    cols = ["ref", "hyp", "segment_file", "source_chunk", "C", "S", "D", "I", "T",
            "segment_wer", "subst_ratio", "deletion_ratio", "insert_ratio",
            "avg_score", "ref_number_words", "wrong_number_words"]
    csv_path = RESULTS / f"{a.exp_name}.wer_segment_report.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    # ---- 5 WER variants (corpus = sum edits / sum ref words over included segments) ----
    def wer(rows_, ignore_plural):
        e = sum(s["edits"] - (s["plural"] if ignore_plural else 0) for s in rows_)
        n = sum(s["T"] for s in rows_)
        return e, n, (100 * e / n if n else 0.0), len(rows_)

    v1 = wer(seg, False)                                              # usual
    v2 = wer(seg, True)                                               # ignore plural
    ge2 = [s for s in seg if s["T"] >= 2]
    v3 = wer(ge2, False)                                             # drop 1-word refs
    v4 = wer(ge2, True)                                              # drop 1-word + plural
    numeric = [s for s in ge2 if s["has_num"]]                       # numbers-only, then v4
    v5 = wer(numeric, True)

    base = v1[2]
    lines = [
        f"WER variants report : {a.exp_name}",
        f"source              : {src}",
        f"segments (scored)   : {len(seg)}   (of {len(rows)}; {len(rows)-len(seg)} empty-ref excluded)",
        "",
        f"{'#':<3}{'variant':<52}{'WER':>8}{'segs':>8}{'ref_w':>10}{'edits':>9}{'dWER':>8}",
        "-" * 98,
    ]
    for n, label, v in [
        (1, "usual WER", v1),
        (2, "ignore plural-only (s) substitutions", v2),
        (3, "ignore 1-word reference sentences", v3),
        (4, "ignore 1-word refs + plural (s) substitutions", v4),
        (5, "numbers-only refs, then ignore 1-word + plural", v5),
    ]:
        e, nn, w_, ns = v
        lines.append(f"{n:<3}{label:<52}{w_:>7.2f}%{ns:>8}{nn:>10}{e:>9}{w_-base:>+7.2f}")
    lines += [
        "-" * 98,
        "",
        "notes:",
        "  WER = sum(S+D+I) / sum(T) over included segments (same estimator as asrpipe.asr.eval).",
        "  plural-only (s) sub = substitution where ref/hyp differ only by a trailing 's'",
        "    (article<->articles, lord<->lords); reclassified as correct (removed from edits).",
        "  number-related word = cardinal/ordinal (one, two, hundred, first, nineteenth, lakh,",
        "    crore, ...) OR any token containing a digit.",
        "  variant 5 = variant 4 restricted to references containing >=1 number-related word.",
    ]
    report = "\n".join(lines) + "\n"
    rpt_path = RESULTS / f"{a.exp_name}.wer_variants_report.txt"
    rpt_path.write_text(report)
    print("\n" + report)
    print(f"per-segment CSV -> {csv_path}  ({len(out_rows)} rows)")
    print(f"summary report  -> {rpt_path}")


if __name__ == "__main__":
    main()
