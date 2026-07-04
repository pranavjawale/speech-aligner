"""jiwer-based S/D/I error analysis over all segments of an eval experiment.

For every segment it word-aligns hyp against ref (jiwer.process_words) and records
each error operation:
  S  substitution  ref_word -> hyp_word
  D  deletion      ref_word present in ref, missing from hyp
  I  insertion     hyp_word present in hyp, absent from ref
Then it ranks the most prominent substitutions / deletions / insertions across the
whole corpus and dumps a raw per-operation CSV (op, ref_word, hyp_word listed first).

Input is results/<exp>.csv (written by asrpipe.asr.eval): its ref_norm / hyp_norm
columns are already normalized with asrpipe.common.wer.normalize, so the alignment
is byte-consistent with the reported corpus WER. Word-level Levenshtein == S+D+I, so
sum(S+D+I)/sum(ref_words) reproduces the experiment's corpus WER (a built-in check).

Run under the NeMo venv (has jiwer 3.x), CPU-only:
  source setup_env.sh
  .venv-nemo/bin/python -m asrpipe.reports.wer_error_analysis \
      --exp-name all-speakers_parakeetrnnt1.1b_disablePass1Extra
Outputs (results/<exp>.*):
  wer_alignment_raw.csv     one row per S/D/I op (op, ref_word, hyp_word, file, speaker, seg_index)
  wer_top_substitutions.csv ref_word,hyp_word,count,pct_of_subs,pct_of_edits
  wer_top_deletions.csv     ref_word,count,pct_of_dels,pct_of_edits
  wer_top_insertions.csv    hyp_word,count,pct_of_ins,pct_of_edits
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import jiwer

try:
    import config as cfg
    RESULTS = Path(cfg.EXPT_RESULTS_DIR)
except Exception:                       # config not importable -> assume repo-root cwd
    RESULTS = Path("results")

OP_ORDER = {"S": 0, "D": 1, "I": 2}     # for "S, D, I listed first" raw ordering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True, help="reads results/<exp>.csv")
    ap.add_argument("--top", type=int, default=30, help="rows to print per error type")
    a = ap.parse_args()

    src = RESULTS / f"{a.exp_name}.csv"
    rows = list(csv.DictReader(src.open()))
    print(f"loaded {len(rows)} utterances from {src}")

    S = Counter()          # (ref_word, hyp_word) -> count
    D = Counter()          # ref_word -> count
    I = Counter()          # hyp_word -> count
    raw = []               # (op, ref_word, hyp_word, file, speaker, seg_index)
    tot_ref = 0
    n_scored = empty_ref = empty_hyp = 0

    # jiwer errors on empty strings, so batch only the both-non-empty pairs;
    # empty-hyp rows are all-deletions (handled inline), empty-ref rows are unscored.
    b_ref, b_hyp, b_meta = [], [], []
    for r in rows:
        rw = (r.get("ref_norm") or "").split()
        hw = (r.get("hyp_norm") or "").split()
        meta = (r.get("file"), r.get("speaker"), r.get("seg_index"))
        if not rw:
            empty_ref += 1
            continue
        n_scored += 1
        tot_ref += len(rw)
        if not hw:
            empty_hyp += 1
            for w in rw:
                D[w] += 1
                raw.append(("D", w, "", *meta))
            continue
        b_ref.append(r["ref_norm"]); b_hyp.append(r["hyp_norm"]); b_meta.append(meta)

    out = jiwer.process_words(b_ref, b_hyp)
    for chunks, ref_w, hyp_w, meta in zip(out.alignments, out.references, out.hypotheses, b_meta):
        for ch in chunks:
            if ch.type == "equal":
                continue
            if ch.type == "substitute":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    rwd, hwd = ref_w[ch.ref_start_idx + k], hyp_w[ch.hyp_start_idx + k]
                    S[(rwd, hwd)] += 1
                    raw.append(("S", rwd, hwd, *meta))
            elif ch.type == "delete":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    rwd = ref_w[ch.ref_start_idx + k]
                    D[rwd] += 1
                    raw.append(("D", rwd, "", *meta))
            elif ch.type == "insert":
                for k in range(ch.hyp_end_idx - ch.hyp_start_idx):
                    hwd = hyp_w[ch.hyp_start_idx + k]
                    I[hwd] += 1
                    raw.append(("I", "", hwd, *meta))

    totS, totD, totI = sum(S.values()), sum(D.values()), sum(I.values())
    edits = totS + totD + totI
    # cross-check vs jiwer's own totals on the batched subset (+ inline empty-hyp deletions)
    assert totS == out.substitutions and totI == out.insertions, "S/I mismatch vs jiwer"

    print(f"\nscored {n_scored} segments  (empty-ref {empty_ref}, empty-hyp {empty_hyp})")
    print(f"ref words = {tot_ref}")
    print(f"  substitutions S = {totS:>6}  ({100*totS/edits:5.1f}% of edits)")
    print(f"  deletions     D = {totD:>6}  ({100*totD/edits:5.1f}% of edits)")
    print(f"  insertions    I = {totI:>6}  ({100*totI/edits:5.1f}% of edits)")
    print(f"  total edits     = {edits:>6}")
    print(f"CORPUS WER = (S+D+I)/ref = {100*edits/tot_ref:.2f}%   "
          f"[S {100*totS/tot_ref:.2f} + D {100*totD/tot_ref:.2f} + I {100*totI/tot_ref:.2f}]")

    def _print(title, items, fmt):
        print(f"\n=== top {a.top} {title} ===")
        for key, c in items[:a.top]:
            print(f"  {c:>5}  {100*c/edits:4.1f}%  {fmt(key)}")

    _print("substitutions (ref -> hyp)", S.most_common(), lambda k: f"{k[0]!r} -> {k[1]!r}")
    _print("deletions (ref word dropped)", D.most_common(), lambda k: repr(k))
    _print("insertions (spurious hyp word)", I.most_common(), lambda k: repr(k))

    # ---- write CSVs ----
    RESULTS.mkdir(exist_ok=True)
    raw.sort(key=lambda t: OP_ORDER[t[0]])          # S rows, then D, then I
    raw_path = RESULTS / f"{a.exp_name}.wer_alignment_raw.csv"
    with raw_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["op", "ref_word", "hyp_word", "file", "speaker", "seg_index"])
        w.writerows(raw)

    sub_path = RESULTS / f"{a.exp_name}.wer_top_substitutions.csv"
    with sub_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ref_word", "hyp_word", "count", "pct_of_subs", "pct_of_edits"])
        for (rwd, hwd), c in S.most_common():
            w.writerow([rwd, hwd, c, round(100*c/totS, 3), round(100*c/edits, 3)])

    del_path = RESULTS / f"{a.exp_name}.wer_top_deletions.csv"
    with del_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ref_word", "count", "pct_of_dels", "pct_of_edits"])
        for wd, c in D.most_common():
            w.writerow([wd, c, round(100*c/totD, 3), round(100*c/edits, 3)])

    ins_path = RESULTS / f"{a.exp_name}.wer_top_insertions.csv"
    with ins_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hyp_word", "count", "pct_of_ins", "pct_of_edits"])
        for wd, c in I.most_common():
            w.writerow([wd, c, round(100*c/totI, 3), round(100*c/edits, 3)])

    print(f"\nraw alignment ({len(raw)} ops) -> {raw_path}")
    print(f"top substitutions ({len(S)} distinct) -> {sub_path}")
    print(f"top deletions     ({len(D)} distinct) -> {del_path}")
    print(f"top insertions    ({len(I)} distinct) -> {ins_path}")


if __name__ == "__main__":
    main()
