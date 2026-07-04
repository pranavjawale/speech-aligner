"""Number-recognition accuracy for an ASR eval (e.g. baseline vs a fine-tune).

Reuses the canonical number-word definition + jiwer word alignment from
asrpipe.reports.wer_segment_report, and additionally counts NUMBER INSERTIONS.

A "number word" = a cardinal/ordinal token (one, two, hundred, first, nineteenth, lakh,
crore, dozen, ...) OR any token containing a digit  (is_number_word, imported verbatim).
Alignment = jiwer.process_words on the eval CSV's already-normalized ref_norm/hyp_norm.
  - a REF number word is "wrong" if it is NOT in an 'equal' chunk -> substituted or deleted.
  - a NUMBER INSERTION = a hyp number word inside an 'insert' chunk (no ref counterpart).
    (A non-number->number substitution is a substitution, not an insertion; not counted.)

Run (NeMo venv; needs jiwer/rapidfuzz):
  .venv-nemo/bin/python -m asrpipe.reports.number_recognition \
      --exps finaltestpadded_parakeetctc0.6b_base finaltestpadded_parakeetctc0.6b_ft
Reads results/<exp>.csv for each; validated to reproduce ref_number_words/wrong_number_words
in the committed results/<exp>.wer_segment_report.csv.
"""
from __future__ import annotations

import argparse
import csv

import jiwer

from asrpipe.reports.wer_segment_report import is_number_word

try:
    from asrpipe.common import paths as P
    RESULTS = P.RESULTS if hasattr(P, "RESULTS") else None
except Exception:  # pragma: no cover
    RESULTS = None


def _csv_path(exp: str) -> str:
    return f"results/{exp}.csv"


def analyse(exp: str) -> dict:
    """Number-recognition stats for one experiment's per-segment eval CSV."""
    rows = list(csv.DictReader(open(_csv_path(exp))))
    total_segments = len(rows)
    ref_num_total = 0      # total number words in the reference (metric 3)
    segs_with_num = 0      # segments containing >=1 ref number word (metric 2)
    wrong_num = 0          # ref number words substituted or deleted
    num_ins = 0            # number insertions

    b_ref, b_hyp, b_ctx = [], [], []
    for r in rows:
        rw = (r.get("ref_norm") or "").split()
        hw = (r.get("hyp_norm") or "").split()
        npos = [i for i, w in enumerate(rw) if is_number_word(w)]
        ref_num_total += len(npos)
        if npos:
            segs_with_num += 1
        if not rw:
            continue
        if not hw:                       # empty hyp -> every ref word (incl. numbers) deleted
            wrong_num += len(npos)
            continue
        b_ref.append(r["ref_norm"]); b_hyp.append(r["hyp_norm"]); b_ctx.append((rw, hw, npos))

    out = jiwer.process_words(b_ref, b_hyp)
    for chunks, ref_w, hyp_w, (rw, hw, npos) in zip(out.alignments, out.references,
                                                    out.hypotheses, b_ctx):
        ref_correct = [False] * len(ref_w)
        for ch in chunks:
            if ch.type == "equal":
                for i in range(ch.ref_start_idx, ch.ref_end_idx):
                    ref_correct[i] = True
            elif ch.type == "insert":
                for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    if is_number_word(hyp_w[j]):
                        num_ins += 1
        wrong_num += sum(1 for i in npos if not ref_correct[i])

    return dict(exp=exp, total_segments=total_segments, segs_with_num=segs_with_num,
                ref_num_total=ref_num_total, wrong_num=wrong_num, num_ins=num_ins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exps", nargs="+", required=True,
                    help="one or more experiment names (reads results/<exp>.csv each)")
    a = ap.parse_args()

    res = [analyse(e) for e in a.exps]

    # metrics 1-3 are properties of the reference/test set: identical across experiments
    ref0 = res[0]
    print("=" * 72)
    print("NUMBER-RECOGNITION ACCURACY")
    print("=" * 72)
    print(f"1. total segments in test set ............... {ref0['total_segments']}")
    print(f"2. segments containing >=1 number .......... {ref0['segs_with_num']}")
    print(f"3. total number words in reference ......... {ref0['ref_num_total']}")
    for d in res:
        if (d["total_segments"], d["ref_num_total"]) != (ref0["total_segments"], ref0["ref_num_total"]):
            print(f"   !! WARNING: {d['exp']} has a different test set/reference "
                  f"(segs={d['total_segments']}, ref_nums={d['ref_num_total']})")
    print("-" * 72)
    print(f"{'experiment':<40} {'wrong%(S+D)':>12} {'num_ins':>9}")
    for d in res:
        pct = 100 * d["wrong_num"] / d["ref_num_total"] if d["ref_num_total"] else 0.0
        print(f"{d['exp']:<40} {pct:10.2f}%  ({d['wrong_num']}/{d['ref_num_total']})"
              .ljust(66) + f"{d['num_ins']:>6}")
    print("=" * 72)


if __name__ == "__main__":
    main()
