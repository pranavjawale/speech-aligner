"""Per-speaker ASR WER from an eval results JSONL, at avg_score thresholds.

Corpus WER (sum edits / sum ref words) per (file, speaker) and overall, including a
segment only if its avg_score >= T (T=0 includes all; avg_score=None kept at T=0).
Input rows carry file/speaker/ref_norm/hyp_norm/avg_score (from S13 on a segment
manifest). Faithful port of /workspace/code/.../per_speaker_wer_report.py.

Port source : per_speaker_wer_report.py
Reorg phase : RP5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def _wer(cell):
    return cell[0] / cell[1] if cell[1] else None


def build(results_path: Path, thr: list[float]) -> str:
    rows = [json.loads(l) for l in Path(results_path).open() if l.strip()]
    spk = defaultdict(lambda: {t: [0, 0, 0] for t in thr})
    overall = {t: [0, 0, 0] for t in thr}
    for r in rows:
        rw = (r.get("ref_norm") or "").split()
        hw = (r.get("hyp_norm") or "").split()
        if not rw:
            continue
        d = Levenshtein.distance(rw, hw)
        avg = r.get("avg_score")
        key = (r.get("file"), r.get("speaker"))
        for t in thr:
            if t <= 0 or (avg is not None and avg >= t):
                a = spk[key][t]; a[0] += d; a[1] += len(rw); a[2] += 1
                o = overall[t]; o[0] += d; o[1] += len(rw); o[2] += 1

    lines = ["=" * 92,
             f"PER-SPEAKER ASR WER by avg_score threshold   source: {Path(results_path).name}",
             f"thresholds (avg_score >=): {thr}   |   WER = corpus (sum edits / sum ref words)",
             "Speaker ids are per-file -> grouped by (file, speaker).", "=" * 92]
    hdr = f"{'file':<16}{'speaker':<12}"
    for t in thr:
        hdr += f"{'segs@'+str(t):>9}{'WER%@'+str(t):>10}"
    lines += [hdr, "-" * len(hdr)]
    for key in sorted(spk):
        fl, sp = key
        line = f"{str(fl):<16}{str(sp):<12}"
        for t in thr:
            c = spk[key][t]; w = _wer(c)
            line += f"{c[2]:>9}{(w*100 if w is not None else float('nan')):>9.2f}%"
        lines.append(line)
    lines.append("-" * len(hdr))
    tot = f"{'OVERALL':<16}{'':<12}"
    for t in thr:
        c = overall[t]; w = _wer(c)
        tot += f"{c[2]:>9}{(w*100 if w is not None else float('nan')):>9.2f}%"
    lines += [tot, ""]
    for t in thr:
        c = overall[t]; w = _wer(c)
        lines.append(f"  OVERALL WER @ avg_score>={t}: {w*100:.2f}%  "
                     f"({c[2]} segs, {c[1]} ref words)" if w is not None else
                     f"  OVERALL WER @ avg_score>={t}: n/a")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="eval results JSONL (from S13 on a segment manifest)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresholds", default="0,0.4")
    a = ap.parse_args()
    thr = [float(x) for x in a.thresholds.split(",")]
    text = build(Path(a.results), thr)
    out = Path(a.out) if a.out else Path(a.results).with_suffix(".per_speaker_wer.txt")
    out.write_text(text)
    print(text)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
