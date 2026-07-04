"""Confidence-filter report: avg_score histogram (0.1 bins) + threshold trade-offs.

Segment counts and cumulative duration% per avg_score bin, plus a drop/keep table
for thresholds 0.3-0.7. Faithful port of /workspace/code/conf_filter_report.py.

Port source : conf_filter_report.py
Reorg phase : RP5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as cfg
from asrpipe.common import paths as P


def hms(s):
    s = int(round(s))
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"


def build(start: int, end: int) -> str:
    bc = [0] * 10; bd = [0.0] * 10
    none_c = 0; none_d = 0.0; tc = 0; td = 0.0; nfiles = 0
    for n in range(start, end):
        p = P.speaker_manifest(n)
        if not p.exists():
            continue
        nfiles += 1
        for s in json.loads(p.read_text())["segments"]:
            a = s.get("avg_score"); d = float(s["duration"]); tc += 1; td += d
            if a is None:
                none_c += 1; none_d += d; continue
            b = min(int(a * 10), 9); bc[b] += 1; bd[b] += d

    L = ["=" * 78,
         "FILTERING SPEAKER SEGMENTS BY AVERAGE ALIGNMENT CONFIDENCE (avg_score)",
         "=" * 78,
         f"Files: {nfiles}   segments: {tc}   total mapped: {hms(td)} ({td/3600:.2f} h)"]
    if none_c:
        L.append(f"(avg_score=None: {none_c} segs, {hms(none_d)})")
    L += ["cum-dur% is bottom-up (share of duration at/below the bin upper edge).", "",
          f"{'bin':<10}{'segs':>8}{'seg%':>7}{'dur':>10}{'dur%':>7}{'cum-seg%':>9}{'cum-dur%':>9}",
          "-" * 60]
    cc = 0; cd = 0.0
    for i in range(10):
        cc += bc[i]; cd += bd[i]
        L.append(f"{i/10:.1f}-{(i+1)/10:.1f}".ljust(10)
                 + f"{bc[i]:>8}{bc[i]/tc*100:>6.1f}%{hms(bd[i]):>10}{bd[i]/td*100:>6.1f}%"
                 + f"{cc/tc*100:>8.1f}%{cd/td*100:>8.1f}%")
    L.append("-" * 60)
    L.append(f"{'TOTAL':<10}{tc:>8}{'100%':>7}{hms(td):>10}{'100%':>7}")
    L += ["", "THRESHOLD TRADE-OFF (drop segments with avg_score < T):"]
    cum = 0.0; cumlist = []
    for i in range(10):
        cum += bd[i]; cumlist.append(cum)
    for t in [3, 4, 5, 6, 7]:
        dropped = cumlist[t - 1] / td * 100
        L.append(f"  keep avg_score >= 0.{t}:  dropped {dropped:>5.1f}%   kept {100-dropped:>5.1f}%")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = build(a.start, a.end)
    out = Path(a.out) if a.out else cfg.REPORTS_DIR / "conf-filter.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
