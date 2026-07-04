"""Per-speaker speech-recovery report at one or more avg_score thresholds.

Speaker ids are per-file -> grouped by (file, speaker). Two tables: cumulative by
speaker_id across all files, and per-(file, speaker). For each, the mapped duration
retained at each threshold (keep segments with avg_score >= T; T=0 keeps all).

Faithful port of /workspace/code/speaker_duration_report.py.

Port source : speaker_duration_report.py
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


def _keep(avg, t):
    if t <= 0:
        return True
    return avg is not None and avg >= t


def build(start: int, end: int, thr: list[float]) -> str:
    rows = {}
    files_seen = []
    for n in range(start, end):
        p = P.speaker_manifest(n)
        if not p.exists():
            continue
        stem = P.stem(n)
        files_seen.append(stem)
        segs = json.loads(p.read_text())
        segs = segs["segments"] if isinstance(segs, dict) else segs
        for s in segs:
            key = (stem, s["speaker_id"])
            d = float(s["duration"]); avg = s.get("avg_score")
            r = rows.setdefault(key, {t: 0.0 for t in thr} | {"_segs": 0})
            r["_segs"] += 1
            for t in thr:
                if _keep(avg, t):
                    r[t] += d

    lines = []
    # cumulative by speaker_id
    glob = {}
    for (stem, sp), r in rows.items():
        g = glob.setdefault(sp, {t: 0.0 for t in thr} | {"_segs": 0, "_files": set()})
        g["_segs"] += r["_segs"]; g["_files"].add(stem)
        for t in thr:
            g[t] += r[t]
    lines += ["=" * 78,
              "CUMULATIVE SPEAKER DURATION — all files combined (by speaker_id)",
              f"thresholds(avg_score >=): {thr}   distinct speaker_ids: {len(glob)}",
              "NOTE: speaker ids are per-file; the same id across files is merged here.",
              "=" * 78]
    ghdr = f"{'speaker':<12}{'files':>6}{'segs':>8}" + "".join(f"{'>='+str(t):>12}" for t in thr)
    if len(thr) == 2:
        ghdr += f"{'retain%':>9}"
    lines += [ghdr, "-" * len(ghdr)]
    gtot = {t: 0.0 for t in thr}; gseg = 0
    for sp in sorted(glob, key=lambda s: -glob[s][thr[0]]):
        g = glob[sp]; gseg += g["_segs"]
        line = f"{sp:<12}{len(g['_files']):>6}{g['_segs']:>8}" + "".join(f"{hms(g[t]):>12}" for t in thr)
        if len(thr) == 2:
            b = g[thr[0]]
            line += f"{(g[thr[1]]/b*100 if b else 0):>8.1f}%"
        lines.append(line)
        for t in thr:
            gtot[t] += g[t]
    lines.append("-" * len(ghdr))
    gline = f"{'TOTAL':<12}{'':>6}{gseg:>8}" + "".join(f"{hms(gtot[t]):>12}" for t in thr)
    if len(thr) == 2 and gtot[thr[0]]:
        gline += f"{gtot[thr[1]]/gtot[thr[0]]*100:>8.1f}%"
    lines += [gline, ""]

    # per-(file, speaker)
    lines += ["=" * 78, "PER-(FILE, SPEAKER) SPEECH RECOVERY BY CONFIDENCE THRESHOLD",
              f"files: {len(files_seen)}  speakers(file,spk): {len(rows)}  "
              f"thresholds(avg_score >=): {thr}",
              "Speaker ids are per-file -> grouped by (file, speaker).", "=" * 78]
    hdr = f"{'file':<16}{'speaker':<12}{'segs':>6}" + "".join(f"{'>='+str(t):>12}" for t in thr)
    if len(thr) == 2:
        hdr += f"{'retain%':>9}"
    lines += [hdr, "-" * len(hdr)]
    grand = {t: 0.0 for t in thr}; gseg = 0
    for (stem, spk) in sorted(rows):
        r = rows[(stem, spk)]; gseg += r["_segs"]
        line = f"{stem:<16}{spk:<12}{r['_segs']:>6}" + "".join(f"{hms(r[t]):>12}" for t in thr)
        if len(thr) == 2:
            base = r[thr[0]]
            line += f"{(r[thr[1]]/base*100 if base else 0):>8.1f}%"
        lines.append(line)
        for t in thr:
            grand[t] += r[t]
    lines.append("-" * len(hdr))
    tot = f"{'TOTAL':<16}{'':<12}{gseg:>6}" + "".join(f"{hms(grand[t]):>12}" for t in thr)
    if len(thr) == 2 and grand[thr[0]]:
        tot += f"{grand[thr[1]]/grand[thr[0]]*100:>8.1f}%"
    lines += [tot, ""]
    for t in thr:
        lines.append(f"  total recovered @ avg_score>={t}: {hms(grand[t])} ({grand[t]/3600:.2f} h)")
    if len(thr) == 2 and grand[thr[0]]:
        drop = grand[thr[0]] - grand[thr[1]]
        lines.append(f"  dropped by raising {thr[0]}->{thr[1]}: {hms(drop)} "
                     f"({drop/grand[thr[0]]*100:.1f}% of total)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresholds", default="0,0.4")
    a = ap.parse_args()
    thr = [float(x) for x in a.thresholds.split(",")]
    text = build(a.start, a.end, thr)
    out = Path(a.out) if a.out else cfg.REPORTS_DIR / "speaker-duration-report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
