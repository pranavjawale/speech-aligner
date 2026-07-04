#!/usr/bin/env python3
"""chunk-wise proxy-WER pattern across an audio file's duration + change-points.

Proxy-WER = ref_match.norm_edit_dist stored per chunk by S4 (align_p1) — the
normalized edit distance between the best ASR hypothesis and the matched reference
window (localization quality; 0=perfect, ~0.7+=garbage). It is NOT a decoded WER,
but it tracks alignment/audio quality per chunk and needs no GPU.

For each requested audio file we build (chunk_start_s, proxy_wer) pairs ordered by
time, smooth them, and detect where the proxy-WER SUDDENLY RISES (a clean recording
going bad) via hysteresis thresholding on the smoothed series.

Outputs (under data/review/):
  <stem>_chunk_proxywer.csv   chunk_index,start_s,end_s,proxy_wer,smoothed,region
  <stem>_chunk_proxywer.txt   detected low/high regions + rising-edge times + top jumps

Usage:
  python scripts/chunk_wer_pattern.py --audio-files 4
  python scripts/chunk_wer_pattern.py --audio-files 4 5 12 --smooth 3 --high 0.55 --low 0.45
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "src"))
import config as cfg  # noqa: E402


def load_series(fno: int):
    stem = cfg.stem(fno)
    rows = []
    for jf in glob.glob(str(cfg.ALIGN_DIR / stem / f"{stem}_chunk_*.json")):
        d = json.loads(Path(jf).read_text())
        rm = d.get("ref_match") or {}
        rows.append((d["chunk_start_s"], d["chunk_end_s"], d["chunk_index"],
                     rm.get("norm_edit_dist")))
    rows.sort()                                  # by chunk_start_s
    return stem, rows


def smooth(vals, w):
    """centered rolling mean, window w (odd)."""
    h = w // 2
    out = []
    for i in range(len(vals)):
        seg = [v for v in vals[max(0, i - h): i + h + 1] if v is not None]
        out.append(statistics.mean(seg) if seg else None)
    return out


def detect_regions(sm, hi, lo):
    """hysteresis: enter 'high' when smoothed>=hi, return to 'low' when smoothed<=lo."""
    regions, state, start = [], "low", 0
    for i, v in enumerate(sm):
        if v is None:
            continue
        if state == "low" and v >= hi:
            regions.append(("low", start, i - 1)); state, start = "high", i
        elif state == "high" and v <= lo:
            regions.append(("high", start, i - 1)); state, start = "low", i
    regions.append((state, start, len(sm) - 1))
    return [r for r in regions if r[2] >= r[1]]


def analyze(fno, out_dir, w, hi, lo, with_resync=False):
    stem, rows = load_series(fno)
    if not rows:
        print(f"{stem}: no chunk alignments found", file=sys.stderr); return
    vals = [r[3] for r in rows]
    sm = smooth(vals, w)

    # optional: old-vs-new proxy-WER from the re-sync plan (CPU-only, report-mode).
    plan = {}
    if with_resync:
        from asrpipe.align import resync
        plan = {p["chunk_index"]: p for p in resync.plan_file(fno)}

    csv_path = out_dir / f"{stem}_chunk_proxywer.csv"
    regions = detect_regions(sm, hi, lo)
    region_of = {}
    for kind, a, b in regions:
        for i in range(a, b + 1):
            region_of[i] = kind
    hdr = ["chunk_index", "start_s", "end_s", "proxy_wer", "smoothed", "region"]
    if with_resync:
        hdr += ["old_proxy_wer", "new_proxy_wer", "resync_offset", "resync_status"]
    with csv_path.open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(hdr)
        for i, (st, en, ci, nd) in enumerate(rows):
            base = [ci, f"{st:.1f}", f"{en:.1f}", "" if nd is None else f"{nd:.4f}",
                    "" if sm[i] is None else f"{sm[i]:.4f}", region_of.get(i, "")]
            if with_resync:
                p = plan.get(ci)
                base += ([f"{p['old_nd']:.4f}", f"{p['new_nd']:.4f}",
                          f"{p['offset']:+d}", p["status"]] if p else ["", "", "", ""])
            wtr.writerow(base)

    # rising edges = low->high transitions; report the high-region onset time
    rising = [rows[a][0] for k, a, b in regions if k == "high" and a > 0]
    # biggest single jumps in the smoothed series
    jumps = sorted(
        ((sm[i] - sm[i - 1], rows[i][0], rows[i][2]) for i in range(1, len(sm))
         if sm[i] is not None and sm[i - 1] is not None),
        reverse=True)[:5]

    txt_path = out_dir / f"{stem}_chunk_proxywer.txt"
    L = [f"{stem}: {len(rows)} chunks | proxy-WER = ref_match.norm_edit_dist "
         f"(smooth w={w}, hi={hi}, lo={lo})", ""]
    L.append("regions (contiguous smoothed low/high):")
    for k, a, b in regions:
        t0, t1 = rows[a][0], rows[b][1]
        mean_nd = statistics.mean(v for v in vals[a:b + 1] if v is not None)
        L.append(f"  {k:>4}  chunks {rows[a][2]:>3}..{rows[b][2]:<3}  "
                 f"t={t0/60:6.1f}..{t1/60:6.1f} min  meanWER={mean_nd:.3f}  ({b-a+1} chunks)")
    L.append("")
    if rising:
        L.append("SUDDEN INCREASES (proxy-WER rising edges), at chunk start time:")
        for t in rising:
            L.append(f"  -> t = {t:.1f} s  ({t/60:.1f} min)")
    else:
        L.append("no sustained rising edge detected (file uniformly low or high)")
    L.append("")
    L.append("top 5 single-chunk jumps (Δsmoothed):")
    for dv, t, ci in jumps:
        L.append(f"  chunk {ci:>3}  t={t/60:6.1f} min  Δ=+{dv:.3f}")

    if with_resync:
        n_res = sum(1 for p in plan.values() if p["status"] == "resynced")
        drift_old = [p["old_nd"] for p in plan.values() if p["status"] != "clean"]
        drift_new = [(p["new_nd"] if p["status"] == "resynced" else p["old_nd"])
                     for p in plan.values() if p["status"] != "clean"]
        L += ["", "per-chunk OLD vs NEW proxy-WER (re-sync report mode; ASR hyp unchanged, "
              "only the matched reference position moves):",
              f"  drift chunks={len(drift_old)}  resynced={n_res}  "
              f"mean proxy-WER {statistics.mean(drift_old):.3f} -> "
              f"{statistics.mean(drift_new):.3f}" if drift_old else "  (no drift)",
              "",
              f"{'idx':>5}{'t_min':>8}{'region':>8}{'old_pWER':>10}{'new_pWER':>10}"
              f"{'delta':>8}{'offset':>8}  status"]
        for i, (st, en, ci, nd) in enumerate(rows):
            p = plan.get(ci)
            if not p:
                continue
            new = p["new_nd"] if p["status"] == "resynced" else p["old_nd"]
            L.append(f"{ci:>5}{st/60:>8.1f}{region_of.get(i,''):>8}{p['old_nd']:>10.3f}"
                     f"{new:>10.3f}{new-p['old_nd']:>+8.3f}{p['offset']:>+8}  {p['status']}")
    txt = "\n".join(L)
    txt_path.write_text(txt + "\n")
    print(txt)
    print(f"\n-> {csv_path}\n-> {txt_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-files", nargs="+", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smooth", type=int, default=3, help="rolling-mean window (odd)")
    ap.add_argument("--high", type=float, default=0.55, help="enter-high threshold")
    ap.add_argument("--low", type=float, default=0.45, help="return-to-low threshold")
    ap.add_argument("--resync", action="store_true",
                    help="add old/new proxy-WER columns from the re-sync plan (CPU)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else (cfg.DATA / "review")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for tok in a.audio_files:
        t = str(tok).lower().replace("audio_file_", "")
        ids += (list(range(int(t.split('-')[0]), int(t.split('-')[1]) + 1)) if "-" in t else [int(t)])
    for fno in ids:
        analyze(fno, out_dir, a.smooth, a.high, a.low, a.resync)


if __name__ == "__main__":
    main()
