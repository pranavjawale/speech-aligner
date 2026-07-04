"""Speech-recovery report: speaker-mapped speech vs original audio vs VAD speech.

Per file + totals: original wav duration, VAD speech (sum chunk durations), and
speaker-mapped speech (sum segment durations), with the recovery ratios.

Faithful port of /workspace/code/speech_recovery_report.py.

Port source : speech_recovery_report.py
Reorg phase : RP5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

import config as cfg
from asrpipe.common import paths as P


def hms(s):
    s = int(round(s))
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"


def build(start: int, end: int) -> str:
    rows = []
    T_o = T_v = T_m = 0.0
    for n in range(start, end):
        wav, spk, cm = P.source_wav(n), P.speaker_manifest(n), P.chunk_manifest(n)
        if not (wav.exists() and spk.exists() and cm.exists()):
            continue
        orig = sf.info(str(wav)).duration
        vad = sum(float(c["duration_s"]) for c in json.loads(cm.read_text())["chunks"])
        mapped = sum(float(s["duration"]) for s in json.loads(spk.read_text())["segments"])
        rows.append((f"{n:03d}", orig, vad, mapped))
        T_o += orig; T_v += vad; T_m += mapped

    L = ["=" * 80,
         "SPEECH RECOVERY — speaker-mapped speech vs original / VAD-detected speech",
         "=" * 80,
         f"Files: {len(rows)}   (denominators: original wav, VAD speech = sum chunk durations,",
         " mapped = sum speaker-segment durations)", "",
         f"{'file':<6}{'orig':>10}{'VADspeech':>11}{'mapped':>10}{'VAD/orig':>9}{'map/VAD':>9}{'map/orig':>9}",
         "-" * 64]
    for n, o, v, m in rows:
        L.append(f"{n:<6}{hms(o):>10}{hms(v):>11}{hms(m):>10}{v/o*100:>8.1f}%{m/v*100:>8.1f}%{m/o*100:>8.1f}%")
    L.append("-" * 64)
    L.append(f"{'TOT':<6}{hms(T_o):>10}{hms(T_v):>11}{hms(T_m):>10}"
             f"{T_v/T_o*100:>8.1f}%{T_m/T_v*100:>8.1f}%{T_m/T_o*100:>8.1f}%")
    L += ["",
          f"original audio      : {hms(T_o)}  ({T_o/3600:.2f} h)",
          f"VAD-detected speech : {hms(T_v)}  ({T_v/3600:.2f} h)  = {T_v/T_o*100:.1f}% of audio",
          f"speaker-mapped      : {hms(T_m)}  ({T_m/3600:.2f} h)",
          f"  mapped / VAD speech : {T_m/T_v*100:.1f}%   <- recovery of DETECTED speech",
          f"  mapped / full audio : {T_m/T_o*100:.1f}%"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = build(a.start, a.end)
    out = Path(a.out) if a.out else cfg.REPORTS_DIR / "speech-recovery.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
