"""Audacity label-file I/O (the per-chunk word list with CTC confidence).

Line format:  "{start:.3f}\\t{end:.3f}\\t{word}[ (score)]"
Shared by S5 (writes labels), S6/S7 (confidence trimming), and S8 (timeline).
Ported verbatim from the parse_label_file / write_labels helpers duplicated across
/workspace/code (confidence_*_realign.py, build_timeline.py).

Port source : confidence_trim_realign.py / build_timeline.py (dedup)
Reorg phase : RP3
"""
from __future__ import annotations

import re
from pathlib import Path

_LABEL_RE = re.compile(r"([\d.]+)\t([\d.]+)\t(.+)")
_SCORE_RE = re.compile(r"\(([0-9.]+)\)")
_SCORE_STRIP = re.compile(r"\s*\([0-9.]+\)")


def parse_label_file(path: Path) -> list[dict]:
    """Parse a label file -> [{start, end, word, score}]."""
    words = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LABEL_RE.match(line)
        if not m:
            continue
        rest = m.group(3)
        sm = _SCORE_RE.search(rest)
        score = float(sm.group(1)) if sm else None
        word = _SCORE_STRIP.sub("", rest).strip()
        words.append({"start": float(m.group(1)), "end": float(m.group(2)),
                      "word": word, "score": score})
    return words


def write_labels(words: list[dict], out_path: Path) -> None:
    """Write [{start, end, word, score?}] as an Audacity label file."""
    lines = []
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is not None and e is not None:
            score = w.get("score")
            score_str = f" ({score:.3f})" if score is not None else ""
            lines.append(f"{s:.3f}\t{e:.3f}\t{w['word']}{score_str}")
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""))
