"""S6/S7 — confidence-based boundary trimming (max-contrast partition).

Two symmetric passes that remove spurious extra reference words the forced aligner
was forced onto silence / already-covered frames (near-zero CTC confidence):

  S6 tail_pass: reads the Pass-2 JSON word scores, finds the split maximising
     left_avg - right_avg (bad TAIL), trims matched_text, re-aligns the shortened
     reference -> {chunk}_labels_corrected1.txt.
  S7 head_pass: reads corrected1, finds the split maximising right_avg - left_avg
     (bad HEAD), trims the leading words, re-aligns -> {chunk}_labels_corrected2.txt.

Both use an adaptive quality gate (skip if top-half confidence < TOP_HALF_MIN) so
uniformly poor chunks aren't falsely trimmed. Faithful port of
confidence_trim_realign.py + confidence_head_trim_realign.py (single-sourced here).

Port source : confidence_trim_realign.py + confidence_head_trim_realign.py
Reorg phase : RP3
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import torch
import whisperx

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.audio_io import mask_silence
from asrpipe.common.labels import parse_label_file, write_labels


def _device() -> str:
    return "cuda" if (cfg.DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"


def _prefix(scores):
    p = [0.0] * (len(scores) + 1)
    for i, s in enumerate(scores):
        p[i + 1] = p[i] + s
    return p


def find_tail_trim_point(scores):
    """Keep words[:trim_at], discard the low-confidence tail. Returns (trim_at|None, info)."""
    n = len(scores)
    info = {"n": n}
    if n < cfg.MIN_KEEP_WORDS + cfg.MIN_TRIM_WORDS:
        return None, {**info, "reason": "too_few_words"}
    top_half_conf = statistics.mean(sorted(scores, reverse=True)[: max(1, n // 2)])
    info["top_half_conf"] = round(top_half_conf, 4)
    if top_half_conf < cfg.TOP_HALF_MIN:
        return None, {**info, "reason": "top_half_too_low"}
    guard = min(cfg.RIGHT_AVG_MAX, cfg.RIGHT_AVG_ADAPTIVE_K * top_half_conf)
    prefix = _prefix(scores)
    best_i, best_c = cfg.MIN_KEEP_WORDS, -999.0
    for i in range(cfg.MIN_KEEP_WORDS, n - cfg.MIN_TRIM_WORDS + 1):
        c = prefix[i] / i - (prefix[n] - prefix[i]) / (n - i)
        if c > best_c:
            best_c, best_i = c, i
    right_avg = (prefix[n] - prefix[best_i]) / (n - best_i)
    info.update({"split": best_i, "contrast": round(best_c, 4),
                 "right_avg": round(right_avg, 4), "adaptive_guard": round(guard, 4)})
    if best_c > cfg.CONTRAST_THRESH and right_avg < guard:
        return best_i, {**info, "reason": "trim", "trim_words": n - best_i}
    return None, {**info, "reason": "no_trim"}


def find_head_trim_point(scores):
    """Discard words[:trim_at] (low-confidence head), keep the rest. Returns (trim_at|None, info)."""
    n = len(scores)
    info = {"n": n}
    if n < cfg.MIN_KEEP_WORDS + cfg.MIN_TRIM_WORDS:
        return None, {**info, "reason": "too_few_words"}
    top_half_conf = statistics.mean(sorted(scores, reverse=True)[: max(1, n // 2)])
    info["top_half_conf"] = round(top_half_conf, 4)
    if top_half_conf < cfg.TOP_HALF_MIN:
        return None, {**info, "reason": "top_half_too_low"}
    guard = min(cfg.RIGHT_AVG_MAX, cfg.RIGHT_AVG_ADAPTIVE_K * top_half_conf)
    prefix = _prefix(scores)
    best_i, best_c = cfg.MIN_TRIM_WORDS, -999.0
    for i in range(cfg.MIN_TRIM_WORDS, n - cfg.MIN_KEEP_WORDS + 1):
        c = (prefix[n] - prefix[i]) / (n - i) - prefix[i] / i
        if c > best_c:
            best_c, best_i = c, i
    left_avg = prefix[best_i] / best_i
    info.update({"split": best_i, "contrast": round(best_c, 4),
                 "left_avg": round(left_avg, 4), "adaptive_guard": round(guard, 4)})
    if best_c > cfg.CONTRAST_THRESH and left_avg < guard:
        return best_i, {**info, "reason": "trim", "trim_words": best_i}
    return None, {**info, "reason": "no_trim"}


def _speech_map(file_no):
    manifest = json.loads(P.chunk_manifest(file_no).read_text())
    return {c["chunk_file"]: c.get("speech_segments", []) for c in manifest["chunks"]}


def _realign(text, chunk_wav, speech_segs, chunk_dur, align_model, align_metadata):
    audio = whisperx.load_audio(str(chunk_wav))
    masked = mask_silence(audio, speech_segs)
    try:
        aligned = whisperx.align([{"text": text, "start": 0.0, "end": chunk_dur}],
                                 align_model, align_metadata, masked, _device(),
                                 return_char_alignments=False)
        return aligned.get("word_segments", [])
    except Exception as e:
        print(f"  [align error] {chunk_wav.name}: {e}", flush=True)
        return []


def tail_pass(file_no, align_model, align_metadata):
    """S6: tail trim -> corrected1 label files."""
    stem = P.stem(file_no)
    aln_dir = P.align_dir(file_no)
    json_files = sorted(aln_dir.glob("*chunk_*.json"))
    if not json_files:
        print(f"[{stem}] no JSONs — skipping tail trim.", flush=True)
        return
    label_dir = P.labels_dir(file_no)
    label_dir.mkdir(parents=True, exist_ok=True)
    speech_map = _speech_map(file_no)
    trimmed = 0
    t0 = time.time()
    for jf in json_files:
        data = json.loads(jf.read_text())
        words = data.get("words", [])
        rm = data["ref_match"]
        chunk_stem = jf.stem
        chunk_dur = data["chunk_end_s"] - data["chunk_start_s"]
        scores = [w.get("score", 0.5) for w in words if w.get("start") is not None]
        trim_at, info = (None, {"reason": "ablated"}) if cfg.ABLATE_TRIM else find_tail_trim_point(scores)
        c1 = label_dir / f"{chunk_stem}_labels_corrected1.txt"
        if trim_at is None:
            existing = label_dir / f"{chunk_stem}_labels.txt"
            if existing.exists():
                c1.write_text(existing.read_text())
            else:
                write_labels(words, c1)
            continue
        trimmed += 1
        new_text = " ".join(rm["matched_text"].split()[:trim_at])
        speech = speech_map.get(data.get("chunk_file", f"{chunk_stem}.wav"), [])
        new_words = _realign(new_text, P.chunk_dir(file_no) / f"{chunk_stem}.wav",
                             speech, chunk_dur, align_model, align_metadata)
        write_labels(new_words, c1)
    print(f"[{stem}] tail trim done ({time.time()-t0:.1f}s): {trimmed}/{len(json_files)} trimmed", flush=True)


def head_pass(file_no, align_model, align_metadata):
    """S7: head trim (reads corrected1) -> corrected2 label files."""
    stem = P.stem(file_no)
    aln_dir = P.align_dir(file_no)
    json_files = sorted(aln_dir.glob("*chunk_*.json"))
    if not json_files:
        print(f"[{stem}] no JSONs — skipping head trim.", flush=True)
        return
    label_dir = P.labels_dir(file_no)
    speech_map = _speech_map(file_no)
    trimmed = 0
    t0 = time.time()
    for jf in json_files:
        data = json.loads(jf.read_text())
        chunk_stem = jf.stem
        chunk_dur = data["chunk_end_s"] - data["chunk_start_s"]
        c1 = label_dir / f"{chunk_stem}_labels_corrected1.txt"
        c2 = label_dir / f"{chunk_stem}_labels_corrected2.txt"
        src = c1 if c1.exists() else label_dir / f"{chunk_stem}_labels.txt"
        c1_words = parse_label_file(src) if src.exists() else []
        scores = [w["score"] for w in c1_words if w.get("score") is not None]
        trim_at, info = (None, {"reason": "ablated"}) if cfg.ABLATE_TRIM else find_head_trim_point(scores)
        if trim_at is None:
            c2.write_text(src.read_text() if src.exists() else "")
            continue
        trimmed += 1
        kept = c1_words[trim_at:]
        new_text = " ".join(w["word"] for w in kept)
        speech = speech_map.get(data.get("chunk_file", f"{chunk_stem}.wav"), [])
        new_words = _realign(new_text, P.chunk_dir(file_no) / f"{chunk_stem}.wav",
                             speech, chunk_dur, align_model, align_metadata)
        write_labels(new_words, c2)
    print(f"[{stem}] head trim done ({time.time()-t0:.1f}s): {trimmed}/{len(json_files)} trimmed", flush=True)


def run_range(start: int, end: int) -> None:
    dev = _device()
    print(f"Conf-trim S6/S7: files {start:03d}-{end-1:03d}  device={dev}", flush=True)
    align_model, align_metadata = whisperx.load_align_model(language_code=cfg.LANGUAGE, device=dev)
    for file_no in range(start, end):
        if not P.align_dir(file_no).exists():
            continue
        try:
            tail_pass(file_no, align_model, align_metadata)   # S6 -> corrected1
            head_pass(file_no, align_model, align_metadata)   # S7 -> corrected2
        except Exception as e:
            print(f"ERROR file {file_no:03d}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        gc.collect()
    print("S6/S7 done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    a = ap.parse_args()
    run_range(a.start, a.end)


if __name__ == "__main__":
    main()
