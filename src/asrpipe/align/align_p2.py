"""S5 — Pass-2 gap-fill alignment (with silence masking).

Chunks where the speaker is faster than the session-average WPS exhaust their
reference window early, leaving 3-8s of audio at the tail (or head) with no word
timestamps. This pass detects tail/head gaps >= GAP_THRESHOLD_S, extracts the
gap sub-audio, silence-masks it, and re-runs wav2vec2 forced alignment on the
extra reference words — merging the new (pass=2) words back into the chunk JSON.

Faithful port of /workspace/code/align_chunks_pass2_withmask.py. Note the quirk
preserved verbatim: ref_wps here uses the FULL chunk duration_s (not VAD speech
duration as in S4). Also regenerates the Audacity label files.

Reads/writes data/alignments/{stem}/{chunk}.json in place; labels -> that dir's
labels/ subdir; sub-chunk wavs -> data/chunks/{stem}/subchunks/.

Port source : align_chunks_pass2_withmask.py
Reorg phase : RP3
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import soundfile as sf
import torch
import whisperx

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.audio_io import mask_silence
from asrpipe.align import matcher

SR = cfg.SAMPLE_RATE


def _device() -> str:
    return "cuda" if (cfg.DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"


def subchunk_speech_segments(chunk_speech, sub_start_s, sub_end_s) -> list[dict]:
    """Intersect chunk-level speech_segments with [sub_start_s, sub_end_s];
    return sub-chunk-relative segments."""
    result = []
    for seg in chunk_speech:
        overlap_start = max(seg["start"], sub_start_s)
        overlap_end = min(seg["end"], sub_end_s)
        if overlap_end > overlap_start:
            result.append({"start": round(overlap_start - sub_start_s, 4),
                           "end": round(overlap_end - sub_start_s, 4)})
    return result


def write_labels(words: list[dict], out_path) -> None:
    lines = []
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is not None and e is not None:
            score = w.get("score")
            score_str = f" ({score:.3f})" if score is not None else ""
            lines.append(f"{s:.3f}\t{e:.3f}\t{w['word']}{score_str}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def process_file(file_no, align_model, align_metadata) -> None:
    stem = P.stem(file_no)
    aln_dir = P.align_dir(file_no)
    if not aln_dir.exists():
        print(f"[{stem}] no pass-1 output — skipping.", flush=True)
        return
    json_files = sorted(aln_dir.glob("*chunk_*.json"))
    if not json_files:
        print(f"[{stem}] no JSON files — skipping.", flush=True)
        return

    label_dir = P.labels_dir(file_no)
    label_dir.mkdir(parents=True, exist_ok=True)
    subchunk_dir = P.subchunk_dir(file_no)
    subchunk_dir.mkdir(parents=True, exist_ok=True)

    _, ref_words = matcher.load_reference(file_no)          # natural (alignment) list

    manifest = json.loads(P.chunk_manifest(file_no).read_text())
    total_speech_s = sum(c["duration_s"] for c in manifest["chunks"])   # FULL dur (quirk)
    ref_wps = len(ref_words) / total_speech_s if total_speech_s > 0 else cfg.FALLBACK_WPS
    speech_seg_map = {c["chunk_file"]: c.get("speech_segments", []) for c in manifest["chunks"]}

    chunk_ref_bounds = {}
    for jf in json_files:
        try:
            d = json.loads(jf.read_text())
            idx = d.get("chunk_index")
            rm = d.get("ref_match", {})
            if idx is not None and rm:
                chunk_ref_bounds[idx] = {"start": rm.get("word_start_idx"),
                                         "end": rm.get("word_end_idx")}
        except Exception:
            pass

    print(f"\n[{stem}] {len(json_files)} chunks  ref={len(ref_words)} words  ref_wps={ref_wps:.2f}", flush=True)
    tail_fixed = head_fixed = tail_added = head_added = 0
    t0 = time.time()

    for jf in json_files:
        data = json.loads(jf.read_text())
        words = data.get("words", [])
        rm = data["ref_match"]
        ms, me = rm["word_start_idx"], rm["word_end_idx"]
        chunk_dur = data["chunk_end_s"] - data["chunk_start_s"]
        chunk_stem = jf.stem
        chunk_wav = P.chunk_dir(file_no) / f"{chunk_stem}.wav"
        chunk_speech = speech_seg_map.get(data.get("chunk_file", f"{chunk_stem}.wav"), [])

        valid_ends = [w["end"] for w in words if w.get("end") is not None]
        valid_starts = [w["start"] for w in words if w.get("start") is not None]
        last_word_end = valid_ends[-1] if valid_ends else 0.0
        first_word_start = valid_starts[0] if valid_starts else chunk_dur
        tail_gap = chunk_dur - last_word_end
        head_gap = first_word_start

        need_audio = (not cfg.ABLATE_PASS2) and (
                      (tail_gap >= cfg.GAP_THRESHOLD_S and me < len(ref_words)) or
                      (head_gap >= cfg.GAP_THRESHOLD_S and ms > 0))
        chunk_audio = whisperx.load_audio(str(chunk_wav)) if need_audio else None
        modified = False

        # ---- tail gap ----
        if not cfg.ABLATE_PASS2 and tail_gap >= cfg.GAP_THRESHOLD_S and me < len(ref_words):
            expected_words = max(1, int(tail_gap * ref_wps * cfg.GAP_OVERSHOOT))
            gap_ref_end = min(me + expected_words, len(ref_words))
            nxt_start = (chunk_ref_bounds.get(data["chunk_index"] + 1) or {}).get("start")
            if nxt_start is not None:
                gap_ref_end = min(gap_ref_end, nxt_start)
            gap_text = " ".join(ref_words[me:gap_ref_end])
            if gap_text.strip():
                audio_start_s = max(0.0, last_word_end - cfg.GAP_BACKOFF_S)
                sub_audio = chunk_audio[int(audio_start_s * SR):]
                sub_dur = len(sub_audio) / SR
                if sub_dur >= cfg.GAP_MIN_SUB_DUR_S:
                    sub_name = f"{chunk_stem}_subchunk_endsection_duration_{int(round(tail_gap))}sec.wav"
                    sf.write(str(subchunk_dir / sub_name), sub_audio, SR, subtype="PCM_16")
                    sub_speech = subchunk_speech_segments(chunk_speech, audio_start_s, audio_start_s + sub_dur)
                    masked_sub = mask_silence(sub_audio, sub_speech)
                    try:
                        aligned = whisperx.align([{"text": gap_text, "start": 0.0, "end": sub_dur}],
                                                 align_model, align_metadata, masked_sub, _device(),
                                                 return_char_alignments=False)
                        new_words = aligned.get("word_segments", [])
                    except Exception as e:
                        print(f"  [tail align error] chunk {data['chunk_index']}: {e}", flush=True)
                        new_words = []
                    for w in new_words:
                        if w.get("start") is not None:
                            w["start"] = round(w["start"] + audio_start_s, 3)
                        if w.get("end") is not None:
                            w["end"] = round(w["end"] + audio_start_s, 3)
                        w["pass"] = 2
                    if new_words:
                        words = words + new_words
                        rm["word_end_idx"] = gap_ref_end
                        rm["matched_text"] = (rm["matched_text"] + " " + gap_text).strip()
                        tail_fixed += 1
                        tail_added += len(new_words)
                        modified = True

        # ---- head gap ----
        if not cfg.ABLATE_PASS2 and head_gap >= cfg.GAP_THRESHOLD_S and ms > 0:
            expected_words = max(1, int(head_gap * ref_wps * cfg.GAP_OVERSHOOT))
            gap_ref_start = max(0, ms - expected_words)
            prv_end = (chunk_ref_bounds.get(data["chunk_index"] - 1) or {}).get("end")
            if prv_end is not None:
                gap_ref_start = max(gap_ref_start, prv_end)
            gap_text = " ".join(ref_words[gap_ref_start:ms])
            if gap_text.strip():
                audio_end_s = min(chunk_dur, first_word_start + cfg.GAP_FORWARD_S)
                sub_audio = chunk_audio[:int(audio_end_s * SR)]
                sub_dur = len(sub_audio) / SR
                if sub_dur >= cfg.GAP_MIN_SUB_DUR_S:
                    sub_name = f"{chunk_stem}_subchunk_startsection_duration_{int(round(head_gap))}sec.wav"
                    sf.write(str(subchunk_dir / sub_name), sub_audio, SR, subtype="PCM_16")
                    sub_speech = subchunk_speech_segments(chunk_speech, 0.0, audio_end_s)
                    masked_sub = mask_silence(sub_audio, sub_speech)
                    try:
                        aligned = whisperx.align([{"text": gap_text, "start": 0.0, "end": sub_dur}],
                                                 align_model, align_metadata, masked_sub, _device(),
                                                 return_char_alignments=False)
                        new_words = aligned.get("word_segments", [])
                    except Exception as e:
                        print(f"  [head align error] chunk {data['chunk_index']}: {e}", flush=True)
                        new_words = []
                    for w in new_words:
                        w["pass"] = 2
                    if new_words:
                        words = new_words + words
                        rm["word_start_idx"] = gap_ref_start
                        rm["matched_text"] = (gap_text + " " + rm["matched_text"]).strip()
                        head_fixed += 1
                        head_added += len(new_words)
                        modified = True

        if modified:
            data["words"] = words
            jf.write_text(json.dumps(data, indent=2))
        write_labels(words, label_dir / f"{chunk_stem}_labels.txt")
        (label_dir / f"{chunk_stem}.txt").write_text(rm["matched_text"])
        del chunk_audio

    print(f"[{stem}] done ({time.time()-t0:.1f}s)  tail={tail_fixed}(+{tail_added}w) "
          f"head={head_fixed}(+{head_added}w)  labels={len(json_files)}", flush=True)


def run_range(start: int, end: int) -> None:
    dev = _device()
    print(f"Pass-2 S5: files {start:03d}-{end-1:03d}  device={dev}  mask>={cfg.SILENCE_MASK_THRESHOLD_S}s", flush=True)
    align_model, align_metadata = whisperx.load_align_model(language_code=cfg.LANGUAGE, device=dev)
    for file_no in range(start, end):
        try:
            process_file(file_no, align_model, align_metadata)
        except Exception as e:
            print(f"ERROR file {file_no:03d}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        gc.collect()
    print("S5 done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    a = ap.parse_args()
    run_range(a.start, a.end)


if __name__ == "__main__":
    main()
