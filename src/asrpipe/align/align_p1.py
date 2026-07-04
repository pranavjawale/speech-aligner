"""S4 — Pass-1 alignment: Whisper ASR + best-of-two localization + masked align.

Canonical 6c arm (design C) only — the strategy-6a rescorer/lm_fusion path is
dropped. Per chunk:
  1. Whisper-greedy (beam 5) hypothesis + the S3 Parakeet-CTC+KenLM hypothesis.
  2. best-of-two: run the sliding-window matcher on BOTH, keep the lower-nd window.
  3. silence-mask the audio (>= threshold) and wav2vec2 forced-align the matched
     REFERENCE text -> per-word timestamps + CTC confidence.

Writes per-chunk JSON to data/alignments/{stem}/{chunk_file}.json and a
{stem}_hyp_winner_stats.json. Hyp options are cached in {stem}_asr.jsonl.

Faithful port of /workspace/code/align_chunks_withmask_lm.py --design C (matcher /
masking / forced-alignment / output schema identical; base helpers now come from
asrpipe.align.matcher + common + config).

Port source : align_chunks_withmask_lm.py (design C)
Reorg phase : RP3
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import whisperx
from faster_whisper import WhisperModel

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.audio_io import mask_silence
from asrpipe.common.textnorm import clean_text
from asrpipe.align import matcher


def _device() -> str:
    return "cuda" if (cfg.DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"


def _vad_speech_dur(chunk_meta: dict) -> float:
    segs = chunk_meta.get("speech_segments", [])
    if segs:
        return sum(s["end"] - s["start"] for s in segs)
    return chunk_meta["duration_s"]


def build_hyp_options(file_no, chunks_meta, chunk_audios, asr_model) -> list[list[str]]:
    """Per-chunk candidate hypotheses [whisper_greedy, ctc_lm] (deduped). Cached."""
    n = len(chunks_meta)
    if cfg.ABLATE_WHISPER:
        # Exp3 (whisperDisable): single-ASR localization — CTC(+KenLM) hyp only. Bypass
        # both the Whisper transcribe and the _asr.jsonl cache (which holds the whisper hyp).
        ctc_path = P.ctc_lm_jsonl(file_no)
        if not ctc_path.exists():
            raise FileNotFoundError(
                f"[{P.stem(file_no)}] ABLATE_WHISPER needs the S3 pre-pass {ctc_path.name}.")
        ctc_by_idx = {}
        for l in ctc_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                ctc_by_idx[r["chunk_index"]] = r.get("ctc_lm_hyp", "")
        print(f"[{P.stem(file_no)}] ABLATE_WHISPER: CTC-only hyps ({n})", flush=True)
        return [[ctc_by_idx.get(m["chunk_index"], "")] for m in chunks_meta]
    cache_path = P.asr_cache_jsonl(file_no)
    if cache_path.exists():
        try:
            cached = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
            if len(cached) == n:
                print(f"[{P.stem(file_no)}] ASR cache hit ({n})", flush=True)
                return [c["hyp_options"] for c in cached]
        except Exception:
            pass

    # S3 Parakeet-CTC+KenLM hypotheses (required)
    ctc_path = P.ctc_lm_jsonl(file_no)
    if not ctc_path.exists():
        raise FileNotFoundError(
            f"[{P.stem(file_no)}] S4 requires the S3 pre-pass {ctc_path.name}. "
            f"Run: asrpipe.align.lm_prepass in the NeMo venv first.")
    ctc_by_idx = {}
    for l in ctc_path.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            ctc_by_idx[r["chunk_index"]] = r.get("ctc_lm_hyp", "")

    print(f"[{P.stem(file_no)}] building hyp options ...", flush=True)
    t1 = time.time()
    options, cache_lines = [], []
    for meta, audio in zip(chunks_meta, chunk_audios):
        segs, _ = asr_model.transcribe(audio, language=cfg.LANGUAGE,
                                       beam_size=cfg.ASR_BEAM_SIZE, vad_filter=False)
        whisper_hyp = clean_text(" ".join(s.text for s in segs))
        ctc_lm_hyp = ctc_by_idx.get(meta["chunk_index"], "")
        opts = [h for h in [whisper_hyp, ctc_lm_hyp] if h]
        opts = list(dict.fromkeys(opts)) or [""]          # dedup, keep order
        options.append(opts)
        cache_lines.append(json.dumps({
            "chunk_index": meta["chunk_index"], "chunk_file": meta["chunk_file"],
            "start_s": meta["start_s"], "end_s": meta["end_s"], "hyp_options": opts,
        }))
    cache_path.write_text("\n".join(cache_lines) + "\n")
    print(f"[{P.stem(file_no)}] hyp options done ({time.time()-t1:.1f}s)", flush=True)
    return options


def match_best_option(ref_words_clean, hyp_options, expected_pos, eff_speech_dur,
                      ref_wps, committed_pos):
    """Sliding-window matcher over each candidate hyp; return the lowest-nd match
    as (ms, me, nd, chosen_hyp, best_idx, per_option_nds)."""
    best = None
    nds = []
    for i, hyp in enumerate(hyp_options):
        hyp_words = hyp.split()
        n_ref_words = max(int(eff_speech_dur * ref_wps), len(hyp_words))  # was max(1,...); no-op (hyp non-empty => >=1)
        s_start = max(0, expected_pos - cfg.SEARCH_BUFFER_BACK)
        s_end = min(len(ref_words_clean), expected_pos + cfg.SEARCH_BUFFER_FWD + n_ref_words)
        unique_ng = matcher.build_unique_ngrams(ref_words_clean, s_start, s_end)
        if hyp_words:
            ms, me, nd = matcher.sliding_window_match(
                ref_words_clean, hyp_words, expected_pos, n_ref_words,
                unique_ngrams=unique_ng, committed_pos=committed_pos)
        else:
            ms = min(expected_pos, len(ref_words_clean))
            me = min(expected_pos + n_ref_words, len(ref_words_clean))
            nd = 1.0
        nds.append(nd)
        # Tie-break: options are ordered [whisper, parakeet+kenlm], so `<=` (not `<`)
        # awards an exact-nd tie to the LATER option = parakeet+kenlm (was: whisper).
        if best is None or nd <= best[2]:
            best = (ms, me, nd, hyp, i)
    return best[0], best[1], best[2], best[3], best[4], nds


def process_file(file_no, asr_model, align_model, align_metadata, executor):
    stem = P.stem(file_no)
    out_dir = P.align_dir(file_no)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_meta = json.loads(P.chunk_manifest(file_no).read_text())["chunks"]
    chunk_paths = [P.chunk_dir(file_no) / c["chunk_file"] for c in chunks_meta]
    print(f"\n[{stem}] {len(chunk_paths)} chunks", flush=True)

    t0 = time.time()
    chunk_audios = list(executor.map(whisperx.load_audio, map(str, chunk_paths)))
    print(f"[{stem}] audio loaded ({time.time()-t0:.1f}s)", flush=True)

    hyp_options = build_hyp_options(file_no, chunks_meta, chunk_audios, asr_model)

    ref_words_clean, ref_words_natural = matcher.load_reference(file_no)
    total_speech_s = sum(_vad_speech_dur(c) for c in chunks_meta)
    ref_wps = len(ref_words_clean) / total_speech_s if total_speech_s > 0 else cfg.FALLBACK_WPS
    print(f"[{stem}] ref {len(ref_words_clean)} words  speech={total_speech_s:.1f}s "
          f"wps={ref_wps:.2f}. matching ...", flush=True)
    t2 = time.time()

    win = {"whisper": 0, "parakeet+kenlm": 0, "tie": 0, "single_option": 0}
    delta_sum = 0.0
    matches, chosen_hyps = [], []
    cumulative_speech, committed_pos = 0.0, 0
    for meta, opts in zip(chunks_meta, hyp_options):
        eff_speech_dur = _vad_speech_dur(meta)
        expected_pos = int(cumulative_speech * ref_wps)
        ms, me, nd, chosen, best_idx, nds = match_best_option(
            ref_words_clean, opts, expected_pos, eff_speech_dur, ref_wps, committed_pos)
        if len(opts) >= 2:
            if abs(nds[0] - nds[1]) < 1e-9:
                win["tie"] += 1
            elif nds[0] < nds[1]:
                win["whisper"] += 1
            else:
                win["parakeet+kenlm"] += 1
            delta_sum += nds[0] - nds[1]
        else:
            win["single_option"] += 1
        if nd <= cfg.COMMIT_THRESHOLD:
            committed_pos = me
        matched_text = " ".join(ref_words_natural[ms:me])
        matches.append((ms, me, nd, matched_text))
        chosen_hyps.append(chosen)
        cumulative_speech += eff_speech_dur
    print(f"[{stem}] matching done ({time.time()-t2:.1f}s)", flush=True)

    n_choice = win["whisper"] + win["parakeet+kenlm"] + win["tie"]
    mean_delta = (delta_sum / n_choice) if n_choice else 0.0
    print(f"[{stem}] best-of-two: whisper={win['whisper']} "
          f"parakeet+kenlm={win['parakeet+kenlm']} tie={win['tie']} "
          f"single={win['single_option']} mean_nd_delta={mean_delta:+.3f}", flush=True)
    stats = {"stem": stem, "n_chunks": len(chunks_meta), "two_way_choices": n_choice,
             "winners": win, "mean_nd_delta_whisper_minus_lm": round(mean_delta, 4)}
    (out_dir.parent / f"{stem}_hyp_winner_stats.json").write_text(json.dumps(stats, indent=2))

    dev = _device()
    print(f"[{stem}] aligning (mask >= {cfg.SILENCE_MASK_THRESHOLD_S}s) ...", flush=True)
    t3 = time.time()
    save_futures = []
    for meta, hyp, audio, (ms, me, nd, matched_text) in \
            zip(chunks_meta, chosen_hyps, chunk_audios, matches):
        chunk_dur = meta["end_s"] - meta["start_s"]
        masked_audio = mask_silence(audio, meta.get("speech_segments", []))
        try:
            aligned = whisperx.align(
                [{"text": matched_text, "start": 0.0, "end": chunk_dur}],
                align_model, align_metadata, masked_audio, dev,
                return_char_alignments=False)
            word_segments = aligned.get("word_segments", [])
        except Exception as e:
            print(f"[{stem}] align error chunk {meta['chunk_index']}: {e}", flush=True)
            word_segments = []
        result = {
            "source_file": f"{stem}.wav",
            "chunk_index": meta["chunk_index"],
            "chunk_file": meta["chunk_file"],
            "chunk_start_s": meta["start_s"],
            "chunk_end_s": meta["end_s"],
            "asr_hypothesis": hyp,
            "ref_match": {
                "word_start_idx": ms, "word_end_idx": me,
                "matched_text": matched_text, "norm_edit_dist": nd,
            },
            "words": word_segments,
        }
        out_path = P.align_json(file_no, meta["chunk_file"])
        save_futures.append(executor.submit(out_path.write_text, json.dumps(result, indent=2)))
    for f in save_futures:
        f.result()
    print(f"[{stem}] align+save done ({time.time()-t3:.1f}s)", flush=True)
    del chunk_audios
    gc.collect()


def run_range(start: int, end: int) -> None:
    dev = _device()
    print(f"Align S4 (design C): files {start:03d}-{end-1:03d}  device={dev}", flush=True)
    asr_model = None if cfg.ABLATE_WHISPER else WhisperModel(
        cfg.ASR_MODEL_SIZE, device=dev, compute_type=cfg.ASR_COMPUTE_TYPE)
    align_model, align_metadata = whisperx.load_align_model(language_code=cfg.LANGUAGE, device=dev)
    print("models ready.", flush=True)
    with ThreadPoolExecutor(max_workers=cfg.IO_WORKERS) as executor:
        for file_no in range(start, end):
            if not P.chunk_manifest(file_no).exists():
                print(f"[{P.stem(file_no)}] SKIP (no manifest)", flush=True)
                continue
            try:
                process_file(file_no, asr_model, align_model, align_metadata, executor)
            except Exception as e:
                print(f"ERROR file {file_no:03d}: {e}", flush=True)
                import traceback
                traceback.print_exc()
    print("S4 done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    a = ap.parse_args()
    run_range(a.start, a.end)


if __name__ == "__main__":
    main()
