"""S4.5 — RE-SYNC: re-localize chunks whose pass-1 alignment drifted.

Problem/design/why: docs/resync-fix.txt.

In repetitive reference text the unique-anchor matcher (align_p1 / matcher.py) has no
distinctive anchors, so a run of chunks gets localized to the wrong reference span and
the error cascades via expected_pos. The audio + ASR are fine; only the localization
is wrong, and the true words sit a (variable) offset away in the SAME reference.

This stage, run AFTER pass-1 and BEFORE pass-2:
  1. DETECT drift runs: smoothed norm_edit_dist, hysteresis (enter >=HIGH, exit <=LOW),
     runs bracketed by confident (low-nd) chunks.
  2. RE-LOCALIZE each drifted chunk by n-gram anchor VOTING (rare n-grams weighted
     1/occurrences), CONSTRAINED to lie between the confident neighbours' reference
     positions (monotonicity guard -> rejects spurious votes, preserves order).
  3. Accept only if the re-localized window lowers norm_edit_dist by >= MARGIN.
  4. RE-ALIGN (apply mode): silence-masked wav2vec2 FA on the corrected matched_text;
     rewrite the chunk JSON (ref_match + words) and tag resynced=true.

Modes:
  report (default): compute + report nd before/after; NO file changes, NO GPU.
  --apply         : rewrite the chunk JSONs and re-run forced alignment (needs GPU).

Additive & opt-in: chunks NOT in a detected drift run are never touched, so files
without drift reproduce the pre-resync outputs byte-for-byte.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import defaultdict

from rapidfuzz.distance import Levenshtein

import config as cfg
from asrpipe.common import paths as P
from asrpipe.align import matcher

# tunables (mirrors scripts/chunk_wer_pattern.py detection)
HIGH = 0.55          # enter-high (drift) threshold on smoothed nd
LOW = 0.45           # return-to-low (confident) threshold
SMOOTH_W = 3         # rolling-mean window
MIN_RUN = 2          # ignore single-chunk spikes
NGRAMS = (5, 4, 3)   # anchor-voting n-gram lengths, tried longest-first (most distinctive)
MARGIN = 0.15        # required nd improvement to accept a re-localization


def _smooth(vals, w):
    h = w // 2
    return [statistics.mean(vals[max(0, i - h): i + h + 1]) for i in range(len(vals))]


def _drift_runs(sm):
    """Hysteresis over the smoothed nd -> list of (a, b) high-run index ranges."""
    runs, state, start = [], "low", 0
    for i, v in enumerate(sm):
        if state == "low" and v >= HIGH:
            state, start = "high", i
        elif state == "high" and v <= LOW:
            runs.append((start, i - 1)); state = "low"
    if state == "high":
        runs.append((start, len(sm) - 1))
    return [(a, b) for a, b in runs if b - a + 1 >= MIN_RUN]


def _ref_indexes(ref_clean, ns):
    idxs = {n: defaultdict(list) for n in ns}
    for n in ns:
        idx = idxs[n]
        for i in range(len(ref_clean) - n + 1):
            idx[tuple(ref_clean[i:i + n])].append(i)
    return idxs


def _vote_start(hyp_words, idxs, ns, lo, hi):
    """rare-weighted n-gram vote for chunk_start = ref_pos - hyp_pos, in [lo, hi].
    Tries the longest n first (most distinctive); falls back to shorter n only if a
    longer one yields no in-bracket vote. Returns (best_start, score, n_used)."""
    for n in ns:
        if len(hyp_words) < n:
            continue
        idx = idxs[n]
        votes = defaultdict(float)
        for j in range(len(hyp_words) - n + 1):
            ps = idx.get(tuple(hyp_words[j:j + n]))
            if not ps:
                continue
            wt = 1.0 / len(ps)
            for p in ps:
                s = p - j
                if lo <= s <= hi:
                    votes[s] += wt
        if votes:
            best = max(votes, key=lambda s: sum(votes.get(s2, 0.0) for s2 in range(s - 2, s + 3)))
            return best, sum(votes.get(s2, 0.0) for s2 in range(best - 2, best + 3)), n
    return None, 0.0, None


def _nd(hyp_str, ref_clean, ms, me):
    """char-level normalized edit distance — identical convention to matcher.py."""
    ref_str = " ".join(ref_clean[ms:me])
    d = Levenshtein.distance(hyp_str, ref_str)
    return round(d / max(len(hyp_str), len(ref_str), 1), 4)


def plan_file(file_no):
    """CPU-only re-localization plan for ALL chunks (no FA, no writes).
    Returns rows ordered by time: {chunk_index, start_s, old_start, old_nd,
    new_start, new_nd, offset, status}. status in
    clean|resynced|not_improved|failed_no_vote. Shared by the report/apply path
    and by scripts/chunk_wer_pattern.py (old-vs-new proxy-WER columns)."""
    aln_dir = P.align_dir(file_no)
    jfs = sorted(aln_dir.glob("*chunk_*.json"))
    if not jfs:
        return []
    chunks = [(jf, json.loads(jf.read_text())) for jf in jfs]
    chunks.sort(key=lambda pd: pd[1]["chunk_start_s"])
    nds = [pd[1]["ref_match"].get("norm_edit_dist") or 0.0 for pd in chunks]
    runs = _drift_runs(_smooth(nds, SMOOTH_W))
    ref_clean, _ = matcher.load_reference(file_no)
    idxs = _ref_indexes(ref_clean, NGRAMS)
    in_run = {}
    for a, b in runs:
        lo = chunks[a - 1][1]["ref_match"]["word_end_idx"] if a > 0 else 0
        hi = chunks[b + 1][1]["ref_match"]["word_start_idx"] if b < len(chunks) - 1 else len(ref_clean)
        for i in range(a, b + 1):
            in_run[i] = (a, b, lo, hi)
    running = {}
    rows = []
    for i, (jf, d) in enumerate(chunks):
        rm = d["ref_match"]; cur_ms, cur_me = rm["word_start_idx"], rm["word_end_idx"]
        hyp_words = (d.get("asr_hypothesis") or "").split(); hyp_str = " ".join(hyp_words)
        old_nd = _nd(hyp_str, ref_clean, cur_ms, cur_me)
        row = {"chunk_index": d["chunk_index"], "start_s": d["chunk_start_s"],
               "old_start": cur_ms, "old_nd": old_nd, "new_start": cur_ms,
               "new_nd": old_nd, "offset": 0, "status": "clean"}
        if i in in_run:
            a, b, lo, hi = in_run[i]
            running_lo = running.get((a, b), lo)
            L = max(cur_me - cur_ms, len(hyp_words))
            voted, score, _n = _vote_start(hyp_words, idxs, NGRAMS, running_lo, min(hi, len(ref_clean) - 1))
            if voted is None:
                row["status"] = "failed_no_vote"
            else:
                new_ms = voted; new_me = min(voted + L, len(ref_clean))
                new_nd = _nd(hyp_str, ref_clean, new_ms, new_me)
                row.update(new_start=new_ms, new_nd=new_nd, offset=new_ms - cur_ms)
                if new_nd < old_nd - MARGIN:
                    row["status"] = "resynced"; running[(a, b)] = new_ms
                else:
                    row["status"] = "not_improved"
        rows.append(row)
    return rows


def resync_file(file_no, apply=False, align_model=None, align_metadata=None):
    stem = P.stem(file_no)
    aln_dir = P.align_dir(file_no)
    jfs = sorted(aln_dir.glob("*chunk_*.json"))
    if not jfs:
        return None
    chunks = []                                    # (path, data), ordered by time
    for jf in jfs:
        d = json.loads(jf.read_text())
        chunks.append((jf, d))
    chunks.sort(key=lambda pd: pd[1]["chunk_start_s"])
    nds = [pd[1]["ref_match"].get("norm_edit_dist") or 0.0 for pd in chunks]
    sm = _smooth(nds, SMOOTH_W)
    runs = _drift_runs(sm)

    ref_clean, ref_nat = matcher.load_reference(file_no)
    idxs = _ref_indexes(ref_clean, NGRAMS)

    speech_map = _rl = None
    if apply:
        from asrpipe.align.conf_trim import _speech_map, _realign  # reuse FA machinery
        speech_map, _rl = _speech_map(file_no), _realign

    report = []                                    # per-chunk rows for accepted/failed
    accepted = failed = 0
    for a, b in runs:
        lo = chunks[a - 1][1]["ref_match"]["word_end_idx"] if a > 0 else 0
        hi = chunks[b + 1][1]["ref_match"]["word_start_idx"] if b < len(chunks) - 1 else len(ref_clean)
        running_lo = lo
        for i in range(a, b + 1):
            jf, d = chunks[i]
            rm = d["ref_match"]
            hyp_words = (d.get("asr_hypothesis") or "").split()
            hyp_str = " ".join(hyp_words)
            cur_ms, cur_me = rm["word_start_idx"], rm["word_end_idx"]
            L = max(cur_me - cur_ms, len(hyp_words))
            old_nd = _nd(hyp_str, ref_clean, cur_ms, cur_me)
            start_hi = min(hi, len(ref_clean) - 1)
            voted, score, _n = _vote_start(hyp_words, idxs, NGRAMS, running_lo, start_hi)
            row = {"chunk_index": d["chunk_index"], "start_s": d["chunk_start_s"],
                   "old_start": cur_ms, "old_nd": old_nd}
            if voted is None:
                failed += 1
                row.update({"status": "failed_no_vote", "new_start": None,
                            "new_nd": None, "offset": None, "vote": round(score, 1)})
                report.append(row)
                continue
            new_ms = voted
            new_me = min(voted + L, len(ref_clean))
            new_nd = _nd(hyp_str, ref_clean, new_ms, new_me)
            row.update({"new_start": new_ms, "new_nd": new_nd,
                        "offset": new_ms - cur_ms, "vote": round(score, 1)})
            if new_nd < old_nd - MARGIN:
                accepted += 1
                row["status"] = "resynced"
                running_lo = new_ms                # monotonic: next chunk starts >= here
                if apply and _rl is not None:
                    new_text = " ".join(ref_nat[new_ms:new_me])
                    chunk_dur = d["chunk_end_s"] - d["chunk_start_s"]
                    speech = speech_map.get(d.get("chunk_file"), [])
                    new_words = _rl(new_text, P.chunk_dir(file_no) / d["chunk_file"],
                                    speech, chunk_dur, align_model, align_metadata)
                    d["ref_match"] = {"word_start_idx": new_ms, "word_end_idx": new_me,
                                      "matched_text": new_text, "norm_edit_dist": new_nd}
                    d["words"] = new_words
                    d["resync"] = {"resynced": True, "old_start_idx": cur_ms,
                                   "old_norm_edit_dist": old_nd, "vote_score": round(score, 1)}
                    jf.write_text(json.dumps(d, indent=2))
            else:
                row["status"] = "not_improved"
            report.append(row)

    # write a per-file report (non-destructive, always)
    outdir = cfg.DATA / "review"
    outdir.mkdir(parents=True, exist_ok=True)
    drift_chunks = sum(b - a + 1 for a, b in runs)
    before = statistics.mean([r["old_nd"] for r in report]) if report else 0.0
    after = statistics.mean([(r["new_nd"] if r["status"] == "resynced" else r["old_nd"])
                             for r in report]) if report else 0.0
    lines = [f"{stem}: {len(chunks)} chunks | drift runs={len(runs)} covering "
             f"{drift_chunks} chunks | resynced={accepted} failed={failed} "
             f"not_improved={len(report)-accepted-failed}",
             f"drift-region mean nd: before={before:.3f}  after={after:.3f}  "
             f"(mode={'APPLY' if apply else 'report'})", ""]
    for a, b in runs:
        lines.append(f"  run chunks {chunks[a][1]['chunk_index']}..{chunks[b][1]['chunk_index']}  "
                     f"t={chunks[a][1]['chunk_start_s']/60:.1f}..{chunks[b][1]['chunk_end_s']/60:.1f} min")
    lines.append("")
    lines.append(f"{'idx':>5}{'t_min':>8}{'old_start':>10}{'new_start':>10}"
                 f"{'offset':>8}{'old_nd':>8}{'new_nd':>8}  status")
    for r in report:
        nn = "" if r["new_start"] is None else r["new_start"]
        nd = "" if r["new_nd"] is None else f"{r['new_nd']:.3f}"
        off = "" if r["offset"] is None else f"{r['offset']:+d}"
        lines.append(f"{r['chunk_index']:>5}{r['start_s']/60:>8.1f}{r['old_start']:>10}"
                     f"{str(nn):>10}{off:>8}{r['old_nd']:>8.3f}{nd:>8}  {r['status']}")
    (outdir / f"{stem}_resync_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))
    print(f"  -> {outdir / f'{stem}_resync_report.txt'}")
    return {"stem": stem, "runs": len(runs), "drift_chunks": drift_chunks,
            "accepted": accepted, "failed": failed, "before": before, "after": after}


def run_range(start, end, apply=False):
    align_model = align_metadata = None
    if apply:
        import torch, whisperx
        dev = "cuda" if (cfg.DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"
        print(f"RE-SYNC (APPLY) files {start:03d}-{end-1:03d}  device={dev}", flush=True)
        align_model, align_metadata = whisperx.load_align_model(language_code=cfg.LANGUAGE, device=dev)
    else:
        print(f"RE-SYNC (report) files {start:03d}-{end-1:03d}", flush=True)
    for file_no in range(start, end):
        if not P.align_dir(file_no).exists():
            continue
        t0 = time.time()
        try:
            resync_file(file_no, apply, align_model, align_metadata)
        except Exception as e:
            print(f"ERROR file {file_no:03d}: {e}", flush=True)
            import traceback; traceback.print_exc()
        gc.collect()
    print("re-sync done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite chunk JSONs + re-align (needs GPU); default = report only")
    a = ap.parse_args()
    run_range(a.start, a.end, a.apply)


if __name__ == "__main__":
    main()
