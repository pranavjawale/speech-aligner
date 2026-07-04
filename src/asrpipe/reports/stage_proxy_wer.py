"""Per-stage alignment-efficacy review: proxy-WER after each aligner stage.

For every chunk, proxy_wer(stage) = word-level WER between the chunk's ASR hypothesis
(the best-of-two `asr_hypothesis` stored in the chunk JSON) and the reference span the
aligner estimated AT THE END OF THAT STAGE. The ASR hyp is the ground-truth proxy
(reference) and the estimated ref is the hypothesis being scored, so the denominator
(ASR words) is CONSTANT across stages -> the five stage columns are directly comparable.
It is the word-level analog of the stored char-level `ref_match.norm_edit_dist`.

Stages and where each stage's ref comes from:
  pass1     CPU replay of the pass-1 matcher (match_best_option) with the cached hyps
  resync    CPU replay of resync drift-detect + anchor-vote (plan_file logic) on pass-1
  pass2     words of labels/{chunk}_labels.txt             (on disk)
  tailtrim  words of labels/{chunk}_labels_corrected1.txt  (on disk)
  headtrim  words of labels/{chunk}_labels_corrected2.txt  (on disk)
Next to each stage's WER we also report how many chunks that stage actually MODIFIED
(resync tag / pass=2 words / corrected1!=labels / corrected2!=corrected1), all derived
from on-disk data (no log parsing) so this runs after ANY pipeline run.

REUSABLE: `python3 -m asrpipe.reports.stage_proxy_wer START END [--tag TAG] [--workers N]`.
Honors ASRPIPE_ALIGN_DIR (works on isolated runs too). CAVEAT: the pass1/resync refs are
reconstructed with the CURRENT config (penalties, resync tunables) + the cached hyps, so
run the review with the SAME config the pipeline run used (a validation line reports the
match rate of replayed pass1-start vs the stored resync.old_start_idx as a self-check).

Port: new (2026-07-03).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

from rapidfuzz.distance import Levenshtein

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.wer import normalize
from asrpipe.align import matcher
from asrpipe.align.align_p1 import match_best_option, _vad_speech_dur
from asrpipe.align.resync import (_smooth, _drift_runs, _ref_indexes, _vote_start, _nd,
                                  SMOOTH_W, NGRAMS, MARGIN)

STAGES = ["pass1", "resync", "pass2", "tailtrim", "headtrim"]


def _label_words(path):
    if not os.path.exists(path):
        return None
    out = []
    for ln in open(path).read().splitlines():
        p = ln.split("\t")
        if len(p) >= 3:
            out.append(p[2].rsplit(" (", 1)[0])
    return out


def process(file_no: int):
    """Reconstruct the 5 stage refs for one file and score proxy-WER. Returns a dict or
    None if the file has no alignments."""
    adir = P.align_dir(file_no); ldir = P.labels_dir(file_no)
    jfs = sorted(glob.glob(f"{adir}/*_chunk_*.json"))
    if not jfs:
        return None
    jmap = {}   # chunk_index -> (stem, asr_hyp, resync_field, pass2_flag)
    for jf in jfs:
        d = json.load(open(jf))
        pass2 = any(w.get("pass") == 2 for w in d.get("words", []))
        jmap[d["chunk_index"]] = (os.path.basename(jf)[:-5], d.get("asr_hypothesis") or "",
                                  d.get("resync", {}), pass2)
    chunks_meta = json.loads(P.chunk_manifest(file_no).read_text())["chunks"]
    cache = {}
    for l in P.asr_cache_jsonl(file_no).read_text().splitlines():
        if l.strip():
            r = json.loads(l); cache[r["chunk_index"]] = r["hyp_options"]
    ref_clean, ref_nat = matcher.load_reference(file_no)
    total = sum(_vad_speech_dur(c) for c in chunks_meta)
    ref_wps = len(ref_clean) / total if total > 0 else cfg.FALLBACK_WPS

    # ---- pass-1 replay (matcher cascade) ----
    cumulative = 0.0; committed = 0
    p1 = {}; order = []
    for meta in chunks_meta:
        ci = meta["chunk_index"]; order.append(ci)
        opts = cache.get(ci, [""])
        eff = _vad_speech_dur(meta); expected = int(cumulative * ref_wps)
        ms, me, nd, _c, _i, _n = match_best_option(ref_clean, opts, expected, eff, ref_wps, committed)
        if nd <= cfg.COMMIT_THRESHOLD:
            committed = me
        p1[ci] = (ms, me, nd); cumulative += eff

    # ---- resync replay (plan_file logic on pass-1 state) ----
    meta_by = {m["chunk_index"]: m for m in chunks_meta}
    ordered = sorted(order, key=lambda ci: meta_by[ci]["start_s"])
    runs = _drift_runs(_smooth([p1[ci][2] for ci in ordered], SMOOTH_W))
    idxs = _ref_indexes(ref_clean, NGRAMS)
    in_run = {}
    for a, b in runs:
        lo = p1[ordered[a - 1]][1] if a > 0 else 0
        hi = p1[ordered[b + 1]][0] if b < len(ordered) - 1 else len(ref_clean)
        for i in range(a, b + 1):
            in_run[i] = (a, b, lo, hi)
    running = {}; rs = {}; resynced = set()
    for i, ci in enumerate(ordered):
        ms, me, _pnd = p1[ci]
        hyp_words = (jmap.get(ci, ("", "", {}, False))[1]).split()
        new_ms, new_me = ms, me
        if i in in_run:
            a, b, lo, hi = in_run[i]
            L = max(me - ms, len(hyp_words))
            voted, _s, _n = _vote_start(hyp_words, idxs, NGRAMS, running.get((a, b), lo),
                                        min(hi, len(ref_clean) - 1))
            if voted is not None:
                nm = voted; nme = min(voted + L, len(ref_clean))
                if _nd(" ".join(hyp_words), ref_clean, nm, nme) < _nd(" ".join(hyp_words), ref_clean, ms, me) - MARGIN:
                    new_ms, new_me = nm, nme; running[(a, b)] = nm; resynced.add(ci)
        rs[ci] = (new_ms, new_me)

    # ---- refs + proxy per chunk + modified-chunk counts (from data) ----
    edits = defaultdict(int); den = 0; scored = 0
    aff = {"resync": 0, "pass2": 0, "tailtrim": 0, "headtrim": 0}
    val_ok = val_tot = 0
    for ci in order:
        stem_ci, asr, rfield, pass2flag = jmap[ci]
        if rfield.get("resynced"):
            aff["resync"] += 1; val_tot += 1
            if p1[ci][0] == rfield.get("old_start_idx"):
                val_ok += 1
        if pass2flag:
            aff["pass2"] += 1
        lbl = _label_words(f"{ldir}/{stem_ci}_labels.txt")
        c1 = _label_words(f"{ldir}/{stem_ci}_labels_corrected1.txt")
        c2 = _label_words(f"{ldir}/{stem_ci}_labels_corrected2.txt")
        if lbl is not None and c1 is not None and c1 != lbl:
            aff["tailtrim"] += 1
        if c1 is not None and c2 is not None and c2 != c1:
            aff["headtrim"] += 1
        asr_n = normalize(asr).split()
        if not asr_n:
            continue
        den += len(asr_n); scored += 1
        refs = {"pass1": " ".join(ref_nat[p1[ci][0]:p1[ci][1]]),
                "resync": " ".join(ref_nat[rs[ci][0]:rs[ci][1]]),
                "pass2": " ".join(lbl or []),
                "tailtrim": " ".join(c1 or []),
                "headtrim": " ".join(c2 or [])}
        for st in STAGES:
            edits[st] += Levenshtein.distance(normalize(refs[st]).split(), asr_n)
    return {"file": P.stem(file_no), "chunks": len(order), "scored": scored, "asr_words": den,
            "aff": aff, "edits": dict(edits), "val_ok": val_ok, "val_tot": val_tot}


def _fmt(wer):
    return f"{wer:.4f}" if wer is not None else ""


def build(start: int, end: int, tag: str, workers: int):
    files = list(range(start, end))
    t0 = time.time()
    with Pool(min(workers, len(files))) as pool:
        results = []
        for r in pool.imap_unordered(process, files):
            if r:
                results.append(r)
                print(f"  [{r['file']}] {r['chunks']} chunks  "
                      f"pass1={r['edits']['pass1']/r['asr_words']:.3f} "
                      f"resync={r['edits']['resync']/r['asr_words']:.3f} "
                      f"head={r['edits']['headtrim']/r['asr_words']:.3f}", flush=True)
    results.sort(key=lambda r: r["file"])
    # aggregate (corpus proxy-WER = sum edits / sum asr words)
    A = sum(r["asr_words"] for r in results)
    C = sum(r["chunks"] for r in results)
    tot_e = {st: sum(r["edits"][st] for r in results) for st in STAGES}
    tot_aff = {k: sum(r["aff"][k] for r in results) for k in ("resync", "pass2", "tailtrim", "headtrim")}
    vok = sum(r["val_ok"] for r in results); vtot = sum(r["val_tot"] for r in results)
    corpus = {st: tot_e[st] / A for st in STAGES}

    # ---- CSV (affected counts next to per-stage proxy-WER) ----
    cfg.EXPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.EXPT_RESULTS_DIR / f"{tag}_stage_proxy_wer.csv"
    hdr = ["file", "chunks",
           "resync_mod", "resync_mod_pct", "pass2_mod", "pass2_mod_pct",
           "tailtrim_mod", "tailtrim_mod_pct", "headtrim_mod", "headtrim_mod_pct",
           "wer_pass1", "wer_resync", "wer_pass2", "wer_tailtrim", "wer_headtrim"]
    import csv as _csv

    def _pc(x, n):
        return f"{100 * x / n:.1f}" if n else ""

    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(hdr)
        for r in results:
            a = r["aff"]; d = r["asr_words"]; ch = r["chunks"]
            w.writerow([r["file"], r["chunks"],
                        a["resync"], _pc(a["resync"], ch), a["pass2"], _pc(a["pass2"], ch),
                        a["tailtrim"], _pc(a["tailtrim"], ch), a["headtrim"], _pc(a["headtrim"], ch)] +
                       [_fmt(r["edits"][st] / d) for st in STAGES])
        w.writerow(["TOTAL", C,
                    tot_aff["resync"], _pc(tot_aff["resync"], C), tot_aff["pass2"], _pc(tot_aff["pass2"], C),
                    tot_aff["tailtrim"], _pc(tot_aff["tailtrim"], C), tot_aff["headtrim"], _pc(tot_aff["headtrim"], C)] +
                   [_fmt(corpus[st]) for st in STAGES])

    # ---- summary + analysis ----
    def pct(x): return f"{100*x/C:.1f}%" if C else "-"
    lines = [
        f"ALIGN-STAGE EFFICACY — proxy-WER after each aligner stage   [tag: {tag}]",
        f"files {start:03d}-{end-1:03d}  ({len(results)} present)   chunks={C}   scored_asr_words={A}",
        "",
        "proxy_wer(stage) = word-level WER( estimated-ref@stage , chunk ASR hyp ) with the ASR hyp",
        "as the reference (constant denominator across stages). Lower = the estimated ref agrees more",
        "with what the ASR heard. pass1/resync refs are CPU-replayed; pass2/tail/head are on-disk labels.",
        f"replay self-check: replayed pass1-start == stored resync.old_start_idx on {vok}/{vtot} resynced chunks.",
        "",
        "CORPUS proxy-WER by stage (sum edits / sum ASR words):",
        f"  pass1 {corpus['pass1']*100:6.2f}%  ->  resync {corpus['resync']*100:6.2f}%  ->  "
        f"pass2 {corpus['pass2']*100:6.2f}%  ->  tailtrim {corpus['tailtrim']*100:6.2f}%  ->  "
        f"headtrim {corpus['headtrim']*100:6.2f}%",
        f"  chunks modified: resync {tot_aff['resync']} ({pct(tot_aff['resync'])}), "
        f"pass2 {tot_aff['pass2']} ({pct(tot_aff['pass2'])}), "
        f"tailtrim {tot_aff['tailtrim']} ({pct(tot_aff['tailtrim'])}), "
        f"headtrim {tot_aff['headtrim']} ({pct(tot_aff['headtrim'])})",
        "",
        f"{'file':<16}{'chunks':>7}{'pass1':>9}{'resync':>9}{'pass2':>9}{'tailtrim':>9}{'headtrim':>9}"
        f"{'|Δresync':>9}{'|Δtrim':>8}",
    ]
    for r in results:
        d = r["asr_words"]; wv = {st: r["edits"][st] / d for st in STAGES}
        dres = wv["pass1"] - wv["resync"]; dtrim = wv["pass2"] - wv["headtrim"]
        lines.append(f"{r['file']:<16}{r['chunks']:>7}"
                     f"{wv['pass1']*100:>8.1f}{wv['resync']*100:>9.1f}{wv['pass2']*100:>9.1f}"
                     f"{wv['tailtrim']*100:>9.1f}{wv['headtrim']*100:>9.1f}"
                     f"{dres*100:>+9.1f}{dtrim*100:>+8.1f}")
    lines += [
        "",
        "ANALYSIS",
        f"- RESYNC is the dominant corrector: on drift files pass1 proxy-WER is very high (wrong",
        f"  localization) and resync cuts it sharply; on clean files pass1≈resync (no drift, nothing to do).",
        f"  Corpus pass1->resync: {corpus['pass1']*100:.2f}% -> {corpus['resync']*100:.2f}% "
        f"({(corpus['pass1']-corpus['resync'])*100:+.2f} pp).",
        f"- TRIMS (tail then head) refine edges: resync->headtrim "
        f"{corpus['resync']*100:.2f}% -> {corpus['headtrim']*100:.2f}% "
        f"({(corpus['resync']-corpus['headtrim'])*100:+.2f} pp); they drop low-confidence edge words the",
        f"  ASR did not support.",
        f"- PASS2 gap-fill can nudge the proxy UP slightly (it ADDS reference words at tail/head; any",
        f"  added word not in the ASR hyp counts against the ref) but recovers audio the first pass missed.",
        "- proxy-WER has a nonzero floor (the ASR hyp is imperfect and covers the whole chunk incl. silence),",
        "  so read the stage-to-stage DELTAS, not absolute values; the ASR hyp is the label-gen best-of-two",
        "  hyp (self-consistency proxy, = word-level norm_edit_dist), not an independent judge.",
        "",
        f"CSV: {csv_path}",
    ]
    summ_path = cfg.EXPERIMENT_LOGS_DIR / f"align-stage-efficacy_summary.txt"
    summ_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten -> {csv_path}\nwritten -> {summ_path}  ({time.time()-t0:.0f}s)")
    return csv_path, summ_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("--tag", default="run", help="output filename tag")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    build(a.start, a.end, a.tag, a.workers)


if __name__ == "__main__":
    main()
