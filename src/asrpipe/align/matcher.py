"""Sliding-window reference matcher (localizes each chunk in the transcript).

Ported verbatim from /workspace/code/align_chunks_withmask.py (the base module the
strategy-6 script reused). The ASR hypothesis is used ONLY here — to find where in
the reference each chunk sits — via an anchored, boundary-aware edit-distance scan.
Constants come from config; text normalization from common.textnorm.

Public:
  load_reference(file_no)        -> (ref_words_clean, ref_words_natural)
  build_unique_ngrams(w, s, e)   -> set of once-only n-grams in the window
  sliding_window_match(...)      -> (match_start, match_end, norm_edit_dist)

Port source : align_chunks_withmask.py (load_reference, anchor + sliding-window)
Reorg phase : RP3 (S4 dependency)
"""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.textnorm import clean_text, clean_text_for_align


def load_reference(file_no: int) -> tuple[list[str], list[str]]:
    """Return (matcher word list, alignment word list) for a transcript.
    Both have identical length (textnorm invariant), so indices are interchangeable."""
    raw = P.transcript(file_no).read_text()
    return clean_text(raw).split(), clean_text_for_align(raw).split()


# --------------------------------------------------------------- anchors
def build_unique_ngrams(ref_words: list[str], start: int, end: int) -> set[tuple]:
    """n-grams (len ANCHOR_MIN..MAX) that occur exactly once in the search window."""
    region = ref_words[start: min(end + cfg.ANCHOR_MAX_LEN, len(ref_words))]
    counts: dict[tuple, int] = {}
    for i in range(len(region)):
        for nlen in range(cfg.ANCHOR_MIN_LEN, min(cfg.ANCHOR_MAX_LEN + 1, len(region) - i + 1)):
            ph = tuple(region[i: i + nlen])
            counts[ph] = counts.get(ph, 0) + 1
    return {ph for ph, c in counts.items() if c == 1}


def _ngram_first_positions(words: list[str]) -> dict[tuple, int]:
    pos: dict[tuple, int] = {}
    for i in range(len(words)):
        for nlen in range(cfg.ANCHOR_MAX_LEN, cfg.ANCHOR_MIN_LEN - 1, -1):
            if i + nlen > len(words):
                continue
            ph = tuple(words[i: i + nlen])
            if ph not in pos:
                pos[ph] = i
    return pos


def compute_anchor_bonus(hyp_pos, ref_pos, unique_ng, n_hyp, n_ref) -> float:
    """Reward candidate windows where unique n-grams appear at matching relative
    positions in hyp and ref (guides matches away from repeated boilerplate)."""
    if not hyp_pos or not ref_pos or not unique_ng:
        return 0.0
    common_unique = (set(hyp_pos) & set(ref_pos)) & unique_ng
    if not common_unique:
        return 0.0
    sorted_phrases = sorted(common_unique, key=lambda p: -len(p))
    bonus = 0.0
    used_hyp: set[int] = set()
    for ph in sorted_phrases:
        plen = len(ph)
        hp = hyp_pos[ph]
        if any(i in used_hyp for i in range(hp, hp + plen)):
            continue
        rp = ref_pos[ph]
        hr = hp / max(n_hyp - plen, 1)
        rr = rp / max(n_ref - plen, 1)
        if abs(hr - rr) <= cfg.ANCHOR_POS_THRESH:
            bonus += plen
            used_hyp.update(range(hp, hp + plen))
    return min(bonus * cfg.ANCHOR_ALPHA, cfg.MAX_ANCHOR_BONUS)


# --------------------------------------------------------------- sliding window
def sliding_window_match(ref_words, hyp_words, expected_pos, n_ref_words,
                         unique_ngrams=None, committed_pos=0):
    """Find the ref window best matching hyp. Returns (start, end, norm_edit_dist)."""
    if n_ref_words <= 0:
        return expected_pos, expected_pos, 1.0
    search_start = max(0, expected_pos - cfg.SEARCH_BUFFER_BACK)
    search_end = min(len(ref_words) - n_ref_words + 1, expected_pos + cfg.SEARCH_BUFFER_FWD)
    search_end = max(search_end, search_start + 1)
    n_ref_words = min(n_ref_words, len(ref_words) - search_start)
    hyp_str = " ".join(hyp_words)
    n_hyp = len(hyp_words)
    hyp_pos = _ngram_first_positions(hyp_words) if unique_ngrams else {}

    bK = min(cfg.BOUNDARY_K, n_hyp // 3) if n_hyp >= 3 else 0
    if bK > 0:
        hyp_head_str = " ".join(hyp_words[:bK])
        hyp_tail_str = " ".join(hyp_words[-bK:])
        hyp_head_len = len(hyp_head_str)
        hyp_tail_len = len(hyp_tail_str)
        hyp_str_len = len(hyp_str)

    best_pos = search_start
    best_effective = float("inf")
    best_raw_dist = float("inf")
    for p in range(search_start, search_end):
        overlap = max(0, committed_pos - p)
        penalty = overlap * cfg.OVERLAP_PENALTY_WORD

        if bK > 0:
            ref_head_str = " ".join(ref_words[p: p + bK])
            head_d = Levenshtein.distance(hyp_head_str, ref_head_str)
            head_max = max(hyp_head_len, len(ref_head_str), 1)
            ref_tail_str = " ".join(ref_words[p + n_ref_words - bK: p + n_ref_words])
            tail_d = Levenshtein.distance(hyp_tail_str, ref_tail_str)
            tail_max = max(hyp_tail_len, len(ref_tail_str), 1)
            boundary_pen = (head_d / head_max + tail_d / tail_max) / 2 * hyp_str_len * cfg.BOUNDARY_ALPHA
        else:
            boundary_pen = 0.0

        if best_effective < float("inf"):
            cutoff = int(best_effective + cfg.MAX_ANCHOR_BONUS - penalty - boundary_pen)
            if cutoff < 0:
                continue
        else:
            cutoff = None
        ref_str = " ".join(ref_words[p: p + n_ref_words])
        dist = Levenshtein.distance(hyp_str, ref_str, score_cutoff=cutoff)
        if cutoff is not None and dist > cutoff:
            continue
        if unique_ngrams and hyp_pos:
            ref_pos = _ngram_first_positions(ref_words[p: p + n_ref_words])
            bonus = compute_anchor_bonus(hyp_pos, ref_pos, unique_ngrams, n_hyp, n_ref_words)
        else:
            bonus = 0.0
        effective = dist + penalty + boundary_pen - bonus
        if effective < best_effective:
            best_effective = effective
            best_raw_dist = dist
            best_pos = p
    match_end = min(best_pos + n_ref_words, len(ref_words))
    ref_str = " ".join(ref_words[best_pos: match_end])
    denom = max(len(hyp_str), len(ref_str), 1)
    return best_pos, match_end, round(best_raw_dist / denom, 4)
