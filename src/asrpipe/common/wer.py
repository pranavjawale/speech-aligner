"""Word-error-rate: scoring-time normalization + corpus/utterance WER.

Ported from /workspace/code/data-for-asr-baseline/run_asr_baseline_eval.py.
This is the EVAL normalizer (distinct from textnorm's matcher/aligner ones):
lowercase, drop everything except [a-z0-9 ], collapse whitespace. Applied
IDENTICALLY to reference and hypothesis so the comparison is fair.

Word-level edit distance via rapidfuzz.distance.Levenshtein (no jiwer dependency).

Two WER definitions (both reported in the frozen experiments):
  corpus WER = sum(word edits) / sum(reference words)      -> WER.corpus_wer
  mean utt WER = mean over utterances of (edits / ref_words) -> WER.mean_utt_wer

Port source : run_asr_baseline_eval.py (normalize + aggregation loop)
Reorg phase : RP2
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation/apostrophes, collapse whitespace (eval-side)."""
    t = text.lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def utt_wer(ref: str, hyp: str) -> float | None:
    """Per-utterance WER. Returns None when the reference is empty (unscored)."""
    rw = normalize(ref).split()
    hw = normalize(hyp).split()
    if not rw:
        return None
    return Levenshtein.distance(rw, hw) / len(rw)


@dataclass
class WerAccumulator:
    """Aggregate edits + reference words across utterances for corpus & mean WER."""
    tot_edits: int = 0
    tot_ref_words: int = 0
    per_utt: list[float] = field(default_factory=list)
    n_scored: int = 0
    n_empty_ref: int = 0

    def add(self, ref: str, hyp: str) -> float | None:
        """Add one (ref, hyp) pair; returns that utterance's WER (or None)."""
        rw = normalize(ref).split()
        hw = normalize(hyp).split()
        if not rw:
            self.n_empty_ref += 1
            return None
        d = Levenshtein.distance(rw, hw)
        self.tot_edits += d
        self.tot_ref_words += len(rw)
        w = d / len(rw)
        self.per_utt.append(w)
        self.n_scored += 1
        return w

    @property
    def corpus_wer(self) -> float | None:
        return self.tot_edits / self.tot_ref_words if self.tot_ref_words else None

    @property
    def mean_utt_wer(self) -> float | None:
        return sum(self.per_utt) / len(self.per_utt) if self.per_utt else None


def corpus_wer(pairs) -> WerAccumulator:
    """Convenience: fold an iterable of (ref, hyp) into a WerAccumulator."""
    acc = WerAccumulator()
    for ref, hyp in pairs:
        acc.add(ref, hyp)
    return acc
