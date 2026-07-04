"""Reference/hypothesis text normalization.

Two normalizers, ported verbatim from /workspace/code/align_chunks_withmask.py:

  clean_text            — for the sliding-window matcher: strips speaker labels,
                          removes apostrophes SILENTLY ("i'll" -> "ill"), replaces
                          every non-alpha char with a space, lowercases.
  clean_text_for_align  — for forced alignment: strips speaker labels, KEEPS
                          within-word apostrophes ("i'll", "don't"), drops leading/
                          trailing apostrophes and sentence punctuation, lowercases.

INVARIANT (relied on by the matcher): both functions produce the SAME word count
for any input, so match indices from clean_text are valid indices into the
clean_text_for_align word list. Verified by tests/test_common.py.

Port source : align_chunks_withmask.py (lines 73-84)
Reorg phase : RP2
"""
from __future__ import annotations

import re

_SPK_LABEL = re.compile(r"spk_\w+:\s*")      # "spk_1023: " -> ""
_APOS      = re.compile(r"'")
_NON_ALPHA = re.compile(r"[^a-zA-Z]")
_NON_ALPHA_KEEP_APOS = re.compile(r"[^a-zA-Z\s']")
_EDGE_APOS = re.compile(r"(?<!\w)'|'(?!\w)")   # apostrophe not inside a word
_WS        = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Matcher normalization: apostrophes removed silently, non-alpha -> space."""
    text = _SPK_LABEL.sub("", text)
    text = _APOS.sub("", text)
    text = _NON_ALPHA.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def clean_text_for_align(text: str) -> str:
    """Alignment normalization: within-word apostrophes kept (contractions intact)."""
    text = _SPK_LABEL.sub("", text)
    text = _NON_ALPHA_KEEP_APOS.sub(" ", text)
    text = _EDGE_APOS.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def words(text: str, *, for_align: bool = False) -> list[str]:
    """Convenience: normalized word list."""
    return (clean_text_for_align(text) if for_align else clean_text(text)).split()
