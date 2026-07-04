"""NON-CIRCULAR WER bake-off — NOT PORTED (belongs in archive/).

phase13_indep_wer.py compares MULTIPLE alignment arms (strat-5, 6a, 6b, 6c) by
scoring each arm's located reference against an independent rnnt-1.1b decode. code2
is 6c-ONLY (Q8) — there are no other arms to compare — so this multi-arm bake-off
tool does not apply to the clean single-strategy pipeline.

Its historical results are recorded in docs/experiment-logs/ and
docs/better-asr-alignment-plan.txt. The original script remains in /workspace/code
for reference; archive/ is populated at RP8.

Reorg decision : RP5 — intentionally NOT ported (multi-arm; out of scope for 6c-only).
"""
raise NotImplementedError(
    "indep_wer is a multi-arm bake-off tool — not applicable to code2 (6c-only). "
    "See docstring; original at /workspace/code/phase13_indep_wer.py.")
