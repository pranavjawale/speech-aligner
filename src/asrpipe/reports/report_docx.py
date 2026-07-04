"""Generate the pipeline report .docx — DEFERRED (low priority).

gen_report_docx.py builds the human-facing Audio_Pipeline_Report.docx from the
pipeline analyses. It is a one-off documentation artifact, not part of the data
pipeline or the acceptance gate, so it is deferred: the underlying numbers are all
reproduced by the ported reports (speech_recovery / conf_filter / speaker_duration /
per_speaker_wer). Port on demand if the .docx needs regenerating.

Reorg decision : RP5 — deferred (documentation artifact; original at
                 /workspace/code/gen_report_docx.py).
"""
raise NotImplementedError(
    "report_docx is a deferred documentation artifact — port on demand. "
    "Original at /workspace/code/gen_report_docx.py.")
