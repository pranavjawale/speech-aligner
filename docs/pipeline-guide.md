# asrpipe pipeline guide

A developer's guide to the `code2` / `asrpipe` pipeline: take raw audio + reference
transcripts, **chunk** the audio, **align** it to the text at word level, build a
speaker-disjoint **train/dev/test** set, and **evaluate** ASR WER on it.

> Audience: a developer who wants to run or adapt this pipeline on their own audio.


---

## 1. What the pipeline does (in one picture)

```
                 S2          S3–S8                    S9                 S10–S12         S13
 wav-files/  →  chunks/  →  alignments/           speaker-segments/  →  asr/splits/  →  asr/results/
 transcripts-v3/            (word-level JSON        + manifests/          (train/dev/     (WER +
   (input)                   + labels + timeline)                         test.jsonl)     per-speaker)
```

Concretely, one audio file (`audio_file_NNN.wav`) is:
1. split by voice-activity detection into **~30 s chunks** (S2),
2. transcribed and **force-aligned** to its reference transcript, producing a
   word-level timeline with per-word confidence scores (S3–S8),
3. regrouped into **per-speaker segments** with their own audio clips (S9),
4. partitioned across files into **speaker-disjoint** train/dev/test splits,
   filtered by confidence, then optionally silence-padded (S10–S12),
5. **scored** for WER by two Parakeet models (S13).

The output splits (`data/asr/splits/`) are the training data; the WER summaries
(`data/asr/results/`) are the evaluation.

---

## 2. Prerequisites

### 2.1 Environments (two, kept separate)
- **main env** — system `python3` with `requirements.txt` installed
  (whisperx, faster-whisper, pyannote, kenlm, pyctcdecode). Runs S2, S4–S12, reports.
- **NeMo venv** — `.venv-nemo/` (NeMo 2.7.3 + torch 2.8, from `requirements-nemo.txt`).
  Runs S3 (Parakeet-CTC pre-pass) and S13 (Parakeet eval). Invoked as
  `.venv-nemo/bin/python`.

A GPU is required for S3, S4 (Whisper), S4.5 (re-sync `--apply`), and S13. The rest is CPU-bound.

### 2.2 One-time setup
```bash
source /workspace/code2/setup_env.sh
```
This exports `HF_TOKEN` (needed to download pyannote/whisper/NeMo models),
writes it to `~/.cache/huggingface/token`, and puts `config/` + `src/asrpipe/`
on `PYTHONPATH` (so `import asrpipe` and `import config` work without a
`pip install -e .`). It also prints whether the NeMo venv is present.

### 2.3 Inputs the pipeline expects
| Input | Location | Notes |
|---|---|---|
| Source audio | `data/wav-files/` | 16 kHz mono PCM_S16LE WAV, named `audio_file_NNN.wav`. In this repo it is a **symlink** to `/workspace/wav-files` (Q3) — do not move/delete that target. |
| Reference transcripts | `data/transcripts-v3/` | one `transcript_NNN.txt` per audio file, the ground-truth text to align against. |
| Language model | `data/lm/` | `court_6gram.bin` (KenLM 6-gram) + `unigrams.txt`, feeding the S3 LM pre-pass. Pre-copied (Q2); not rebuilt by code2. |

If your audio is mp3/other, convert to the WAV spec first (S1 — see §4.4). The
file-numbering convention `audio_file_NNN` / `transcript_NNN` is how stages pair
audio with its transcript; keep it consistent.

---

## 3. How to run

### 3.1 Everything, end to end (the verified path)
```bash
source /workspace/code2/setup_env.sh
bash scripts/run_rp6.sh            # S2→S13 for all 24 files, then prints WER
```
Per-stage logs land in `logs/rp6/`; the driver aborts on the first stage failure
and prints the final WER-vs-frozen comparison at the end.

### 3.2 By track (thin wrappers over the same modules)
Each wrapper sources `setup_env.sh` and logs under `logs/<name>/`. The `START END`
range is **end-exclusive**, so `1 25` means files 001–024.
```bash
bash scripts/run_pipeline.sh 1 25          # S2–S9   : audio → speaker segments
bash scripts/run_training_data.sh 1 25     # S10–S12 : segments → splits (+pad)
bash scripts/run_eval.sh                   # S13     : WER of ctc + rnnt on padded test
bash scripts/run_reports.sh 1 25           # 4 corpus reports (reads S9/S13 outputs)
```

### 3.3 A single stage / a single file
Every stage is a module CLI. Use the NeMo python for S3 and S13:
```bash
python3               -m asrpipe.split.vad_split           2 3    # just file_002
.venv-nemo/bin/python -m asrpipe.align.lm_prepass          2 3    # S3 (NeMo)
python3               -m asrpipe.align.align_p1             2 3
python3               -m asrpipe.align.resync               2 3 --apply   # S4.5 (drift-only; no-op if none)
python3               -m asrpipe.align.align_p2             2 3
python3               -m asrpipe.align.conf_trim            2 3
python3               -m asrpipe.align.timeline             2 3
python3               -m asrpipe.segments.speaker_segments  2 3
```
This is the fastest way to iterate on one stage — verify on a single small file
before a full-corpus run.

### 3.4 Not wired
- `00_build_lm.sh` — only checks the LM assets are present; the LM is pre-copied,
  not rebuilt. To rebuild, use `/workspace/code/build_ngram_lm.py`.
- `run_finetune.sh` — RP7 follow-up placeholder; fine-tuning is not implemented yet.

---

## 4. Stage reference

Notation: ⚙️ = runs under the NeMo venv. `NNN` = zero-padded file number.
Every chunk/segment filename encodes `start<sec>_end<sec>` (and, for segments, the
speaker id), so files are self-describing and joinable without a database.

### S2 — VAD split  (`asrpipe.split.vad_split`)
- **In:** `data/wav-files/audio_file_NNN.wav`
- **Out:** `data/chunks/audio_file_NNN/`
  - chunk WAVs: `audio_file_NNN_chunk_000001_start000028_end000053.wav`
  - `audio_file_NNN_asr.jsonl` — per-chunk speech-segment metadata
- **What:** pyannote VAD groups speech into ≤30 s chunks.
- **Key config:** `VAD_ONSET`, `VAD_OFFSET`, `VAD_CHUNK_SIZE`, `VAD_MIN_DUR_ON/OFF`.

### S3 — LM pre-pass ⚙️  (`asrpipe.align.lm_prepass`)
- **In:** chunk WAVs
- **Out:** the "ctc_lm" hypothesis per chunk (consumed by S4's best-of-two)
- **What:** Parakeet-CTC-0.6b forward pass + `pyctcdecode` beam search with the
  KenLM 6-gram and unigrams — a strong text hypothesis to help localize the chunk
  in the reference transcript.
- **Key config:** `CTC_LM_MODEL`, `KENLM_ALPHA/BETA/BEAM_SIZE`, `PREPASS_WORKERS`.
- **Note:** GPU kernel autotuning makes this the one stage with tiny run-to-run
  drift (see §7).

### S4 — Align pass 1  (`asrpipe.align.align_p1`, + `asrpipe.align.matcher`)
- **In:** chunks, S3 hypothesis
- **Out:** `data/alignments/audio_file_NNN/audio_file_NNN_chunk_*.json`
- **Format:** per-chunk JSON —
  ```json
  {"source_file": "...", "chunk_index": 1, "chunk_start_s": 28.0, "chunk_end_s": 53.0,
   "asr_hypothesis": "...", "ref_match": "...",
   "words": [{"word": "there's", "start": 0.28, "end": 0.641, "score": 0.397, "pass": 2}]}
  ```
- **What:** Whisper (faster-whisper) greedy ASR → **best-of-two** reference
  localization (Parakeet+KenLM hyp vs Whisper hyp, sliding-window matcher) →
  silence-masked wav2vec2 forced alignment → per-word timestamps + confidence.
- **Key config:** `ASR_MODEL_SIZE`, `SEARCH_BUFFER_BACK/FWD`, the `COMMIT_*` matcher knobs, `SILENCE_MASK_THRESHOLD_S`.
- **Writes only JSON** — the per-chunk `.json`, `{stem}_hyp_winner_stats.json`, and
  the `{stem}_asr.jsonl` hyp cache. **S4 does NOT create any `labels/` files** — the
  pass-1 alignment lives only in the JSON `words` array at this point.

### S4.5 — Re-sync  (`asrpipe.align.resync`)   *(drift correction; automatic, drift-only)*
- **In:** the S4 alignment JSONs (reads `ref_match.norm_edit_dist` per chunk + the
  cached `{stem}_asr.jsonl` hyps — no Whisper re-run).
- **Out:** for chunks inside a detected **drift run**, the same JSONs rewritten in
  place (corrected `ref_match` localization + re-run forced alignment) and tagged
  `"resync": {"resynced": true, ...}`, plus a per-file
  `data/review/{stem}_resync_report.txt`. Clean files are **not touched** (byte-identical).
- **What:** fixes *localization drift* — long runs of chunks matched against the wrong
  span of the reference (repetitive text starves the unique-anchor matcher, and the
  error cascades via `expected_pos`). Detects drift runs from the smoothed
  `norm_edit_dist` (hysteresis), re-localizes each drifted chunk by rare-weighted
  n-gram anchor **voting** constrained between the confident neighbours (monotonicity
  guard), then re-aligns with silence-masked wav2vec2.
- **Self-gating:** a no-op when no drift run is found, so it always runs but only
  changes drifted files (preserves byte-identity for clean ones).
- **Modes:** `--apply` rewrites JSONs + re-aligns (needs GPU); default (no flag) is a
  non-destructive **report** (CPU) that also serves as a drift-vs-bad-audio discriminator.
- **Why here (between S4 and S5):** localization is decided in S4; S5 gap-fill and the
  S6/S7 trims only refine edges and cannot relocate a chunk — so drift must be
  corrected *before* S5 runs.
- **Key config (module constants):** `HIGH=0.55`, `LOW=0.45`, `SMOOTH_W=3`,
  `MIN_RUN=2`, `NGRAMS=(5,4,3)`, `MARGIN=0.15`.
- **Full detail:** `docs/resync-fix.txt`.

### S5 — Align pass 2  (`asrpipe.align.align_p2`)
- **In:** the S4 alignment JSONs
- **Out:** the same JSONs rewritten **in place** (gap-fill words tagged `"pass": 2`),
  **and** — this is where the `labels/` directory is first populated — for each chunk:
  - `labels/{chunk}_labels.txt` — Audacity label file for the **post-pass2** words
    (i.e. pass-1 + pass-2 combined), via `common.labels.write_labels`.
  - `labels/{chunk}.txt` — the plain matched **reference text** (`ref_match.matched_text`), no timing.
- **What:** gap-fills tail/head audio the first pass left unaligned, then emits the
  first on-disk label files.
- **Key config:** `GAP_THRESHOLD_S`, `GAP_OVERSHOOT`, `GAP_BACKOFF_S/FORWARD_S`.
- **Note:** because pass 2 writes back over the S4 JSONs, always verify S4+S5
  together, not S4 alone. `_labels.txt` therefore reflects pass-1 **and** pass-2 — it
  is not a "pass-1 only" artifact.

### S6/S7 — Confidence trim  (`asrpipe.align.conf_trim`)
- **In:** the S4/S5 JSON `words` + the `labels/{chunk}_labels.txt` from S5.
- **Out (same `labels/` dir):**
  - **S6 `tail_pass`** → `labels/{chunk}_labels_corrected1.txt` — trims a
    low-confidence **tail** run and re-aligns. If no tail trim is warranted, it copies
    `_labels.txt` verbatim, so **corrected1 always exists**.
  - **S7 `head_pass`** → `labels/{chunk}_labels_corrected2.txt` — reads corrected1,
    trims a low-confidence **head** run and re-aligns. If no head trim, it copies
    corrected1, so **corrected2 always exists** (= the final label after all 4 passes).
- **What:** drops low-confidence runs of words at the tail (S6) then head (S7) of a
  chunk and re-aligns, improving boundary quality.
- **Key config:** `TOP_HALF_MIN`, `CONTRAST_THRESH`, `RIGHT_AVG_MAX`,
  `RIGHT_AVG_ADAPTIVE_K`, `MIN_KEEP_WORDS`, `MIN_TRIM_WORDS`.

#### Label files under `data/alignments/audio_file_NNN/labels/` — who writes what
For each chunk there are four files; this is the exact producer of each (verified in code):

| File (per chunk) | Written by | Stage | Content / format |
|---|---|---|---|
| `{chunk}.txt` | `align_p2.py` | **S5** | plain matched **reference text** (`ref_match.matched_text`), no timing |
| `{chunk}_labels.txt` | `align_p2.py` → `common.labels.write_labels` | **S5** | Audacity `start⇥end⇥word (score)`; words after **pass-1 + pass-2** |
| `{chunk}_labels_corrected1.txt` | `conf_trim.py` `tail_pass` | **S6 (tail trim)** | same Audacity format, after **tail** trim + re-align (or a copy of `_labels.txt` if no trim) |
| `{chunk}_labels_corrected2.txt` | `conf_trim.py` `head_pass` | **S7 (head trim)** | same format, after **head** trim + re-align on top of corrected1 — the **final** label; consumed by S8 |

All Audacity files use `common.labels.write_labels`: tab-separated `start⇥end⇥word (score)`,
times to 3 decimals, confidence in parentheses. S8 reads **corrected2** (falling back
corrected1 → `_labels.txt` if a stage was skipped).

### S8 — Timeline  (`asrpipe.align.timeline`)
- **In:** corrected labels
- **Out:** `data/alignments/audio_file_NNN_timeline.json`
- **What:** stitches all chunks into one **absolute-time** word timeline for the
  file (word, start/end, score, chunk_index, ref_word_idx).

### S9 — Speaker segments  (`asrpipe.segments.speaker_segments`)
- **In:** the file timeline
- **Out:**
  - segment audio: `data/speaker-segments/audio_file_NNN/audio_file_NNN_spk_1012_seg_000175_start...end....wav`
  - `data/manifests/audio_file_NNN_speaker_manifest.json`
- **Format (manifest):** `total_segments`, `speaker_stats{spk → segments/duration/words}`,
  and per-segment `source_chunks`, `avg_score`, `min_score`, `scored_words`.
- **What:** merges consecutive same-speaker words into speaker-turn segments, cuts
  the audio clip for each, and computes confidence stats.
- **Key config:** `SEGMENT_MERGE_GAP_S`, `MIN_SEG_DUR_S`, `CONF_THRESHOLD` (0.4).

### S10 — Manifests  (`asrpipe.asr.manifests`)
- **In:** speaker manifests
- **Out:** `data/asr/manifests/segments/` and `.../chunks/`
- **What:** flattens the per-file manifests into NeMo-style rows
  (`audio_filepath`, `duration`, `text`, `avg_score`, …).

### S11 — Partition  (`asrpipe.asr.partition`)  *(corpus-level; no file range)*
- **In:** all segment rows + the fixed `speaker_assignment.json`
- **Out:** `data/asr/splits/{train,dev,test}.jsonl` + `partition_summary.txt`
  + `speaker_assignment.json`
- **What:** **speaker-disjoint** train/dev/test split, keeping only segments with
  `avg_score ≥ 0.4`. Reuses a fixed speaker→split assignment for comparability.
- **Key config:** `SPLIT_SEED` (42), `SPLIT_THRESHOLD` (0.4), `SPLIT_DEV_MIN_S`.
- **Current numbers:** train 94 spk / 28 410 segs / 32.1 h · dev 6 / 536 / 0.61 h ·
  test 20 / 3 390 / 3.65 h.

### S12 — Pad  (`asrpipe.asr.pad`)
- **In:** a split jsonl
- **Out:** `data/asr/splits-padded/<split>.jsonl` (+ new padded WAVs)
- **What:** prepends and appends `PAD_SECONDS` (0.5 s) of silence to each segment
  (a small, consistent WER win). `--only test` pads just the test split (the eval
  gate); pass `train,dev,test` to pad all.

### S13 — Eval ⚙️  (`asrpipe.asr.eval`)
- **In:** a (padded) manifest
- **Out:** `data/asr/results/{ctc06b,rnnt11b}_padded_test_6C.jsonl` + `.summary.json`
- **Format (summary):** `corpus_wer`, `mean_utt_wer`, `utterances`, `rtfx`, and a
  `per_speaker{ "audio_file|spk": {segs, ref_words, wer} }` block.
- **What:** decodes the manifest with `ctc-0.6b` (the fine-tune candidate) and
  `rnnt-1.1b` (independent, non-circular judge) and reports WER.
- **Key config:** `EVAL_MODEL_CTC/RNNT`, `EVAL_DECODING`, `EVAL_BATCH_SIZE`.

### S1 — Convert (not usually run)
`convert_to_wav.sh` in the frozen tree turns mp3/other → 16 kHz mono WAV. In this
repo the source WAVs already exist, so S1 is a no-op — run it only if new mp3s
appear.

---

## 5. Data layout reference

All data lives under `data/` (paths derived from `config.py`):
```
data/
  wav-files/                 SYMLINK → source 16k mono WAVs (input)
  transcripts-v3/            reference text, transcript_NNN.txt (input)
  lm/                        court_6gram.bin + unigrams.txt (input, pre-copied)
  chunks/audio_file_NNN/     S2: chunk WAVs + _asr.jsonl
  alignments/audio_file_NNN/ S4/S5: per-chunk *.json ; labels/ (S6/S7)
  alignments/audio_file_NNN_timeline.json   S8
  speaker-segments/audio_file_NNN/          S9: per-speaker segment WAVs
  manifests/audio_file_NNN_speaker_manifest.json   S9
  asr/manifests/{segments,chunks}/          S10
  asr/splits/{train,dev,test}.jsonl         S11 (+ partition_summary.txt, speaker_assignment.json)
  asr/splits-padded/<split>.jsonl           S12
  asr/results/*.summary.json                S13
```
A training row (what a fine-tune would consume):
```json
{"audio_filepath": ".../audio_file_022_spk_1035_seg_001001_....wav", "duration": 1.621,
 "text": "industrial alcohol is not potable", "speaker": "spk_1035",
 "avg_score": 0.6552, "min_score": 0.264}
```

---

## 6. Reading the results

- **Headline WER** → `corpus_wer` in `data/asr/results/*.summary.json`
  (and echoed in `logs/rp6/driver.log`).
- **Reference numbers** to reproduce (the acceptance gate, `config.FROZEN`):
  ctc-0.6b padded-test **36.73 %**, rnnt-1.1b **34.37 %** on the 20-speaker test set.
- **Per-speaker WER** → the `per_speaker` block. This is where data-quality problems
  surface: a handful of speakers/files sit at 80–99 % WER (bad source audio) and
  drag the corpus number — e.g. `spk_1026`/file_015, `spk_1031`/file_004,
  `spk_1032`/file_007, `spk_1019`/file_024. These are candidates to
  drop/quarantine before training (see `fine-tune-phase.txt`).
- **Corpus reports** (`scripts/run_reports.sh`) write to `reports-out/`:
  speech-recovery, confidence-filter, speaker-duration, per-speaker-WER.

---

## 7. Gotchas & notes

- **End-exclusive ranges.** `START END` means `[START, END)`. `1 25` = files 001–024;
  `2 3` = only file_002.
- **NeMo venv for S3 & S13.** Running these under the main `python3` will fail —
  use `.venv-nemo/bin/python` (the wrappers already do).
- **S3 GPU env-drift.** The LM pre-pass is not bit-reproducible run to run (cuDNN
  autotuning); it can flip a few best-of-two windows, so the final WER lands
  *within run-to-run noise* of the frozen numbers, not exactly equal. This is
  expected — acceptance is "within noise," not equality.
- **Symlinked input.** `data/wav-files` points at `/workspace/wav-files`. Moving or
  deleting that target breaks regeneration from S2 onward.
- **Confidence threshold 0.4** is the single most important data-quality knob
  (`CONF_THRESHOLD` / `SPLIT_THRESHOLD`): segments below it are dropped at split time.
- **Config is the control panel.** Every tunable (VAD, LM weights, matcher anchors,
  trim thresholds, split seed, pad seconds, eval models) lives in `config/config.py`,
  grouped by stage. Change behavior there, not in the stage modules.

---

## 8. Adapting to your own data — checklist

1. Put 16 kHz mono WAVs in `data/wav-files/` as `audio_file_NNN.wav` (or convert with S1).
2. Put matching reference transcripts in `data/transcripts-v3/` as `transcript_NNN.txt`.
3. Provide a language model in `data/lm/` (or rebuild via the frozen tree's
   `build_ngram_lm.py`) — or relax the S3 LM dependency if you don't have one.
4. Review the stage constants in `config/config.py` (VAD granularity, the 30 s chunk
   size, the 0.4 confidence threshold, split seed, pad seconds).
5. Run `scripts/run_pipeline.sh 1 <N+1>` on your file range, then
   `scripts/run_training_data.sh`, then `scripts/run_eval.sh`.
6. Inspect `data/asr/splits/partition_summary.txt` and the WER summaries; use the
   per-speaker WER to spot and quarantine bad-audio files before training.

---

## 9. Phase 15 — fine-tune + evaluation (downstream of the pipeline)

Once the splits + pseudo-labels exist, fine-tune an ASR model on them. Full running record
(hyperparameters, per-epoch telemetry, results, and a **DEVELOPER RUNBOOK** for launching your
own experiments) is in **`docs/fine-tune-phase.txt`**; every script + how to use it is in
**`scripts/list-of-scripts.txt`**. In brief:

1. **Edit** `src/asrpipe/finetune/conf/ctc_finetune.yaml` (manifests, batch/max_duration, LR,
   freeze_encoder_epochs, exp_manager.name). **Launch**: `bash scripts/run_finetune.sh`
   (`--fast-dev-run` first). **Watch**: `bash scripts/tb_metrics.sh` (browser-free).
2. **Convert** the best `.ckpt` → deployable `.nemo`:
   `.venv-nemo/bin/python -m asrpipe.finetune.ckpt_to_nemo --ckpt <...>.ckpt --base <model> --out <...>.nemo`.
3. **Evaluate** (`asr.eval` now loads a local `.nemo`, and supports `--kenlm` shallow fusion):
   `.venv-nemo/bin/python -m asrpipe.asr.eval --manifest <test.jsonl> --model <best.nemo>
   --exp-name <name> --finetuned --padded --batch-size 32`.
4. **Compare / analyze**: `asrpipe.reports.compare_experiments` (multi-expt table + gate),
   `asrpipe.reports.wer_segment_report` (per-segment C/S/D/I + 5 WER variants),
   `wer_error_analysis` (top S/D/I errors), `wer_by_pass1_winner` (WER by which ASR won pass-1).

Reference result (2026-07-03): parakeet-ctc-0.6b fine-tuned on `legal2023_38hrs` reached
**18.27%** padded-test corpus WER vs **23.39%** base. A 3-gram-LM shallow fusion on top HURT
it (20.09%) — a strong in-domain fine-tune usually doesn't benefit from an in-domain n-gram.
