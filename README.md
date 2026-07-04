# README — Alignment: from raw audio + transcript to speaker segments with text

This guide walks you, step by step, through running the **alignment pipeline** on a folder of audio files and their reference transcripts. The end result is: for every audio file, a set of **per-speaker audio segments**, each paired with **its transcribed text and word-level
timings**.


> If you also want to build a train/dev/test dataset and measure ASR accuracy (WER),
> that is a downstream extension covered by `docs/pipeline-guide.md` (stages S10–S13).
> This document stops at the point where you have speaker segments + transcripts,
> which is the goal here.

---

## 0. The big picture (read this once)

You give the pipeline **audio + a reference transcript** for each file. It figures
out **which words were spoken when**, groups consecutive words by speaker into
**segments**, cuts an **audio clip** for each segment, and writes the segment's
**text**. You get back aligned `(audio clip ⇄ transcript)` pairs, per speaker.

```
  INPUT                         ALIGNMENT PIPELINE (this guide)                  OUTPUT
  ───────────────────           ─────────────────────────────────────           ─────────────────────────
  data/wav-files/         ┌──► S2  split audio into ~30s chunks                  data/speaker-segments/
    audio_file_NNN.wav    │    S3  ASR + language-model pre-pass                   audio_file_NNN/
  data/transcripts-v3/    ├──► S4  align pass 1  (locate + time each word)          *_spk_XXXX_seg_*.wav   ← clip
    transcript_NNN.txt    │    S4.5 re-sync      (fix drifted chunks)             data/manifests/
  data/lm/                │    S5  align pass 2  (gap-fill edges)                   audio_file_NNN_
    court_6gram.bin       ├──► S6/S7 confidence trim (clean segment ends)            speaker_manifest.json ← text
    unigrams.txt          │    S8  timeline      (stitch into one word timeline)                        + timings
                          └──► S9  speaker segments (group by speaker, cut clips)
```

The stages are numbered **S2–S9** (S1 is an optional audio-format converter; the
higher-numbered S10–S13 belong to the downstream dataset/eval task). One command
runs all of S2–S9; you can also run any single stage on a single file while learning.

---

## 1. Prerequisites

### 1.1 Hardware
- A **GPU is required** for stages S3, S4, S4.5, and S9's alignment work. The rest is CPU-bound. 

### 1.2 Two Python environments (already built in this repo)
This pipeline uses **two** interpreters. You do not create them — they already exist:
- **main env** — the system `python3` (has whisperx / faster-whisper / pyannote /
  rapidfuzz). Runs **S2, S4, S4.5, S5, S6/S7, S8, S9**.
- **NeMo venv** — `.venv-nemo/bin/python` (NeMo 2.7.3 + torch 2.8). Runs **S3** only
  (the Parakeet-CTC + language-model pre-pass).

The orchestrator script picks the right interpreter for each stage automatically, so
for the normal path you never invoke these by hand.

### 1.3 One-time environment setup — **do this first, every session**

```bash
source /workspace/code2/setup_env.sh
```

**Where we are:** step 0 of the task — turning the machine on.
**What it achieves:** exports your Hugging Face token (needed to download the
pyannote / Whisper / NeMo models on first use), writes it to
`~/.cache/huggingface/token`, and puts `config/` + `src/asrpipe/` on `PYTHONPATH` so
`import config` and `import asrpipe` work. It prints a status summary; check that it
says `NeMo venv ... (ready)`.

---

## 2. Input format — what the pipeline expects

Put your inputs in these three folders (all under `/workspace/code2/data/`). The file number `NNN` is how the pipeline pairs an audio file with
its transcript.

| Input | Folder | Filename | Format / notes |
|---|---|---|---|
| **Source audio** | `data/wav-files/` | `audio_file_NNN.wav` | **16 kHz, mono, 16-bit PCM WAV** (`PCM_S16LE`). `NNN` is zero-padded, e.g. `audio_file_007.wav`. If your audio is mp3/other or a different sample rate, convert it first (see §2.1). |
| **Reference transcript** | `data/transcripts-v3/` | `transcript_NNN.txt` | Plain UTF-8 text, the ground-truth words for that audio. See §2.2 for the exact text layout. `NNN` must match the audio. |
| **Language model** | `data/lm/` | `court_6gram.bin` + `unigrams.txt` | A KenLM n-gram binary + unigram list, used by the S3 pre-pass to produce a strong text hypothesis. Pre-supplied in this repo. |

> In this repo `data/wav-files` is a **symlink** to `/workspace/wav-files`. Don't move
> or delete that target. To add your own audio, drop new `audio_file_NNN.wav` files
> into the folder the symlink points at (or replace the symlink with a real folder).

### 2.1 If your audio isn't 16 kHz mono WAV (S1 — convert)
Convert each file to the required spec, e.g. with ffmpeg:
```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 data/wav-files/audio_file_001.wav
```
(The repo's frozen tree also has a `convert_to_wav.sh` helper.) The pipeline assumes
16 kHz mono throughout (`config.SAMPLE_RATE = 16000`), so this must be done before S2.

### 2.2 Transcript text layout
The transcript is the **reference text to align against**. In this repo the transcripts
are speaker-labelled: a `spk_XXXX:` line, then that speaker's text, then a blank line,
repeating. Example (`transcript_002.txt`):

```
spk_1012:
there's a note .

spk_1001:
yes let's have a quick . a note has been uploaded for us on the screen .

spk_1012:
after a series of discussions with the government on the many provisions of the act ...
```

Notes:
- Text is **lowercase, minimally punctuated** — it matches the Parakeet ASR output
  style (lowercase, no rich punctuation). Keep your references in the same style.
- The `spk_XXXX:` labels carry the speaker identity that ends up on each segment.
  **Speaker ids are treated as global/consistent across files** (the same `spk_1012`
  in two files is the same person), which matters if you later build speaker-disjoint
  splits.

---

## 3. Run the alignment — the one-command path

**Where we are:** the core of the task. Inputs are in place; now we run all alignment
stages S2→S9.

**What it achieves:** takes every audio file in your range and produces the final
speaker segments + transcripts (Section 5 describes exactly what lands on disk).

```bash
cd /workspace/code2
bash scripts/run_pipeline.sh 1 25      # runs files 001..024
```

- The range is **`START END`, end-exclusive**: `1 25` means files 001–024;
  `2 3` means only file_002; `1 4` means files 001, 002, 003.
- Per-stage logs are written to `logs/pipeline/` (`s2.log`, `s3.log`, …). The script
  **aborts on the first failing stage** and prints the tail of that stage's log.
- First run is slower because the models download into `~/.cache`.

That single command runs the eight stages below **in order**. You can run it and skip
to Section 5, or read on to understand what each stage contributes.

### What each stage does (and where you are as it runs)

Progress through the task as the script advances:

```
  [S2]──[S3]──[S4]──[S4.5]──[S5]──[S6/S7]──[S8]──[S9]
   split  asr   align  resync  fill   trim   timeline  segments   ✓ done
```

| Stage | Where we are | What it achieves | Reads → Writes |
|---|---|---|---|
| **S2 — VAD split** | Cut the audio into workable pieces | Voice-activity detection groups speech into **≤30 s chunks** so alignment runs on short spans, not a 1-hour file. | `data/wav-files/*.wav` → `data/chunks/audio_file_NNN/` (chunk WAVs + `_asr.jsonl`) |
| **S3 — LM pre-pass** ⚙️ | Get a strong text guess per chunk | Runs Parakeet-CTC + KenLM beam search to produce a good ASR hypothesis, used next to **locate** each chunk inside the reference transcript. | chunk WAVs → per-chunk hypothesis cache |
| **S4 — Align pass 1** | Place each chunk in the transcript & time its words | Whisper ASR + "best-of-two" localization finds **which span of the reference** a chunk covers, then forced alignment assigns a **start/end time + confidence to every word**. | chunks + hyp → `data/alignments/audio_file_NNN/*.json` |
| **S4.5 — Re-sync** | Repair chunks that landed in the wrong place | Detects **localization drift** (runs of chunks matched to the wrong span of the transcript) and re-places + re-aligns them. Automatic and self-gating: a no-op on clean files. | S4 JSONs → same JSONs (corrected in place) |
| **S5 — Align pass 2** | Recover words missed at chunk edges | Gap-fills head/tail audio the first pass left unaligned, then writes the first on-disk **label files**. | S4 JSONs → JSONs + `alignments/.../labels/` |
| **S6/S7 — Confidence trim** | Clean up ragged segment ends | Trims low-confidence runs of words at the **tail (S6)** then **head (S7)** of each chunk and re-aligns, improving boundary quality. | labels → `*_labels_corrected2.txt` (final per-chunk label) |
| **S8 — Timeline** | Stitch chunks back into one file | Merges all chunks into a single **absolute-time word timeline** for the whole file (word, start, end, score, chunk index). | corrected labels → `data/alignments/audio_file_NNN_timeline.json` |
| **S9 — Speaker segments** | Produce the deliverable | Merges consecutive **same-speaker** words into speaker-turn **segments**, cuts an **audio clip** per segment, and records each segment's text, timings, and confidence. | timeline → `data/speaker-segments/` + `data/manifests/` |

⚙️ = runs under the NeMo venv (handled for you by the script).

> **Tip — run a single stage on a single file.** Every stage is a module CLI.
> To watch one file go through one stage (end-exclusive range `2 3` = file_002 only):
> ```bash
> python3               -m asrpipe.split.vad_split           2 3   # S2
> .venv-nemo/bin/python -m asrpipe.align.lm_prepass          2 3   # S3 (NeMo venv)
> python3               -m asrpipe.align.align_p1             2 3   # S4
> python3               -m asrpipe.align.resync              2 3 --apply   # S4.5
> python3               -m asrpipe.align.align_p2             2 3   # S5
> python3               -m asrpipe.align.conf_trim            2 3   # S6/S7
> python3               -m asrpipe.align.timeline             2 3   # S8
> python3               -m asrpipe.segments.speaker_segments  2 3   # S9
> ```
> This is the fastest way to sanity-check on one small file before a full-corpus run.

---

## 4. (Optional) Flatten to a simple `audio ⇄ text` list

**Where we are:** the alignment task is already complete after S9. This optional step
just repackages the result into the most convenient form.

**What it achieves:** flattens the per-file manifests into **one JSON row per segment**
(`audio_filepath`, `duration`, `text`, `speaker`, …) — the cleanest
"here is a clip and here is its transcript" table, one file per audio file.

```bash
python3 -m asrpipe.asr.manifests 1 25
```

Output: `data/asr/manifests/segments/segments_NNN.jsonl`. Each line looks like:

```json
{"audio_filepath": "/workspace/code2/data/speaker-segments/audio_file_001/audio_file_001_spk_0100_seg_000000_start000001_end000003.wav",
 "duration": 1.666, "text": "my lord this is a demolition matter",
 "file": "audio_file_001", "unit": "segment", "speaker": "spk_0100",
 "seg_index": 0, "avg_score": 0.5723, "min_score": 0.12}
```

This is the format most downstream tools (dataset builders, ASR trainers) expect.

---

## 5. The final output — where it is and what it looks like

After S9 (Section 3), for each input file you get **two** things:

### 5.1 The segment audio clips
```
data/speaker-segments/audio_file_NNN/
    audio_file_NNN_spk_1012_seg_000000_start000029_end000033.wav
    audio_file_NNN_spk_1001_seg_000001_start000036_end000037.wav
    ...
```
- One **WAV clip per speaker segment**, 16 kHz mono.
- The filename is **self-describing**: source file, `spk_<id>` (who spoke), a segment
  index, and `start<sec>_end<sec>` (position in the original audio). You can pair a
  clip with its transcript from the filename alone — no database needed.

### 5.2 The transcript + metadata (per-file manifest)
```
data/manifests/audio_file_NNN_speaker_manifest.json
```
A JSON document with:
- `source_file`, `total_segments`, and a `speaker_stats` block
  (`{spk → {segments, total_duration_s, total_words}}`),
- a `segments` array — **one entry per clip in 5.1** — carrying the transcript and
  timing. One segment entry:

```json
{
  "seg_index": 0,
  "speaker_id": "spk_1012",
  "audio_file": "audio_file_002_spk_1012_seg_000000_start000029_end000033.wav",
  "start_abs": 29.1078, "end_abs": 33.1718, "duration": 4.064,
  "word_count": 3, "avg_score": 0.621, "min_score": 0.524,
  "words": [
    {"word": "there's", "start_abs": 29.1078, "end_abs": 32.2518, "score": 0.615},
    {"word": "a",       "start_abs": 32.3118, "end_abs": 32.4318, "score": 0.524},
    {"word": "note",    "start_abs": 32.5118, "end_abs": 33.1718, "score": 0.724}
  ]
}
```

**How to read it:**
- The segment's **transcript** is the `words[*].word` joined in order (`"there's a
  note"`). If you ran Section 4, that same text is the explicit `text` field.
- `audio_file` names the matching clip in `data/speaker-segments/audio_file_NNN/`.
- `start_abs` / `end_abs` are **absolute seconds in the original audio**; each word
  also carries its own time span and a **confidence `score`** (0–1).
- `avg_score` / `min_score` summarize the segment's alignment confidence. This is the
  main **quality knob**: the canonical dataset keeps only segments with
  `avg_score ≥ 0.4` (`config.CONF_THRESHOLD`). Low-confidence segments are where you'd
  expect alignment mistakes.

### 5.3 In one sentence
> For each speaker turn you get a **WAV clip** in `data/speaker-segments/…` and its
> **word-by-word transcript with timings and confidence** in the matching
> `data/manifests/audio_file_NNN_speaker_manifest.json` (or, after §4, as a flat
> `{audio_filepath, text}` row in `data/asr/manifests/segments/segments_NNN.jsonl`).

---

## 6. Good to know (gotchas)

- **End-exclusive ranges.** `START END` = `[START, END)`. `1 25` = files 001–024;
  `2 3` = only file_002.
- **Wrong interpreter for S3 fails.** S3 must run under `.venv-nemo/bin/python`; the
  orchestrator handles this, but if you run stages by hand, use the NeMo venv for S3.
- **`config/config.py` is the control panel.** Every tunable — VAD granularity, the
  30 s chunk cap, the 0.4 confidence threshold, speaker-merge gap, minimum segment
  duration — lives there, grouped by stage. Change behaviour there, not in the modules.
- **Quarantine.** `config.QUARANTINE_FILES` (currently `{15}`) marks known bad-audio
  files to exclude from the downstream dataset. It does not stop S2–S9 from processing
  them; it drops them later. Inspect per-segment `avg_score` to spot bad audio yourself.
- **Confidence is your quality signal.** If a file's segments are mostly low
  `avg_score`, suspect bad audio or a transcript that doesn't match the recording.

## 7. Quick reference — the whole thing

```bash
# 0. every session
source /workspace/code2/setup_env.sh

# 1. put inputs in place
#    data/wav-files/audio_file_NNN.wav        (16 kHz mono 16-bit WAV)
#    data/transcripts-v3/transcript_NNN.txt   (matching reference text)

# 2. run all alignment stages S2–S9 for your file range (end-exclusive)
cd /workspace/code2
bash scripts/run_pipeline.sh 1 25

# 3. (optional) flatten to one-row-per-segment audio⇄text manifests
python3 -m asrpipe.asr.manifests 1 25

# 4. read the output
#    clips : data/speaker-segments/audio_file_NNN/*.wav
#    text  : data/manifests/audio_file_NNN_speaker_manifest.json
#    flat  : data/asr/manifests/segments/segments_NNN.jsonl   (after step 3)
```

For deeper stage internals and the downstream dataset/WER work, see
`docs/pipeline-guide.md`.
