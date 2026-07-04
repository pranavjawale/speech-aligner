# README — Fine-tune: adapting an ASR model to your data (and evaluating it)

This guide is for someone **new to this repo** who wants to **fine-tune a speech-recognition
model** on the speaker segments the alignment pipeline produced, and then **measure how much
it improved** — using plain greedy decoding, **no language model** (the simple case).

It picks up where `README-alignment.md` leaves off: that guide gave you speaker segments +
transcripts and split them into `train / dev / test` manifests; this guide turns those into a
better model. Each step tells you **(a) where you are** in the overall task, **(b) what the
step achieves**, and **(c) the exact command**.

> Scope: fine-tuning `nvidia/parakeet-ctc-0.6b` on our `legal2023_38hrs` dataset, the path
> that is actually implemented and validated in this repo. Full running record (per-epoch
> curve, results, deep runbook): `docs/fine-tune-phase.txt`.

---

## 0. The big picture (read this once)

An off-the-shelf ASR model is good at general English but makes domain mistakes on our
legal/court audio. **Fine-tuning** continues training that model on **our** audio+transcript
pairs so it adapts to the vocabulary, speakers, and acoustics — while being careful **not to
forget** what it already knows. Then we **evaluate** the fine-tuned model on a held-out test
set and compare its word-error-rate (WER) to the original.

```
  INPUT (from the alignment pipeline)          FINE-TUNE (this guide)                 OUTPUT
  ─────────────────────────────────────        ─────────────────────────────         ─────────────────────────
  data/asr/splits-<dataset>/                    train_ctc.py:                          exp/finetune/<name>/
    train.jsonl   (audio ⇄ text, per seg)  ┌──►   load base parakeet-ctc-0.6b            checkpoints/
    dev.jsonl     (early-stopping set)      │      freeze encoder 2 epochs                  ...val_wer=0.1359-epoch=14.ckpt
  data/asr/splits-<dataset>-padded/         ├──►   unfreeze, train to early-stop            best_val_wer0.1359_ep14.nemo  ← deployable
    test.jsonl    (held-out eval set)       │    eval.py (greedy, no LM):
  nvidia/parakeet-ctc-0.6b  (base model) ───┘      decode test.jsonl, score WER          data/asr/results/ + docs/experiment-logs/
```

Two things flow in: the **base model** (downloaded) and the **manifests** (from the alignment
pipeline). Two things come out: a **fine-tuned model** (`.nemo`) and a **WER report**.

---

## 1. Prerequisites

### 1.1 Hardware
- A **GPU is required**. This was developed on one NVIDIA RTX A4500 (20 GB). Memory is the
  main constraint (see the OOM notes in §7); 20 GB is enough with the settings shipped here.

### 1.2 The NeMo environment
Fine-tuning and evaluation run **only** under the **NeMo venv** (`.venv-nemo/bin/python`,
NeMo 2.7.3 + torch 2.8) — *not* the system `python3`. The wrapper scripts already use the
right interpreter, so you rarely invoke it by hand.

### 1.3 One-time setup — do this first, every session
```bash
source /workspace/code2/setup_env.sh
```
**Where we are:** step 0 — turning the machine on.
**What it achieves:** exports the Hugging Face token (needed to download the base model on
first use) and puts `config/` + `src/asrpipe/` on `PYTHONPATH`. Confirm the summary line
prints `NeMo venv ... (ready)`.

---

## 2. Input format — what fine-tuning consumes

Fine-tuning does **not** read raw audio + transcripts directly. It reads the **NeMo manifests**
the alignment pipeline produced (`README-alignment.md` → `run_training_data.sh`). Each split is
a `.jsonl` file, **one JSON object per line = one training example**:

```json
{"audio_filepath": ".../audio_file_022_spk_1035_seg_001001_....wav", "duration": 1.621,
 "text": "industrial alcohol is not potable", "speaker": "spk_1035",
 "avg_score": 0.6552, "min_score": 0.264}
```

| Field | Meaning | Used for |
|---|---|---|
| `audio_filepath` | the segment WAV (16 kHz mono) | the audio input |
| `text` | the segment transcript (lowercase, no punctuation — matches the model's output style) | the training target |
| `duration` | seconds | length filtering (`min_duration`/`max_duration`) |
| `avg_score` | alignment confidence | already filtered to ≥ 0.4 at split time |

The three manifests you need:

| Split | Path (this repo) | Role |
|---|---|---|
| **train** | `data/asr/splits-legal2023_38hrs/train.jsonl` | what the model learns from (35.66 h) |
| **dev** | `data/asr/splits-legal2023_38hrs/dev.jsonl` | validation each epoch → early-stopping + best-checkpoint pick (0.57 h) |
| **test** | `data/asr/splits-legal2023_38hrs-padded/test.jsonl` | held-out; **only** used at eval time (2.97 h, +0.5 s padded) |

> **Speaker-disjoint** splits matter: no speaker appears in more than one split, so a good
> dev/test score reflects generalization, not memorized voices. To fine-tune on a **different**
> dataset, point the manifest paths at your own splits (see §6.1). These are **pseudo-labels**
> (pipeline-derived, not hand-checked) — the model can only get as good as they are.

### The base model
We start from **`nvidia/parakeet-ctc-0.6b`** — a NeMo **FastConformer-CTC** English model
(~0.6 B params, BPE tokenizer) whose native output is **lowercase / unpunctuated**, matching
our transcripts. It downloads automatically on the first run. **We keep its tokenizer** (no
vocab rebuild) so fine-tuning only *adapts* existing knowledge.

---

## 3. Fine-tune — step by step

Everything is driven by **one config file** and **one wrapper script**:
- `src/asrpipe/finetune/conf/ctc_finetune.yaml` — the knobs (edit this).
- `scripts/run_finetune.sh` — the launcher (runs `asrpipe.finetune.train_ctc` under the NeMo venv).

### Step 3.1 — Understand the three decisions the config already makes for you

**Where we are:** about to train; first, know *why* the defaults are what they are. These are
the three things a reviewer will ask about.

**(a) Starting model.** `init_from_pretrained_model: nvidia/parakeet-ctc-0.6b` — we adapt a
pretrained model rather than train from scratch.

**(b) Learning rate — low, warmed up, then decayed.** `optim.lr: 1e-4` with AdamW. This peak
is **10–50× lower** than a from-scratch rate: the encoder is already trained, so a big step
would overwrite (catastrophically forget) it. The schedule is **CosineAnnealing** with
`warmup_steps: 1000` — a linear ramp 0 → 1e-4 over ~the first 2 epochs (so the first batches
don't jolt the weights), then a cosine decay toward `min_lr: 1e-6`.

**(c) Freeze then unfreeze the encoder.** `freeze_encoder_epochs: 2`. The model has a big
**encoder** (acoustic features, most of the pretrained knowledge) and a small **CTC head**
(the output layer). For the **first 2 epochs the encoder is frozen** — only the head trains,
so it settles on our data first. **From epoch 2 the encoder unfreezes** and the whole network
trains end-to-end at the low LR, adapting the acoustics to the domain. This freeze→unfreeze is
the **main anti-forgetting lever** (`0` = never freeze; longer = safer but slower to adapt).

Other shipped defaults: `batch_size: 8` × `accumulate_grad_batches: 8` (effective batch ~64),
`precision: bf16-mixed`, `max_duration: 16` (caps attention memory), `max_epochs: 40` with
**early-stopping on `val_wer`** (`patience: 8`), keeping only the single best checkpoint.

### Step 3.2 — Smoke-test the wiring (always do this first)
```bash
cd /workspace/code2
bash scripts/run_finetune.sh --fast-dev-run
```
**Where we are:** pre-flight check.
**What it achieves:** runs a **single batch** through the whole path (data loading →
exp_manager → one train + val step) in ~a minute. It catches manifest typos, tokenizer issues,
and out-of-memory (OOM) problems **before** you commit to a multi-hour run. Fix any error here
first.

### Step 3.3 — Launch the full fine-tune
```bash
bash scripts/run_finetune.sh
```
**Where we are:** the core of the task.
**What it achieves:** trains the model to natural early-stop. On this dataset/GPU that was
**23 epochs, ~5.1 h**. During the run:
- epochs 0–1: encoder **frozen**, head-only (fast, ~5 min/epoch),
- epochs 2+: encoder **unfrozen**, full backprop (~14 min/epoch; GPU use ~8 GB → ~16 GB),
- it **validates every epoch** and keeps the checkpoint with the **lowest dev `val_wer`**.

Where things land:
- log: `logs/finetune/train_<stamp>.log`
- checkpoints + TensorBoard: `exp/finetune/parakeet-ctc-0.6b_legal2023_38hrs_ft/`

> `resume_if_exists: true` — re-running the same `exp_manager.name` **resumes** from the last
> checkpoint rather than starting over. Use `--max-epochs 3` for a short run.

### Step 3.4 — Watch progress (browser-free)
```bash
bash scripts/tb_metrics.sh          # prints per-epoch val_wer / val_loss + latest train_loss/lr
```
**Where we are:** monitoring.
**What it achieves:** reads the TensorBoard event file and prints the per-epoch curve in the
terminal — no port-forwarding needed. (For the full web UI: `bash scripts/run_tensorboard.sh 6006`.)
On the reference run, `val_wer` fell 19.8% → **13.59%** by epoch 14, then early-stopping
triggered after 8 more epochs with no improvement.

### Step 3.5 — Export the best checkpoint to a deployable `.nemo`
```bash
.venv-nemo/bin/python -m asrpipe.finetune.ckpt_to_nemo \
    --ckpt exp/finetune/parakeet-ctc-0.6b_legal2023_38hrs_ft/checkpoints/parakeet-ctc-0.6b_legal2023_38hrs_ft--val_wer=0.1359-epoch=14.ckpt \
    --base nvidia/parakeet-ctc-0.6b \
    --out  exp/finetune/parakeet-ctc-0.6b_legal2023_38hrs_ft/checkpoints/best.nemo
```
**Where we are:** packaging the result.
**What it achieves:** the training checkpoint (`.ckpt`, ~7.3 GB, includes optimizer state) is
not directly loadable by the evaluator. This rebuilds a clean, self-contained **`.nemo`**
(~2.4 GB) — base architecture + tokenizer with the fine-tuned weights copied in — and verifies
it restores. **This is the file you evaluate and deploy.**
*(In the reference run this was already produced as `best_val_wer0.1359_ep14.nemo`.)*

---

## 4. The training output — where it is and what it is

```
exp/finetune/parakeet-ctc-0.6b_legal2023_38hrs_ft/
    checkpoints/
        parakeet-ctc-0.6b_legal2023_38hrs_ft--val_wer=0.1359-epoch=14.ckpt   # best training ckpt (~7.3 GB)
        best_val_wer0.1359_ep14.nemo                                          # deployable model (~2.4 GB)
    events.out.tfevents...                                                    # TensorBoard metrics
```
- The **`.nemo`** is the deliverable — a single portable file you load for inference/eval.
- The filename encodes the **best dev WER (0.1359 = 13.59%)** and the **epoch (14)**.
- Per-epoch telemetry is also saved to `results/finetune_ctc0.6b_legal2023_38hrs_per_epoch.csv`.

---

## 5. Evaluate the model — greedy, **no language model** (the simple case)

**Where we are:** the model is trained; now measure whether it actually improved, honestly, on
the **held-out padded test set**. "Simple case" = plain **greedy** decoding, **no LM fusion** —
just the acoustic model. (This repo also supports `--kenlm` shallow fusion, but on a strong
in-domain fine-tune it *hurt*, so leave it off.)

### Step 5.1 — Score the fine-tuned model on the test set
```bash
.venv-nemo/bin/python -m asrpipe.asr.eval \
    --manifest data/asr/splits-legal2023_38hrs-padded/test.jsonl \
    --model    exp/finetune/parakeet-ctc-0.6b_legal2023_38hrs_ft/checkpoints/best_val_wer0.1359_ep14.nemo \
    --exp-name myfinetune_ft --finetuned --padded --batch-size 32 \
    --data-description 'legal2023_38hrs padded test (2258 segs), greedy, no LM'
```
- `--model <best.nemo>` — the fine-tuned model (a local `.nemo`; a raw `.ckpt` is **not** accepted).
- `--finetuned` — flags it as a fine-tuned model (provenance in the report).
- `--padded` — the test audio is silence-padded (+0.5 s), the primary eval condition.
- **No `--kenlm`** → greedy decoding, no language model. This is the simple case.

**What it achieves:** decodes all 2258 test segments, computes WER against the references, and
writes a full report.

### Step 5.2 — Read the result
Outputs:
- `results/myfinetune_ft.csv` — per-segment `ref / hyp / wer`.
- `docs/experiment-logs/myfinetune_ft.summary.txt` — the headline. Look for:
  - **`CORPUS WER`** — the single headline number (total edits ÷ total reference words).
  - **`mean utt`** — average per-utterance WER.
  - a **`per_speaker`** block — WER per test speaker; this is where data-quality problems show
    up (a few hard speakers sit much higher and drag the average).

### Step 5.3 — Compare fine-tuned vs. base (the actual claim)
First make sure you have a **baseline** number on the *same* test set (score the base model the
same way, `--model nvidia/parakeet-ctc-0.6b`, without `--finetuned`), then:
```bash
python3 -m asrpipe.reports.compare_experiments --name ft_vs_base \
    --exps myfinetune_ft finaltestpadded_parakeetctc0.6b_base \
    --baseline finaltestpadded_parakeetctc0.6b_base
```
**What it achieves:** a side-by-side table (corpus + mean-utt WER, per-speaker deltas) with a
comparability gate, so the improvement is stated against a like-for-like baseline.

### Reference result (already on disk)
On the `legal2023_38hrs` padded test (2258 segments, greedy, no LM):

| Model | Corpus WER | mean-utt |
|---|---:|---:|
| base `parakeet-ctc-0.6b` | 23.39% | 32.41% |
| **fine-tuned (this guide)** | **18.27%** | **24.28%** |
| *(non-circular judge)* rnnt-1.1b | 20.03% | 27.99% |

The fine-tune beats the base by **5.1 WER points (≈22% relative)** and also beats the
independent **rnnt-1.1b judge** — a genuinely good sign (see the circularity caveat below).

---

## 6. Adapting to your own experiment

### 6.1 Most experiments = edit only the YAML
Copy `ctc_finetune.yaml` to `ctc_finetune_<expt>.yaml`, change what you need, and run
`bash scripts/run_finetune.sh --config <path>` (keeps the default intact). Common edits:

| Want to… | Change |
|---|---|
| Fine-tune a different base (e.g. ctc-1.1b) | `init_from_pretrained_model` |
| Train on your own data | `model.train_ds` / `validation_ds.manifest_filepath` |
| Fix OOM | lower `train_ds.batch_size`, raise `accumulate_grad_batches` to keep effective batch ~64; lower `max_duration` |
| Change the LR schedule | `model.optim.lr`, `sched.{warmup_steps,min_lr}` |
| Freeze longer/shorter | `freeze_encoder_epochs` (0 = never) |
| Keep runs separate | `exp_manager.name` → new `exp/finetune/<name>/` dir |

Change **training logic** (the freeze/unfreeze callback, optimizer wiring) in
`src/asrpipe/finetune/train_ctc.py` — rarely needed.

---

## 7. Gotchas & notes

- **NeMo venv only.** Running fine-tune/eval under the system `python3` fails — use
  `.venv-nemo/bin/python` (the wrappers do).
- **Always `--fast-dev-run` first.** It exercises data + exp_manager end-to-end in a minute.
- **CUDA OOM at the unfreeze (epoch 2).** Memory jumps ~2.5× when the encoder unfreezes. Lower
  `batch_size` (16→8→4) and raise `accumulate_grad_batches` to hold the effective batch; lower
  `max_duration` (attention memory is O(T²)). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is already exported by the wrapper.
- **Disk-quota on checkpoint save (`EDQUOT`).** Each `.ckpt` is ~7.3 GB. Keep `save_top_k: 1`,
  `save_last: false`, `always_save_nemo: false`; delete stale `exp/finetune/<old>` dirs.
  (Quota ≠ free space.)
- **No `.nemo` at train end?** Expected — convert with `ckpt_to_nemo.py` (Step 3.5).
- **Early-stop on `val_wer`, not `val_loss`.** On the reference run `val_loss` bottomed at
  epoch 8 but `val_wer` kept improving to epoch 14 — stopping on WER picked the right model.
- **Circularity caveat.** The training labels are whisper+parakeet-CTC-derived, so a CTC
  fine-tune partly learns the *label style*. Always also report the **rnnt-1.1b judge**
  (independent, 20.03% base) — the fine-tune beating it is the honest signal.

---

## 8. Quick reference — the whole thing

```bash
# 0. every session
source /workspace/code2/setup_env.sh
cd /workspace/code2

# 1. (once) confirm the manifests exist (produced by the alignment pipeline)
ls data/asr/splits-legal2023_38hrs/{train,dev}.jsonl data/asr/splits-legal2023_38hrs-padded/test.jsonl

# 2. smoke test, then fine-tune to early-stop
bash scripts/run_finetune.sh --fast-dev-run
bash scripts/run_finetune.sh
bash scripts/tb_metrics.sh                       # watch val_wer per epoch

# 3. export the best checkpoint to a deployable .nemo
.venv-nemo/bin/python -m asrpipe.finetune.ckpt_to_nemo \
    --ckpt exp/finetune/parakeet-ctc-0.6b_legal2023_38hrs_ft/checkpoints/<best>.ckpt \
    --base nvidia/parakeet-ctc-0.6b --out exp/finetune/.../best.nemo

# 4. evaluate — greedy, NO language model (simple case)
.venv-nemo/bin/python -m asrpipe.asr.eval \
    --manifest data/asr/splits-legal2023_38hrs-padded/test.jsonl \
    --model exp/finetune/.../best.nemo \
    --exp-name myfinetune_ft --finetuned --padded --batch-size 32

# 5. read docs/experiment-logs/myfinetune_ft.summary.txt  (CORPUS WER + per_speaker)
#    compare vs baseline with asrpipe.reports.compare_experiments
```

For the full running record (per-epoch table, exact results, deep debugging runbook) see
`docs/fine-tune-phase.txt`; for the upstream data see `README-alignment.md`.
