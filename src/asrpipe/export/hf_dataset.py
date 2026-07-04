"""Export a NeMo-manifest split to a HuggingFace `datasets` DatasetDict and (optionally)
push it to the Hub. This is the STANDARD `datasets` + `push_to_hub` path (Parquet with an
embedded `Audio` feature) — no custom format — wrapped so it is one reproducible command.

Reads {train,dev,test}.jsonl (dev -> "validation"), builds a Dataset per split with an
`Audio(16 kHz)` column plus metadata columns (audio_path, text, speaker, avg_score, min_score,
duration, source_file, segment_wer), writes a dataset card (README.md), and either
save_to_disk (dry-run) or push_to_hub. Share the UNPADDED split (padding is an eval-time
trick, not part of the data).

Requires `datasets` (+ `huggingface_hub` for --push): pip install datasets huggingface_hub.

Usage:
  # inspect locally without pushing (builds + save_to_disk + writes README.md):
  python3 -m asrpipe.export.hf_dataset --splits data/asr/splits-legal2023_38hrs \
      --name legal2023_38hrs --dry-run --out /tmp/hf_legal2023_38hrs
  # push (private by default; needs `huggingface-cli login` with a write token):
  python3 -m asrpipe.export.hf_dataset --splits data/asr/splits-legal2023_38hrs \
      --repo your-org/legal2023_38hrs --name legal2023_38hrs --push --private
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


def _rows(path: Path):
    return [json.loads(l) for l in path.open() if l.strip()]


def _load_wer_map(wer_csv: Path):
    """audio_filepath -> per-segment WER, from an asrpipe.asr.eval raw-results CSV."""
    with open(wer_csv) as f:
        return {row["audio_filepath"]: float(row["wer"]) for row in csv.DictReader(f)}


def build_wer_sidecar(splits_dir, wer_csv):
    """audio_path,segment_wer,split rows for all splits — no Audio decode/embed, so this can be
    pushed as a standalone file without re-uploading the audio already on the Hub."""
    splits_dir = Path(splits_dir)
    wer_map = _load_wer_map(wer_csv)
    out_rows = []
    for src, hf in SPLIT_MAP.items():
        man = splits_dir / f"{src}.jsonl"
        if not man.exists():
            continue
        for r in _rows(man):
            w = wer_map.get(r["audio_filepath"])
            out_rows.append({
                "audio_path": Path(r["audio_filepath"]).name,
                "split": hf,
                "segment_wer": round(w, 2) if w is not None else "",
            })
    return out_rows


def build(splits_dir, sr: int = 16000, wer_csv=None):
    """Build a DatasetDict from {train,dev,test}.jsonl. dev -> 'validation'."""
    from datasets import Dataset, DatasetDict, Audio
    splits_dir = Path(splits_dir)
    wer_map = _load_wer_map(wer_csv) if wer_csv else None
    dd = {}
    for src, hf in SPLIT_MAP.items():
        man = splits_dir / f"{src}.jsonl"
        if not man.exists():
            print(f"  skip {src}: no {man}")
            continue
        rows = _rows(man)
        data = {
            "audio":       [r["audio_filepath"] for r in rows],
            "audio_path":  [Path(r["audio_filepath"]).name for r in rows],
            "text":        [r.get("text", "") for r in rows],
            "speaker":     [r.get("speaker") for r in rows],
            "avg_score":   [(float(r["avg_score"]) if r.get("avg_score") is not None else None) for r in rows],
            "min_score":   [(float(r["min_score"]) if r.get("min_score") is not None else None) for r in rows],
            "duration":    [float(r["duration"]) for r in rows],
            "source_file": [r.get("file") for r in rows],
        }
        if wer_map is not None:
            missing = 0
            segment_wer = []
            for r in rows:
                w = wer_map.get(r["audio_filepath"])
                if w is None:
                    missing += 1
                else:
                    w = round(w, 2)
                segment_wer.append(w)
            data["segment_wer"] = segment_wer
            if missing:
                print(f"  WARNING: {missing}/{len(rows)} rows in {hf} had no match in {wer_csv}")
        dd[hf] = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=sr))
        print(f"  {hf:<11} {len(rows)} rows")
    return DatasetDict(dd)


def dataset_card(name: str, dd) -> str:
    """Generate a dataset card (README.md) with YAML frontmatter + prose."""
    def stat(split):
        ds = dd.get(split)
        if ds is None:
            return (0, 0.0, 0)
        dur = sum(ds["duration"]) / 3600
        return (len(ds), dur, len(set(ds["speaker"])))
    tr, va, te = stat("train"), stat("validation"), stat("test")
    total_h = tr[1] + va[1] + te[1]
    fm = "\n".join([
        "---",
        "license: other",
        "license_name: restricted-court-audio",
        "task_categories:", "- automatic-speech-recognition",
        "language:", "- en",
        f"pretty_name: {name}",
        "size_categories:", "- 10K<n<100K",
        "tags:", "- speech", "- asr", "- legal", "- court", "- pseudo-labels",
        "---",
    ])
    body = f"""
# {name}

Court-audio ASR dataset: **{total_h:.1f} h** of English legal/court speech cut into
per-speaker segments, with **speaker-disjoint** train / validation / test splits.

> ⚠️ **Pseudo-labels, not gold.** Transcripts are produced by an automatic pipeline
> (Parakeet-CTC + KenLM + Whisper best-of-two localization → wav2vec2 forced alignment),
> not human annotation. Corpus WER vs an independent judge (parakeet-rnnt-1.1b) is ~20%.
> A per-segment confidence `avg_score` is provided; only segments with `avg_score >= 0.4`
> are included. Filter further if you need cleaner labels.

> ⚠️ **Usage / privacy.** This is real court audio with real speakers. Confirm you have the
> right to (re)distribute the audio and that there are no privacy/PII constraints before
> making a copy public. Default share mode is **private/gated**.

## Splits
| split | segments | speakers | duration |
|---|--:|--:|--:|
| train | {tr[0]} | {tr[2]} | {tr[1]:.2f} h |
| validation | {va[0]} | {va[2]} | {va[1]:.2f} h |
| test | {te[0]} | {te[2]} | {te[1]:.2f} h |

Splits are **speaker-disjoint** (a speaker's segments are entirely within one split) and
were stratified by per-speaker WER so dev/test carry reliable labels while test also spans
a bounded hard tail.

## Columns
- `audio`: 16 kHz mono; decoded to a waveform by the `Audio` feature.
- `audio_path`: original segment filename (e.g. `audio_file_001_spk_0100_seg_000000_start000001_end000003.wav`),
  unique per row — encodes source file, speaker, segment index, and start/end (s) for traceability.
- `text`: pseudo-label transcript (lowercase-normalizable).
- `speaker`: global speaker id (`spk_XXXX`), consistent across source files.
- `avg_score` / `min_score`: per-segment mean / min word confidence (0–1).
- `duration`: seconds. `source_file`: originating audio file id.
- `segment_wer`: per-segment WER vs the parakeet-rnnt-1.1b judge (word-level Levenshtein;
  same normalizer as the corpus-level WER reported above), from the all-speakers eval run.

## Load
```python
from datasets import load_dataset
ds = load_dataset("<org>/{name}")           # ds["train"], ds["validation"], ds["test"]
ds = ds.filter(lambda x: x["avg_score"] >= 0.6)   # optional: tighter quality filter
```

## Provenance / limitations
Segments come from a from-scratch alignment pipeline; the split, per-set WER, and full
methodology are documented alongside the corpus. Labels are noisy (ASR + alignment errors);
treat WER on this set's `text` as a proxy, and prefer an independent judge for model eval.
"""
    return fm + "\n" + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/asr/splits-legal2023_38hrs")
    ap.add_argument("--name", default="legal2023_38hrs")
    ap.add_argument("--repo", default=None, help="hub repo id, e.g. org/legal2023_38hrs (for --push)")
    ap.add_argument("--private", action="store_true", help="push as a private dataset (recommended)")
    ap.add_argument("--push", action="store_true", help="push_to_hub (else dry-run save_to_disk)")
    ap.add_argument("--skip-card", action="store_true",
                    help="push (push_to_hub only) but do NOT overwrite README.md on the Hub "
                         "(use when the repo's card was hand-edited and differs from the auto-generated one)")
    ap.add_argument("--out", default=None, help="local save_to_disk dir (dry-run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="dry-run only: save just the first N rows/split (card still reflects the FULL dataset)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--wer-csv", default="results/all-speakers_parakeetrnnt1.1b_disablePass1Extra.csv",
                    help="raw per-segment eval-results CSV (audio_filepath,wer,...) to join in as "
                         "segment_wer; pass '' to skip")
    ap.add_argument("--wer-sidecar-only", action="store_true",
                    help="don't touch the dataset/README at all; just write+push a small "
                         "audio_path,split,segment_wer CSV (path_in_repo=segment_wer.csv) so "
                         "existing embedded audio is never re-uploaded")
    a = ap.parse_args()

    if a.wer_sidecar_only:
        rows = build_wer_sidecar(a.splits, a.wer_csv)
        out = Path(a.out or f"/tmp/hf_{a.name}_segment_wer.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio_path", "split", "segment_wer"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {out}")
        if a.push:
            if not a.repo:
                ap.error("--push needs --repo org/name")
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=str(out), path_in_repo="segment_wer.csv",
                                repo_id=a.repo, repo_type="dataset")
            print(f"pushed segment_wer.csv -> https://huggingface.co/datasets/{a.repo}")
        return

    dd = build(a.splits, a.sr, wer_csv=(a.wer_csv or None))
    print(dd)
    readme = dataset_card(a.name, dd)   # always from the FULL dataset

    if a.push:
        if not a.repo:
            ap.error("--push needs --repo org/name")
        dd.push_to_hub(a.repo, private=a.private)
        if a.skip_card:
            print("  (--skip-card: leaving the Hub's existing README.md untouched)")
        else:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                                repo_id=a.repo, repo_type="dataset")
        print(f"pushed -> https://huggingface.co/datasets/{a.repo}  (private={a.private})")
    else:
        from datasets import DatasetDict
        out = Path(a.out or f"/tmp/hf_{a.name}")
        save = dd
        if a.limit:
            save = DatasetDict({k: v.select(range(min(a.limit, len(v)))) for k, v in dd.items()})
            print(f"(dry-run --limit {a.limit}: saving a {a.limit}-row/split sample; card reflects the full dataset)")
        save.save_to_disk(str(out))
        (out / "README.md").write_text(readme)
        print(f"dry-run: DatasetDict + README.md saved to {out}  (use --push --repo to upload)")


if __name__ == "__main__":
    main()
