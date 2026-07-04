"""S11 — build train/dev/test splits by REUSING the fixed speaker assignment.

code2 is 6c-only (Q8), so the speaker->split decision is an INHERITED fixed input
(data/asr/speaker_assignment.json, carried over from the strategy-5 era) — this
keeps the Phase-15 baseline comparable (same 94/6/20 speakers). The original tiered
partitioner that CREATED the assignment needed strat-5 + rnnt WER and lives in
archive/ for reference; here we only rebuild split membership from 6c segments.

Logic (verbatim from rebuild_splits_reuse_assignment.py):
  - keep segments with avg_score >= SPLIT_THRESHOLD (0.4)
  - route each to its speaker's split (speakers absent from the assignment -> train)
  - shuffle each split with a FRESH random.Random(SEED) (per-split, not shared)
  - write {train,dev,test}.jsonl + carry the assignment forward

Port source : rebuild_splits_reuse_assignment.py (canonical) [+ partition_train_dev_test.py -> archive]
Reorg phase : RP4
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import config as cfg
from asrpipe.common import paths as P


def run(segments_path=None, assignment_path=None, out_dir=None,
        threshold=None, seed=None) -> None:
    segments_path = Path(segments_path) if segments_path else P.asr_manifest("segments")
    assignment_path = Path(assignment_path) if assignment_path else cfg.ASR_DIR / "speaker_assignment.json"
    out = Path(out_dir) if out_dir else cfg.SPLITS_DIR
    threshold = cfg.SPLIT_THRESHOLD if threshold is None else threshold
    seed = cfg.SPLIT_SEED if seed is None else seed

    assign = json.loads(assignment_path.read_text())        # {spk: {split, ...}}
    split_of = {sp: d["split"] for sp, d in assign.items()}

    rows = [json.loads(l) for l in open(segments_path) if l.strip()]
    quarantined = {cfg.stem(n) for n in getattr(cfg, "QUARANTINE_FILES", set())}
    n_all = len(rows)
    if quarantined:
        rows = [r for r in rows if r.get("file") not in quarantined]        # drop bad-audio files
    n_quar = n_all - len(rows)
    kept = [r for r in rows if r.get("avg_score") is not None and r["avg_score"] >= threshold]

    by_split = defaultdict(list)
    unknown = defaultdict(lambda: [0, 0.0])
    for r in kept:
        sp = r["speaker"]
        split = split_of.get(sp)
        if split is None:
            unknown[sp][0] += 1
            unknown[sp][1] += float(r["duration"])
            split = "train"
        by_split[split].append(r)

    out.mkdir(parents=True, exist_ok=True)
    summ = []
    for split in ("train", "dev", "test"):
        rs = by_split.get(split, [])
        random.Random(seed).shuffle(rs)                     # FRESH rng per split (verbatim)
        with (out / f"{split}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        dur = sum(float(r["duration"]) for r in rs)
        nspk = len({r["speaker"] for r in rs})
        summ.append(f"  {split:6s}: {nspk:3d} spk  {len(rs):6d} segs  {dur/3600:6.2f} h")

    (out / "speaker_assignment.json").write_text(json.dumps(assign, indent=2))
    lines = ["split rebuild — REUSED fixed speaker assignment on 6c segments",
             f"threshold avg_score >= {threshold}", *summ,
             f"  kept segments: {len(kept)} / {len(rows)} total"]
    if quarantined:
        lines.append(f"  QUARANTINED {sorted(quarantined)}: dropped {n_quar} segments "
                     f"({n_all} -> {len(rows)} before threshold)")
    if unknown:
        lines.append(f"  speakers not in assignment -> defaulted to TRAIN: {len(unknown)}")
        for sp, (n, d) in sorted(unknown.items()):
            lines.append(f"    {sp}: {n} segs, {d/60:.1f} min")
    else:
        lines.append("  all speakers found in assignment (no defaults).")
    (out / "partition_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default=None)
    ap.add_argument("--assignment", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    run(a.segments, a.assignment, a.out_dir, a.threshold, a.seed)


if __name__ == "__main__":
    main()
