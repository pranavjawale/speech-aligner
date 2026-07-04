"""Multi-experiment comparison (baseline vs fine-tuned side-by-side, N-way).

Reads finished single-experiment metadata JSONs from docs/experiment-logs/
(produced by asrpipe.asr.eval --exp-name), runs a comparability gate, and emits
an Excel-portable comparison table + provenance, per
docs/multi-expt-comparison-metadata-format.txt (all 10 rules).

Outputs (under docs/experiment-logs/comparisons/<name>.*):
  <name>.csv               one row per experiment (Model, Test data, WER, time)  [rule 6]
  <name>.per_speaker.csv   per-(file|speaker) WER + delta, when test data matches [rule 8]
  <name>.metadata.json     provenance: sources+md5, git, reproduce command        [rule 10]
  <name>.summary.txt       human-readable comparability verdict + table
and appends to docs/experiment-logs/comparison-track-list.txt                      [rule 10]

Usage:
  python -m asrpipe.reports.compare_experiments \
     --name 6c-test_ctc-vs-rnnt_baseline \
     --exps all-speakers_parakeetrnnt1.1b_baselineexpt1 all-speakers_parakeetctc0.6b_baselineexpt1 \
     --baseline all-speakers_parakeetrnnt1.1b_baselineexpt1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import config as cfg


def _md5(path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit() -> dict:
    root = str(cfg.ROOT)
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as e:
        return {"commit": None, "dirty": None, "error": str(e)}


def _load(exp_name: str) -> dict:
    """Resolve an experiment name to its metadata JSON (rule 9)."""
    p = cfg.EXPERIMENT_LOGS_DIR / f"{exp_name}.metadata.json"
    if not p.exists():
        raise FileNotFoundError(
            f"experiment '{exp_name}' not found / not finished: missing {p}")
    meta = json.loads(p.read_text())
    meta["_metadata_path"] = str(p)
    meta["_metadata_md5"] = _md5(p)
    return meta


def _comparability(metas: list[dict]) -> dict:
    """Rules 4 + 7 — is a WER delta valid across these experiments?"""
    def field(m, *path):
        d = m
        for k in path:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return d
    checks = {
        "same_test_data":  ("data", "manifest_md5"),
        "same_wer_method": ("eval", "normalizer"),
        "same_filtering":  ("data", "conf_filter_threshold"),
        "same_padding":    ("data", "padded"),
        "same_gpu":        ("compute", "gpu"),
    }
    reasons, results = [], {}
    for label, path in checks.items():
        vals = {m["experiment_name"]: field(m, *path) for m in metas}
        same = len(set(json.dumps(v, sort_keys=True) for v in vals.values())) == 1
        results[label] = {"same": same, "values": vals}
        if not same:
            reasons.append(f"{label} differs: {vals}")
    # WER comparable requires identical test data AND identical WER method
    wer_comparable = results["same_test_data"]["same"] and results["same_wer_method"]["same"]
    time_comparable = results["same_gpu"]["same"]
    return {"wer_comparable": wer_comparable, "time_comparable": time_comparable,
            "checks": results, "reasons": reasons}


def _row(m: dict) -> dict:
    s = m.get("summary", {})
    d = m.get("data", {})
    md = m.get("model", {})
    ev = m.get("eval", {})
    cp = m.get("compute", {})
    tm = m.get("timing", {})
    return {
        "experiment": m["experiment_name"],
        "model": md.get("name"),
        "is_finetuned": md.get("is_finetuned"),
        "model_revision": md.get("revision"),
        "test_data": d.get("manifest"),
        "test_data_md5": d.get("manifest_md5"),
        "utterances": s.get("utterances"),
        "scored": s.get("scored"),
        "empty_ref": s.get("empty_ref"),
        "empty_hyp": s.get("empty_hyp"),
        "padded": d.get("padded"),
        "conf_filter_threshold": d.get("conf_filter_threshold"),
        "normalizer": ev.get("normalizer"),
        "decoding": ev.get("decoding"),
        "batch_size": ev.get("batch_size"),
        "gpu": cp.get("gpu"),
        "decode_s": tm.get("decode_s"),
        "total_s": tm.get("total_s"),
        "rtfx": s.get("rtfx"),
        "corpus_wer": s.get("corpus_wer"),
        "mean_utt_wer": s.get("mean_utt_wer"),
    }


def run(name: str, exps: list[str], baseline: str | None = None,
        reproduce_command: str | None = None) -> dict:
    if baseline and baseline not in exps:
        exps = [baseline] + exps
    if not baseline:
        baseline = exps[0]
    metas = [_load(e) for e in exps]
    cmp = _comparability(metas)

    rows = [_row(m) for m in metas]
    base_row = next(r for r in rows if r["experiment"] == baseline)
    base_wer = base_row["corpus_wer"]
    # delta vs baseline + rank (only meaningful if wer_comparable, but always emit numbers)
    for r in rows:
        r["delta_corpus_vs_baseline"] = (
            round(r["corpus_wer"] - base_wer, 6)
            if (r["corpus_wer"] is not None and base_wer is not None) else None)
    ranked = sorted([r for r in rows if r["corpus_wer"] is not None],
                    key=lambda r: r["corpus_wer"])
    for i, r in enumerate(ranked, 1):
        r["rank_by_corpus_wer"] = i

    out_dir = cfg.EXPERIMENT_LOGS_DIR / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    cols = ["rank_by_corpus_wer", "experiment", "model", "is_finetuned", "model_revision",
            "test_data", "test_data_md5", "utterances", "scored", "empty_ref", "empty_hyp",
            "padded", "conf_filter_threshold", "normalizer", "decoding", "batch_size",
            "gpu", "decode_s", "total_s", "rtfx",
            "corpus_wer", "mean_utt_wer", "delta_corpus_vs_baseline"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r.get("rank_by_corpus_wer") or 1e9):
            w.writerow(r)

    # rule 8 — per-speaker delta table, only when test data is identical
    per_spk_path = None
    if cmp["checks"]["same_test_data"]["same"]:
        per_spk_path = out_dir / f"{name}.per_speaker.csv"
        speakers = set()
        pmap = {}
        for m in metas:
            ps = m.get("summary", {}).get("per_speaker", {})
            pmap[m["experiment_name"]] = ps
            speakers.update(ps.keys())
        base_ps = pmap.get(baseline, {})
        with per_spk_path.open("w", newline="") as f:
            hdr = ["file_speaker", "ref_words"] + \
                  [f"wer[{e}]" for e in exps] + \
                  [f"delta[{e}]" for e in exps if e != baseline]
            w = csv.writer(f)
            w.writerow(hdr)
            for sp in sorted(speakers):
                refw = next((pmap[e][sp]["ref_words"] for e in exps
                             if sp in pmap[e]), None)
                wers = [pmap[e].get(sp, {}).get("wer") for e in exps]
                bw = base_ps.get(sp, {}).get("wer")
                deltas = [(pmap[e].get(sp, {}).get("wer") - bw)
                          if (e != baseline and bw is not None
                              and pmap[e].get(sp, {}).get("wer") is not None) else None
                          for e in exps if e != baseline]
                w.writerow([sp, refw] + [round(x, 6) if x is not None else None for x in wers]
                           + [round(x, 6) if x is not None else None for x in deltas])

    # rule 10 — provenance + summary + track-list
    finished_at = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    metadata = {
        "comparison_name": name,
        "finished_at": finished_at,
        "baseline": baseline,
        "experiments": [{"name": m["experiment_name"],
                         "metadata_path": m["_metadata_path"],
                         "metadata_md5": m["_metadata_md5"]} for m in metas],
        "comparability": cmp,
        "git": _git_commit(),
        "outputs": {"table_csv": str(csv_path),
                    "per_speaker_csv": str(per_spk_path) if per_spk_path else None},
        "reproduce_command": reproduce_command,
        "rows": rows,
    }
    meta_path = out_dir / f"{name}.metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    txt_path = out_dir / f"{name}.summary.txt"
    lines = [
        f"comparison : {name}",
        f"baseline   : {baseline}",
        f"experiments: {', '.join(exps)}",
        f"WER comparable : {cmp['wer_comparable']}",
        f"time comparable: {cmp['time_comparable']}",
    ]
    if cmp["reasons"]:
        lines.append("caveats:")
        lines += [f"  - {r}" for r in cmp["reasons"]]
    lines.append("")
    lines.append(f"{'rank':>4}  {'corpus_wer':>10}  {'mean_utt':>9}  {'Δvsbase':>8}  experiment")
    for r in sorted(rows, key=lambda r: r.get("rank_by_corpus_wer") or 1e9):
        cw = f"{r['corpus_wer']*100:.2f}%" if r["corpus_wer"] is not None else "n/a"
        mu = f"{r['mean_utt_wer']*100:.2f}%" if r["mean_utt_wer"] is not None else "n/a"
        dv = r.get("delta_corpus_vs_baseline")
        dl = f"{dv*100:+.2f}" if dv is not None else ""
        lines.append(f"{r.get('rank_by_corpus_wer','?'):>4}  {cw:>10}  {mu:>9}  {dl:>8}  {r['experiment']}")
    lines += ["", "reproduce  :", f"  {reproduce_command}"]
    txt_path.write_text("\n".join(lines) + "\n")

    track = cfg.EXPERIMENT_LOGS_DIR / "comparison-track-list.txt"
    if not track.exists():
        track.write_text("comparison-name\ttable-csv\tfinished-at\n")
    with track.open("a") as f:
        f.write(f"{name}\t{csv_path}\t{finished_at}\n")

    print(f"table    -> {csv_path}")
    if per_spk_path:
        print(f"per-spk  -> {per_spk_path}")
    print(f"metadata -> {meta_path}\nsummary  -> {txt_path}\ntrack    -> {track}")
    print(f"\nWER comparable: {cmp['wer_comparable']}  |  time comparable: {cmp['time_comparable']}")
    return metadata


def _reproduce_cmd(a) -> str:
    parts = ["python", "-m", "asrpipe.reports.compare_experiments", "--name", a.name,
             "--exps", *a.exps]
    if a.baseline:
        parts += ["--baseline", a.baseline]
    return " ".join(shlex.quote(p) for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="comparison-name (output file stem)")
    ap.add_argument("--exps", nargs="+", required=True, help="experiment names to compare")
    ap.add_argument("--baseline", default=None, help="baseline experiment for deltas")
    a = ap.parse_args()
    run(a.name, a.exps, a.baseline, reproduce_command=_reproduce_cmd(a))


if __name__ == "__main__":
    main()
