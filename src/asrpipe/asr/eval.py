"""S13 — decode a NeMo manifest and score WER (the acceptance-gate eval).

Loads a NeMo ASR model (ctc-0.6b = fine-tune candidate; rnnt-1.1b = independent
non-circular judge), decodes greedy_batch, and scores against the manifest 'text':
  corpus WER  = sum word-edits / sum reference words   (primary metric)
  mean utt WER= mean of per-utterance WER
  per-(file, speaker) WER breakdown (speaker ids are per-file -> group by both).

Reuses common.wer.normalize (byte-identical to /code). Writes {out}.jsonl (per-row)
and {out}.summary.json. Faithful port of run_asr_baseline_eval.py.

Port source : run_asr_baseline_eval.py
Reorg phase : RP4
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

import config as cfg
from asrpipe.common.wer import normalize


def _hyp_text(o):
    """transcribe() returns Hypothesis objects (.text) in recent NeMo, or plain str."""
    if hasattr(o, "text"):
        return o.text
    if isinstance(o, (list, tuple)) and o:
        return _hyp_text(o[0])
    return str(o)


# ---------------------------------------------------------------------------
# provenance helpers (expt-metadata-format.txt: compute specs + reproducibility)
# ---------------------------------------------------------------------------
def _git_provenance():
    """C1 — code2 git commit + dirty flag (pins the pipeline/eval code)."""
    root = str(cfg.ROOT)
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as e:
        return {"commit": None, "dirty": None, "error": str(e)}


def _md5(path) -> str | None:
    """C2 — content hash of a file (proves the exact manifest, not just its path)."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


def _model_revision(model_name: str):
    """C3 — resolve the HF model commit sha (best-effort; None if offline/unknown)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_name).sha
    except Exception as e:
        return None


def _compute_specs() -> dict:
    """Spec #3 — GPU / CUDA / library / host compute fingerprint."""
    specs = {
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
        specs["ram_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        import torch
        specs["torch"] = torch.__version__
        specs["cuda"] = torch.version.cuda
        specs["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            specs["gpu"] = p.name
            specs["gpu_mem_gb"] = round(p.total_memory / 1024**3, 1)
        else:
            specs["gpu"] = "cpu"
    except Exception:
        pass
    try:
        specs["gpu_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()[0].strip()
    except Exception:
        pass
    try:
        import nemo
        specs["nemo"] = nemo.__version__
    except Exception:
        pass
    return specs


def _reproduce_cmd(a) -> str:
    """Spec #10 — the exact single command to reproduce this run."""
    parts = [".venv-nemo/bin/python", "-m", "asrpipe.asr.eval",
             "--manifest", a.manifest, "--model", a.model or cfg.EVAL_MODEL_CTC]
    if a.exp_name:            parts += ["--exp-name", a.exp_name]
    if a.out:                 parts += ["--out", a.out]
    if a.batch_size:          parts += ["--batch-size", str(a.batch_size)]
    if a.workers:             parts += ["--workers", str(a.workers)]
    if a.limit:               parts += ["--limit", str(a.limit)]
    if a.finetuned:           parts += ["--finetuned"]
    if a.padded:              parts += ["--padded"]
    if a.conf_filter_threshold is not None:
        parts += ["--conf-filter-threshold", str(a.conf_filter_threshold)]
    if a.data_description:    parts += ["--data-description", a.data_description]
    if a.model_description:   parts += ["--model-description", a.model_description]
    if a.kenlm:               parts += ["--kenlm", a.kenlm]
    if a.unigrams:            parts += ["--unigrams", a.unigrams]
    if a.lm_alpha is not None: parts += ["--lm-alpha", str(a.lm_alpha)]
    if a.lm_beta is not None:  parts += ["--lm-beta", str(a.lm_beta)]
    if a.lm_beam_size:        parts += ["--lm-beam-size", str(a.lm_beam_size)]
    if a.gpu_batch:           parts += ["--gpu-batch", str(a.gpu_batch)]
    return " ".join(shlex.quote(p) for p in parts)


def _append_track_list(exp_name, log_file, summary_path, finished_at):
    """Spec #11 — register the experiment in experiment-logs/expt-track-list.txt."""
    track = cfg.EXPERIMENT_LOGS_DIR / "expt-track-list.txt"
    header = "expt-name\texpt-log-file\tfinished-at\n"
    if not track.exists():
        track.write_text(header)
    log_ref = log_file or str(summary_path)
    with track.open("a") as f:
        f.write(f"{exp_name}\t{log_ref}\t{finished_at}\n")
    print(f"track-list -> {track}", flush=True)


def _write_experiment(exp_name, summary, results, meta, log_file=None):
    """Spec #8 raw CSV -> results/, spec #9 metadata json + summary txt -> experiment-logs/,
    spec #11 append to expt-track-list.txt."""
    cfg.EXPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    finished_at = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # (8) raw per-utterance CSV
    csv_path = cfg.EXPT_RESULTS_DIR / f"{exp_name}.csv"
    cols = ["audio_filepath", "file", "speaker", "seg_index", "duration",
            "avg_score", "min_score", "ref_words", "wer", "ref_norm", "hyp", "hyp_norm"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # (9) metadata json (provenance + summary) + human-readable summary txt
    meta_path = cfg.EXPERIMENT_LOGS_DIR / f"{exp_name}.metadata.json"
    metadata = {"experiment_name": exp_name, "finished_at": finished_at, **meta,
                "summary": summary, "raw_results_csv": str(csv_path),
                "log_file": log_file}
    meta_path.write_text(json.dumps(metadata, indent=2))

    txt_path = cfg.EXPERIMENT_LOGS_DIR / f"{exp_name}.summary.txt"
    d = meta
    lines = [
        f"experiment : {exp_name}",
        f"model      : {d['model']['name']}  ({'finetuned' if d['model']['is_finetuned'] else 'base'})",
        f"             {d['model'].get('description','')}",
        f"             revision={d['model'].get('revision')}",
        f"data       : {d['data']['manifest']}",
        f"             {d['data'].get('description','')}",
        f"             utterances={d['data']['utterances']}  duration={summary['audio_min']} min"
        f"  md5={d['data']['manifest_md5']}",
        f"             padded={d['data']['padded']} (pad_s={d['data']['pad_seconds']})"
        f"  conf_filter_threshold={d['data']['conf_filter_threshold']}"
        f"  observed_avg_score=[{d['data']['observed_avg_score_min']},{d['data']['observed_avg_score_max']}]",
        f"mode       : streaming={d['eval']['streaming']}  vad_applied={d['eval']['vad_applied']}",
        f"hyperparams: decoding={d['eval']['decoding']}  batch_size={d['eval']['batch_size']}"
        f"  workers={d['eval']['workers']}  normalizer={d['eval']['normalizer']}",
        f"scoring    : utterances={summary['utterances']}  scored={summary['scored']}"
        f"  empty_ref={summary['empty_ref']}  empty_hyp={summary['empty_hyp']}",
        f"compute    : {d['compute'].get('gpu')} ({d['compute'].get('gpu_mem_gb')}GB)"
        f"  cuda={d['compute'].get('cuda')}  torch={d['compute'].get('torch')}"
        f"  nemo={d['compute'].get('nemo')}  driver={d['compute'].get('gpu_driver')}",
        f"code       : git {d['code']['git']['commit']} (dirty={d['code']['git']['dirty']})",
        f"             script={d['code']['eval_script']}",
        f"timing     : load={d['timing']['load_s']}s  decode={d['timing']['decode_s']}s"
        f"  total={d['timing']['total_s']}s  rtfx={summary['rtfx']}",
        "",
        f"CORPUS WER : {summary['corpus_wer']*100:.2f}%",
        f"mean utt   : {summary['mean_utt_wer']*100:.2f}%",
        "",
        "reproduce  :",
        f"  {d['reproduce_command']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"CSV -> {csv_path}\nmetadata -> {meta_path}\nsummary  -> {txt_path}", flush=True)
    _append_track_list(exp_name, log_file, txt_path, finished_at)     # spec #11


def _kenlm_decode(asr, paths, kenlm_path, unigrams_path, alpha, beta, beam_size,
                  gpu_batch, workers):
    """CTC + word-level KenLM shallow fusion via pyctcdecode (same mechanism as the
    S3 lm_prepass). Returns list[str] hyps. Score = acoustic + alpha*LM + beta*|words|."""
    import os
    import soundfile as sf
    import torch
    import pyctcdecode
    from multiprocessing import Pool

    vocab = list(asr.decoder.vocabulary)
    unis = [w.strip() for w in open(unigrams_path) if w.strip()] if unigrams_path else None
    decoder = pyctcdecode.build_ctcdecoder(
        vocab, kenlm_model_path=kenlm_path, unigrams=unis, alpha=alpha, beta=beta)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr.eval()
    asr = asr.to(dev)
    print(f"KenLM shallow fusion: {kenlm_path}  unigrams={len(unis) if unis else 0}  "
          f"alpha={alpha} beta={beta} beam={beam_size} gpu_batch={gpu_batch}", flush=True)

    @torch.inference_mode()
    def logits_for(bp):
        sigs = [sf.read(p)[0].astype("float32") for p in bp]
        lens = [len(s) for s in sigs]
        batch = torch.zeros(len(sigs), max(lens))
        for j, s in enumerate(sigs):
            batch[j, :len(s)] = torch.from_numpy(s)
        lp, enc_len, *_ = asr.forward(input_signal=batch.to(dev),
                                      input_signal_length=torch.tensor(lens).to(dev))
        lp = lp.cpu().numpy()
        return [lp[j, :int(enc_len[j])] for j in range(len(sigs))]

    logits = []
    for i in range(0, len(paths), gpu_batch):
        logits.extend(logits_for(paths[i:i + gpu_batch]))
    nw = max(1, min(workers, (os.cpu_count() or 4) - 2))
    with Pool(nw) as pool:
        texts = decoder.decode_batch(pool, logits, beam_width=beam_size)
    return list(texts)


def run(manifest, out=None, model=None, limit=0, batch_size=None, workers=None,
        exp_name=None, data_description=None, finetuned=False,
        model_description=None, conf_filter_threshold=None, padded=False,
        reproduce_command=None, log_file=None,
        kenlm=None, unigrams=None, lm_alpha=None, lm_beta=None,
        lm_beam_size=None, gpu_batch=None) -> dict:
    model_name = model or cfg.EVAL_MODEL_CTC
    batch_size = cfg.EVAL_BATCH_SIZE if batch_size is None else batch_size
    workers = cfg.EVAL_NUM_WORKERS if workers is None else workers

    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    if limit and limit > 0:
        rows = rows[:limit]
    paths = [r["audio_filepath"] for r in rows]
    print(f"Loaded {len(rows)} rows from {manifest}", flush=True)

    import nemo.collections.asr as nemo_asr
    print(f"Loading model {model_name} ...", flush=True)
    t_load = time.perf_counter()
    if str(model_name).endswith(".nemo"):          # local fine-tuned checkpoint
        asr = nemo_asr.models.ASRModel.restore_from(restore_path=model_name)
    else:                                          # NGC/HF pretrained name
        asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    load_s = time.perf_counter() - t_load

    lm_meta = None
    if kenlm:                                          # CTC + KenLM shallow fusion
        lm_alpha = cfg.KENLM_ALPHA if lm_alpha is None else lm_alpha
        lm_beta = cfg.KENLM_BETA if lm_beta is None else lm_beta
        lm_beam_size = cfg.KENLM_BEAM_SIZE if lm_beam_size is None else lm_beam_size
        gpu_batch = batch_size if gpu_batch is None else gpu_batch
        decoding_desc = (f"beam+kenlm(pyctcdecode) lm={Path(kenlm).name} alpha={lm_alpha} "
                         f"beta={lm_beta} beam={lm_beam_size} "
                         f"unigrams={'yes' if unigrams else 'no'}")
        t0 = time.perf_counter()
        hyps = _kenlm_decode(asr, paths, kenlm, unigrams, lm_alpha, lm_beta,
                             lm_beam_size, gpu_batch, workers)
        decode_s = time.perf_counter() - t0
        lm_meta = {"kenlm_model": str(kenlm),
                   "unigrams": str(unigrams) if unigrams else None,
                   "lm_alpha": lm_alpha, "lm_beta": lm_beta, "lm_beam_size": lm_beam_size}
    else:                                              # greedy (NeMo native decoding)
        try:
            from omegaconf import open_dict
            dcfg = asr.cfg.decoding
            with open_dict(dcfg):
                dcfg.strategy = cfg.EVAL_DECODING
            asr.change_decoding_strategy(dcfg)
            print(f"decoding strategy -> {cfg.EVAL_DECODING}", flush=True)
        except Exception as e:
            print(f"decoding strategy unchanged ({e})", flush=True)
        decoding_desc = cfg.EVAL_DECODING
        t0 = time.perf_counter()
        outputs = asr.transcribe(paths, batch_size=batch_size, num_workers=workers)
        decode_s = time.perf_counter() - t0
        hyps = [_hyp_text(o) for o in outputs]
    assert len(hyps) == len(rows), f"hyp/row mismatch {len(hyps)} vs {len(rows)}"

    results = []
    tot_d = tot_n = 0
    per_spk = defaultdict(lambda: [0, 0, 0])
    per_utt = []
    audio_s = 0.0
    empty_ref = empty_hyp = 0                          # C9 — failure counts
    scores = [float(r["avg_score"]) for r in rows if r.get("avg_score") is not None]
    for r, h in zip(rows, hyps):
        ref_n, hyp_n = normalize(r["text"]), normalize(h)
        rw, hw = ref_n.split(), hyp_n.split()
        audio_s += float(r.get("duration", 0) or 0)
        if not hw:
            empty_hyp += 1
        if rw:
            d = Levenshtein.distance(rw, hw)
            tot_d += d
            tot_n += len(rw)
            per_utt.append(d / len(rw))
            key = (r.get("file"), r.get("speaker"))
            per_spk[key][0] += d
            per_spk[key][1] += len(rw)
            per_spk[key][2] += 1
            utt_wer = d / len(rw)
        else:
            empty_ref += 1
            utt_wer = None
        results.append({**r, "hyp": h, "ref_norm": ref_n, "hyp_norm": hyp_n,
                        "ref_words": len(rw), "wer": utt_wer})

    corpus_wer = tot_d / tot_n if tot_n else None
    mean_wer = sum(per_utt) / len(per_utt) if per_utt else None
    rtfx = (audio_s / decode_s) if decode_s else None

    print("\n" + "=" * 64)
    print(f"MODEL      : {model_name}")
    print(f"MANIFEST   : {manifest}")
    print(f"utterances : {len(rows)} ({len(per_utt)} scored, "
          f"{empty_ref} empty-ref, {empty_hyp} empty-hyp)")
    print(f"CORPUS WER : {corpus_wer*100:.2f}%   (sum edits / sum ref words)")
    print(f"mean utt   : {mean_wer*100:.2f}%   decode {decode_s:.1f}s"
          + (f" (RTFx {rtfx:.1f})" if rtfx else ""), flush=True)

    summary = {
        "model": model_name, "manifest": str(manifest),
        "utterances": len(rows), "scored": len(per_utt),
        "empty_ref": empty_ref, "empty_hyp": empty_hyp,          # C9
        "audio_min": round(audio_s / 60, 2), "decode_s": round(decode_s, 1),
        "rtfx": round(rtfx, 1) if rtfx else None,
        "corpus_wer": corpus_wer, "mean_utt_wer": mean_wer,
        "per_speaker": {f"{fl}|{spk}": {"segs": segs, "ref_words": n,
                                        "wer": (d / n if n else None)}
                        for (fl, spk), (d, n, segs) in per_spk.items()},
    }

    # legacy/gate output: JSONL rows + summary.json next to --out (unchanged behavior)
    if out:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            for x in results:
                f.write(json.dumps(x) + "\n")
        Path(str(outp) + ".summary.json").write_text(json.dumps(summary, indent=2))
        print(f"rows -> {outp}  |  summary -> {outp}.summary.json", flush=True)

    # experiment mode: full provenance per expt-metadata-format.txt
    if exp_name:
        meta = {
            "data": {                                            # spec #1 + C2,C6,C7
                "manifest": str(manifest),
                "description": data_description,
                "utterances": len(rows),
                "manifest_md5": _md5(manifest),
                "padded": bool(padded),
                "pad_seconds": cfg.PAD_SECONDS if padded else 0.0,
                "conf_filter_threshold": conf_filter_threshold,
                "observed_avg_score_min": round(min(scores), 4) if scores else None,
                "observed_avg_score_max": round(max(scores), 4) if scores else None,
            },
            "model": {                                           # spec #2 + C3
                "name": model_name,
                "path": model_name,
                "is_finetuned": bool(finetuned),
                "description": model_description or (
                    "fine-tuned checkpoint" if finetuned else "base pretrained (NGC/HF)"),
                "revision": _model_revision(model_name),
            },
            "compute": _compute_specs(),                         # spec #3
            "timing": {                                          # spec #4
                "load_s": round(load_s, 1),
                "decode_s": round(decode_s, 1),
                "total_s": round(load_s + decode_s, 1),
            },
            "code": {                                            # spec #5 + C1
                "eval_script": str(Path(__file__).resolve()),
                "git": _git_provenance(),
            },
            "eval": {                                            # spec #6, #7 + C5
                "decoding": decoding_desc,
                "kenlm": lm_meta,                                # None unless shallow fusion
                "batch_size": batch_size,
                "workers": workers,
                "limit": limit,
                "streaming": False,
                "vad_applied": False,
                "normalizer": "asrpipe.common.wer.normalize "
                              "(lowercase, strip to [a-z0-9 ], collapse ws; "
                              "word-level Levenshtein)",
            },
            "reproduce_command": reproduce_command,              # spec #10
        }
        _write_experiment(exp_name, summary, results, meta, log_file=log_file)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None, help="optional JSONL+summary.json (gate mode)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    # experiment-metadata mode (expt-metadata-format.txt)
    ap.add_argument("--exp-name", default=None,
                    help="dataTag_modelTag_exptTag; triggers CSV + metadata emitters")
    ap.add_argument("--data-description", default=None)
    ap.add_argument("--model-description", default=None)
    ap.add_argument("--finetuned", action="store_true", help="model is a fine-tuned ckpt")
    ap.add_argument("--padded", action="store_true", help="manifest audio is silence-padded")
    ap.add_argument("--conf-filter-threshold", type=float, default=None,
                    help="avg_score filter applied when the manifest was built (None = unfiltered)")
    ap.add_argument("--log-file", default=None,
                    help="path to this run's stdout/stderr log (recorded in the track-list, spec #11)")
    # CTC + KenLM shallow-fusion decoding (pyctcdecode); default off -> greedy
    ap.add_argument("--kenlm", default=None, help="KenLM .bin/.arpa for CTC shallow fusion")
    ap.add_argument("--unigrams", default=None, help="unigram list (pyctcdecode, optional)")
    ap.add_argument("--lm-alpha", type=float, default=None, help=f"LM weight (default cfg {cfg.KENLM_ALPHA})")
    ap.add_argument("--lm-beta", type=float, default=None, help=f"word-insertion bonus (default cfg {cfg.KENLM_BETA})")
    ap.add_argument("--lm-beam-size", type=int, default=None, help=f"beam width (default cfg {cfg.KENLM_BEAM_SIZE})")
    ap.add_argument("--gpu-batch", type=int, default=None, help="logits forward batch (KenLM mode; default = --batch-size)")
    a = ap.parse_args()
    if not a.out and not a.exp_name:
        ap.error("need --out (gate mode) or --exp-name (experiment mode)")
    run(a.manifest, a.out, a.model, a.limit, a.batch_size, a.workers,
        exp_name=a.exp_name, data_description=a.data_description,
        finetuned=a.finetuned, model_description=a.model_description,
        conf_filter_threshold=a.conf_filter_threshold, padded=a.padded,
        reproduce_command=_reproduce_cmd(a), log_file=a.log_file,
        kenlm=a.kenlm, unigrams=a.unigrams, lm_alpha=a.lm_alpha, lm_beta=a.lm_beta,
        lm_beam_size=a.lm_beam_size, gpu_batch=a.gpu_batch)


if __name__ == "__main__":
    main()
