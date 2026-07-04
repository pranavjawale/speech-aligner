"""S3 — parallel Parakeet-CTC + KenLM(+unigrams) pre-pass.

Two decoupled stages so the CPU beam search runs across many cores:
  1. GPU: batched Parakeet-CTC forward -> log-probs (B, T, V+1).
  2. CPU: pyctcdecode decode_batch(Pool) over all chunks (the parallelized part).

Writes data/chunks/{stem}/{stem}_ctc_lm.jsonl:
  {chunk_index, chunk_file, ctc_lm_hyp}

The ctc_lm_hyp is used ONLY to localize each chunk in the reference (S4 matcher);
biasing it toward court vocabulary (6-gram KenLM + 12.6k unigrams) improves that
localization. Idempotent: a complete {stem}_ctc_lm.jsonl is a cache hit.

Runs in the NeMo venv (code2/.venv-nemo). Faithful port of
/workspace/code/ctc_kenlm_prepass_parallel.py — same model/LM/beam, so hypotheses
are identical; uses the shared textnorm.clean_text and config-driven paths.

Port source : ctc_kenlm_prepass_parallel.py
Reorg phase : RP3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import config as cfg
from asrpipe.common import paths as P
from asrpipe.common.textnorm import clean_text


def run_range(start: int, end: int, *, alpha: float | None = None,
              beta: float | None = None, beam_size: int | None = None,
              gpu_batch: int | None = None, workers: int | None = None,
              lm: str | None = None, unigrams: str | None = None) -> None:
    """Decode files [start, end) (end EXCLUSIVE) -> per-chunk ctc_lm hypotheses."""
    alpha = cfg.KENLM_ALPHA if alpha is None else alpha
    beta = cfg.KENLM_BETA if beta is None else beta
    beam_size = cfg.KENLM_BEAM_SIZE if beam_size is None else beam_size
    gpu_batch = cfg.PREPASS_GPU_BATCH if gpu_batch is None else gpu_batch
    workers = cfg.PREPASS_WORKERS if workers is None else workers
    lm = str(cfg.KENLM_BIN) if lm is None else lm
    unigrams_path = str(cfg.UNIGRAMS_TXT) if unigrams is None else unigrams

    import soundfile as sf
    import torch
    import nemo.collections.asr as nemo_asr
    import pyctcdecode

    print(f"Loading {cfg.CTC_LM_MODEL} ...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.CTC_LM_MODEL)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    vocab = list(model.decoder.vocabulary)
    unigrams = [w.strip() for w in open(unigrams_path) if w.strip()] if unigrams_path else None
    print(f"pyctcdecode decoder (KenLM + {len(unigrams) if unigrams else 0} unigrams, "
          f"alpha={alpha} beta={beta} beam={beam_size})", flush=True)
    decoder = pyctcdecode.build_ctcdecoder(
        vocab, kenlm_model_path=lm, unigrams=unigrams, alpha=alpha, beta=beta)

    n_workers = min(workers, (os.cpu_count() or 4) - 2)
    pool = Pool(n_workers)
    print(f"CPU decode pool: {n_workers} workers", flush=True)

    @torch.inference_mode()
    def file_logits(paths):
        """List of (T_i, V+1) float32 log-prob arrays for the given chunk wavs."""
        out = []
        for i in range(0, len(paths), gpu_batch):
            bp = paths[i:i + gpu_batch]
            sigs = [sf.read(p)[0].astype("float32") for p in bp]
            lens = [len(s) for s in sigs]
            batch = torch.zeros(len(sigs), max(lens))
            for j, s in enumerate(sigs):
                batch[j, :len(s)] = torch.from_numpy(s)
            lp, enc_len, *_ = model.forward(
                input_signal=batch.to(dev),
                input_signal_length=torch.tensor(lens).to(dev))
            lp = lp.cpu().numpy()
            for j in range(len(sigs)):
                out.append(lp[j, :int(enc_len[j])])
        return out

    for n in range(start, end):
        stem = P.stem(n)
        man = P.chunk_manifest(n)
        chunk_dir = P.chunk_dir(n)
        if not man.exists() or not chunk_dir.is_dir():
            print(f"[{stem}] SKIP (no manifest/chunk dir)", flush=True)
            continue
        chunks_meta = json.loads(man.read_text())["chunks"]
        out_path = P.ctc_lm_jsonl(n)
        if out_path.exists():
            try:
                have = len([l for l in out_path.read_text().splitlines() if l.strip()])
                if have == len(chunks_meta):
                    print(f"[{stem}] cache hit ({have}) -> skip", flush=True)
                    continue
            except Exception:
                pass
        paths = [str(chunk_dir / c["chunk_file"]) for c in chunks_meta]
        t0 = time.time()
        logits = file_logits(paths)
        t1 = time.time()
        texts = decoder.decode_batch(pool, logits, beam_width=beam_size)
        t2 = time.time()
        lines = [json.dumps({"chunk_index": m["chunk_index"],
                             "chunk_file": m["chunk_file"],
                             "ctc_lm_hyp": clean_text(t)})
                 for m, t in zip(chunks_meta, texts)]
        out_path.write_text("\n".join(lines) + "\n")
        print(f"[{stem}] {len(paths)} chunks  fwd={t1-t0:.1f}s "
              f"decode={t2-t1:.1f}s -> {out_path.name}", flush=True)

    pool.close()
    pool.join()
    print("S3 DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", type=int, default=1)
    ap.add_argument("end", nargs="?", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--gpu-batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--unigrams", default=None)
    a = ap.parse_args()
    run_range(a.start, a.end, alpha=a.alpha, beta=a.beta, beam_size=a.beam_size,
              gpu_batch=a.gpu_batch, workers=a.workers, lm=a.lm, unigrams=a.unigrams)


if __name__ == "__main__":
    main()
