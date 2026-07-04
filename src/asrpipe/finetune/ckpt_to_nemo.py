"""Convert a Lightning .ckpt (from the fine-tune) into a loadable .nemo.

exp_manager's save_best_model did not emit a .nemo, so we rebuild it: load the
pretrained architecture + tokenizer (KEPT unchanged during fine-tuning) and copy
the fine-tuned weights from the checkpoint's state_dict, then save_to(.nemo).

  .venv-nemo/bin/python -m asrpipe.finetune.ckpt_to_nemo \
      --ckpt exp/finetune/.../checkpoints/....ckpt \
      --base nvidia/parakeet-ctc-0.6b \
      --out  exp/finetune/.../checkpoints/best.nemo
"""
from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="nvidia/parakeet-ctc-0.6b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    print(f"loading base arch+tokenizer: {a.base}", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=a.base)

    print(f"loading checkpoint weights: {a.ckpt}", flush=True)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # ignore known non-parameter buffers (metrics), fail loudly on real weight gaps
    real_missing = [k for k in missing if not k.startswith(("_wer", "wer."))]
    if real_missing:
        raise SystemExit(f"ABORT: {len(real_missing)} missing weight keys, e.g. {real_missing[:5]}")
    print(f"loaded (missing={len(missing)} unexpected={len(unexpected)}; "
          f"non-metric missing={len(real_missing)})", flush=True)

    model.save_to(a.out)
    print(f"saved -> {a.out}", flush=True)

    # verify it restores
    del model
    m2 = nemo_asr.models.ASRModel.restore_from(restore_path=a.out)
    print(f"verify OK: restored {type(m2).__name__}", flush=True)


if __name__ == "__main__":
    main()
