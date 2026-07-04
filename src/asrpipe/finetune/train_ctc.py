"""Phase 15 — fine-tune nvidia/parakeet-ctc-0.6b on legal2023_38hrs (NeMo FastConformer-CTC).

Loads the pretrained CTC-BPE model, rewires it to the legal2023_38hrs train/dev manifests
(KEEPING the pretrained BPE tokenizer + vocab), optionally freezes the encoder for the first
N epochs (train the CTC head, then gradual-unfreeze), and trains with exp_manager
(TensorBoard + val_wer checkpointing + early stopping). Config: conf/ctc_finetune.yaml.

Run under the NeMo venv:
  .venv-nemo/bin/python -m asrpipe.finetune.train_ctc                         # full run
  .venv-nemo/bin/python -m asrpipe.finetune.train_ctc --fast-dev-run          # 1-batch smoke test
  .venv-nemo/bin/python -m asrpipe.finetune.train_ctc --config <path> --max-epochs 3

Baseline to beat (parakeet-ctc-0.6b, legal2023_38hrs padded test): corpus WER 23.39%.
Port source : NEW (RP7 / Phase 15).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf, open_dict

try:                                   # NeMo 2.x -> lightning.pytorch; older -> pytorch_lightning
    import lightning.pytorch as pl
except ImportError:                    # pragma: no cover
    import pytorch_lightning as pl

DEFAULT_CFG = str(Path(__file__).parent / "conf" / "ctc_finetune.yaml")


class UnfreezeEncoder(pl.Callback):
    """Gradual unfreezing: unfreeze the encoder at the start of epoch `at_epoch`."""
    def __init__(self, at_epoch: int):
        self.at = at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if self.at > 0 and trainer.current_epoch == self.at:
            pl_module.encoder.unfreeze()
            print(f"[freeze] encoder UNFROZEN at epoch {self.at}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--fast-dev-run", action="store_true",
                    help="smoke test: run 1 train + 1 val batch, no checkpoints")
    ap.add_argument("--max-epochs", type=int, default=None, help="override trainer.max_epochs")
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    if a.max_epochs is not None:
        cfg.trainer.max_epochs = a.max_epochs

    import nemo.collections.asr as nemo_asr
    from nemo.utils.exp_manager import exp_manager

    freeze_epochs = int(cfg.get("freeze_encoder_epochs", 0))
    callbacks = [UnfreezeEncoder(freeze_epochs)] if freeze_epochs > 0 else []

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    if a.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True
    trainer = pl.Trainer(callbacks=callbacks, **trainer_kwargs)

    print(f"loading pretrained {cfg.init_from_pretrained_model} ...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(cfg.init_from_pretrained_model)
    model.set_trainer(trainer)

    # rewire data (tokenizer/vocab UNCHANGED) + optimizer
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)
    with open_dict(model.cfg):
        model.cfg.optim = cfg.model.optim
    model.setup_optimization(cfg.model.optim)

    if freeze_epochs > 0:
        model.encoder.freeze()
        print(f"[freeze] encoder FROZEN for first {freeze_epochs} epoch(s) "
              f"(training the CTC head only)", flush=True)

    exp_manager(trainer, cfg.exp_manager)   # also exercised by --fast-dev-run (catches logger/config issues)

    trainer.fit(model)

    if not a.fast_dev_run:
        # exp_manager saves the best .ckpt (save_top_k=1, monitor=val_wer) + exports the best
        # .nemo (save_best_model). No extra explicit save (keeps disk/quota footprint minimal).
        run_dir = Path(cfg.exp_manager.exp_dir) / cfg.exp_manager.name
        print(f"training done. best .ckpt + best .nemo under: {run_dir}/checkpoints/", flush=True)
    else:
        print("fast-dev-run OK — training pipeline is wired correctly.", flush=True)


if __name__ == "__main__":
    main()
