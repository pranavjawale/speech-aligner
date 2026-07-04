"""unit tests for the asrpipe.common layer.

Run:  source setup_env.sh && python -m pytest tests/ -q
  or:  python tests/test_common.py     (no pytest needed — has a __main__ runner)

Coverage:
  - textnorm: both normalizers + the SAME-word-count invariant (matcher relies on it)
  - wer:      normalize, per-utt WER, corpus vs mean-utt aggregation, empty-ref None
  - audio_io: mask_silence zeroes the right regions; pad_silence adds the right samples
  - paths:    builders derive from config and follow the /code naming conventions
"""
import numpy as np

import config as cfg
from asrpipe.common import textnorm as tn
from asrpipe.common import wer as W
from asrpipe.common import audio_io as aio
from asrpipe.common import paths as P


# ---------------------------------------------------------------- textnorm
def test_clean_text_basic():
    assert tn.clean_text("spk_1023: I'll go, my Lords!") == "ill go my lords"
    # apostrophe removed silently -> "i'll" collapses to "ill"
    assert "ill" in tn.clean_text("I'll").split()


def test_clean_text_for_align_keeps_contractions():
    out = tn.clean_text_for_align("spk_1042: I'll don't go.")
    assert out == "i'll don't go"


def test_same_word_count_invariant():
    # The matcher assumes clean_text and clean_text_for_align yield identical
    # word counts so indices are interchangeable. Check on tricky inputs.
    for s in [
        "spk_1: I'll don't won't",
        "yes, my lords -- I appear in IA No. 123!",
        "'quoted' words and it's fine",
        "spk_99:   multiple    spaces\tand\nnewlines",
    ]:
        assert len(tn.clean_text(s).split()) == len(tn.clean_text_for_align(s).split()), s


# ---------------------------------------------------------------- wer
def test_normalize():
    assert W.normalize("Hello, WORLD!  it's 12.") == "hello world it s 12"


def test_utt_wer_exact_and_empty():
    assert W.utt_wer("the cat sat", "the cat sat") == 0.0
    assert W.utt_wer("", "anything") is None          # empty ref -> unscored
    # one substitution out of three ref words
    assert abs(W.utt_wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9


def test_corpus_vs_mean_utt():
    # utt A: 1 edit / 4 ref = 0.25 ; utt B: 1 edit / 1 ref = 1.0
    acc = W.corpus_wer([("a b c d", "a b c x"), ("z", "q")])
    assert acc.tot_edits == 2 and acc.tot_ref_words == 5
    assert abs(acc.corpus_wer - 2 / 5) < 1e-9          # 0.40
    assert abs(acc.mean_utt_wer - (0.25 + 1.0) / 2) < 1e-9  # 0.625 (differs from corpus)
    assert acc.n_scored == 2 and acc.n_empty_ref == 0


# ---------------------------------------------------------------- audio_io
def test_mask_silence_zeros_gaps():
    sr = cfg.SAMPLE_RATE
    audio = np.ones(5 * sr, dtype=np.float32)          # 5s of ones
    # speech 0.0-1.0 and 3.0-4.0 -> internal gap 1.0-3.0 (2s, >0.8) zeroed;
    # trailing 4.0-5.0 (1s, >0.8) zeroed; no leading gap.
    segs = [{"start": 0.0, "end": 1.0}, {"start": 3.0, "end": 4.0}]
    m = aio.mask_silence(audio, segs)
    assert m[int(0.5 * sr)] == 1.0                      # speech kept
    assert m[int(2.0 * sr)] == 0.0                      # internal gap zeroed
    assert m[int(4.5 * sr)] == 0.0                      # trailing zeroed
    assert audio[int(2.0 * sr)] == 1.0                  # original untouched (copy)


def test_mask_silence_short_gap_preserved():
    sr = cfg.SAMPLE_RATE
    audio = np.ones(2 * sr, dtype=np.float32)
    # gap 1.0-1.5 = 0.5s < 0.8 threshold -> preserved
    segs = [{"start": 0.0, "end": 1.0}, {"start": 1.5, "end": 2.0}]
    m = aio.mask_silence(audio, segs)
    assert m[int(1.25 * sr)] == 1.0


def test_mask_silence_no_segments_passthrough():
    audio = np.ones(100, dtype=np.float32)
    assert aio.mask_silence(audio, []) is audio        # unmodified passthrough


def test_pad_silence_adds_samples():
    sr = cfg.SAMPLE_RATE
    audio = np.ones(sr, dtype=np.float32)              # 1s
    out = aio.pad_silence(audio)                       # default 0.5s each side
    pad_n = int(round(cfg.PAD_SECONDS * sr))
    assert len(out) == len(audio) + 2 * pad_n
    assert out[0] == 0.0 and out[-1] == 0.0
    assert out[pad_n] == 1.0                           # original starts after pad


# ---------------------------------------------------------------- paths
def test_paths_derive_from_config():
    assert P.stem(3) == "audio_file_003"
    assert P.chunk_manifest(3) == cfg.MANIFEST_DIR / "audio_file_003_manifest.json"
    assert P.chunk_wav(3, 12, 40, 70).name == \
        "audio_file_003_chunk_000012_start000040_end000070.wav"
    assert P.split_manifest("test", padded=True) == cfg.SPLITS_PAD_DIR / "test.jsonl"
    assert P.timeline_json(3).parent == cfg.ALIGN_DIR
    # per-chunk alignment JSON is named after the chunk wav (.wav -> .json)
    assert P.align_json(3, "audio_file_003_chunk_000012_start000040_end000070.wav").name == \
        "audio_file_003_chunk_000012_start000040_end000070.json"
    # every builder stays under DATA
    for path in (P.chunk_dir(1), P.align_json(1, "audio_file_001_chunk_000000.wav"),
                 P.speaker_manifest(1), P.segments_dir(1, padded=True), P.result_jsonl("x")):
        assert str(path).startswith(str(cfg.DATA))


# ---------------------------------------------------------------- runner
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
