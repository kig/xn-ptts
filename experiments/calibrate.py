#!/usr/bin/env python3
"""Calibration: map text length -> audio duration and per-utterance cost.

Single-threaded (RAYON_NUM_THREADS=1), one voice. Measures RTF at several
text lengths for f32 and q8_0 so we can pick utterance texts that yield
~30s of audio and know the per-utterance baseline cost.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import build_utterances, load_model, measure_rtf, save_result

os.environ.setdefault("RAYON_NUM_THREADS", "1")

LENGTHS = [200, 400, 600, 800, 1000, 1200]  # target chars
REPS = 2


def run() -> None:
    results = {"lengths": {}, "quant": os.environ.get("QUANT", "f32")}
    for quant in [None, "q8_0"]:
        t0 = time.perf_counter()
        model = load_model(quant)
        print(f"[{quant}] model load {time.perf_counter()-t0:.1f}s")
        state = model.get_state_for_voice("a")
        # warmup: one short utterance to fault in weights
        w = measure_rtf(state, build_utterances(1, 200)[0])
        print(f"[{quant}] warmup {w['audio_s']:.1f}s audio in {w['wall_s']:.1f}s (rtf {w['rtf']:.2f})")
        per_len = []
        for chars in LENGTHS:
            utts = build_utterances(REPS, chars, seed=chars)
            row = []
            for text in utts:
                r = measure_rtf(state, text)
                row.append(r)
                print(f"[{quant}] {chars}c -> audio {r['audio_s']:.1f}s wall {r['wall_s']:.1f}s rtf {r['rtf']:.2f} rms {r['rms']:.3f}")
            per_len.append({"chars": chars, "runs": row})
        results["lengths"][str(quant)] = per_len
    p = save_result("calibration", results)
    print("saved", p)


if __name__ == "__main__":
    run()
