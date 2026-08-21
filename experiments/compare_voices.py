#!/usr/bin/env python3
"""Voice quality matrix: xn-ptts (v1 / 2026-01 / 2026-04 voices) vs pip pocket-tts.

Same text, 5 voices, objective metrics + spectrograms.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from quality import metrics, SAMPLE_RATE  # reuse metric functions

OUT = Path(__file__).parent / "quality"
TEXT = ("It was a bright cold day in April, and the clocks were striking thirteen. "
        "She walked through the garden gate and into a world of green hedges and quiet paths. "
        "The old lighthouse stood alone on the cliff, its beam sweeping the dark water every night.")


async def gen(uri: str, voice: str, text: str, out: Path) -> float:
    import websockets
    t0 = time.perf_counter()
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice": voice}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "ready":
                break
        await ws.send(json.dumps({"type": "text", "text": text}))
        await ws.send(json.dumps({"type": "flush", "flush_id": 1}))
        pcm = np.zeros(0, dtype=np.float32)
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "audio":
                c = np.frombuffer(base64.b64decode(m["audio"]), dtype="<i2").astype(np.float32) / 32768.0
                pcm = np.concatenate([pcm, c])
            elif m["type"] == "flushed" and m["flush_id"] == 1:
                break
        await ws.send(json.dumps({"type": "end_of_stream"}))
    pcm.astype("<f4").tofile(out)
    return time.perf_counter() - t0


def read_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path)) as w:
        assert w.getframerate() == SAMPLE_RATE, w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


async def main() -> None:
    servers = {
        "xn_v1": "ws://127.0.0.1:8002/speech/tts",
        "xn_2026-01": "ws://127.0.0.1:8005/speech/tts",
        "xn_2026-04": "ws://127.0.0.1:8004/speech/tts",
    }
    voices = ["alba", "anna", "charles", "eve", "fantine"]
    rows = []
    for label, uri in servers.items():
        for v in voices:
            p = OUT / f"voice_{v}_{label}.f32"
            dt = await gen(uri, v, TEXT, p)
            m = metrics(np.fromfile(p, dtype="<f4"))
            m.update({"voice": v, "source": label, "wall_s": dt})
            rows.append(m)
            print(f"{label:12s} {v:8s}: lufs={m['lufs'] and round(m['lufs'],1):<6} "
                  f"noise={m['noise_floor_db'] and round(m['noise_floor_db'],1)}dB "
                  f"hiss={m['hiss_ratio'] and round(m['hiss_ratio'],3)} "
                  f"glitch={m['glitch_count']} rtf={dt/m['duration_s']:.2f}")
    for lang in ["english_2026-01", "english"]:
        for v in voices:
            p = OUT / f"pip_{v}_{lang}.wav"
            if not p.exists():
                continue
            pcm = read_wav(p)
            m = metrics(pcm)
            m.update({"voice": v, "source": f"pip_{lang}", "wall_s": None})
            rows.append(m)
            print(f"{'pip_'+lang:12s} {v:8s}: lufs={m['lufs'] and round(m['lufs'],1):<6} "
                  f"noise={m['noise_floor_db'] and round(m['noise_floor_db'],1)}dB "
                  f"hiss={m['hiss_ratio'] and round(m['hiss_ratio'],3)} "
                  f"glitch={m['glitch_count']}")
    (OUT / "voice_matrix.json").write_text(json.dumps(rows, indent=1))
    print("saved", OUT / "voice_matrix.json")


if __name__ == "__main__":
    asyncio.run(main())
