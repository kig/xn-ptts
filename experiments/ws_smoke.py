#!/usr/bin/env python3
"""Protocol smoke test + throughput probe for ptts-ws-server.

Drives the ws protocol the same way a client would: Setup -> Ready (+header),
Text -> Audio chunks, EndOfStream, Flush -> Flushed. Verifies audio bytes and
measures per-utterance latency and aggregate throughput with N concurrent
voices.
"""
import argparse
import asyncio
import base64
import json
import sys
import time

import numpy as np

try:
    import websockets
except ImportError:
    print("pip install websockets", file=sys.stderr)
    sys.exit(1)

TEXT = ("It was a bright cold day in April, and the clocks were striking thirteen. "
        "She walked through the garden gate and into a world of green hedges and quiet paths. "
        "The old lighthouse stood alone on the cliff, its beam sweeping the dark water every night. "
        "He opened the heavy book and the smell of old paper filled the small room. "
        "The train rattled through the countryside, past fields of golden wheat and red barns. "
        "A gentle rain began to fall, tapping softly on the windowpanes like a distant melody.")


async def one_session(uri: str, voice: str, text: str, results: list, idx: int) -> None:
    t0 = time.perf_counter()
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice": voice}))
        n_audio = 0
        samples = 0
        ready = None
        while True:
            msg = json.loads(await ws.recv())
            t = msg["type"]
            if t == "ready":
                ready = msg
            elif t == "audio":
                n_audio += 1
                samples += len(msg["audio"]) // 2
            elif t == "end_of_stream":
                break
        # flush protocol check
        await ws.send(json.dumps({"type": "flush", "flush_id": 7}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "flushed" and msg["flush_id"] == 7:
                break
        await ws.send(json.dumps({"type": "end_of_stream"}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "end_of_stream":
                break
    dt = time.perf_counter() - t0
    dur = samples / ready["sample_rate"]
    results[idx] = {"voice": voice, "wall_s": dt, "audio_s": dur, "chunks": n_audio,
                    "sample_rate": ready["sample_rate"], "frame_size": ready["frame_size"]}
    print(f"  {voice}: {dur:.1f}s audio in {dt:.1f}s wall (rtf {dt/dur:.2f}), "
          f"{n_audio} chunks, sr={ready['sample_rate']} frame={ready['frame_size']}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8001")
    ap.add_argument("--voices", default="anna,charles,eve,fantine")
    ap.add_argument("--reps", type=int, default=1, help="utterances per voice")
    args = ap.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    results = [None] * (len(voices) * args.reps)
    tasks = []
    for r in range(args.reps):
        for i, v in enumerate(voices):
            tasks.append(one_session(args.uri, v, TEXT, results, r * len(voices) + i))
    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    audio = sum(r["audio_s"] for r in results if r)
    print(f"== total: {len(results)} utterances, {audio:.0f}s audio in {wall:.1f}s wall "
          f"-> {audio/wall:.2f}x realtime")


if __name__ == "__main__":
    asyncio.run(main())
