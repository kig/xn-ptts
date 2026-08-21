#!/usr/bin/env python3
"""Minimal probe: setup -> text -> flush -> audio, against a ptts-ws-server."""
import asyncio, json, base64, sys, time
import websockets

TEXT = ("It was a bright cold day in April, and the clocks were striking thirteen. "
        "She walked through the garden gate and into a world of green hedges and quiet paths. "
        "The old lighthouse stood alone on the cliff, its beam sweeping the dark water every night.")

async def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://pocket-tts:8000/speech/tts"
    t0 = time.perf_counter()
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice": "anna"}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "ready":
                print(f"ready: sr={m['sample_rate']} frame={m['frame_size']}")
                break
        await ws.send(json.dumps({"type": "text", "text": TEXT}))
        await ws.send(json.dumps({"type": "flush", "flush_id": 1}))
        nb = 0; chunks = 0
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "audio":
                nb += len(base64.b64decode(m["audio"])); chunks += 1
            elif m["type"] == "flushed" and m["flush_id"] == 1:
                break
        await ws.send(json.dumps({"type": "end_of_stream"}))
    dt = time.perf_counter() - t0
    dur = nb / 2 / 24000
    print(f"OK: {dur:.1f}s audio in {dt:.1f}s wall (rtf {dt/dur:.2f}), {chunks} chunks")
    assert dur > 15, "audio too short"

if __name__ == "__main__":
    asyncio.run(main())
