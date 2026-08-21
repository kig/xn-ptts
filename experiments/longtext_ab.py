#!/usr/bin/env python3
"""Long-text A/B: old prod server (postprocess, no chunking) vs new build.

Long complex paragraph (>150 tokens) + a short control. Metrics via quality.py.
"""
import asyncio, base64, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from quality import metrics, SAMPLE_RATE

LONG = (
    "In this paper, the authors present a comprehensive framework for agentic artificial intelligence "
    "that integrates large language models with chain-of-thought reasoning and verification mechanisms "
    "to address the complex scheduling problem of unmanned aerial vehicles in mobile edge computing "
    "environments, where the fundamental challenge lies in the tight coupling between delivery route "
    "optimization and computational task offloading decisions, since the route determines both when each "
    "station can offload its computing workload and how the drones can simultaneously serve as both "
    "transport vehicles and flying computational resources, and they demonstrate through extensive "
    "experiments on synthetic and real-world workloads that their approach consistently outperforms "
    "traditional optimization methods across multiple performance metrics including makespan, energy "
    "consumption, and service quality, while also providing interpretable explanations of the decisions "
    "made by the system at each stage of the planning process, which is particularly valuable for "
    "industrial deployment scenarios where operators need to understand and trust the automated "
    "scheduling decisions before they can be safely integrated into production environments, and the "
    "authors further discuss the limitations of their current implementation and outline several "
    "promising directions for future research, including the extension of their framework to "
    "multi-drone coordination, dynamic re-planning under uncertainty, and the integration of "
    "reinforcement learning components for continuous policy improvement."
)
SHORT = "The quick brown fox jumps over the lazy dog, and the dog sleeps peacefully by the fire."


async def gen(uri: str, text: str) -> tuple[np.ndarray, float]:
    import websockets
    t0 = time.perf_counter()
    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice": "jane"}))
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
    return pcm, time.perf_counter() - t0


def report(label: str, pcm: np.ndarray, wall: float) -> None:
    m = metrics(pcm)
    print(f"{label:38s}: {m['duration_s']:.1f}s audio rtf={wall/m['duration_s']:.2f} "
          f"lufs={m['lufs'] and round(m['lufs'],1)} peak={m['peak']:.2f} "
          f"clip={m['clip_frac']:.4f} glitch={m['glitch_count']} "
          f"noise={m['noise_floor_db'] and round(m['noise_floor_db'],1)}dB "
          f"hiss={m['hiss_ratio'] and round(m['hiss_ratio'],3)}")


async def main() -> None:
    old = "ws://pocket-tts:8000/speech/tts"  # via backend network (old image, still deployed)
    new = "ws://127.0.0.1:8008/speech/tts"
    for label, uri in [("OLD prod (postproc, no chunk)", old), ("NEW (raw, chunked)", new)]:
        for name, text in [("SHORT", SHORT), ("LONG (>150 tok)", LONG)]:
            try:
                pcm, wall = await gen(uri, text)
                report(f"{label} {name}", pcm, wall)
            except Exception as e:
                print(f"{label} {name}: FAILED {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
