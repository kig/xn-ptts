#!/usr/bin/env python3
"""Quality + CPU measurement for the TTS post-processing pipeline.

Generates utterances (raw and post-processed) via the ws server, plus pip
pocket-tts reference outputs, and computes objective audio quality metrics:
integrated LUFS / LRA / true peak, clipping, glitch discontinuities, noise
floor + hiss-band energy, spectral flatness. Saves WAVs + a metrics table.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SAMPLE_RATE = 24000
OUT = Path(__file__).parent / "quality"
OUT.mkdir(exist_ok=True)

TEXTS = [
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "She walked through the garden gate and into a world of green hedges and quiet paths. "
    "The old lighthouse stood alone on the cliff, its beam sweeping the dark water every night.",
    "The train rattled through the countryside, past fields of golden wheat and red barns. "
    "A gentle rain began to fall, tapping softly on the windowpanes like a distant melody. "
    "The baker rose before dawn to light the ovens and knead the day's first loaves.",
]


async def generate(uri: str, voice: str, text: str, out_path: Path, n: int = 1) -> float:
    """One session: setup -> n texts flushed individually -> pcm out. Returns wall s."""
    t0 = time.perf_counter()
    async with websockets_connect(uri) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice": voice}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "ready":
                break
        pcm = np.zeros(0, dtype=np.float32)
        for i in range(n):
            await ws.send(json.dumps({"type": "text", "text": text}))
            await ws.send(json.dumps({"type": "flush", "flush_id": i}))
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "audio":
                    chunk = np.frombuffer(base64.b64decode(m["audio"]), dtype="<i2").astype(
                        np.float32
                    ) / 32768.0
                    pcm = np.concatenate([pcm, chunk])
                elif m["type"] == "flushed" and m["flush_id"] == i:
                    break
        await ws.send(json.dumps({"type": "end_of_stream"}))
    dt = time.perf_counter() - t0
    pcm.astype("<f4").tofile(out_path)
    return dt


def websockets_connect(uri):
    import websockets
    return websockets.connect(uri, max_size=None)


def metrics(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    import pyloudnorm as pyln
    m = {}
    x = np.asarray(pcm, dtype=np.float64)
    m["samples"] = int(x.size)
    m["duration_s"] = float(x.size / sr)
    m["nan"] = int(np.isnan(x).sum()) + int(np.isinf(x).sum())
    m["peak"] = float(np.abs(x).max())
    # clipping + near-clip
    m["clip_frac"] = float((np.abs(x) >= 0.999).mean())
    # glitches: adjacent-sample jumps too big for speech at 24 kHz
    if x.size > 1:
        d = np.abs(np.diff(x))
        m["glitch_count"] = int((d > 0.9).sum())
        m["glitch_p95"] = float(np.percentile(d, 99.99))
    else:
        m["glitch_count"] = 0
        m["glitch_p95"] = 0.0
    # loudness (ITU BS.1770-4) via pyloudnorm
    try:
        meter = pyln.Meter(sr)
        m["lufs"] = float(meter.integrated_loudness(x))
        m["lra"] = float(meter.loudness_range(x))
    except Exception:
        m["lufs"] = None
        m["lra"] = None
    # true peak via 4x oversampling (pyloudnorm here lacks true_peak)
    try:
        from scipy.signal import resample_poly
        ov = resample_poly(x, 4, 1)
        m["true_peak_db"] = float(20 * np.log10(np.abs(ov).max() + 1e-12))
    except Exception:
        m["true_peak_db"] = None
    # noise: RMS of the quietest 5% of 20ms frames; hiss = 8-12 kHz energy share
    frame = int(0.02 * sr)
    nf = x.size // frame
    if nf > 20:
        frames = x[: nf * frame].reshape(nf, frame)
        frms = np.sqrt((frames ** 2).mean(axis=1))
        q = np.quantile(frms, 0.05)
        quiet = frames[frms <= q]
        m["noise_floor_db"] = float(20 * np.log10(np.sqrt((quiet ** 2).mean()) + 1e-12))
        # hiss ratio in quiet frames
        spec = np.abs(np.fft.rfft(quiet, axis=1))
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        hi = (freqs >= 8000) & (freqs <= 12000)
        lo = freqs <= 4000
        m["hiss_ratio"] = float(spec[:, hi].sum() / (spec[:, lo].sum() + 1e-12))
        # spectral flatness (geometric/arithmetic mean of power)
        pw = spec ** 2 + 1e-12
        m["flatness"] = float(np.exp(np.mean(np.log(pw.mean(axis=0)))) / pw.mean(axis=0).mean())
    else:
        m["noise_floor_db"] = None
        m["hiss_ratio"] = None
        m["flatness"] = None
    return m


def make_spectrogram(path: Path, pcm: np.ndarray, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.specgram(pcm, Fs=SAMPLE_RATE, NFFT=512, noverlap=384, cmap="magma")
    ax.set_title(path.stem)
    ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-uri", default="ws://127.0.0.1:8002/speech/tts", help="server WITHOUT postprocess")
    ap.add_argument("--pp-uri", default="ws://127.0.0.1:8003/speech/tts", help="server WITH postprocess")
    ap.add_argument("--voices", default="anna,charles,eve,fantine,alba")
    ap.add_argument("--pip", action="store_true", help="also generate with pip pocket-tts")
    args = ap.parse_args()
    import websockets  # noqa

    rows = []
    for voice in args.voices.split(","):
        for ti, text in enumerate(TEXTS):
            for label, uri in [("raw", args.raw_uri), ("pp", args.pp_uri)]:
                p = OUT / f"{voice}_{ti}_{label}.f32"
                dt = await generate(uri, voice, text, p)
                pcm = np.fromfile(p, dtype="<f4")
                m = metrics(pcm)
                m.update({"voice": voice, "text": ti, "label": label, "wall_s": dt})
                rows.append(m)
                print(f"{voice} t{ti} {label}: lufs={m['lufs'] and round(m['lufs'],1)} "
                      f"noise={m['noise_floor_db'] and round(m['noise_floor_db'],1)}dB "
                      f"glitch={m['glitch_count']} clip={m['clip_frac']:.4f} "
                      f"wall={dt:.1f}s")
    if args.pip:
        from pocket_tts import main as ppm  # noqa: F401  (import side effects)
        # generate via the pip package's python API
        import importlib
        mm = importlib.import_module("pocket_tts.main")
        gen = getattr(mm, "generate", None) or getattr(mm, "main", None)
        print("pip pocket-tts API:", [a for a in dir(mm) if "generate" in a.lower() or "tts" in a.lower()][:8])
        for voice in args.voices.split(","):
            for ti, text in enumerate(TEXTS):
                p = OUT / f"{voice}_{ti}_pip.f32"
                # try common API shapes
                try:
                    out = mm.generate(text=text, voice=voice)
                except TypeError:
                    try:
                        out = mm.generate(text, voice)
                    except Exception as e:
                        print(f"pip {voice} t{ti} failed: {e}")
                        continue
                pcm = np.asarray(out, dtype=np.float32)
                pcm.tofile(p)
                m = metrics(pcm)
                m.update({"voice": voice, "text": ti, "label": "pip"})
                rows.append(m)
                print(f"pip {voice} t{ti}: lufs={m['lufs'] and round(m['lufs'],1)} "
                      f"noise={m['noise_floor_db'] and round(m['noise_floor_db'],1)}dB glitch={m['glitch_count']}")
    # save table + spectrograms
    with open(OUT / "metrics.json", "w") as f:
        json.dump(rows, f, indent=1)
    for voice in args.voices.split(","):
        for ti in range(len(TEXTS)):
            for label in ["raw", "pp"]:
                p = OUT / f"{voice}_{ti}_{label}.f32"
                if p.exists():
                    pcm = np.fromfile(p, dtype="<f4")
                    make_spectrogram(p, pcm, OUT / f"{voice}_{ti}_{label}.png")
    print("saved", OUT / "metrics.json")


if __name__ == "__main__":
    asyncio.run(main())
