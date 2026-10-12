#!/usr/bin/env python3
"""Measure the 6-11 kHz decoder bursts on a live ptts-ws-server.

Synthesises one websocket request per line (as paperradio does: one line per
request, voice jane, PCM out) and reports, per 20 ms frame, the 6-11 kHz band
against the 300-4000 Hz speech band.

A "burst event" is a run of frames whose 6-11 kHz level sits in the
[-32, -19.6] dBFS window and is 2.5-6x the concurrent 300-4000 Hz band, at
least 20 ms long, with events closer than 0.3 s merged. That is the detector
with which paperradio's audit counted 86-88 bursts in episode 37155005.

Usage:
  experiments/burstcheck.py --uri ws://127.0.0.1:8008/speech/tts --out /tmp/tts
  experiments/burstcheck.py --text "hello world"            # single utterance
  experiments/burstcheck.py --lines-json-dir DIR            # default: that episode

The default line set is the input of paperradio episode 37155005 (paper
2610.11130), whose lines.json survives in the paperradio deployment
(`deploy/radiodata/episodes/37155005-.../segment-*/lines.json`).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import numpy as np

SR = 24000
EPISODE_LINES = Path(
    "/repos/radio_textgen/arxiv/paperradio/deploy/radiodata/episodes/"
    "37155005-3eb9-4940-8d0f-60d8a928fac0"
)


def load_lines(lines_dir: Path) -> list[dict]:
    lines = []
    for f in sorted(
        lines_dir.glob("segment-*/lines.json"), key=lambda p: int(p.parent.name.split("-")[1])
    ):
        for ln in json.loads(f.read_text()):
            ln["segment"] = f.parent.name
            lines.append(ln)
    return lines


async def gen(uri: str, text: str, voice: str) -> np.ndarray:
    import websockets

    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice_id": voice}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "ready":
                break
            if m["type"] == "error":
                raise RuntimeError(m["message"])
        await ws.send(json.dumps({"type": "text", "text": text}))
        await ws.send(json.dumps({"type": "flush", "flush_id": 1}))
        chunks = [np.zeros(0, dtype=np.float32)]
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "audio":
                c = np.frombuffer(base64.b64decode(m["audio"]), dtype="<i2").astype(np.float32)
                chunks.append(c / 32768.0)
            elif m["type"] == "flushed" and m["flush_id"] == 1:
                break
            elif m["type"] == "error":
                raise RuntimeError(m["message"])
        await ws.send(json.dumps({"type": "end_of_stream"}))
    return np.concatenate(chunks)


def band_levels(pcm: np.ndarray, frame: int = 480, hop: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame 6-11 kHz and 300-4000 Hz RMS levels in dBFS."""
    win = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    burst = (freqs >= 6000) & (freqs <= 11000)
    speech = (freqs >= 300) & (freqs <= 4000)
    scale = 2.0 / win.sum()
    n = 1 + max(0, (len(pcm) - frame) // hop)
    b = np.empty(n)
    s = np.empty(n)
    for i in range(n):
        mag = np.abs(np.fft.rfft(pcm[i * hop : i * hop + frame] * win))
        b[i] = 20 * np.log10(np.sqrt((mag[burst] ** 2).sum()) * scale + 1e-12)
        s[i] = 20 * np.log10(np.sqrt((mag[speech] ** 2).sum()) * scale + 1e-12)
    return b, s


def events(b: np.ndarray, s: np.ndarray, frame_s: float = 0.01, gap: float = 0.3) -> list[float]:
    ratio = b - s
    mask = (b >= -32) & (b <= -19.6) & (ratio >= 20 * np.log10(2.5)) & (ratio <= 20 * np.log10(6))
    out: list[float] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        if (j - i + 1) * frame_s >= 0.015:
            t = (i + j) / 2 * frame_s
            if not out or t - out[-1] >= gap:
                out.append(t)
        i = j + 1
    return out


def report(label: str, pcm: np.ndarray) -> int:
    b, s = band_levels(pcm)
    active = s > -60
    ev = events(b, s)
    print(
        f"{label}: {len(pcm)/SR:6.2f}s events={len(ev):2d} "
        f"6-11k p95={np.percentile(b[active], 95):6.1f} max={b.max():6.1f} dBFS "
        f"| 300-4000 mean={s[active].mean():6.1f} dBFS"
    )
    return len(ev)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8008/speech/tts")
    ap.add_argument("--voice", default="jane")
    ap.add_argument("--out", default="", help="directory for the raw PCM (.npy)")
    ap.add_argument("--lines-json-dir", default=str(EPISODE_LINES))
    ap.add_argument("--limit", type=int, default=0, help="first N lines only")
    ap.add_argument("--text", default=None, help="synthesise this instead of the episode lines")
    args = ap.parse_args()
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.text is not None:
        pcm = await gen(args.uri, args.text, args.voice)
        if args.out:
            np.save(Path(args.out) / "text.npy", pcm)
        report("text", pcm)
        return

    lines = load_lines(Path(args.lines_json_dir))
    if args.limit:
        lines = lines[: args.limit]
    total = 0
    audio = 0.0
    for idx, ln in enumerate(lines):
        pcm = await gen(args.uri, ln["text"], args.voice)
        if args.out:
            np.save(Path(args.out) / f"line-{idx:03d}.npy", pcm)
        total += report(f"[{idx:03d} {ln['speaker']:5s}] {ln['text'][:44]!r}", pcm)
        audio += len(pcm) / SR
    print(f"TOTAL: {len(lines)} lines, {audio:.1f}s audio, {total} burst events")


if __name__ == "__main__":
    asyncio.run(main())
