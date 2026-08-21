import asyncio, base64, json, sys
import numpy as np
from longtext_ab import LONG
async def main():
    import websockets
    uri = "ws://pocket-tts:8000/speech/tts"
    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "setup", "output_format": "pcm", "voice": "jane"}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "ready":
                break
        await ws.send(json.dumps({"type": "text", "text": LONG}))
        await ws.send(json.dumps({"type": "flush", "flush_id": 1}))
        pcm = np.zeros(0, dtype=np.float32)
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "audio":
                c = np.frombuffer(base64.b64decode(m["audio"]), dtype="<i2").astype(np.float32) / 32768.0
                pcm = np.concatenate([pcm, c])
            elif m["type"] == "flushed" and m["flush_id"] == 1:
                break
    pcm.astype("<f4").tofile("/exp/quality/longtext_old.f32")
    print("saved", len(pcm)/24000, "s")
asyncio.run(main())
