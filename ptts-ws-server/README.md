# ptts-ws-server

WebSocket server for Pocket TTS, optimized for this EPYC 7302 (see
`experiments/notebook.md` for the measurement campaign that motivates the
defaults).

## Protocol

JSON messages over WebSocket (all messages tagged `type`):

- client → `setup` (`output_format`, `voice`/`voice_id`, `json_config`,
  `model_name`, `voice_emb`)
- server → `ready` (`sample_rate`, `frame_size`, `request_id`), optionally an
  `audio` header for `wav` output
- client → `text` (utterance), `flush` (`flush_id`), `end_of_stream`
- server → `audio` (base64 chunks, `start_s`/`stop_s`/`stream_id`), `text`
  (subtitle frames), `flushed`, `end_of_stream`, `error`

`output_format`: `pcm` (s16le 24 kHz, default), `pcm_24000`, `wav`, `opus`,
`ulaw_8000`, `alaw_8000`.

Voices are loaded from `<config-dir>/voices/*.safetensors` (keyed by file
stem) plus any `--voice-dir`.

## Docker (recommended)

Drop-in replacement for the existing image (`Dockerfile.xnptts.current`):
same bases, same CMD shape, same `/models` mount. Optimized defaults baked in:
`RAYON_NUM_THREADS=1` (single-threaded generation streams), q8_0 weights,
physical-core cpuset.

```sh
docker compose up -d --build     # serve /models on :8000
```

The image `ENV RAYON_NUM_THREADS=1`; quantization and address are deployment
knobs (compose `command` or `docker run` args): `--quant q8_0` (or `q4k` for
≈12% more throughput at high concurrency, or `f32` for parity with the old
image). Recommended cpuset: `--cpuset-cpus=0-15` (physical cores only).

Output handling: raw decoder output (no post-processing — loudness
normalization and denoising were removed after they proved to overboost and
clip). Long texts are split into sentence-aligned chunks of at most
`--max-tokens-per-chunk` tokens (default 150; mirrors pip pocket-tts
chunking — without it, long paragraphs degrade into stuttering noise:
measured 146s of -40dB-noise audio vs 79s clean on the same 433-token text).
`--default-voice` sets the fallback for unknown/omitted voices.

Voices: both classic `audio_prompt`-style embeddings and pip pocket-tts
**precomputed-state** voices are supported (`transformer.layers.N.self_attn/
cache` files — the 2026-01/2026-04 voice sets).

Model mount layout (`/models`):

```
/models/config.json
/models/model.safetensors        # or model.q8.gguf
/models/tokenizer.model
/models/voices/<name>.safetensors
```

To swap over the live server (as root):

```sh
kill $(pgrep -f "ptts-ws-server --addr 0.0.0.0:8000")
docker compose up -d --build
```

## Bare binary

```sh
cargo run --release -p ptts-ws-server -- --addr 0.0.0.0:8000 \
    --config /models/config.json --quant q8_0
```

## Tuning notes (measured on EPYC 7302, 16 cores / 8×16MB L3)

- Independent single-threaded streams beat multi-threaded streams at equal core
  counts: 16 streams × 1 thread = 18.5× realtime vs 4 streams × 4 threads =
  13.2× vs 1 stream × 16 threads = 4.2× (spread pinning).
- With `RAYON_NUM_THREADS=1` each ws connection is its own single-threaded
  stream; 4 concurrent voices measured 10.0× realtime vs 7.4× for the old
  f32/32-thread-rayon image.
- Pin to physical cores only (`--cpuset-cpus=0-15`); SMT siblings hurt.
- The flow-LM KV cache is q8_0-quantized in `ptts` (3.8× less KV memory per
  session, speed-neutral).
