# TTS throughput experiments — EPYC 7302

Scientific study of core pinning + L3 cache strategies for Pocket-TTS on this
EPYC 7302 (16 cores, 8×16MB L3 slices, single NUMA). Results and analysis in
`notebook.md`; raw per-utterance timings in `results/*.json`.

## Setup
- Model: `/tmp/model-local/` (config.json + symlinked `tts_b6369a24.safetensors`,
  `tokenizer.model` from `kyutai/pocket-tts`).
- Voices: `/tmp/voice-out/{anna,charles,eve,fantine}.safetensors`.
- Python: venv at `/tmp/ptts-venv` with `ptts` (maturin develop --release).

## Reproduce
```
RAYON_NUM_THREADS=1 python experiments/calibrate.py          # text length -> duration
python experiments/run_deployment.py --name X --workers "a:0:1|b:2:1|c:4:1|d:6:1" --utts 2
python experiments/summarize.py                               # table + plot
bash experiments/gather.sh                                    # full battery + validation
```

Worker spec: `speaker[:label]:cpus:threads` joined by `|` (`a2` maps to voice `a`).
Default: one process per worker (own model, pinned). `--share-model` runs all workers
as threads in one process (RAYON_NUM_THREADS=1 implied).

## Key results (see notebook.md for full tables)
- 16 streams × 1 thread, q8_0, physical-core pinning: **18.5× realtime** (q4k: 20.7×).
- Never multi-thread a single stream (1×16t = 0.74×).
- q8 KV cache (committed) is speed-neutral but cuts per-state KV memory 3.8×.
- Prod server: `RAYON_NUM_THREADS=1`, pin each ws connection's thread to its own core.
