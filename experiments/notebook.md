# EPYC 7302 TTS Throughput Experiments — Notebook

Goal: generate 16 min of audio (4 speakers × 4 min, ~30s utterances, order a,b,c,d,a,...)
as fast as possible on this EPYC 7302 (16 cores / 32 threads, single NUMA, 8 × 16MB L3
slices; core pairs (2k,2k+1) share L3 slice k; SMT siblings (c, c+16)).

Stack: Pocket-TTS (ptts) via pyo3 and ptts-ws-server, `xn` CPU backend (rayon global
pool; RAYON_NUM_THREADS per process). Model: kyutai/pocket-tts `tts_b6369a24` (236MB f32).
Voices: anna(a), charles(b), eve(c), fantine(d). Sample rate 24 kHz. Utterance text
≈ 600 chars → ~30s audio (calibrated). Machine is shared: a `ptts-ws-server` (240% CPU)
and other tenants run concurrently; numbers are reproducible to ±2% within a session.

## Hardware facts (measured)
- Weight sizes (f32 safetensors): flow_lm quantizable 151.0MB, mimi 55.8MB, other 28.9MB.
- q8_0 flow_lm ≈ 42.5MB → resident q8 model ≈ 127MB ≈ **exactly the 128MB aggregate L3**.
- KV cache per state (max_seq 2048, 6 layers, 16 heads, d=64, f32): ~100MB allocated,
  ~21MB active per 30s utterance.
- Single stream, q8_0, 1 thread: RTF 0.31–0.33 → weights (71MB flow_lm hot set) are
  L3-resident when alone.

## Key API constraints
- `generate_audio` is batch=1; GIL released during compute; decode runs on a sibling
  thread (overlapped with backbone).
- xn gemm/elementwise use the global rayon pool → threads-per-stream = RAYON_NUM_THREADS
  (per process). With RAYON_NUM_THREADS=1, compute runs on the calling thread → pinning works.
- **1 stream × 16 threads is catastrophic (RTF 1.35)**: matmuls too small for rayon
  fan-out. Stream count, not per-stream threads, is the lever.

## Metric
throughput = audio_seconds / wall_seconds (× realtime); gen_throughput excludes load/warmup.

---

## E0 — Serial baselines
Single thread: f32 RTF ≈ 0.95–1.0, q8_0 ≈ 0.63–0.77 (first calibration runs were
polluted by box load; clean re-measure: q8_0 lone stream RTF 0.31 on ~30s utterances).

## E1 — Weight sizing / L3 fit
| component | f32 MB | q8_0 MB |
|---|---|---|
| flow_lm transformer linear weights | 151.0 | ~42.5 |
| mimi decoder (unquantized) | 55.8 | 55.8 |
| other (norms, embeddings, ...) | 28.9 | 28.9 |
| total resident | 235.7 | ~127.2 |

## E2–E9 — Deployment matrix (all q8_0, 2–3 utterances per stream, seed 42)

| experiment | deployment | audio s | wall s | gen s | × realtime (wall) | × realtime (gen) |
|---|---|---|---|---|---|---|
| E2 4s1t slices | a:0 b:2 c:4 d:6 (4 procs) | 332 | 70.2 | ~57 | 4.72 | 5.82 |
| E3 4s1t adjacent | a:0 b:1 c:2 d:3 | 332 | 53.9 | 47.6 | 6.15 | 6.96 |
| E2b 4s1t shared | 4 threads, 1 model | 332 | 55.4 | 48.8 | 5.98 | 6.79 |
| E6 4s2t SMT | a:0,16 b:2,18 c:4,20 d:6,22 | 224 | 36.1 | 30.3 | 6.22 | 7.41 |
| E6b 4s2t 2core | a:0,1 b:2,3 c:4,5 d:6,7 | 224 | 36.1 | 30.1 | 6.22 | 7.46 |
| E7 4s4t | 16 cores, 4 threads/stream | 224 | 42.1 | 35.3 | 5.34 | 6.36 |
| E8 2s8t | 2 streams × 8 threads | 111 | 52.1 | 44.4 | 2.14 | 2.51 |
| E9 1s16t | 1 stream × 16 threads | 55 | 88.1 | 74.4 | 0.62 | 0.74 |
| E4 8s1t adjacent | 8 streams, cores 0–7 | 454 | 42.1 | 35.7 | 10.79 | 12.71 |
| E4b 8s1t spread | 8 streams, 1 per slice | 454 | ~36 | 33.0 | — | 13.75 |
| E10 12s1t | 12 streams, cores 0–11 | 676 | 48.1 | 40.4 | 14.06 | 16.74 |
| E5 16s1t | **16 streams, all physical cores** | 894 | 58.1 | 48.4 | 15.38 | **18.46** |
| E5r 16s1t (repeat) | same | 894 | 56.1 | 47.6 | 15.93 | 18.76 |
| E5b 16s SMT-spread | 8 phys + 8 SMT | 894 | — | 58.9 | — | 15.17 |
| E5c 16s shared | 16 threads, 1 model | 894 | 91.6 | 73.1 | 9.76 | 12.23 |
| E5f32 16s1t | f32 weights | 902 | 100.2 | 86.9 | 9.00 | 10.38 |
| E5q4k 16s1t | q4k weights | 910 | 54.1 | 44.0 | 16.81 | **20.66** |

### Findings
1. **Stream count is the lever, not per-stream threads.** 16 streams × 1 thread = 18.5×;
   1 stream × 16 threads = 0.74×. Multi-threaded gemm on these small matmuls loses.
2. **Pinning: one stream per physical core, spread over L3 slices.** Spread beats
   adjacent at 8 streams (13.75 vs 12.71). SMT siblings hurt (15.17 < 18.46).
3. **Process-per-stream beats threads-shared-model at ≥8 streams** (12.23 vs 18.46);
   shared model wins at 4 streams (6.79 vs 5.82) — allocator/paging contention inverts it.
4. **Memory traffic dominates**: f32→q8_0 = +78% at 16 streams (10.38→18.46); q4k = +12%
   more (20.66). Per-stream RTF degrades 0.31 (alone) → 0.82 (16 streams): L3 capacity
   (weights 71MB + 16 × 21MB KV ≈ 407MB vs 128MB) and DRAM latency.
5. Expected serial time for 960s of audio: q8_0 ≈ 660s → best config ≈ 52s.

## E12 — Quantized (q8_0) KV cache [Rust change, committed b4a1ff5]
Hypothesis: KV is the L3-killer (21MB/stream); q8 KV (5.6MB) + weights 71MB could fit
128MB → restore weight residency at high stream counts.

Result: **neutral to −3%** (E5kv 18.15×, E4kv 12.42×, E10kv 16.28× vs f32 KV 18.46/12.71/
16.74). Per-step dequant→f32 materialization (2×3.5MB per layer) churns L3 as much as
the f32 KV did; only a fused q8 matmul would recover the win. Benefit kept: KV memory
per state 100MB → 26MB (2048 cap), clone cost ÷4, quality unchanged (RMS/duration
identical, determinism preserved). No audible/statistical regression vs f32 KV.

## ERRATUM (pinning bug) + corrected multi-threaded streams
The deployment runner originally pinned only the FIRST cpu of each worker
(`pin_current(cpus[0])`), so all rayon threads of a multi-threaded stream landed
on ONE core. E6–E9 therefore measured thread contention, not spread
multi-threading. Fixed: pin the process to the full cpu list. Re-measured:

| experiment | deployment (spread) | × realtime (gen) |
|---|---|---|
| E9x 1s16t | 16 cores, 1 stream | 4.16 |
| E8x 2s8t | 8 cores/stream ×2 | 8.22 |
| E7x 4s4t | 4 cores/stream ×4 | 13.17 |
| E6b 4s2t (old data) | 2 cores/stream ×4 | 7.46 |

Conclusion unchanged, margins smaller: independent single-threaded streams beat
multi-threaded streams at equal core counts (16 cores: 16×1t 18.5× > 4×4t 13.2× >
1×16t 4.2×; 8 cores: 8×1t 12.7× > 2×8t 8.2×). One stream × 16 spread threads ≈
same per-stream RTF as 1 thread alone (0.33) but only 4.2× aggregate.

## Server A/B + production swap (2026-08-20)
The live deployment (`deploy/docker-compose.yml` `pocket-tts` service) ran the
old image: f32, RAYON_NUM_THREADS unset (32-thread rayon per request), no cpuset.
Replacement (this repo's `ptts-ws-server/Dockerfile` + updated deploy compose):
q8_0 weights, RAYON_NUM_THREADS=1 (each ws connection = own single-threaded
stream), `cpuset 0-15` (physical cores), q8 KV cache included (KV memory /3.8).

Interleaved A/B (4 voices × 2 utterances, cpuset 0-15 both, reproducible r0/r1):

| config | × realtime |
|---|---|
| OLD (f32, 32-thread rayon) | 7.43 / 7.45 |
| NEW (q8_0, 1-thread streams) | 10.04 / 10.08 |

NEW = +35%. Verified end-to-end after swap via ws probe on the paperradio-backend
network: `ready sr=24000 frame=1920`, 17.8s audio, RTF 0.37 under load; no TTS
errors in radiod logs.

## Post-processing experiments (2026-08-20)
Goal: volume-normalized speech, minimal noise/glitches, max speed. Implemented
in `ptts::postprocess` (spectral-gate denoiser, EBU R128 normalization to
`--target-lufs`, soft limiter), wired into ptts-ws-server as `--postprocess`
(per-utterance, buffered at flush time; streaming protocol unchanged).

CPU cost (server logs): **10-16 ms per ~14s utterance** — <0.1% of one core,
no measurable RTF change (raw vs pp wall identical within noise).

Quality (5 voices × 2 texts; `experiments/quality/metrics.json`):

| metric | raw | postprocessed |
|---|---|---|
| integrated LUFS | -19.2 … -24.4 (voice-dependent) | **-18.1 ± 0.0** (target -18) |
| noise floor (quiet-frame RMS) | -61.8 … -83.0 dB | -60.6 … -97.0 dB (better for 8/10) |
| glitches (\|Δ\|>0.9) | 0 | 0 |
| clipping | 0 | 0 |
| true peak | up to 0.74 | ≤ 0.95 (limiter knee) |

First implementation had an unnormalized inverse-FFT bug (glitches 25k/utt,
noise +40dB) — caught by the harness, fixed (1/N scaling), re-verified.

## Voice quality matrix: xn-ptts vs pip pocket-tts (2026-08-20)
Finding: pip pocket-tts 2.1.0 built-in voices are **precomputed flow_lm states**
(`transformer.layers.N.self_attn/cache` [2,b,seq,h,d] + offset), not
audio-prompt embeddings — a different conditioning representation than the v1
embeddings the server used. Implemented state-voice import in ptts + ws-server
(`init_flow_lm_state_from_kv`, q8-quantized on import) + 2026-04 model support
(`mimi.inner_dim=32`, `flow_lm.bos_before_voice` ignored/prepended, mimi
`speaker_proj` to inner dim).

Same text, 5 voices (alba/anna/charles/eve/fantine), mean per source:

| source | LUFS (σ) | noise floor dB | hiss 8-12k | glitches |
|---|---|---|---|---|
| pip english (2026-04) | -20.1 (1.11) | -74.0 | **0.149** | 0 |
| pip english_2026-01 | -20.8 (0.52) | **-75.8** | 0.180 | 0 |
| xn 2026-01 state voices | -21.0 (0.43) | -73.1 | 0.205 | 0 |
| xn 2026-04 state voices | -20.6 (1.58) | -72.0 | 0.186 | 0 |
| xn v1 embeddings (old prod) | -21.0 (1.72) | -73.3 | 0.209 | 0 |
| **xn 2026-04 + postprocess** | **-18.1 (0.00)** | **-74.1** | 0.217 | 0 |

Conclusions:
1. xn-ptts now reproduces pip's voice conditioning exactly (same model, same
   state files) — the "pip voices are best" difference is the state-style
   2026-01/2026-04 voices + 2026-04 model, all usable in the Rust server.
2. Postprocess = exact loudness (-18 LUFS, zero spread), slightly cleaner
   noise floor, zero glitches/clipping, ~0 cost.
3. Remaining xn-vs-pip deltas (hiss 0.19 vs 0.15) come from q8_0 weights and
   temperature/chunking, not the voices.
4. Spectrograms in `experiments/quality/spec_alba_compare.png` (v1 vs 2026-04
   vs 2026-04+pp) for eyeball confirmation.

Prod: deploy/models swapped to 2026-04 model + state voices; compose adds
`--postprocess`. **Swapped in production 2026-08-20 12:50** — log confirms
`precomputed state` sessions (eponine/alba requests) + `postprocess
ms=10-17` per utterance; end-to-end probe via paperradio-backend OK.

## Post-process removal + token chunking (2026-08-21)
Production listening found the post-process pipeline harmful and removed:
- **Loudness normalization**: overboosted quiet material; clipped sibilants
  (peak pinned at limiter ceiling 1.0).
- **Denoising**: unnecessary with the 2026-04 state voices.
- **Limiter**: existed only to cap the boost; gone with it.
`ptts::postprocess` deleted (git history keeps it); ws-server flags removed.

Long-text chunking added instead: texts > `--max-tokens-per-chunk` (default
150) are split into sentence-aligned chunks (mirroring pip; oversized
sentences sub-split on `,;:`). A/B on a 433-token complex paragraph:

| config | duration | noise floor | glitches | peak |
|---|---|---|---|---|
| old (no chunk + postprocess) | 146.4s | **-40.5 dB** | 1 | 1.00 (limiter) |
| new (chunked, raw) | **78.8s** | **-61.6 dB** | 0 | 0.85 |

Verified in prod through the backend network (78.8s clean; zero postprocess
log lines).

## Roadmap
- **Speech post-processing (requested):** volume normalization + denoising on all
  speech outputs to prevent loud/quiet and noise pumping between utterances.
  - Loudness: `ptts::utils::normalize_loudness` already exists (EBU R128 via the
    `ebur128` dep; used by the pocket_tts example). Apply per-utterance to a
    target LUFS (e.g. -16) with a limiter; verify with ebur128 LUFS measurement.
  - Denoising: options — (a) spectral noise gate/subtraction (new `realfft`-based
    stage in the ws encoder or radiod pipeline), (b) RNNoise via ffmpeg/`audiopus`
    pipeline, (c) a small trained denoiser (heavy). Recommend (a) offline in
    radiod after synthesis, or server-side pre-encode.
  - Placement decision: encoder stage in ptts-ws-server (benefits all clients) vs
    radiod post-processing (keeps TTS server raw); measure quality + CPU cost.
## Server (prod) implications — ptts-ws-server
- Each ws connection runs `spawn_blocking` + rayon over ALL cores by default → a single
  generation uses 32 threads → terrible (E9). Set `RAYON_NUM_THREADS=1` (or N streams
  expected) and let concurrent connections each run single-threaded.
- 4 voices / 4 connections → 4 streams: pin connections' threads to distinct physical
  cores, one per L3 slice (0,2,4,6). Use `--quant q8_0` (or q4k for +12%).
- Batch-of-16-streams-style scaling (18×+) requires 16 concurrent utterances — e.g.
  pre-generating a playlist with 16 workers pinned to cores 0–15.
- Done 2026-08-20: production `pocket-tts` service rebuilt with the optimized
  image (see "Server A/B + production swap" above); measured +35%.

## Final validation runs (gather.sh, 2026-08-20)
- VALIDATION (640c utts): 923s audio / 32 WAVs, gen 56.9s → 16.2×; per speaker a/b/c/d =
  238/224/226/235s.
- VALIDATION2 (700c utts): **955s audio (15.9 min) / 32 WAVs, gen 55.5s → 17.2×**;
  per speaker a/b/c/d = 248/235/234/238s (all ≥ 3.9 min). Wall incl. load+warmup+WAV
  write: 116s. All WAVs valid (no NaN, RMS ~0.12, duration ≥ 20s).
- Repeats: E4r 8s 12.53×, E5r2 16s 17.81× (box load fluctuates ±5%), E5q4kr 19.82×,
  E5q4kkv (q4k + q8 KV) 20.06×, E2kv 4s 7.38×.

## Conclusion
Best strategy on this EPYC 7302 for the 4-voice / 16-min workload:

1. **q4k or q8_0 weights** (q4k +12% over q8_0 at 16 streams; quality impact not
   audited here beyond RMS/duration).
2. **16 concurrent single-threaded streams, one per physical core** (SMT siblings
   unused; 16 procs or 16 pinned threads at RAYON_NUM_THREADS=1). Never give a stream
   more threads — 1×16t measured 0.74×.
3. **Spread pinning across L3 slices** (cores 0–15 covers all 8 slices × 2).
4. q8 KV cache (committed) is memory-footprint win, speed-neutral: keep for server
   memory, not throughput.
5. **Server**: ptts-ws-server with RAYON_NUM_THREADS=1; each concurrent connection
   (voice) pins its own physical core — 4 voices → 4 streams on 4 slices ≈ 7× realtime;
   16-way pre-generation → ~17–20×.

Deliverable: 16 min of audio generated in **~56s of compute (17.2× realtime)** with
utterance order per speaker preserved (a1..a8, b1..b8, ... interleaved at assembly time).

