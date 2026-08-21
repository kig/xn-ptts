#!/usr/bin/env python3
"""Deployment runner: run K generation streams across a core/topology plan.

Worker spec grammar:  "speaker:cpu1,cpu2,...:threads"  joined by '|'
  e.g. "a:0,1:2|b:2,3:2|c:4,5:2|d:6,7:2"

Two execution modes:
  --share-model : all workers are threads in ONE process sharing one model
                  (RAYON_NUM_THREADS must be 1 in this mode)
  default       : each worker is its own process with its own model copy;
                  `threads` sets that process's RAYON_NUM_THREADS and the
                  process is pinned to the listed CPUs.

Every worker generates its speaker's utterances serially (fresh state clone
per utterance, like the real a,b,c,d pipeline). Timings are per utterance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    VOICES,
    SAMPLE_RATE,
    build_utterances,
    load_model,
    measure_rtf,
    pin_current,
    save_result,
    write_wav,
)


def worker_job(spec: dict, result_path: Path) -> None:
    """Process worker: pin, load model, generate, write JSON result file."""
    os.environ["RAYON_NUM_THREADS"] = str(spec["threads"])
    # Pin the whole process (all future threads incl. rayon workers) to the
    # full cpu list, not just the first cpu: rayon workers inherit the
    # affinity of the thread that spawns them.
    os.sched_setaffinity(0, set(spec["cpus"]))
    import ptts
    import numpy as np
    model = load_model(spec["quant"])
    state = model.get_state_for_voice(spec["speaker"])
    # warmup
    w = measure_rtf(state, build_utterances(1, 200, seed=1)[0])
    rows = []
    for i, text in enumerate(spec["texts"]):
        r = measure_rtf(state, text, seed=spec["seed"])
        r["text_chars"] = len(text)
        rows.append(r)
        if spec.get("wav_out"):
            pcm = state.generate_audio(text, temperature=0.5, seed=spec["seed"])
            write_wav(Path(spec["wav_out"]) / f"{spec['label']}_{i:02d}.wav",
                      np.asarray(pcm, dtype=np.float32))
    payload = {"speaker": spec["speaker"], "label": spec["label"], "cpus": spec["cpus"],
               "threads": spec["threads"], "rows": rows, "warmup_rtf": w["rtf"]}
    result_path.write_text(json.dumps(payload, default=float))


def thread_worker_job(spec: dict, model, results: list, idx: int) -> None:
    """Thread worker sharing one model (RAYON_NUM_THREADS=1 assumed)."""
    state = model.get_state_for_voice(spec["speaker"])
    pin_current(spec["cpus"][0])
    measure_rtf(state, build_utterances(1, 200, seed=1)[0])  # warmup
    rows = []
    for text in spec["texts"]:
        r = measure_rtf(state, text, seed=spec["seed"])
        r["text_chars"] = len(text)
        rows.append(r)
    results[idx] = {"speaker": spec["speaker"], "cpus": spec["cpus"], "threads": 1,
                    "rows": rows, "warmup_rtf": None}


def parse_workers(specs: str, utts_per_worker: int, target_chars: int, seed: int, quant: str) -> list[dict]:
    workers = []
    for i, spec in enumerate(specs.split("|")):
        speaker, cpus, threads = spec.split(":")
        cpus = [int(c) for c in cpus.split(",")]
        voice = speaker.rstrip("0123456789")  # "a2" -> voice "a"
        workers.append({
            "id": i, "speaker": voice, "label": speaker, "cpus": cpus, "threads": int(threads),
            "texts": build_utterances(utts_per_worker, target_chars, seed=seed + i),
            "seed": seed, "quant": quant,
        })
    return workers


def run(args) -> dict:
    workers = parse_workers(args.workers, args.utts, args.target_chars, args.seed, args.quant)
    if args.wav_out:
        Path(args.wav_out).mkdir(parents=True, exist_ok=True)
        for w in workers:
            w["wav_out"] = args.wav_out
    t_start = time.perf_counter()
    results = []
    if args.share_model:
        assert all(w["threads"] == 1 for w in workers), "share-model implies 1 thread per worker"
        import threading
        model = load_model(args.quant)
        slots = [None] * len(workers)
        threads = [
            threading.Thread(target=thread_worker_job, args=(w, model, slots, i))
            for i, w in enumerate(workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = slots
    else:
        import multiprocessing as mp
        import tempfile
        ctx = mp.get_context("fork")
        tmpdir = Path(tempfile.mkdtemp(prefix=f"dep_{args.name}_"))
        result_paths = [tmpdir / f"w{i}.json" for i in range(len(workers))]
        procs = [ctx.Process(target=worker_job, args=(w, result_paths[i]))
                 for i, w in enumerate(workers)]
        for p in procs:
            p.start()
        deadline = time.perf_counter() + 1800
        while len(results) < len(workers):
            if time.perf_counter() > deadline:
                alive = sum(1 for p in procs if p.is_alive())
                raise RuntimeError(f"worker timeout: {len(workers)-len(results)} missing, {alive} alive")
            time.sleep(2)
            for i, p in enumerate(procs):
                if not p.is_alive() and not result_paths[i].exists() and p.exitcode != 0:
                    raise RuntimeError(f"worker {i} ({workers[i]['label']}) died, exitcode {p.exitcode}")
            results = [json.loads(p.read_text()) for p in result_paths if p.exists()]
        for p in procs:
            p.join(timeout=10)
    wall = time.perf_counter() - t_start

    total_audio = sum(r["audio_s"] for w in results for r in w["rows"])
    gen_wall = max(sum(r["wall_s"] for r in w["rows"]) for w in results)
    agg = {
        "meta": {
            "name": args.name, "quant": args.quant, "workers": args.workers,
            "share_model": args.share_model, "utts_per_worker": args.utts,
            "target_chars": args.target_chars, "seed": args.seed, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "workers": results,
        "global": {
            "wall_s": wall,
            "gen_wall_s": gen_wall,
            "total_audio_s": total_audio,
            "throughput": total_audio / wall,          # audio seconds per wall second
            "gen_throughput": total_audio / gen_wall,
            "effective_rtf": wall / total_audio,        # wall per audio second
            "n_utterances": sum(len(w["rows"]) for w in results),
        },
    }
    p = save_result(args.name, agg)
    print(f"\n=== {args.name}: wall {wall:.1f}s, gen {gen_wall:.1f}s, audio {total_audio:.0f}s, "
          f"throughput {total_audio/wall:.2f}x realtime (gen {total_audio/gen_wall:.2f}x, rtf {wall/total_audio:.3f}) ===")
    for w in results:
        rtfs = [r["rtf"] for r in w["rows"]]
        print(f"  {w['speaker']} cpus={w['cpus']} thr={w['threads']}: "
              f"{len(w['rows'])} utts, audio {sum(r['audio_s'] for r in w['rows']):.0f}s, "
              f"wall {sum(r['wall_s'] for r in w['rows']):.1f}s, rtf med {np.median(rtfs):.3f}")
    print("saved", p)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", required=True, help="speaker:cpus:threads|...")
    ap.add_argument("--utts", type=int, default=8)
    ap.add_argument("--target-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quant", default="q8_0")
    ap.add_argument("--share-model", action="store_true")
    ap.add_argument("--wav-out", default=None, help="dir to write concatenated WAVs")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
