#!/usr/bin/env python3
"""Summarize experiment results: JSON -> markdown table + throughput plot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR

EXPERIMENT_ORDER = [
    "E2_4s1t_slices", "E3_4s1t_adjacent", "E2b_4s1t_shared", "E4_8s1t", "E5_16s1t",
    "E6_4s2t_smt", "E6b_4s2t_2cores", "E7_4s4t", "E8_2s8t", "E9_1s16t",
]


def load(name: str) -> dict | None:
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def summarize() -> str:
    rows = []
    for name in EXPERIMENT_ORDER:
        d = load(name)
        if d is None:
            continue
        g = d["global"]
        m = d["meta"]
        rows.append({
            "name": name,
            "workers": m["workers"],
            "share": m["share_model"],
            "utts": g["n_utterances"],
            "audio_s": g["total_audio_s"],
            "wall_s": g["wall_s"],
            "gen_wall_s": g.get("gen_wall_s", g["wall_s"]),
            "throughput": g["throughput"],
            "gen_throughput": g.get("gen_throughput", g["throughput"]),
            "eff_rtf": g["effective_rtf"],
        })
    rows.sort(key=lambda r: -r["gen_throughput"])
    out = ["| experiment | deployment | audio s | wall s | gen s | x realtime (wall) | x realtime (gen) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['workers']} | {r['audio_s']:.0f} | {r['wall_s']:.1f} | "
                   f"{r['gen_wall_s']:.1f} | {r['throughput']:.2f} | {r['gen_throughput']:.2f} |")
    return "\n".join(out)


def plot() -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "(matplotlib not installed; skipping plot)"
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, vals = [], []
    for name in EXPERIMENT_ORDER:
        d = load(name)
        if d is None:
            continue
        labels.append(name)
        vals.append(d["global"].get("gen_throughput", d["global"]["throughput"]))
    order = np.argsort(vals)[::-1]
    ax.bar([labels[i] for i in order], [vals[i] for i in order], color="#4c72b0")
    ax.set_ylabel("generation throughput (× realtime)")
    ax.set_title("EPYC 7302 — Pocket-TTS 4-voice throughput by deployment")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = Path(__file__).parent / "results" / "summary.png"
    fig.savefig(p, dpi=110)
    return f"saved {p}"


if __name__ == "__main__":
    print(summarize())
    print(plot())
