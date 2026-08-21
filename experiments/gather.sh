#!/bin/bash
# Long results-gathering run for the xn-ptts notebook.
# Each experiment has a timeout; the run continues after failures.
# Logs to experiments/gather.log
set -u
cd /repos/xn-ptts/experiments
LOG=gather.log
PY=/tmp/ptts-venv/bin/python
E16="a:0:1|b:1:1|c:2:1|d:3:1|a2:4:1|b2:5:1|c2:6:1|d2:7:1|a3:8:1|b3:9:1|c3:10:1|d3:11:1|a4:12:1|b4:13:1|c4:14:1|d4:15:1"
E8="a:0:1|b:1:1|c:2:1|d:3:1|a2:4:1|b2:5:1|c2:6:1|d2:7:1"

echo "=== gather start $(date) ===" | tee -a $LOG
run() { # name, timeout_s, args...
  local name=$1; shift
  local tmo=$1; shift
  echo "### $name $(date +%H:%M:%S)" | tee -a $LOG
  timeout $tmo $PY run_deployment.py --name "$name" "$@" 2>&1 | tee -a $LOG
  local rc=${PIPESTATUS[0]}
  echo "### $name rc=$rc $(date +%H:%M:%S)" | tee -a $LOG
  return 0
}

# 1. Repeat stability: 8 and 16 streams, q8_0
run E4r_8s1t 420 --workers "$E8" --utts 2
run E5r2_16s1t 420 --workers "$E16" --utts 2

# 2. Combined L3 strategy: q4k weights + q8 KV, 16 streams
run E5q4kkv_16s1t 500 --workers "$E16" --utts 2 --quant q4k

# 3. q4k repeat at 16 streams (best weight quant so far)
run E5q4kr_16s1t 500 --workers "$E16" --utts 2 --quant q4k

# 4. q8 KV at 4 streams (L3-resident regime)
run E2kv_4s1t 420 --workers "a:0:1|b:2:1|c:4:1|d:6:1" --utts 3

# 5. FINAL VALIDATION: full 16-min workload, best config
#    16 streams x 1 thread, q8_0, ~31s utterances -> 32 utterances, ~16 min audio
rm -rf wavs && mkdir -p wavs
run VALIDATION_16min 900 --workers "$E16" --utts 2 --target-chars 640 --wav-out wavs

# 6. Summarize + verify WAVs
$PY summarize.py 2>&1 | tee -a $LOG
$PY - <<'EOF' 2>&1 | tee -a $LOG
import sys, json, glob
sys.path.insert(0, ".")
from common import SAMPLE_RATE
import wave
d = json.load(open("results/VALIDATION_16min.json"))
g = d["global"]
print(f"VALIDATION: audio {g['total_audio_s']:.0f}s, gen {g['gen_wall_s']:.1f}s -> {g['gen_throughput']:.2f}x realtime")
files = sorted(glob.glob("wavs/*.wav"))
tot = 0.0; bad = []
for f in files:
    try:
        with wave.open(f) as w:
            dur = w.getnframes()/w.getframerate()
        tot += dur
        if dur < 20: bad.append((f, dur))
    except Exception as e:
        bad.append((f, str(e)))
print(f"WAVs: {len(files)} files, total {tot:.0f}s audio, bad: {bad}")
# per-speaker totals
import collections
by = collections.defaultdict(float)
for f in files:
    spk = f.split("/")[-1][:1]
    with wave.open(f) as w:
        by[spk] += w.getnframes()/w.getframerate()
for spk, s in sorted(by.items()):
    print(f"  speaker {spk}: {s:.0f}s")
EOF
echo "=== gather done $(date) ===" | tee -a $LOG
