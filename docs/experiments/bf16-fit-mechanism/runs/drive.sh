#!/usr/bin/env bash
# Driver: 1s samplers + capped probe. Caps live in probe.py; never loosened here.
set -u
OUT=/tmp/bf16-fit-probe

# GPU sampler (1s)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -l 1 \
  > "$OUT/gpu.log" 2>&1 &
GPU_PID=$!

# Host mem + swap sampler (1s)
(
  while true; do
    ts=$(date +%s)
    ma=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    st=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
    sf=$(grep SwapFree /proc/meminfo | awk '{print $2}')
    echo "$ts MemAvailable_kB=$ma SwapUsed_kB=$((st - sf))"
    sleep 1
  done
) > "$OUT/mem.log" 2>&1 &
MEM_PID=$!

cleanup() { kill "$GPU_PID" "$MEM_PID" 2>/dev/null; }
trap cleanup EXIT

source /srv/dev/shanevcantwell/ComfyUI-DiffusionGemma/.venv/bin/activate
python "$OUT/probe.py" 2>&1 | tee "$OUT/probe.out"
rc=${PIPESTATUS[0]}
echo "probe exit code: $rc"
exit "$rc"
