#!/bin/bash
# Final two DeepEP arms.
#
# The earlier guard refused to start if ANY foreign process was on the GPUs. That is right
# for a competing training job (A2, A4 and an earlier A1 all OOMed against one) but wrong
# for an idle CUDA context: mdong15 has held 932 MiB at 0% utilisation for 8+ minutes with
# GPUs 1-7 untouched, and waiting for that to vanish could block indefinitely.
#
# So the rule is now a memory threshold, not presence: block on >2 GiB of foreign memory,
# tolerate less. The foreign footprint at launch is recorded with each run, because this
# config peaks near 76 GiB of 79 GiB and a marginal OOM must be judged against how much
# someone else was holding rather than blamed on the setting under test.
set -uo pipefail
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}
C=$HOME/assn1-f/configs
ME=$(whoami)
LIMIT_MIB=2048

foreign_mib () {
  local t=0
  while read -r pid mem; do
    [ -z "$pid" ] && continue
    u=$(ps -o user= -p "${pid%,}" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && t=$((t + ${mem%% *}))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr ',' ' ')
  echo "$t"
}

run () {
  name=$1; cfg=$2
  out=/mnt/scratch/chloe_wong/runs/$name
  if [ -f "$out/run.log" ] && [ "$(grep -c 'Total step time' "$out/run.log")" -ge 30 ]; then
    echo "== $name already has 30 steps, skipping"; return
  fi
  rm -f "$out/run.log"
  for w in $(seq 1 180); do
    f=$(foreign_mib)
    [ "$f" -le "$LIMIT_MIB" ] && break
    [ $((w % 10)) -eq 1 ] && echo "== $name waiting: ${f} MiB foreign ($(date +%H:%M:%S))"
    sleep 60
  done
  f=$(foreign_mib)
  if [ "$f" -gt "$LIMIT_MIB" ]; then echo "== $name GAVE UP: ${f} MiB foreign"; return; fi
  bash ~/assn1-f/analysis/gpu_cleanup.sh >/dev/null 2>&1
  f=$(foreign_mib)
  mkdir -p "$out"; git rev-parse HEAD > "$out/git_commit.txt"; cp "$cfg" "$out/"
  echo "$f" > "$out/foreign_mib_at_start.txt"
  echo "== $name start $(date +%H:%M:%S)  (foreign ${f} MiB)"
  uv run examples/run_sft.py --config "$cfg" \
      logger.log_dir="$out/logs" logger.wandb_enabled=False \
      logger.tensorboard_enabled=True logger.monitor_gpus=True \
      checkpointing.enabled=False checkpointing.checkpoint_dir="$out/ckpt" \
      > "$out/run.log" 2>&1
  rc=$?; steps=$(grep -c 'Total step time' "$out/run.log")
  echo "   exit=$rc  steps=$steps  end $(date +%H:%M:%S)"
  if [ "$steps" -lt 30 ] && grep -q "out of memory" "$out/run.log"; then
    echo "   OOM (foreign at start: ${f} MiB): $(grep -o 'this process has [0-9.]* GiB memory in use' "$out/run.log" | sort -u | head -1)"
  fi
}

run dep-A6-balgate $C/dep-A6-balgate.yaml
run dep-A2-sms8    $C/dep-A2-sms8.yaml
echo "== all DeepEP arms done $(date +%H:%M:%S)"
