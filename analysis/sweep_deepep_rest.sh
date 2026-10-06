#!/bin/bash
# Remaining DeepEP arms, consolidated.
#
# The box is shared with cojoandrei03 and palshraya, who both took all 8 GPUs during the
# first pass. A2, A3 and A4 each OOMed against a foreign process, which says nothing about
# DeepEP, so their logs are discarded and re-measured. This runner waits for the box to be
# free and never launches alongside someone else's job -- a timing measured under
# contention is not a measurement, and competing for a teammate's GPUs is not acceptable.
set -uo pipefail
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}
C=$HOME/assn1-f/configs
ME=$(whoami)

foreign () {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && { echo "$u"; return 0; }
  done
  return 1
}

run () {
  name=$1; cfg=$2
  out=/mnt/scratch/chloe_wong/runs/$name
  if [ -f "$out/run.log" ] && [ "$(grep -c 'Total step time' "$out/run.log")" -ge 30 ]; then
    echo "== $name already has 30 steps, skipping"; return
  fi
  rm -f "$out/run.log"
  # Wait up to 4 hours for the box to free up.
  for w in $(seq 1 240); do
    o=$(foreign) || break
    [ $((w % 10)) -eq 1 ] && echo "== $name waiting: GPUs held by $o ($(date +%H:%M:%S))"
    sleep 60
  done
  if o=$(foreign); then echo "== $name GAVE UP: $o still on the box"; return; fi
  bash ~/assn1-f/analysis/gpu_cleanup.sh || { echo "== $name SKIPPED: GPUs dirty"; return; }
  if o=$(foreign); then echo "== $name SKIPPED: $o arrived during cleanup"; return; fi
  mkdir -p "$out"; git rev-parse HEAD > "$out/git_commit.txt"; cp "$cfg" "$out/"
  echo "== $name start $(date +%H:%M:%S)"
  uv run examples/run_sft.py --config "$cfg" \
      logger.log_dir="$out/logs" logger.wandb_enabled=False \
      logger.tensorboard_enabled=True logger.monitor_gpus=True \
      checkpointing.enabled=False checkpointing.checkpoint_dir="$out/ckpt" \
      > "$out/run.log" 2>&1
  rc=$?; steps=$(grep -c 'Total step time' "$out/run.log")
  echo "   exit=$rc  steps=$steps  end $(date +%H:%M:%S)"
  # A run that OOMed against someone else's process is contamination, not a result.
  if [ "$steps" -eq 0 ] && grep -q "out of memory" "$out/run.log"; then
    echo "   OOM: $(grep -o 'Process [0-9]* has [0-9.]* GiB memory in use' "$out/run.log" | sort -u | tr '\n' ' ')"
  fi
}

run dep-A2-sms8      $C/dep-A2-sms8.yaml
run dep-A3-sms32     $C/dep-A3-sms32.yaml
run dep-A4-torchdisp $C/dep-A4-torchdisp.yaml
run dep-A5-gmm       $C/dep-A5-gmm.yaml
run dep-A6-balgate   $C/dep-A6-balgate.yaml
bash ~/assn1-f/analysis/gpu_cleanup.sh
echo "== remaining arms done $(date +%H:%M:%S)"
