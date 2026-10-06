#!/bin/bash
# Re-run the two num_sms arms. Their first attempt collided with another user's job
# (pid 286713, cojoandrei03) that appeared on the box at ~01:44 and took ~17 GiB per
# GPU; A2 OOMed against it and A3 was correctly refused by the dirty-box guard.
# Neither result was about DeepEP, so both logs are discarded and re-measured.
set -uo pipefail
while pgrep -f '[s]weep_deepep.sh' > /dev/null || pgrep -f '[s]weep_deepep2.sh' > /dev/null; do sleep 30; done
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}
C=$HOME/assn1-f/configs

# Refuse to start if anyone else is on the GPUs; a shared box makes timings meaningless.
foreign () {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$(whoami)" ] && { echo "$u"; return 0; }
  done
  return 1
}

run () {
  name=$1; cfg=$2
  out=/mnt/scratch/chloe_wong/runs/$name
  rm -f "$out/run.log"
  for w in $(seq 1 60); do
    o=$(foreign) && { echo "== waiting: GPUs held by $o ($((w*60))s)"; sleep 60; continue; }
    break
  done
  if o=$(foreign); then echo "== $name SKIPPED: $o still on the box"; return; fi
  bash ~/assn1-f/analysis/gpu_cleanup.sh || { echo "== $name SKIPPED: GPUs dirty"; return; }
  mkdir -p "$out"; git rev-parse HEAD > "$out/git_commit.txt"; cp "$cfg" "$out/"
  echo "== $name start $(date +%H:%M:%S)"
  uv run examples/run_sft.py --config "$cfg" \
      logger.log_dir="$out/logs" logger.wandb_enabled=False \
      logger.tensorboard_enabled=True logger.monitor_gpus=True \
      checkpointing.enabled=False checkpointing.checkpoint_dir="$out/ckpt" \
      > "$out/run.log" 2>&1
  echo "   exit=$?  steps=$(grep -c 'Total step time' "$out/run.log")  end $(date +%H:%M:%S)"
}

run dep-A2-sms8  $C/dep-A2-sms8.yaml
run dep-A3-sms32 $C/dep-A3-sms32.yaml
bash ~/assn1-f/analysis/gpu_cleanup.sh
echo "== rerun done $(date +%H:%M:%S)"
