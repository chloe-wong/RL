#!/bin/bash
# A2 (dispatcher_num_sms=8) is the one arm with no valid measurement: its single attempt
# OOMed against another user's job, which says nothing about DeepEP. Retry it once the
# main queue drains and the box is genuinely free.
set -uo pipefail
while pgrep -f '[s]weep_deepep_rest.sh' > /dev/null; do sleep 30; done
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}
ME=$(whoami)
foreign () {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && { echo "$u"; return 0; }
  done
  return 1
}
out=/mnt/scratch/chloe_wong/runs/dep-A2-sms8
rm -f "$out/run.log"
for w in $(seq 1 120); do
  o=$(foreign) || break
  [ $((w % 10)) -eq 1 ] && echo "waiting: GPUs held by $o ($(date +%H:%M:%S))"
  sleep 60
done
if o=$(foreign); then echo "GAVE UP: $o still on the box"; exit 1; fi
bash ~/assn1-f/analysis/gpu_cleanup.sh || { echo "GPUs dirty"; exit 1; }
mkdir -p "$out"; git rev-parse HEAD > "$out/git_commit.txt"
cp $HOME/assn1-f/configs/dep-A2-sms8.yaml "$out/"
echo "== dep-A2-sms8 retry start $(date +%H:%M:%S)"
uv run examples/run_sft.py --config $HOME/assn1-f/configs/dep-A2-sms8.yaml \
    logger.log_dir="$out/logs" logger.wandb_enabled=False \
    logger.tensorboard_enabled=True logger.monitor_gpus=True \
    checkpointing.enabled=False checkpointing.checkpoint_dir="$out/ckpt" \
    > "$out/run.log" 2>&1
echo "   exit=$?  steps=$(grep -c 'Total step time' "$out/run.log")  end $(date +%H:%M:%S)"
bash ~/assn1-f/analysis/gpu_cleanup.sh
echo "== A2 retry done"
