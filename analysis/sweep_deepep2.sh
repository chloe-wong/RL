#!/bin/bash
# Follow-on arm: waits for the main DeepEP sweep to finish, then runs the balanced-gate
# diagnostic. Kept separate so it cannot disturb a sweep already in flight.
set -uo pipefail
while pgrep -f '[s]weep_deepep.sh' > /dev/null; do sleep 30; done
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}
name=dep-A6-balgate
cfg=$HOME/assn1-f/configs/dep-A6-balgate.yaml
out=/mnt/scratch/chloe_wong/runs/$name
bash ~/assn1-f/analysis/gpu_cleanup.sh || { echo "GPUs dirty, refusing"; exit 1; }
mkdir -p "$out"; git rev-parse HEAD > "$out/git_commit.txt"; cp "$cfg" "$out/"
echo "== $name start $(date +%H:%M:%S)"
uv run examples/run_sft.py --config "$cfg" \
    logger.log_dir="$out/logs" logger.wandb_enabled=False \
    logger.tensorboard_enabled=True logger.monitor_gpus=True \
    checkpointing.enabled=False checkpointing.checkpoint_dir="$out/ckpt" \
    > "$out/run.log" 2>&1
echo "   exit=$?  steps=$(grep -c 'Total step time' "$out/run.log")  end $(date +%H:%M:%S)"
bash ~/assn1-f/analysis/gpu_cleanup.sh
echo "== A6 done"
