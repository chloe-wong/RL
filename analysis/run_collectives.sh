#!/bin/bash
# Runner for the collectives lane. Same shape as run_sft_nanov3_baseline.sh, but the
# config is selectable so every arm is measured against sft-nanov3-collectives.yaml
# (the run-F settings) rather than against the untuned upstream recipe.
#
# Usage: EXP_NAME=<name> bash run_collectives.sh [hydra overrides...]
set -euo pipefail
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
# The shared inductor cache has root-owned entries left by the sudo profiling modes;
# keep compilation in a directory this user owns.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}

CONFIG=${CONFIG:-examples/configs/recipes/llm/sft-nanov3-collectives.yaml}
EXP_NAME=${EXP_NAME:-collectives-$(date +%Y%m%d-%H%M%S)}
OUT=/mnt/scratch/chloe_wong/runs/$EXP_NAME
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
git diff > "$OUT/git_diff.patch"

uv run examples/run_sft.py \
    --config "$CONFIG" \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=8 \
    sft.max_num_steps=50 \
    sft.val_at_start=false \
    sft.val_period=1000 \
    logger.log_dir="$OUT/logs" \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    logger.monitor_gpus=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir="$OUT/ckpt" \
    "$@" \
    2>&1 | tee "$OUT/run.log"
