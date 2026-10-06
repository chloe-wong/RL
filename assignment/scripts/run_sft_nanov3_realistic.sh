#!/bin/bash
# Realistic-workload SFT run: Nemotron-Nano-V3 on Nemotron-Cascade-2 chat, 4k context,
# GBS 128, same backend as the baseline (DTensor V2, FSDP2 + EP8, 1 node x 8 H100).
# Config: $CONFIG (default: ~/assn1-f/configs/sft-nanov3-cascade2chat-4k-1n8g.yaml)
#
# Usage: bash ~/assn1-f/run_sft_nanov3_realistic.sh [extra Hydra overrides...]
# Example: bash ~/assn1-f/run_sft_nanov3_realistic.sh sft.max_num_steps=50
# Example: CONFIG=~/assn1-f/configs/sft-nanov3-cascade2chat-8k-chunkedce-1n8g.yaml \
#          bash ~/assn1-f/run_sft_nanov3_realistic.sh
set -euo pipefail
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
CONFIG=${CONFIG:-$HOME/assn1-f/configs/sft-nanov3-cascade2chat-4k-1n8g.yaml}

EXP_NAME=${EXP_NAME:-sft-nanov3-cascade2chat4k-$(date +%Y%m%d-%H%M%S)}
OUT=/mnt/scratch/chloe_wong/runs/$EXP_NAME
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
cp "$CONFIG" ~/assn1-f/configs/sft-nanov3-cascade2chat-4k-1n8g.yaml "$OUT/"

uv run examples/run_sft.py \
    --config "$CONFIG" \
    logger.log_dir="$OUT/logs" \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    logger.monitor_gpus=True \
    checkpointing.checkpoint_dir="$OUT/ckpt" \
    "$@" \
    2>&1 | tee "$OUT/run.log"
