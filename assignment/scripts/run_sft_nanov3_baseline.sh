#!/bin/bash
# Baseline SFT run: Nemotron-3-Nano-30B-A3B (Nano V3), NeMo RL DTensor V2 (Automodel)
# backend, FSDP2 + EP8 on 1 node x 8 H100. Recipe is the upstream
# sft-nanov3-30BA3B-2n8g-fsdp2.yaml, run at 1 node as in its test-suite script.
#
# Usage: bash ~/assn1-f/run_sft_nanov3_baseline.sh [extra Hydra overrides...]
# Example: bash ~/assn1-f/run_sft_nanov3_baseline.sh sft.max_num_steps=50
set -euo pipefail
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL

EXP_NAME=${EXP_NAME:-sft-nanov3-baseline-$(date +%Y%m%d-%H%M%S)}
OUT=/mnt/scratch/chloe_wong/runs/$EXP_NAME
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"

uv run examples/run_sft.py \
    --config examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=8 \
    sft.max_num_steps=20 \
    logger.log_dir="$OUT/logs" \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    logger.monitor_gpus=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir="$OUT/ckpt" \
    "$@" \
    2>&1 | tee "$OUT/run.log"
