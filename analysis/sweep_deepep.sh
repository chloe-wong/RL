#!/bin/bash
# DeepEP sweep on Track B (Cascade-2 chat, 8k), on top of every optimization already
# verified there: chunked CE, bf16 FSDP output, expandable_segments, no per-step
# empty_cache, fused AdamW.
#
# DeepEP itself has always been on -- BackendConfig.dispatcher defaults to "deepep".
# What has never been set is any of its knobs: the base config passes the removed
# `enable_deepep` key, so dispatcher_num_sms sat at 20 and dispatcher_async_dispatch
# at False for every Track B run. That is why measured DeepEP/compute overlap is 0.5%.
#
# A0 is the base config unchanged: the noise floor. No arm counts unless it beats the
# A0-vs-optim-O1 spread.
#
# Ray does not always release GPU memory when a run exits, so the first version of this
# script launched arm N+1 on top of arm N's workers and A1 OOMed with two processes on
# the device. Every arm now tears down and waits for the GPUs to actually drain first.
set -uo pipefail
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/scratch/chloe_wong/tmp/inductor-clean}
C=$HOME/assn1-f/configs

run () {
  name=$1; cfg=$2
  out=/mnt/scratch/chloe_wong/runs/$name
  if [ -f "$out/run.log" ] && [ "$(grep -c 'Total step time' "$out/run.log")" -ge 30 ]; then
    echo "== $name already has 30 steps, skipping"; return
  fi
  if ! bash ~/assn1-f/analysis/gpu_cleanup.sh; then
    echo "== $name SKIPPED: GPUs would not drain, refusing to measure on a dirty box"
    return
  fi
  mkdir -p "$out"
  git rev-parse HEAD > "$out/git_commit.txt"
  cp "$cfg" "$out/"
  echo "== $name  ($(basename $cfg))  start $(date +%H:%M:%S)"
  uv run examples/run_sft.py --config "$cfg" \
      logger.log_dir="$out/logs" logger.wandb_enabled=False \
      logger.tensorboard_enabled=True logger.monitor_gpus=True \
      checkpointing.enabled=False checkpointing.checkpoint_dir="$out/ckpt" \
      > "$out/run.log" 2>&1
  rc=$?
  echo "   exit=$rc  steps=$(grep -c 'Total step time' "$out/run.log" 2>/dev/null)  end $(date +%H:%M:%S)"
  # Record whether more than one process was on the device, so a contaminated OOM is
  # never mistaken for a real memory result.
  if grep -q "out of memory" "$out/run.log" 2>/dev/null; then
    echo "   OOM detail: $(grep -o 'Process [0-9]* has [0-9.]* GiB memory in use' "$out/run.log" | sort -u | tr '\n' ' ')"
  fi
}

run dep-A0-repeat    $C/sft-nanov3-cascade2chat-8k-best-1n8g.yaml
run dep-A1-async     $C/dep-A1-async.yaml
run dep-A2-sms8      $C/dep-A2-sms8.yaml
run dep-A3-sms32     $C/dep-A3-sms32.yaml
run dep-A4-torchdisp $C/dep-A4-torchdisp.yaml
run dep-A5-gmm       $C/dep-A5-gmm.yaml
bash ~/assn1-f/analysis/gpu_cleanup.sh
echo "== sweep done $(date +%H:%M:%S)"
