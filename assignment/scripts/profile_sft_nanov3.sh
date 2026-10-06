#!/bin/bash
# Profile the Nano V3 baseline SFT run (same config as run_sft_nanov3_baseline.sh).
# Profiles training steps 4-5 (after warmup, before the step-10 validation).
#
# Usage: bash ~/assn1-f/profile_sft_nanov3.sh <mode>
#   nsys     Nsight Systems timeline per rank (NVTX, CUDA, cuBLAS, autograd ops),
#            plus a root nsys sampler for GPU metrics (SM, tensor core, NVLink).
#   torch    PyTorch profiler (Kineto) traces, for Perfetto and HTA.
#   memory   CUDA memory snapshot, for pytorch.org/memory_viz.
#   pyspy    py-spy (non-blocking) on one policy worker: host-side Python stacks.
#   metrics  nsys GPU metrics on an untraced run (needs sudo).
#   ncu      Nsight Compute on the compute kernels of step 4, all 8 policy workers
#            (needs sudo: this machine sets RmProfilingAdminOnly=1).
#
# Environment:
#   CONFIG       NeMo RL config to profile (default: the upstream baseline recipe).
#   PROFILE_TAG  Output subdirectory, e.g. "chat8k" -> profiles/chat8k/<mode>
#                (default: profiles/<mode>, the SQuAD baseline profiles).
#   ncu only: NCU_KERNEL_REGEX (kernels to profile), NCU_LAUNCH_COUNT (cap per rank),
#             NCU_SET (e.g. "full"; default: speed-of-light, occupancy, workload sections).
#   GPU_METRICS_DURATION  seconds of GPU-metrics sampling from step 1 (nsys: 150, metrics: 45).
set -euo pipefail
MODE=${1:?usage: profile_sft_nanov3.sh nsys|torch|memory|pyspy|metrics|ncu}
source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL

CONFIG=${CONFIG:-examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml}
OUT=/mnt/scratch/chloe_wong/profiles/${PROFILE_TAG:+$PROFILE_TAG/}$MODE
rm -rf "$OUT" && mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
git diff > "$OUT/git_diff.patch"
cp "$CONFIG" "$OUT/"
PY=$UV_PROJECT_ENVIRONMENT/bin/python

ARGS=(
    examples/run_sft.py
    --config "$CONFIG"
    cluster.num_nodes=1
    cluster.gpus_per_node=8
    sft.max_num_steps=6
    sft.val_at_start=False
    logger.log_dir="$OUT/logs"
    logger.wandb_enabled=False
    logger.tensorboard_enabled=True
    logger.monitor_gpus=False
    checkpointing.enabled=False
)

# Wait for a line in the run log, then run a command.
on_step() {
    local pattern=$1; shift
    until grep -q "$pattern" "$OUT/run.log" 2>/dev/null; do sleep 0.5; done
    "$@"
}

case $MODE in
nsys)
    export NRL_NSYS_WORKER_PATTERNS="*policy*"
    export NRL_NSYS_PROFILE_STEP_RANGE=4:6
    export NRL_NSYS_EXTRA_OPTIONS='{"t": "cuda,nvtx,cublas,cudnn", "pytorch": "autograd-nvtx"}'
    # GPU metrics need root here, and starting nsys as root takes ~9 s. Start the
    # sampler during step 1 (warmup) and run it long enough to cover traced steps 4-5.
    on_step "Step 1/" sudo $CUDA_HOME/bin/nsys profile --trace=none \
        --gpu-metrics-devices=all --gpu-metrics-frequency=5000 \
        --duration=${GPU_METRICS_DURATION:-150} -o "$OUT/gpu_metrics" --force-overwrite=true sleep ${GPU_METRICS_DURATION:-150} \
        > "$OUT/gpu_metrics.log" 2>&1 &
    $PY "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
    wait
    sudo chown -R "$USER": "$OUT"
    cp "$RAY_TMPDIR"/ray/session_latest/logs/nsight/*.nsys-rep "$OUT/"
    ;;
torch)
    # PyTorch profiler (Kineto) only. Traces open in Perfetto and feed HTA.
    export NRL_NSYS_WORKER_PATTERNS="__no_nsys__"  # Enable step-range hooks, attach no nsys.
    export NRL_NSYS_PROFILE_STEP_RANGE=4:6
    export NRL_TORCH_PROFILE_DIR=$OUT/kineto
    $PY "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
    ;;
memory)
    # CUDA caching-allocator history for steps 4-5 (open at pytorch.org/memory_viz).
    export NRL_NSYS_WORKER_PATTERNS="__no_nsys__"
    export NRL_NSYS_PROFILE_STEP_RANGE=4:6
    export NRL_MEMORY_SNAPSHOT_DIR=$OUT/memory
    $PY "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
    ;;
pyspy)
    # Non-blocking py-spy on one policy worker for 6 s from step 4 (host-side Python).
    on_step "Step 4/" bash -c 'pid=$(pgrep -f "ray::DTensorPolicyWorkerV2" | head -1)
        sudo $HOME/.local/bin/py-spy record --pid $pid --duration 6 --rate 100 \
            --nonblocking --format speedscope -o '"$OUT"'/pyspy_worker.speedscope.json' \
        > "$OUT/pyspy.log" 2>&1 &
    $PY "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
    wait
    sudo chown -R "$USER": "$OUT"
    ;;
metrics)
    # GPU metrics on an untraced run: SM, tensor core, DRAM, NVLink. Starting nsys as
    # root takes ~9 s, so start the sampler during step 1 (warmup, ~25 s) and log the
    # wall-clock start of each step so the analysis can clip to steps 4-5.
    on_step "Step 1/" sudo $CUDA_HOME/bin/nsys profile --trace=none \
        --gpu-metrics-devices=all --gpu-metrics-frequency=10000 \
        --duration=${GPU_METRICS_DURATION:-45} -o "$OUT/gpu_metrics" --force-overwrite=true sleep ${GPU_METRICS_DURATION:-45} \
        > "$OUT/gpu_metrics.log" 2>&1 &
    ( for n in 2 3 4 5 6; do on_step "Step $n/" true; echo "$n $(date +%s.%N)"; done
      on_step "Max number of steps" true; echo "end $(date +%s.%N)" ) > "$OUT/step_times.txt" &
    $PY "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
    wait
    sudo chown -R "$USER": "$OUT"
    ;;
ncu)
    # Run only the 8 policy workers under ncu (wrapping the whole Ray tree stalls
    # cuInit in Ray's helper processes). All ranks are profiled so they slow down
    # together and NCCL/DeepEP waits stay short. Skip NCCL and DeepEP kernels:
    # replaying them would deadlock their peers.
    export NRL_NSYS_WORKER_PATTERNS="__no_nsys__"
    export NRL_NSYS_PROFILE_STEP_RANGE=4:5
    NCU_KERNEL_REGEX=${NCU_KERNEL_REGEX:-'^(?!.*(nccl|deep_ep|dispatch|combine|notify)).*$'}
    if [ -n "${NCU_SET:-}" ]; then
        NCU_SECTIONS="--set $NCU_SET"
    else
        NCU_SECTIONS="--section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats \
--section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis"
    fi
    export NRL_WORKER_PY_PREFIX_PATTERN="lm_policy-*"
    export NRL_WORKER_PY_PREFIX="sudo -E env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
$CUDA_HOME/bin/ncu --target-processes application-only --profile-from-start off \
--kernel-name 'regex:$NCU_KERNEL_REGEX' --launch-count ${NCU_LAUNCH_COUNT:-2500} \
$NCU_SECTIONS --import-source no -o $OUT/ncu_rank%q{RANK} --"
    $PY "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
    sudo chown -R "$USER": "$OUT"
    ;;
*)
    echo "unknown mode: $MODE" >&2
    exit 1
    ;;
esac
echo "PROFILE_DONE $MODE -> $OUT"
