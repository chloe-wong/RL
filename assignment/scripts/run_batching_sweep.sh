#!/bin/bash
# Batching sweep: does filling the batch fix the under-fill bottleneck?
#   B  packing on, gbs 16   -- no change to training semantics (same 16 seqs, same gradient)
#   C  packing on, gbs 64   -- workload change, 4x data per update
#   D  packing on, gbs 128  -- workload change, 8x data per update
# Run A (baseline, no packing, gbs 16) is already recorded as sft-nanov3-baseline-mfu.
#
# Each run is 50 steps with validation and checkpointing off, matching the baseline
# exactly so the numbers are comparable. A failure (e.g. OOM at gbs 128) is logged and
# the sweep continues.
set -uo pipefail

run() {
    local name=$1 gbs=$2
    local exp=sft-nanov3-$name
    local out=/mnt/scratch/chloe_wong/runs/$exp
    echo "############## $name : packing=true gbs=$gbs ##############"
    date -u +"start %H:%M:%S UTC"
    EXP_NAME=$exp bash ~/assn1-f/run_sft_nanov3_baseline.sh \
        sft.max_num_steps=50 \
        sft.val_at_start=false \
        sft.val_period=1000 \
        policy.sequence_packing.enabled=true \
        policy.train_global_batch_size="$gbs" \
        > /mnt/scratch/chloe_wong/logs/$name.out 2>&1
    local rc=$?
    date -u +"end   %H:%M:%S UTC"
    if [ $rc -eq 0 ]; then
        echo "$name OK  -> $out"
    else
        echo "$name FAILED rc=$rc (continuing) -- last lines:"
        tail -20 "$out/run.log" 2>/dev/null | cut -c1-160
    fi
    echo
}

run B-pack-gbs16  16
run C-pack-gbs64  64
run D-pack-gbs128 128
echo "SWEEP_DONE"
