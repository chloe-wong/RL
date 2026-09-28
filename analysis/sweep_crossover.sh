#!/bin/bash
# Track A and Track B share only two optimizations (the allocator pair, and fused AdamW).
# Track B additionally has chunked cross-entropy and bf16 FSDP output, and bf16 FSDP
# output is the largest kernel-level win in the whole project (+67% MFU on Track B).
# Neither has ever been tried on Track A. This closes that gap.
set -uo pipefail
run() {
    local name=$1; shift
    echo "############## $name : $* ##############"
    date -u +"start %H:%M:%S UTC"
    EXP_NAME=sft-nanov3-$name CONFIG=examples/configs/recipes/llm/sft-nanov3-collectives-fused.yaml \
        bash ~/assn1-f/analysis/run_collectives.sh "$@" > /mnt/scratch/chloe_wong/logs/$name.out 2>&1
    local rc=$?
    date -u +"end   %H:%M:%S UTC"
    if [ $rc -eq 0 ]; then
        L=/mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log
        printf "%s OK  " "$name"
        grep "Total step time" $L | awk '{print $5}' | tr -d s | tail -n +2 | sort -n \
          | awk '{a[NR]=$1} END{printf "median=%.3f p90=%.3f  ", a[int(NR/2)], a[int(NR*0.9)]}'
        grep "Floating Point Utilization" $L | awk '{print $7}' | tr -d '%' | sort -n \
          | awk '{a[NR]=$1} END{printf "MFU=%.3f%%  ", a[int(NR/2)]}'
        grep "Training FLOPS" $L | awk '{print $4}' | sort -n | awk '{a[NR]=$1} END{printf "TFLOPS=%.1f\n", a[int(NR/2)]}'
    else
        echo "$name FAILED rc=$rc:"
        grep -E "ValueError|RuntimeError|OutOfMemory|KeyError|ConfigAttribute" \
            /mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log 2>/dev/null | grep -v repeated | head -3 | cut -c1-160
    fi
    echo
}
run K1-bf16out  "+policy.dtensor_cfg.fsdp_output_dtype=bfloat16"
run K2-bf16-chunk "+policy.dtensor_cfg.fsdp_output_dtype=bfloat16" "+policy.logprob_chunk_size=1024"
echo SWEEP_DONE
