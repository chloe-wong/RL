#!/bin/bash
# The MoE token dispatcher is the all-to-all that sends each token to the GPU holding
# its expert. Measured: 280 ms of DeepEP's 326 ms is notify/handshake metadata, not
# token movement, at ~1 ms per call for a few hundred tokens per rank.
#
# It also gates CUDA graphs: cuda_graph.modules=[moe_preprocess] requires
# dispatcher='hybridep', and [attn] requires linear='torch'. So the dispatcher choice
# and the launch-starvation fix are the same decision.
#
# All arms sit on top of fused AdamW so each win is measured incrementally.
set -uo pipefail
B=policy.dtensor_cfg.automodel_kwargs.backend
run() {
    local name=$1; shift
    echo "############## $name : $* ##############"
    date -u +"start %H:%M:%S UTC"
    EXP_NAME=sft-nanov3-$name CONFIG=examples/configs/recipes/llm/sft-nanov3-collectives-fused.yaml \
        bash ~/assn1-f/analysis/run_collectives.sh "$@" \
        > /mnt/scratch/chloe_wong/logs/$name.out 2>&1
    local rc=$?
    date -u +"end   %H:%M:%S UTC"
    if [ $rc -eq 0 ]; then
        L=/mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log
        printf "%s OK  " "$name"
        grep "Total step time" $L | awk '{print $5}' | tr -d s | tail -n +2 | sort -n \
          | awk '{a[NR]=$1} END{printf "median=%.3fs p90=%.3fs  ", a[int(NR/2)], a[int(NR*0.9)]}'
        grep "Floating Point Utilization" $L | awk '{print $7}' | tr -d '%' | sort -n \
          | awk '{a[NR]=$1} END{printf "MFU=%.3f%%\n", a[int(NR/2)]}'
    else
        echo "$name FAILED rc=$rc -- reason:"
        grep -E "ValueError|RuntimeError|OutOfMemory|NotImplemented|ImportError" \
            /mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log 2>/dev/null \
            | grep -v "repeated" | head -3 | cut -c1-165
    fi
    echo
}
run I1-cg-router  "+$B.cuda_graph.modules=[moe_router]"
run I2-hybridep   "+$B.dispatcher=hybridep"
run I3-hybrid-cg  "+$B.dispatcher=hybridep" "+$B.cuda_graph.modules=[moe_router,moe_preprocess]"
run I4-disp-torch "+$B.dispatcher=torch"
run I5-async-disp "+$B.dispatcher_async_dispatch=true"
echo SWEEP_DONE
