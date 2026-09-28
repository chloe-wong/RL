#!/bin/bash
# I1 (cuda_graph=[moe_router]) and I5 (dispatcher_async_dispatch) each cut p90 from
# 1.560s to ~1.45s without moving the median. Both reduce launch jitter rather than
# work, which is what the skew analysis predicts should help the tail. Do they compound?
#
# J0 repeats the fused baseline unchanged: p90 is a tail statistic and noisier than
# median, so it needs its own noise floor before a 7% change means anything.
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
          | awk '{a[NR]=$1} END{printf "median=%.3f p90=%.3f p99=%.3f min=%.3f  ", a[int(NR/2)], a[int(NR*0.9)], a[int(NR*0.99)], a[1]}'
        grep "Floating Point Utilization" $L | awk '{print $7}' | tr -d '%' | sort -n \
          | awk '{a[NR]=$1} END{printf "MFU=%.3f%%\n", a[int(NR/2)]}'
    else
        echo "$name FAILED rc=$rc:"
        grep -E "ValueError|RuntimeError|OutOfMemory" /mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log 2>/dev/null \
            | grep -v repeated | head -2 | cut -c1-160
    fi
    echo
}
run J0-fused-repeat
run J1-cg-async   "+$B.cuda_graph.modules=[moe_router]" "+$B.dispatcher_async_dispatch=true"
run J2-cg-async-sms "+$B.cuda_graph.modules=[moe_router]" "+$B.dispatcher_async_dispatch=true" "+$B.dispatcher_num_sms=8"
echo SWEEP_DONE
