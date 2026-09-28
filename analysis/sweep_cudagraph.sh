#!/bin/bash
# Fused AdamW removed 5,968 optimizer kernels and 756 device syncs per step, but
# rank 0's GPU is still idle 68% of the span across 27,879 gaps averaging 87us.
# The step is host-launch-bound end to end, not just in the optimizer.
#
# CUDA graphs capture a module's launches once and replay them as a single launch.
# Automodel exposes scoped capture for attn / te_dpa / moe_router / moe_preprocess.
# This model is 23 MoE + 6 attention layers, so the MoE scopes are where the small
# ops are. All arms run on top of fused AdamW, so the win is measured incrementally.
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
    if [ $rc -eq 0 ]; then echo "$name OK"; else
        echo "$name FAILED rc=$rc -- reason:"
        grep -E "Error|error:|Unsupported|assert" /mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log 2>/dev/null \
            | grep -v "stderr\|monitor" | tail -6 | cut -c1-170
    fi
    echo
}
run H1-cg-moe      "+$B.cuda_graph.modules=[moe_router,moe_preprocess]"
run H2-cg-moe-attn "+$B.cuda_graph.modules=[attn,moe_router,moe_preprocess]"
echo SWEEP_DONE
